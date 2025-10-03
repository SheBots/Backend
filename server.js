import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import { streamChat as streamChatOpenAI } from './src/providers/openaiAdapter.js';
import { streamChat as streamChatGemini } from './src/providers/geminiAdapter.js';
import fetch from 'node-fetch';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || 'http://localhost:5173';
const PROVIDER = process.env.PROVIDER || 'openai';
const MODEL_NAME = process.env.MODEL_NAME || 'gpt-4o-mini';
const MODEL_API_KEY = process.env.MODEL_API_KEY;

// PII redaction patterns
const EMAIL_REGEX = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g;
const PHONE_REGEX = /(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/g;

/**
 * Get the streaming function for the configured provider
 */
function getStreamingProvider(provider) {
  switch (provider.toLowerCase()) {
    case 'openai':
      return streamChatOpenAI;
    case 'gemini':
      return streamChatGemini;
    default:
      throw new Error(`Unsupported provider: ${provider}. Supported providers: openai, gemini`);
  }
}

/**
 * Redact PII from text content
 */
function redactPII(text) {
  return text
    .replace(EMAIL_REGEX, '[redacted-email]')
    .replace(PHONE_REGEX, '[redacted-phone]');
}

/**
 * Validate and normalize chat messages
 */
function validateChatRequest(body) {
  const { message, history = [] } = body;

  // Validate message
  if (!message || typeof message !== 'string') {
    throw new Error('Message is required and must be a string');
  }

  const trimmedMessage = message.trim();
  if (trimmedMessage.length === 0) {
    throw new Error('Message cannot be empty');
  }

  if (trimmedMessage.length > 2000) {
    throw new Error('Message exceeds maximum length of 2000 characters');
  }

  // Validate and normalize history
  if (!Array.isArray(history)) {
    throw new Error('History must be an array');
  }

  const normalizedHistory = history.map((msg, index) => {
    if (!msg || typeof msg !== 'object') {
      throw new Error(`History item ${index} must be an object`);
    }

    const { role, content } = msg;
    
    if (!role || !['system', 'user', 'assistant'].includes(role)) {
      throw new Error(`History item ${index} must have a valid role (system, user, or assistant)`);
    }

    if (!content || typeof content !== 'string') {
      throw new Error(`History item ${index} must have content as a string`);
    }

    return { role, content: content.trim() };
  });

  return {
    message: trimmedMessage,
    history: normalizedHistory
  };
}

// Security middleware
app.use(helmet({
  contentSecurityPolicy: false // Disable CSP to allow SSE
}));

// CORS configuration
app.use(cors({
  origin: ALLOWED_ORIGIN,
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 30, // 30 requests per minute per IP
  message: { error: 'Too many requests, please try again later' },
  standardHeaders: true,
  legacyHeaders: false
});

app.use('/api', limiter);

// Body parser
app.use(express.json({ limit: '1mb' }));

// RAG routes moved to separate RagService; Backend will call RAG via HTTP using RAG_URL/RAG_API_KEY

// Health endpoint
app.get('/api/health', (req, res) => {
  res.json({
    ok: true,
    provider: PROVIDER,
    model: MODEL_NAME
  });
});

// Chat endpoint with SSE streaming
app.post('/api/chat', async (req, res) => {
  try {
    // Check if API key is configured
    if (!MODEL_API_KEY) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
        'Access-Control-Allow-Credentials': 'true'
      });

      res.write(`event: error\n`);
      res.write(`data: ${JSON.stringify({ error: 'API key not configured. Please set MODEL_API_KEY environment variable.' })}\n\n`);
      res.end();
      return;
    }

    // Validate request
    const { message, history } = validateChatRequest(req.body);
    const useDocs = Boolean(req.body.useDocs === true || String(req.body.useDocs) === 'true');

    // If RAG is enabled and useDocs=true, call the RAG service HTTP API to get top snippets
    let preamble = null;
    if (useDocs && (process.env.RAG_ENABLED || 'true') === 'true') {
      try {
        const k = Number(process.env.RAG_TOP_K || 5);
        const ragUrl = process.env.RAG_URL || 'http://localhost:3001';
        const apiKey = process.env.RAG_API_KEY;
        const url = `${ragUrl.replace(/\/$/, '')}/api/search?q=${encodeURIComponent(message)}&k=${k}`;
        const headers = { 'Accept': 'application/json' };
        if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;
        const resp = await fetch(url, { headers, timeout: 5000 });
        if (resp.ok) {
          const json = await resp.json();
          const results = json.results || [];
          if (results.length > 0) {
            const snippets = results.map(r => {
              const sanitized = String(r.text || '').replace(EMAIL_REGEX, '[redacted-email]').replace(PHONE_REGEX, '[redacted-phone]');
              return `----\n${sanitized}\nSource: [${r.title || r.url}](${r.url})\n`;
            }).join('\n');

            preamble = `You are a helpful assistant. Prefer these snippets if relevant.\nCite as [Title](URL). If unsure, say so.\n\n${snippets}\n`;
          }
        } else {
          console.warn('RAG service responded with', resp.status);
        }
      } catch (e) {
        console.error('RAG service error', e.message);
      }
    }

    // Set SSE headers
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
      'Access-Control-Allow-Credentials': 'true'
    });

    // Build messages array with PII redaction
    const messages = [
      ...history.map(msg => ({
        role: msg.role,
        content: redactPII(msg.content)
      }))
    ];

    // If we built a preamble, inject it as a system message to bias the model
    if (preamble) {
      messages.unshift({ role: 'system', content: preamble });
    }

    messages.push({ role: 'user', content: redactPII(message) });

    // Send start event
    res.write(`event: start\n`);
    res.write(`data: ${JSON.stringify({ provider: PROVIDER, model: MODEL_NAME })}\n\n`);

    // Log request (no PII)
    console.log(`[${new Date().toISOString()}] Chat request - Provider: ${PROVIDER}, Model: ${MODEL_NAME}, Messages: ${messages.length}`);

    let tokensStreamed = 0;

    try {
      // Get the appropriate streaming provider
      const streamChat = getStreamingProvider(PROVIDER);
      
      // Stream response
      for await (const token of streamChat({
        messages,
        model: MODEL_NAME,
        apiKey: MODEL_API_KEY,
        temperature: 0.2
      })) {
        res.write(`data: ${JSON.stringify({ token })}\n\n`);
        tokensStreamed++;
      }

      // Send end event
      res.write(`event: end\n`);
      res.write(`data: ${JSON.stringify({ tokensStreamed })}\n\n`);
      
      console.log(`[${new Date().toISOString()}] Chat completed - Tokens streamed: ${tokensStreamed}`);
      
    } catch (streamError) {
      console.error('Streaming error:', streamError.message);
      
      res.write(`event: error\n`);
      res.write(`data: ${JSON.stringify({ error: `Provider error: ${streamError.message}` })}\n\n`);
    }

    res.end();

  } catch (validationError) {
    // Handle validation errors
    const statusCode = validationError.message.includes('exceeds maximum length') ? 413 : 400;
    
    res.writeHead(statusCode, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
      'Access-Control-Allow-Credentials': 'true'
    });

    res.write(`event: error\n`);
    res.write(`data: ${JSON.stringify({ error: validationError.message })}\n\n`);
    res.end();
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  
  if (res.headersSent) {
    return next(err);
  }
  
  res.status(500).json({
    error: 'Internal server error'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Endpoint not found'
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  process.exit(0);
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Chatbot API server running on http://localhost:${PORT}`);
  console.log(`📡 Provider: ${PROVIDER}, Model: ${MODEL_NAME}`);
  console.log(`🔒 CORS allowed origin: ${ALLOWED_ORIGIN}`);
  console.log(`🔑 API key configured: ${MODEL_API_KEY ? '✅' : '❌'}`);
});