# SheBots Backend (FastAPI + transformers)

Run a streaming SSE chat endpoint backed by a local transformers model.

Note: this repository now defaults to the Qwen chat model (`Qwen/Qwen1.5-1.8B-Chat`).
To run that model locally you will need a CUDA-enabled GPU, recent `transformers` and `accelerate`,
and a valid Hugging Face token if the model requires authentication. Override `MODEL_ID` in
`.env` to use a different model (for local CPU-friendly testing, set `MODEL_ID=distilgpt2`).

Setup

1. Create and activate a virtualenv with Python 3.10+.
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
2. Install requirements:
   pip install -r requirements.txt
3. Copy `.env.example` to `.env` and edit if needed.

Run (development):
   uvicorn main:app --reload --port 8000

Notes

- The model specified by MODEL_ID can be large. CPU inference is slow. Use a quantized or smaller model for local runs.
- SSE streaming endpoint: POST /api/chat (Content-Type: application/json). Accepts {message, history?, useDocs?} and returns text/event-stream.

Curl example (no RAG):
curl -N -H "Content-Type: application/json" -d '{"message":"Hello"}' <http://localhost:8000/api/chat>

SheBots Chatbot API

A production-ready chatbot API with Server-Sent Events (SSE) streaming, built with Node.js and Express.

## Features

- 🚀 **SSE Streaming**: Real-time streaming of assistant responses
- 🔒 **Security**: CORS protection, rate limiting, helmet security headers
- 🛡️ **PII Protection**: Automatic redaction of emails and phone numbers
- ⚡ **Performance**: Efficient streaming with minimal latency
- 🔧 **Configurable**: Environment-based configuration
- 📊 **Monitoring**: Health endpoint and structured logging

## Quick Start

### Prerequisites

- Node.js >= 18
- OpenAI API key

### Installation

1. **Clone and setup**:

   ```bash
   git clone <repository-url>
   cd Backend
   npm install
   ```

2. **Configure environment**:

   ```bash
   cp .env.example .env
   ```

3. **Edit `.env`** and set your OpenAI API key:

   ```env
   MODEL_API_KEY=sk-your-openai-api-key-here
   ```

4. **Start the server**:

   ```bash
   npm start
   ```

The API will be available at `http://localhost:3000`

## API Endpoints

### GET /api/health

Returns server status and current configuration.

**Response**:

```json
{
  "ok": true,
  "provider": "openai",
  "model": "gpt-4o-mini"
}
```

**Example**:

```bash
curl http://localhost:3000/api/health
```

### POST /api/chat

Streams chat completions via Server-Sent Events.

**Request Body**:

```json
{
  "message": "Hello, how are you?",
  "history": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user", 
      "content": "Previous message"
    },
    {
      "role": "assistant",
      "content": "Previous response"
    }
  ]
}
```

**Request Validation**:

- `message`: Required string, max 2000 characters
- `history`: Optional array of message objects with `role` and `content`
- Roles must be: `system`, `user`, or `assistant`

**SSE Response Events**:

1. **Start Event**:

   ```text
   event: start
   data: {"provider":"openai","model":"gpt-4o-mini"}
   ```

2. **Token Events** (multiple):

   ```text
   data: {"token":"Hello"}
   data: {"token":" there"}
   data: {"token":"!"}
   ```

3. **End Event**:

   ```text
   event: end
   data: {"tokensStreamed":15}
   ```

4. **Error Event** (if needed):

   ```text
   event: error
   data: {"error":"Error message"}
   ```

**Example with curl**:

```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "message": "Tell me a short joke",
    "history": []
  }'
```

**Example with JavaScript**:

```javascript
const eventSource = new EventSource('http://localhost:3000/api/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: 'Hello!',
    history: []
  })
});

eventSource.addEventListener('start', (event) => {
  const data = JSON.parse(event.data);
  console.log('Started:', data);
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Token:', data.token);
};

eventSource.addEventListener('end', (event) => {
  const data = JSON.parse(event.data);
  console.log('Completed, tokens:', data.tokensStreamed);
  eventSource.close();
});

eventSource.addEventListener('error', (event) => {
  const data = JSON.parse(event.data);
  console.error('Error:', data.error);
  eventSource.close();
});
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Server port |
| `ALLOWED_ORIGIN` | `http://localhost:5173` | CORS allowed origin |
| `PROVIDER` | `openai` | LLM provider name |
| `MODEL_NAME` | `gpt-4o-mini` | Model to use |
| `MODEL_API_KEY` | *(required)* | OpenAI API key |

### Security Features

- **CORS**: Restricts access to specified origin
- **Rate Limiting**: 30 requests per minute per IP for `/api/*`
- **Helmet**: Security headers with CSP disabled for SSE
- **PII Redaction**: Automatic removal of emails and phone numbers
- **Input Validation**: Message length and format validation

## Error Handling

### Common Errors

| Status | Error | Solution |
|--------|-------|----------|
| `400` | Invalid message format | Check message is non-empty string ≤ 2000 chars |
| `413` | Message too long | Reduce message to ≤ 2000 characters |
| `429` | Rate limit exceeded | Wait and retry (max 30 req/min) |
| `500` | API key not configured | Set `MODEL_API_KEY` in `.env` |
| `500` | Provider error | Check API key validity and OpenAI service status |

### CORS Issues

If you get CORS errors:

1. **Check origin**: Ensure your frontend runs on the `ALLOWED_ORIGIN` URL
2. **Update config**: Modify `ALLOWED_ORIGIN` in `.env` to match your frontend
3. **Restart server**: Changes require server restart

Example for different origins:

```env
# For production
ALLOWED_ORIGIN=https://yourdomain.com

# For development on different port
ALLOWED_ORIGIN=http://localhost:3000

# For multiple origins, modify server.js cors config
```

## Development

### Scripts

- `npm start`: Production server
- `npm run dev`: Development server (same as start, with NODE_ENV=development)

### Project Structure

```text
Backend/
├── src/
│   └── providers/
│       └── openaiAdapter.js    # OpenAI streaming implementation
├── server.js                   # Main Express server
├── package.json               # Dependencies and scripts
├── .env.example              # Environment template
└── README.md                 # This file
```

### Adding New Providers

1. Create adapter in `src/providers/`
2. Implement `streamChat` async generator function
3. Update server.js to use new provider
4. Add provider-specific environment variables

## Monitoring

### Logs

The server logs minimal metadata (no PII):

```text
[2024-01-01T00:00:00.000Z] Chat request - Provider: openai, Model: gpt-4o-mini, Messages: 3
[2024-01-01T00:00:05.000Z] Chat completed - Tokens streamed: 25
```

### Health Checks

Monitor server health:

```bash
curl http://localhost:3000/api/health
```

Expected response indicates all systems operational:

```json
{
  "ok": true,
  "provider": "openai", 
  "model": "gpt-4o-mini"
}
```

## Troubleshooting

### Server Won't Start

- Check Node.js version: `node --version` (requires >=18)
- Check port availability: `netstat -an | findstr :3000`
- Review error logs for specific issues

### No Streaming Response

- Verify API key is set and valid
- Check network connectivity to OpenAI
- Ensure Content-Type is `text/event-stream`

### Frontend Integration Issues

- Verify CORS origin matches exactly
- Use EventSource API for SSE
- Handle all event types (start, data, end, error)

## License

MIT License - see LICENSE file for details.
