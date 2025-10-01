/**
 * OpenAI Chat Completions streaming adapter
 * Provides async generator for streaming chat responses
 */

/**
 * Stream chat completions from OpenAI
 * @param {Object} params - Parameters for the chat completion
 * @param {Array} params.messages - Array of message objects
 * @param {string} params.model - Model name (e.g., 'gpt-4o-mini')
 * @param {string} params.apiKey - OpenAI API key
 * @param {number} [params.temperature=0.2] - Temperature for response generation
 * @returns {AsyncIterable<string>} - Async generator yielding content chunks
 */
export async function* streamChat({ messages, model, apiKey, temperature = 0.2 }) {
  const url = 'https://api.openai.com/v1/chat/completions';
  
  const requestBody = {
    model,
    messages,
    stream: true,
    temperature
  };

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI API error (${response.status}): ${errorText}`);
  }

  if (!response.body) {
    throw new Error('No response body received from OpenAI API');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      
      // Keep the last incomplete line in the buffer
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        const trimmedLine = line.trim();
        
        if (trimmedLine === '') continue;
        if (trimmedLine === 'data: [DONE]') return;
        
        if (trimmedLine.startsWith('data: ')) {
          try {
            const jsonStr = trimmedLine.slice(6);
            const data = JSON.parse(jsonStr);
            
            if (data.choices?.[0]?.delta?.content) {
              yield data.choices[0].delta.content;
            }
          } catch (parseError) {
            // Skip malformed JSON lines
            console.warn('Failed to parse SSE data:', trimmedLine);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}