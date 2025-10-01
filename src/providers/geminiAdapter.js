/**
 * Google Gemini Chat Completions streaming adapter
 * Uses direct HTTP API calls to Gemini REST API
 */

/**
 * Stream chat completions from Google Gemini using REST API
 * @param {Object} params - Parameters for the chat completion
 * @param {Array} params.messages - Array of message objects
 * @param {string} params.model - Model name (e.g., 'gemini-2.5-flash')
 * @param {string} params.apiKey - Gemini API key
 * @param {number} [params.temperature=0.2] - Temperature for response generation
 * @returns {AsyncIterable<string>} - Async generator yielding content chunks
 */
export async function* streamChat({ messages, model, apiKey, temperature = 0.2 }) {
  try {
    // Convert messages to simple text prompt
    const prompt = convertMessagesToGeminiFormat(messages);

    // Prepare the request body for Gemini REST API
    const requestBody = {
      contents: [
        {
          parts: [
            {
              text: prompt
            }
          ]
        }
      ],
      generationConfig: {
        temperature: temperature,
        maxOutputTokens: 2048,
      }
    };

    // Make direct HTTP request to Gemini API
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Gemini API error (${response.status}): ${errorText}`);
    }

    const data = await response.json();
    
    // Extract the generated text
    const generatedText = data.candidates?.[0]?.content?.parts?.[0]?.text;
    
    if (generatedText) {
      // Split into words to simulate streaming
      const words = generatedText.split(' ');
      for (let i = 0; i < words.length; i++) {
        if (i === 0) {
          yield words[i];
        } else {
          yield ' ' + words[i];
        }
        // Small delay to simulate streaming
        await new Promise(resolve => setTimeout(resolve, 30));
      }
    } else {
      throw new Error('No text generated from Gemini API');
    }

  } catch (error) {
    console.error('Gemini API error:', error);
    throw new Error(`Gemini API error: ${error.message}`);
  }
}

/**
 * Convert OpenAI-style messages to Gemini format
 * @param {Array} messages - Array of message objects with role and content
 * @returns {string} - Combined content for simple text generation
 */
function convertMessagesToGeminiFormat(messages) {
  let combinedContent = '';
  
  for (const message of messages) {
    const { role, content } = message;
    
    switch (role) {
      case 'system':
        combinedContent += `System: ${content}\n\n`;
        break;
      case 'user':
        combinedContent += `User: ${content}\n\n`;
        break;
      case 'assistant':
        combinedContent += `Assistant: ${content}\n\n`;
        break;
      default:
        combinedContent += `${content}\n\n`;
    }
  }
  
  // Add a prompt for the assistant to respond
  combinedContent += 'Assistant: ';
  
  return combinedContent.trim();
}