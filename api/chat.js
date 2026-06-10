import { pipeline, env } from '@xenova/transformers';
import { Pinecone } from '@pinecone-database/pinecone';
import { Redis } from '@upstash/redis';
import { Ratelimit } from '@upstash/ratelimit';

// Configure transformers to use /tmp for caching models on Vercel Serverless
env.cacheDir = '/tmp';
// Disable local models to ensure it fetches from HF Hub
env.allowLocalModels = false;

// Initialize Upstash Redis & Ratelimiter
const redis = new Redis({
  url: process.env.CodeMateRAG_KV_REST_API_URL,
  token: process.env.CodeMateRAG_KV_REST_API_TOKEN,
});

// Create a sliding window ratelimiter (30 requests per minute per IP)
const ratelimit = new Ratelimit({
  redis: redis,
  limiter: Ratelimit.slidingWindow(30, "1 m"),
  prefix: "@upstash/ratelimit/v2"
});

// Cache the embedding pipeline outside the handler for warm starts
let generateEmbedding = null;

// Initialize Pinecone
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});
const indexName = process.env.PINECONE_INDEX_NAME || 'codemate-index';
const index = pc.Index(indexName);

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    // 1. Rate Limiting
    const ip = req.headers['x-forwarded-for'] || req.socket?.remoteAddress || '127.0.0.1';
    const { success, limit, remaining, reset } = await ratelimit.limit(ip);
    
    // Set headers for rate limit info
    res.setHeader('X-RateLimit-Limit', limit);
    res.setHeader('X-RateLimit-Remaining', remaining);
    res.setHeader('X-RateLimit-Reset', reset);

    if (!success) {
      return res.status(429).json({ error: 'Too many requests. Please try again later.' });
    }

    // 2. Parse User Query
    const { message } = req.body;
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    // 3. Generate Embedding
    if (!generateEmbedding) {
      generateEmbedding = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }
    const output = await generateEmbedding(message, { pooling: 'mean', normalize: true });
    const embeddingArray = Array.from(output.data);

    // 4. Query Pinecone
    const queryResponse = await index.query({
      vector: embeddingArray,
      topK: 6,
      includeMetadata: true
    });
    
    const context = queryResponse.matches.map(m => m.metadata.text).join('\n\n');

    // 5. Query OpenRouter API
    const systemPrompt = `You are a helpful and professional assistant for CodeMate, a tech community. 
Your goal is to answer questions about CodeMate based strictly on the provided context. 
If the answer is not in the context, politely say that you don't have that information.
Keep your answers concise, professional, and friendly.
CRITICAL INSTRUCTION 1: Do NOT start your response with "Hello!", "Hi!", or any similar greetings. Just provide the answer directly.
CRITICAL INSTRUCTION 2: You represent CodeMate. Always use "we", "us", and "our" when referring to CodeMate (e.g., "our contact section", not "their website").

Context:
${context}`;

    const openRouterResponse = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "google/gemma-4-31b-it:free",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: message }
        ]
      })
    });

    if (!openRouterResponse.ok) {
      const errText = await openRouterResponse.text();
      throw new Error(`OpenRouter API error: ${openRouterResponse.status} ${errText}`);
    }

    const data = await openRouterResponse.json();
    const answer = data.choices[0].message.content;

    return res.status(200).json({ answer });

  } catch (error) {
    console.error('Chat API Error:', error);
    return res.status(500).json({ error: 'Internal Server Error' });
  }
}
