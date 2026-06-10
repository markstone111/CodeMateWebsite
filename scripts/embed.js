import { pipeline } from '@xenova/transformers';
import { Pinecone } from '@pinecone-database/pinecone';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.join(__dirname, '../.env') });

async function main() {
    console.log("Initializing Pinecone...");
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY
    });
    const indexName = process.env.PINECONE_INDEX_NAME || 'codemate-index';
    const index = pc.Index(indexName);

    console.log("Loading embedding model...");
    // all-MiniLM-L6-v2 is a great lightweight model (384 dimensions)
    const generateEmbedding = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

    console.log("Reading knowledge base...");
    const kbPath = path.join(__dirname, '../knowledge/codemate_knowledge.md');
    const content = fs.readFileSync(kbPath, 'utf8');

    // Improved chunking strategy: group paragraphs to preserve context (~500 chars max)
    const rawPieces = content.split('\n\n').map(c => c.trim()).filter(c => c.length > 0);
    const chunks = [];
    let currentChunk = "";
    
    for (const piece of rawPieces) {
        if (currentChunk.length + piece.length > 600) {
            if (currentChunk) chunks.push(currentChunk.trim());
            currentChunk = piece;
        } else {
            currentChunk += (currentChunk ? "\n\n" : "") + piece;
        }
    }
    if (currentChunk) chunks.push(currentChunk.trim());

    console.log(`Created ${chunks.length} chunks. Generating embeddings...`);
    
    const vectors = [];
    for (let i = 0; i < chunks.length; i++) {
        const text = chunks[i];
        
        // Generate embedding
        const output = await generateEmbedding(text, { pooling: 'mean', normalize: true });
        // The output is a tensor, we need an array of numbers
        const embeddingArray = Array.from(output.data);
        
        vectors.push({
            id: `chunk-${i}`,
            values: embeddingArray,
            metadata: {
                text: text
            }
        });
    }

    console.log(`Upserting ${vectors.length} vectors to Pinecone...`);
    if (vectors.length > 0) {
        console.log(`Clearing existing vectors from index...`);
        try {
            await index.deleteAll();
        } catch (e) {
            console.log("No existing vectors to delete or index already empty.");
        }
        
        console.log(`Sample vector ID: ${vectors[0].id}, Dimensions: ${vectors[0].values.length}`);
        
        // Pinecone v7 requires the { records: [...] } format
        await index.upsert({records: vectors});
    } else {
        console.log("No vectors to upsert.");
    }

    console.log("Successfully embedded and uploaded all chunks!");
}

main().catch(console.error);
