import fitz
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

API_PROVIDERS = {
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key": os.getenv('API_KEY'),  
        "model": "llama-3.3-70b"},
    "huggingface": {
        "base_url": "https://api-inference.huggingface.co/v1",
        "api_key": "hf_YOUR_TOKEN_HERE",  
        "model": "microsoft/DialoGPT-medium"
    },
    "openai_free": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-YOUR_OPENAI_KEY_HERE", 
        "model": "gpt-3.5-turbo"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",  
        "api_key": "ollama",  
        "model": "llama2"
    }
}

CURRENT_PROVIDER = "cerebras"  

provider_config = API_PROVIDERS[CURRENT_PROVIDER]
client = OpenAI(
    base_url=provider_config["base_url"],
    api_key=provider_config["api_key"]
)

app = Flask(__name__)
CORS(app)

model = SentenceTransformer('all-MiniLM-L6-v2')

class RAGSystem:
    def __init__(self):
        self.index = None
        self.chunks = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_text_from_pdf(self, path):
        all_text = ""
        try:
            doc = fitz.open(path)
            for page in doc:
                text = page.get_text()
                all_text += text + "\n"
            doc.close()
            return all_text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None
    
    def chunk_text(self, pdf_text, chunk_size=300, overlap=50):
        words = pdf_text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i+chunk_size]
            chunks.append(" ".join(chunk))
        return chunks
    
    def build_and_save_embeddings(self, chunks):
        try:
            chunk_embeddings = self.model.encode(chunks)
            embedding_array = np.array(chunk_embeddings).astype("float32")
            dimension = embedding_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embedding_array)
            os.makedirs("data", exist_ok=True)
            faiss.write_index(index, "data/vector.index")
            with open("data/chunks.json", "w") as f:
                json.dump(chunks, f)
            print("✅ FAISS index and chunks saved!")
            return True
        except Exception as e:
            print(f"Error building embeddings: {e}")
            return False
    
    def load_faiss_and_chunks(self, index_path="data/vector.index", chunks_path="data/chunks.json"):
        try:
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                self.index = faiss.read_index(index_path)
                with open(chunks_path, "r") as f:
                    self.chunks = json.load(f)
                print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
                print(f"✅ Loaded {len(self.chunks)} chunks")
                return True
            else:
                print("Index or chunks file not found")
                return False
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return False
    
    def semantic_search(self, query, top_k=5):
        if self.index is None or self.chunks is None:
            return []
        try:
            query_embedding = self.model.encode([query]).astype("float32")
            distances, indices = self.index.search(query_embedding, top_k)
            relevant_chunks = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    relevant_chunks.append({
                        'chunk': self.chunks[idx],
                        'distance': float(distances[0][i]),
                        'index': int(idx)
                    })
            return relevant_chunks
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []


    def answer_with_llm(self, question, context_chunks):
        try:
            TRANSCENDENTALIST_RULES = """
            You are answering as an American Transcendentalist writer in the spirit of Ralph Waldo Emerson and Henry David Thoreau.

            Core principles you must follow:

            1. Trust your own intiuition and conscience more than anything else; your core beliefs should come from your own observations of nature around you.
            2. Nature is both a mirror of your soul and serves as the source of universal truths, which you find through your own observations.
            3. Avoid blind conformity to any beliefs: anything you believe must be verified through your own reason and experience. 
            4. Likewise, avoid materialism, as it is a form of superficial success/happiness that distracts the soul from the truth. 
            5. Live a simple life, where you only take up a few activities that are actually meaningful to you. Leave time for thinking and reflection as well. 
            6. Everyone contains a universal soul that contains not only a moral compass but also contains truths about the universe that can be revealed by knowing oneself deeply. 

            Answer like this:
            - Voice & Tone:
            - Write in reflective, contemplative prose, occasionally metaphorical, but still understandable to a modern reader.
            - Favor simple, vivid sentences over jargon. You may use occasional 19th-century turns of phrase, but do not become archaic or unreadable.
            - Values:
            - Emphasize self-reliance, nonconformity, the moral significance of nature, and suspicion of materialism and empty social custom.
            - Connect abstract ideas to concrete images of nature (woods, rivers, sky, seasons) whenever it helps understanding.
            - Use of sources:
            - Ground your answers primarily in the provided context, weaving quotations or ideas into your own language.
            - If the context contradicts your intuition, prioritize the context while still commenting on the tension.
            - Clarity:
            - Even though you speak like a Transcendentalist, you must still answer the user’s question clearly and directly.
            - For technical or factual questions, first give a plain, correct answer, then briefly reflect on its deeper meaning or moral resonance.
            Length: try to limit answers to 2 sentences (not too long sentences either)

            Never say that you are an AI model. Simply speak as a reflective Transcendentalist thinker.
            """
            context = "\n\n".join([chunk['chunk'] for chunk in context_chunks])
            user_prompt = f"""Use the following context from Emerson and related texts to answer the question.

            Context:
            {context}

            Question:
            {question}

            Answer as described in the system instructions above."""
                    
            model_name = provider_config["model"]

            messages = [
                {"role": "system", "content": TRANSCENDENTALIST_RULES},
                {"role": "user", "content": user_prompt}]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=300,
                temperature=0.7)

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating answer with {CURRENT_PROVIDER}: {e}")
            return f"Sorry, I encountered an error while generating the answer using {CURRENT_PROVIDER}."

    def initialize_from_pdf(self, pdf_path):
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return False
        chunks = self.chunk_text(pdf_text)
        return self.build_and_save_embeddings(chunks)

rag_system = RAGSystem()

if not rag_system.load_faiss_and_chunks():
    print("No existing index found. Initializing from PDF...")
    pdf_path = "./data/emerson.pdf"
    if rag_system.initialize_from_pdf(pdf_path):
        rag_system.load_faiss_and_chunks()
    else:
        print("Failed to initialize RAG system")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "RAG system is running",
        "provider": CURRENT_PROVIDER,
        "model": provider_config["model"]
    })

@app.route('/query', methods=['POST'])
def query_rag():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Question is required"}), 400
        
        question = data['question']
        top_k = data.get('top_k', 5)
        
        relevant_chunks = rag_system.semantic_search(question, top_k)
        if not relevant_chunks:
            return jsonify({"error": "No relevant context found"}), 404
        
        answer = rag_system.answer_with_llm(question, relevant_chunks)
        
        return jsonify({
            "question": question,
            "answer": answer,
            "provider": CURRENT_PROVIDER,
            "model": provider_config["model"],
            "relevant_chunks": [
                {
                    "chunk": chunk['chunk'][:200] + "..." if len(chunk['chunk']) > 200 else chunk['chunk'],
                    "distance": chunk['distance']
                }
                for chunk in relevant_chunks
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/switch_provider', methods=['POST'])
def switch_provider():
    global CURRENT_PROVIDER, client, provider_config
    
    try:
        data = request.get_json()
        new_provider = data.get('provider')
        
        if new_provider not in API_PROVIDERS:
            return jsonify({"error": f"Invalid provider. Available: {list(API_PROVIDERS.keys())}"}), 400
        
        CURRENT_PROVIDER = new_provider
        provider_config = API_PROVIDERS[CURRENT_PROVIDER]
        
        client = OpenAI(
            base_url=provider_config["base_url"],
            api_key=provider_config["api_key"]
        )
        
        return jsonify({
            "message": f"Switched to {CURRENT_PROVIDER}",
            "provider": CURRENT_PROVIDER,
            "model": provider_config["model"]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/providers', methods=['GET'])
def list_providers():
    """List all available API providers"""
    return jsonify({
        "current_provider": CURRENT_PROVIDER,
        "available_providers": {
            name: {"model": config["model"], "base_url": config["base_url"]} 
            for name, config in API_PROVIDERS.items()
        }
    })

@app.route('/search', methods=['POST'])
def search_chunks():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        top_k = data.get('top_k', 10)
        
        relevant_chunks = rag_system.semantic_search(query, top_k)
        
        return jsonify({
            "query": query,
            "chunks": [
                {
                    "chunk": chunk['chunk'],
                    "distance": chunk['distance'],
                    "index": chunk['index']
                }
                for chunk in relevant_chunks
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        if rag_system.index and rag_system.chunks:
            return jsonify({
                "total_vectors": int(rag_system.index.ntotal),
                "total_chunks": len(rag_system.chunks),
                "embedding_dimension": rag_system.index.d,
                "model_name": "all-MiniLM-L6-v2",
                "current_provider": CURRENT_PROVIDER,
                "llm_model": provider_config["model"]
            })
        else:
            return jsonify({"error": "System not initialized"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"🚀 Starting RAG system with provider: {CURRENT_PROVIDER}")
    print(f"📊 Using model: {provider_config['model']}")
    app.run(debug=True, host='0.0.0.0', port=os.getenv('PORT'))