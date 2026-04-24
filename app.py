from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from groq import Groq
import requests as http_requests
import os

app = Flask(__name__)
CORS(app)

# ═══ API KEYS (set in Render dashboard as environment variables) ═══
PINECONE_KEY = os.environ.get("PINECONE_API_KEY", "")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
INDEX_NAME = "clinassist-ai"

# ═══ CONNECT TO PINECONE ═══
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
stats = index.describe_index_stats()
print(f"Pinecone connected: {stats.total_vector_count} vectors")

# ═══ CONNECT TO GROQ ═══
print("Connecting to Groq...")
groq_client = Groq(api_key=GROQ_KEY)
print("Groq connected")

# ═══ EMBEDDING VIA HUGGINGFACE API ═══
HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


def get_embedding(text):
    try:
        response = http_requests.post(
            HF_EMBED_URL,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


# ═══ DEFAULT SYSTEM PROMPT (used when frontend doesn't send one) ═══
DEFAULT_PROMPT = """You are ClinAssist AI, a clinical training assistant.
Use the provided CONTEXT to answer accurately.
Use **bold** for key terms. Use bullet points for lists. Use tables for labs.
When context contains [MASTER SOURCE — AUTHORITY OVERRIDE], follow that source.
ABG requires arterial sample. Flag dangerous drugs via Nurse.
Never include training disclaimer in messages."""

# ═══ SESSION STORAGE ═══
sessions = {}


# ═══ RAG: SEARCH KNOWLEDGE BASE ═══
def search_knowledge(query, top_k=5):
    embedding = get_embedding(query)
    if embedding is None:
        return ""
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    chunks = []
    for match in results.matches:
        if match.score > 0.3:
            chunks.append(match.metadata.get("text", ""))
    return "\n\n---\n\n".join(chunks)


# ═══ ASK AGENT (with mode-specific prompt support) ═══
def ask_agent(user_message, conversation_history, system_prompt=None):
    # Search knowledge base for relevant context
    context = search_knowledge(user_message)

    # Build the augmented message with RAG context
    augmented_message = user_message
    if context:
        augmented_message = f"""CONTEXT FROM CLINICAL KNOWLEDGE BASE:
{context}

---

USER MESSAGE: {user_message}"""

    conversation_history.append({"role": "user", "content": augmented_message})

    # Use frontend's mode-specific prompt if provided, otherwise default
    prompt = system_prompt if system_prompt else DEFAULT_PROMPT

    messages = [{"role": "system", "content": prompt}] + conversation_history

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=2000
    )

    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})
    return reply, conversation_history


# ═══ ROUTES ═══
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "agent": "ClinAssist AI",
        "model": "Groq Llama 3.3 70B + RAG (Pinecone)",
        "features": "mode-specific prompts supported",
        "knowledge_vectors": index.describe_index_stats().total_vector_count
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default")
    system_prompt = data.get("system_prompt", None)  # Accept from frontend
    mode = data.get("mode", "virtual_patient")  # Accept mode info

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    if session_id not in sessions:
        sessions[session_id] = []

    try:
        reply, sessions[session_id] = ask_agent(
            user_message,
            sessions[session_id],
            system_prompt=system_prompt
        )
        return jsonify({
            "reply": reply,
            "session_id": session_id,
            "mode": mode,
            "rag_used": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    data = request.json
    session_id = data.get("session_id", "default")
    sessions[session_id] = []
    return jsonify({"status": "reset"})


# ═══ RUN ═══
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
