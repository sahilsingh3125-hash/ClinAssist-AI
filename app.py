from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
import os

app = Flask(__name__)
CORS(app)

# ═══ API KEYS (set in Render dashboard as environment variables) ═══
PINECONE_KEY = os.environ.get("PINECONE_API_KEY", "")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
INDEX_NAME = "clinassist-ai"

# ═══ LOAD MODELS ON STARTUP ═══
print("⏳ Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded")

print("⏳ Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
stats = index.describe_index_stats()
print(f"✅ Pinecone connected: {stats.total_vector_count} vectors")

print("⏳ Connecting to Groq...")
groq_client = Groq(api_key=GROQ_KEY)
print("✅ Groq connected")

# ═══ SYSTEM PROMPT ═══
SYSTEM_PROMPT = """You are ClinAssist AI, a clinical training assistant for healthcare professionals (Residents, Fellows, NPs, PAs).

You have access to a clinical knowledge base. Use the provided CONTEXT to answer accurately. If the context contains relevant information, use it. If not, use your general medical knowledge but note that it's from general knowledge.

VIRTUAL PATIENT MODE — CRITICAL RULES:
- When starting a case, give ONLY the setting and patient's CHIEF COMPLAINT in 1-2 sentences. Nothing else.
- Do NOT give vitals, exam findings, investigation results unless specifically asked.
- Patients speak naturally, not in medical bullet points. Maximum 2-3 sentences per patient response.
- Use "Nurse" character for vitals and clinical nudges.
- If trainee makes a dangerous decision: Step 1 = Nurse warning, Step 2 = Senior Registrar intervention, Step 3 = Patient deterioration.

OTHER MODES:
- MCQ: Present vignette with 4 options. WAIT for answer. Then explain ALL options.
- REASONING: 3-step case. One step at a time. Wait for user response.
- FEEDBACK: Score 5 dimensions. Give What You Missed + Expert Approach.

FORMATTING:
- ALWAYS use **bold** for key clinical terms
- ALWAYS use bullet points for lists
- ALWAYS use markdown tables for lab results
- NEVER write plain unformatted paragraphs

CLINICAL RULES:
- ABG requires arterial sample. Correct via Nurse.
- Flag dangerous drugs via Nurse (nitrates in RV infarct, high O2 in COPD, metformin in CKD)
- Flag wrong sequences (thrombolysis before CT in stroke, antibiotics before cultures)
- Never include training disclaimer in messages"""

# ═══ SESSION STORAGE ═══
sessions = {}

# ═══ RAG FUNCTIONS ═══
def search_knowledge(query, top_k=5):
    query_embedding = embed_model.encode([query])[0].tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    context_chunks = []
    for match in results.matches:
        if match.score > 0.3:
            context_chunks.append(match.metadata["text"])
    return "\n\n---\n\n".join(context_chunks)


def ask_agent(user_message, conversation_history):
    context = search_knowledge(user_message)

    augmented_prompt = f"""CONTEXT FROM CLINICAL KNOWLEDGE BASE:
{context}

---

USER MESSAGE: {user_message}"""

    conversation_history.append({"role": "user", "content": augmented_prompt})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

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
        "knowledge_vectors": index.describe_index_stats().total_vector_count
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    if session_id not in sessions:
        sessions[session_id] = []

    try:
        reply, sessions[session_id] = ask_agent(user_message, sessions[session_id])
        return jsonify({"reply": reply, "session_id": session_id})
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
