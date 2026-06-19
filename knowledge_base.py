import os
import requests
import chromadb

# 6 common conditions in triage dictionary
TRIAGE_KNOWLEDGE = {
    "Common Cold": {
        "symptoms": ["runny nose", "sneezing", "congestion", "mild sore throat", "mild cough", "low-grade fever"],
        "urgency": "Low (Self-Care)",
        "recommended_action": "Rest, stay hydrated, use saline nasal drops, or over-the-counter pain relievers if needed.",
        "red_flags": ["difficulty breathing", "fever lasting more than 5 days", "inability to keep liquids down"]
    },
    "Influenza (Flu)": {
        "symptoms": ["sudden onset of high fever", "body aches", "chills", "severe fatigue", "dry cough", "headache"],
        "urgency": "Moderate (Consult Doctor)",
        "recommended_action": "Rest, drink plenty of fluids, isolate to prevent spread. Consult a healthcare provider, especially if at high-risk (elderly, chronic conditions).",
        "red_flags": ["shortness of breath", "chest pain", "confusion", "persistent dizziness"]
    },
    "Gastroenteritis (Food Poisoning)": {
        "symptoms": ["nausea", "vomiting", "watery diarrhea", "abdominal cramps", "mild fever"],
        "urgency": "Moderate (Self-Care / Consult if severe)",
        "recommended_action": "Prevent dehydration by drinking Oral Rehydration Salts (ORS) or clear fluids in small, frequent sips. Eat bland foods when tolerated.",
        "red_flags": ["inability to keep fluids down for 24 hours", "signs of severe dehydration (extreme thirst, dry mouth, little to no urination)", "high fever", "bloody stools"]
    },
    "Hypertension (High Blood Pressure)": {
        "symptoms": ["often silent", "headache (especially back of head)", "dizziness", "blurred vision", "tinnitus"],
        "urgency": "Moderate (Monitor & Consult Doctor)",
        "recommended_action": "Monitor blood pressure regularly, reduce sodium intake, avoid stress, and consult a doctor for long-term management and medication.",
        "red_flags": ["severe chest pain", "sudden severe headache with numbness or speech difficulty", "shortness of breath", "severe anxiety"]
    },
    "Migraine": {
        "symptoms": ["throbbing headache on one side of head", "nausea", "sensitivity to light and sound", "visual disturbances (aura)"],
        "urgency": "Moderate (Self-Care / Consult Doctor if frequent)",
        "recommended_action": "Rest in a quiet, dark room. Apply a cool cloth to forehead. Avoid known triggers (stress, certain foods, bright lights). Use doctor-recommended medication.",
        "red_flags": ["headache that is sudden and explosive ('thunderclap')", "headache with fever, stiff neck, or confusion", "first-ever severe headache after age 50"]
    },
    "Type 2 Diabetes (High Blood Sugar)": {
        "symptoms": ["increased thirst (polydipsia)", "frequent urination (polyuria)", "increased hunger", "unexplained weight loss", "fatigue", "blurred vision"],
        "urgency": "Moderate (Consult Doctor for Evaluation)",
        "recommended_action": "Consult a healthcare provider for formal blood sugar testing. Focus on diet modification (reducing refined sugars/carbs) and regular physical activity.",
        "red_flags": ["rapid deep breathing", "confusion or extreme drowsiness", "fruity-smelling breath (signs of diabetic ketoacidosis)", "persistent vomiting"]
    }
}

# Detailed guidelines simulating a health guidelines database for fallback keyword matching
MOH_WHO_GUIDELINES = [
    {
        "source": "WHO Guidelines on Influenza Prevention (2024)",
        "content": "To prevent the spread of influenza, the World Health Organization (WHO) recommends: 1) Annual vaccination for high-risk individuals including pregnant women, the elderly, children, and healthcare workers. 2) Regular hand hygiene using alcohol-based hand rub or soap and water. 3) Practicing respiratory hygiene (covering mouth and nose when coughing or sneezing). 4) Early self-isolation of those feeling unwell, feverish, and having other respiratory symptoms."
    },
    {
        "source": "WHO/UNICEF Guidelines for Diarrhoeal Disease Control",
        "content": "For the treatment of acute gastroenteritis, WHO and UNICEF recommend: 1) Oral Rehydration Therapy (ORT) using low-osmolarity Oral Rehydration Salts (ORS) to prevent dehydration. 2) Continued feeding, including breastfeeding, during the illness. 3) Zinc supplementation (20 mg per day for 10-14 days for children) to reduce the severity and duration of the episode. 4) Avoiding anti-diarrheal drugs (like loperamide) unless explicitly prescribed by a doctor, especially in young children, as they can retain toxins in the gut."
    },
    {
        "source": "WHO Global Brief on Hypertension",
        "content": "Hypertension is a major cause of premature death worldwide. WHO guidance on prevention and management includes: 1) Reducing salt intake to less than 5g per day. 2) Eating more fruit and vegetables. 3) Being physically active on a regular basis. 4) Avoiding tobacco use and reducing alcohol consumption. 5) Regularly monitoring blood pressure to ensure early detection and adherence to pharmacological treatment if prescribed."
    },
    {
        "source": "WHO Guidelines on Diabetes Care and Lifestyle",
        "content": "WHO recommendations for managing and preventing Type 2 Diabetes emphasize: 1) Achieving and maintaining a healthy body weight. 2) Being physically active—at least 30 minutes of regular, moderate-intensity activity on most days. 3) Eating a healthy diet, avoiding sugar and saturated fats intake. 4) Avoiding tobacco use—smoking increases the risk of cardiovascular diseases associated with diabetes. 5) Standard medical management involves blood glucose monitoring, insulin or oral medication, and regular screening for complications (kidney, feet, eyes)."
    },
    {
        "source": "MOH/WHO Headache Disorders Fact Sheet",
        "content": "Migraine is a primary headache disorder that is highly disabling. According to WHO: 1) Effective treatment requires identifying and avoiding triggers, which can include stress, lack of sleep, dietary items (like aged cheese, artificial sweeteners, caffeine, or alcohol), and environmental changes. 2) Acute treatment should be taken early in the attack (e.g., simple analgesics or triptans). 3) Prophylactic treatment is indicated for patients with frequent or severe disabling attacks. 4) Avoid medication-overuse headache by limiting acute pain medications to no more than 2-3 days per week."
    },
    {
        "source": "MOH Emergency Warning Signals",
        "content": "The Ministry of Health (MOH) warning signals specify conditions under which emergency medical services (e.g., 911 or local emergency room) should be contacted immediately: 1) Constant chest pain or pressure, especially if it radiates to the arm, neck, or jaw, or is accompanied by sweating or shortness of breath. 2) Sudden onset of weakness or numbness on one side of the body, difficulty speaking, or sudden confusion. 3) Severe difficulty breathing or gasping for air. 4) Sudden loss of consciousness or fainting. 5) Sudden, severe, and explosive headache without known cause."
    },
    {
        "source": "WHO General Self-Care and Hydration Advice",
        "content": "General health self-care involves: 1) Drinking 2 to 3 liters of clean water daily, depending on activity level and weather. 2) Maintaining 7-8 hours of sleep per night. 3) Eating a balanced diet containing a variety of whole foods. 4) Seeking early consultation for symptoms that do not improve after 48-72 hours. 5) Keeping a list of emergency numbers and a first-aid kit readily available at home."
    }
]

# Path to database (use writable /tmp/chroma_db if available, e.g. on Streamlit Cloud)
if os.path.exists("/tmp") and os.access("/tmp", os.W_OK):
    DB_PATH = "/tmp/chroma_db"
else:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(CURRENT_DIR, "chroma_db")
COLLECTION_NAME = "health_guidelines"

def check_ollama_status(host: str = "http://localhost:11434") -> bool:
    """Always returns True since Sentence-Transformers is used as the local engine (no Ollama server dependency)."""
    return True

_embedding_model = None

def get_local_embedding(text: str) -> list:
    """Fetches text embedding using local sentence-transformers all-MiniLM-L6-v2 model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model.encode(text).tolist()

def get_knowledge_as_text() -> str:
    """Formats the TRIAGE_KNOWLEDGE dictionary into a clean markdown/text format for agent retrieval."""
    text_parts = ["# TRIAGE KNOWLEDGE DATABASE\n"]
    for condition, info in TRIAGE_KNOWLEDGE.items():
        text_parts.append(f"## Condition: {condition}")
        text_parts.append(f"- **Symptoms**: {', '.join(info['symptoms'])}")
        text_parts.append(f"- **Urgency Level**: {info['urgency']}")
        text_parts.append(f"- **Recommended First Action**: {info['recommended_action']}")
        text_parts.append(f"- **Red Flags**: {', '.join(info['red_flags'])}")
        text_parts.append("")
    return "\n".join(text_parts)

def get_db_document_count() -> int:
    """Returns the number of documents currently stored in ChromaDB."""
    try:
        if not os.path.exists(DB_PATH):
            return 0
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0

def simple_keyword_overlap(query: str, document: str) -> float:
    """Calculates a simple word-overlap Jaccard score between query and document."""
    query_words = set(query.lower().split())
    doc_words = set(document.lower().split())
    if not query_words:
        return 0.0
    intersection = query_words.intersection(doc_words)
    return len(intersection) / len(query_words)

class RAGSystem:
    """A RAG system querying ChromaDB locally, calling local sentence-transformers all-MiniLM-L6-v2 embeddings model."""
    def __init__(self, model_name="all-MiniLM-L6-v2", host=None):
        self.model_name = model_name
        self.host = host
        self.chroma_client = None
        self.collection = None
        self.use_fallback = True

        db_doc_count = get_db_document_count()
        
        if db_doc_count > 0:
            try:
                self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
                self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
                self.use_fallback = False
            except Exception as e:
                print(f"Warning: RAG initialization with ChromaDB failed. Falling back. Error: {e}")
                self.use_fallback = True
        else:
            print("Warning: RAG system is running in offline keyword overlap fallback. (DB count is 0)")
            self.use_fallback = True

    def query(self, query_text: str, top_k: int = 2) -> list:
        if not query_text:
            return []
            
        if self.use_fallback:
            # Fall back to Jaccard overlap on base mock guidelines
            scores = []
            for doc in MOH_WHO_GUIDELINES:
                text = f"{doc['source']} {doc['content']}"
                score = simple_keyword_overlap(query_text, text)
                scores.append(score)
            
            top_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:top_k]
            results = []
            for idx in top_indices:
                results.append({
                    "doc": MOH_WHO_GUIDELINES[idx],
                    "score": float(scores[idx]),
                    "method": "Keyword Overlap (Local Fallback)"
                })
            return results
        else:
            try:
                # 1. Fetch embedding for query using sentence-transformers
                query_vector = get_local_embedding(query_text)
                
                # 2. Query ChromaDB
                res = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=top_k
                )
                
                results = []
                # Parse Chroma response
                if res and "documents" in res and len(res["documents"]) > 0:
                    documents = res["documents"][0]
                    metadatas = res["metadatas"][0] if "metadatas" in res else []
                    distances = res["distances"][0] if "distances" in res else []
                    
                    for idx, doc in enumerate(documents):
                        source = metadatas[idx].get("source_file", "Ingested Document") if idx < len(metadatas) else "ChromaDB"
                        dist = distances[idx] if idx < len(distances) else 0.0
                        # Convert distance to similarity score
                        score = 1.0 / (1.0 + dist)
                        
                        results.append({
                            "doc": {
                                "source": f"{source} (Chunk {metadatas[idx].get('chunk_index', idx)})",
                                "content": doc
                            },
                            "score": float(score),
                            "method": "Sentence-Transformers + ChromaDB"
                        })
                return results
            except Exception as e:
                print(f"RAG query failed: {e}. Running keyword fallback...")
                # Immediate fallback
                scores = []
                for doc in MOH_WHO_GUIDELINES:
                    text = f"{doc['source']} {doc['content']}"
                    score = simple_keyword_overlap(query_text, text)
                    scores.append(score)
                top_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:top_k]
                return [{
                    "doc": MOH_WHO_GUIDELINES[idx],
                    "score": float(scores[idx]),
                    "method": "Keyword Overlap (RAG Fallback)"
                } for idx in top_indices]
