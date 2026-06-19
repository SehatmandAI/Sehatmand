import os
import numpy as np
import google.generativeai as genai

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

# Detailed guidelines simulating a health guidelines database for RAG
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

def simple_keyword_overlap(query: str, document: str) -> float:
    """Calculates a simple word-overlap Jaccard score between query and document."""
    query_words = set(query.lower().split())
    doc_words = set(document.lower().split())
    if not query_words:
        return 0.0
    intersection = query_words.intersection(doc_words)
    return len(intersection) / len(query_words)

class RAGSystem:
    """A lightweight RAG system that uses Gemini Embeddings or falls back to word overlap."""
    def __init__(self, guidelines=MOH_WHO_GUIDELINES, api_key=None):
        self.guidelines = guidelines
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.embeddings = []
        self.use_fallback = True
        
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                genai.configure(api_key=self.api_key)
                self._generate_database_embeddings()
                self.use_fallback = False
            except Exception as e:
                print(f"Warning: RAG initialization with Gemini failed. Falling back to keyword matching. Error: {e}")
                self.use_fallback = True
        else:
            print("Warning: GEMINI_API_KEY not found or default placeholder. Using keyword matching fallback.")
            self.use_fallback = True

    def _generate_database_embeddings(self):
        self.embeddings = []
        for doc in self.guidelines:
            text = f"{doc['source']}\n{doc['content']}"
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            self.embeddings.append(result['embedding'])
        self.embeddings = np.array(self.embeddings)

    def query(self, query_text: str, top_k: int = 2):
        if not query_text:
            return []
            
        if self.use_fallback:
            scores = []
            for doc in self.guidelines:
                text = f"{doc['source']} {doc['content']}"
                score = simple_keyword_overlap(query_text, text)
                scores.append(score)
            
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append({
                        "doc": self.guidelines[idx],
                        "score": float(scores[idx]),
                        "method": "Keyword Overlap"
                    })
            
            # If no matches found, return default guidelines
            if not results:
                for idx in range(min(top_k, len(self.guidelines))):
                    results.append({
                        "doc": self.guidelines[idx],
                        "score": 0.0,
                        "method": "Default Fallback"
                    })
            return results
        else:
            try:
                genai.configure(api_key=self.api_key)
                query_result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query_text,
                    task_type="retrieval_query"
                )
                query_vector = np.array(query_result['embedding'])
                
                dots = np.dot(self.embeddings, query_vector)
                norms_docs = np.linalg.norm(self.embeddings, axis=1)
                norm_query = np.linalg.norm(query_vector)
                similarities = dots / (norms_docs * norm_query + 1e-9)
                
                top_indices = np.argsort(similarities)[::-1][:top_k]
                results = []
                for idx in top_indices:
                    results.append({
                        "doc": self.guidelines[idx],
                        "score": float(similarities[idx]),
                        "method": "Gemini Embeddings"
                    })
                return results
            except Exception as e:
                print(f"Embedding query failed: {e}. Falling back...")
                # Run keyword matching fallback
                scores = []
                for doc in self.guidelines:
                    text = f"{doc['source']} {doc['content']}"
                    score = simple_keyword_overlap(query_text, text)
                    scores.append(score)
                top_indices = np.argsort(scores)[::-1][:top_k]
                return [{
                    "doc": self.guidelines[idx],
                    "score": float(scores[idx]),
                    "method": "Keyword Overlap (Fallback)"
                } for idx in top_indices]
