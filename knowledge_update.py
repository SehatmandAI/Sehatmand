import os
import json
import re
import requests
import chromadb
import google.generativeai as genai
from pypdf import PdfReader

# Path to database (use writable /tmp/chroma_db if available, e.g. on Streamlit Cloud)
if os.path.exists("/tmp") and os.access("/tmp", os.W_OK):
    DB_PATH = "/tmp/chroma_db"
else:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(CURRENT_DIR, "chroma_db")
COLLECTION_NAME = "health_guidelines"

# Try loading from local module
try:
    from knowledge_base import get_local_embedding
except ImportError:
    from sehatmand.knowledge_base import get_local_embedding


def extract_text_from_pdf(pdf_path: str) -> str:
    """Reads a PDF file and extracts text page by page."""
    reader = PdfReader(pdf_path)
    text_list = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_list.append(page_text)
    return "\n\n--- PAGE BREAK ---\n\n".join(text_list)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Splits a large string into smaller overlapping text chunks."""
    chunks = []
    if not text:
        return chunks
    
    start = 0
    text_len = len(text)
    while start < text_len:
        # Move end to the nearest space if possible to avoid word-splitting
        end = min(start + chunk_size, text_len)
        if end < text_len:
            space_idx = text.rfind(" ", start, end)
            if space_idx > start + (chunk_size // 2):
                end = space_idx

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        if end >= text_len:
            break
        start = end - overlap
    return chunks

def generate_mock_document(output_path: str):
    """Generates a mock text file representing standard health guidelines for testing."""
    mock_content = """# Official MOH & WHO Universal Health Advisory Manual
This manual compiles global health standards for the prevention and treatment of common primary care cases.

## Chapter 1: Respiratory Conditions and Influenza Management
Influenza is an infectious disease causing fever, body aches, sore throat, and fatigue.
Prevention: The primary prevention is an annual influenza vaccine. Wash hands frequently with soap.
Treatment: Rest, maintain high hydration. Do not self-prescribe antibiotics since influenza is viral, not bacterial.

## Chapter 2: Acute Gastroenteritis and Rehydration Standards
Gastroenteritis is characterized by sudden vomiting, watery diarrhea, and abdominal cramping.
The primary risk is dehydration. The World Health Organization recommends Oral Rehydration Salts (ORS) as the first line of care.
Diet: Bland diet (bananas, rice, applesauce, toast) is advised once vomiting stops. Avoid anti-diarrheals.

## Chapter 3: Hypertension (High Blood Pressure) Guidelines
Hypertension is defined as blood pressure consistently above 140/90 mmHg.
Non-pharmacological management: Limit sodium to under 2000mg per day. Engage in 150 minutes of moderate cardiovascular exercise weekly.
Red Flags: Chest pain, severe headaches, or sudden vision loss.

## Chapter 4: Diabetes Mellitus Type 2 Lifestyle Management
Type 2 Diabetes involves insulin resistance and elevated blood glucose.
Care guidelines: Maintain low-glycemic index diets. Keep weight within healthy BMI parameters. Regular foot checks are mandatory due to neuropathy risk.

## Chapter 5: Migraine and Headache Care Protocols
Migraines are severe throbbing headaches, often unilateral, accompanied by photo/phonophobia.
Triggers: Stress, lack of sleep, caffeine withdrawal, and specific food items like processed meats containing nitrates.
Self-care: Resting in dark, quiet spaces. Cold compresses on the neck or forehead.

## Chapter 6: Emergency Warning Signs (MOH Warning Signals)
Any patient presenting with the following must be transferred immediately to an emergency care facility:
1. Sudden speech difficulty, facial drooping, or hemiparesis.
2. Constant crushing chest pain radiating to the left arm or jaw.
3. Severe gasping or inability to breathe.
4. Loss of consciousness or sudden confusion.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mock_content)

def update_knowledge_base(file_path: str, status_callback=None) -> int:
    """
    Main update pipeline:
    1. Extract text from target file (or generate mock if missing).
    2. Chunk the document.
    3. Wipe and rebuild ChromaDB collection.
    4. Generate embeddings using Sentence-Transformers and upload chunks to ChromaDB.
    """
    # Check file existence, create test mock file if it's missing
    if not os.path.exists(file_path):
        if status_callback:
            status_callback(f"Target document '{os.path.basename(file_path)}' not found. Creating a sample mock guidelines document...")
        mock_path = os.path.join(CURRENT_DIR, "guidelines.txt")
        generate_mock_document(mock_path)
        file_path = mock_path

    if status_callback:
        status_callback(f"Reading {os.path.basename(file_path)}...")
    
    # Read text
    if file_path.lower().endswith(".pdf"):
        full_text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()

    # Chunk text
    if status_callback:
        status_callback("Splitting document into text chunks...")
    chunks = chunk_text(full_text)
    total_chunks = len(chunks)
    
    if total_chunks == 0:
        raise ValueError("The document contains no text after parsing.")

    if status_callback:
        status_callback(f"Connecting to ChromaDB database at {DB_PATH}...")
        
    # Rebuild Chroma collection
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    # Retrieve or create the collection
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    # Delete existing entries in the collection to start fresh
    try:
        existing_data = collection.get()
        if existing_data and "ids" in existing_data and existing_data["ids"]:
            collection.delete(ids=existing_data["ids"])
    except Exception as e:
        if status_callback:
            status_callback(f"Note: Could not empty existing collection: {e}")

    if status_callback:
        status_callback(f"Found {total_chunks} chunks. Generating local embeddings with Sentence-Transformers (all-MiniLM-L6-v2)...")

    # Ingest chunks
    batch_size = 20
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_ids = [f"doc_chunk_{idx}" for idx in range(i, i+len(batch_chunks))]
        batch_embeddings = []
        batch_metadatas = []
        
        for idx, chunk in enumerate(batch_chunks):
            embedding = get_local_embedding(chunk)
            batch_embeddings.append(embedding)
            batch_metadatas.append({
                "chunk_index": i + idx,
                "source_file": os.path.basename(file_path)
            })
            
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_chunks,
            metadatas=batch_metadatas
        )
        
        if status_callback:
            status_callback(f"Ingested chunks {min(i+batch_size, total_chunks)} of {total_chunks}...")

    if status_callback:
        status_callback("Database successfully updated!")
        
    return total_chunks


def get_api_key(name: str, default: str = "") -> str:
    val = os.getenv(name, "")
    if val and val != "your_gemini_api_key_here" and val != "your_openai_api_key_here":
        return val.strip()
    try:
        import streamlit as st
        val = st.secrets.get(name, "")
        if val:
            return val.strip()
    except Exception:
        pass
    return default


def parse_conditions_from_pdf(file_path: str, status_callback=None) -> list:
    """Extract structured conditions from a PDF by chunking and sending to Gemini in one batch call.
    
    Two-pass approach:
    1. Use ChromaDB to find the top-K most relevant chunks for the document's main content
    2. Send those chunks to Gemini for structured extraction
    """
    if not os.path.exists(file_path):
        if status_callback:
            status_callback(f"File not found: {file_path}")
        return []
    
    full_text = extract_text_from_pdf(file_path)
    chunks = chunk_text(full_text)
    
    # Use ChromaDB to find relevant chunks (not just top-K by similarity, but spread across the doc)
    target_chunks = 120
    if len(chunks) > target_chunks and os.path.exists(DB_PATH):
        try:
            chroma_client = chromadb.PersistentClient(path=DB_PATH)
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
            count = collection.count()
            if count > 0:
                # Query with document summary to get relevant chunks
                doc_sample = full_text[:500]
                from knowledge_base import get_local_embedding
                query_vec = get_local_embedding(doc_sample)
                res = collection.query(query_embeddings=[query_vec], n_results=target_chunks)
                if res and "documents" in res and res["documents"]:
                    chunks = res["documents"][0]
                else:
                    chunks = chunks[:target_chunks]
            else:
                chunks = chunks[:target_chunks]
        except Exception:
            chunks = chunks[:target_chunks]
    else:
        chunks = chunks[:target_chunks]
    
    # Combine chunks into a single prompt payload
    combined_text = "\n\n".join(chunks)
    
    # Use Gemini to extract conditions
    gemini_key = get_api_key("GEMINI_API_KEY")
    if not gemini_key:
        if status_callback:
            status_callback("Gemini API key not found. Skipping PDF condition parsing.")
        return []
    
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
You are a medical conditions extraction engine. I'm going to provide you with text chunks from a medical guidelines PDF.

Extract ALL medical conditions, illnesses, diseases, and disorders described in the text. For each condition, extract the following fields:

1. condition_name: The name of the condition (use the most common/recognized name)
2. symptoms: List of specific symptoms associated with this condition. MUST be specific symptoms (not vague like "general symptoms" or "various symptoms"). Include at least 2-3 specific symptoms.
3. urgency: One of: "Low (Self-Care)", "Moderate (Consult Doctor)", "Moderate (Self-Care / Consult if severe)", "Moderate (Monitor & Consult Doctor)", "Moderate (Consult Doctor for Evaluation)", "Moderate (Self-Care / Consult Doctor if frequent)", "High", or "Emergency"
4. recommended_action: The recommended first action or treatment. MUST be specific (not vague like "general management" or "standard treatment"). Include specific actions like "take X", "do Y", "rest", "seek medical care", etc.

IMPORTANT RULES:
- Extract EVERY condition you find, even if it appears in different sections
- Do NOT skip conditions just because they have similar names
- Include conditions that are described with specific symptoms AND treatment/management
- Include emergency conditions, acute conditions, chronic conditions, and general illnesses
- The symptoms MUST be specific (not vague like "general symptoms" or "various symptoms")
- The recommended_action MUST be specific (not vague like "general management" or "standard treatment")
- The urgency should match the context in the text
- DO NOT DEDUPLICATE similar conditions unless they are truly the same (e.g., "Pneumonia" and "Lobar Pneumonia" can both be included)
- Include ALL severities, not just the major ones

If the symptoms in a chunk are vague, use what you can infer from the context. If the action is vague, provide a reasonable specific action based on the condition.

Respond with ONLY a JSON array of objects. NOTHING ELSE. No markdown code fences, no explanations, no preamble.
Example format:
[
  {{"condition_name": "Common Cold", "symptoms": ["runny nose", "sneezing", "congestion"], "urgency": "Low (Self-Care)", "recommended_action": "Rest and hydration"}},
  {{"condition_name": "Influenza", "symptoms": ["fever", "body aches"], "urgency": "Moderate (Consult Doctor)", "recommended_action": "Rest and consult doctor"}}
]

---
PDF CHUNKS:
{combined_text}
"""
        
        if status_callback:
            status_callback(f"Sending {len(chunks)} chunks to Gemini for condition extraction...")
        
        response = model.generate_content(prompt)
        
        # Parse the JSON response with robust error handling
        response_text = response.text.strip()
        
        def try_parse_json(text):
            """Try multiple strategies to parse JSON from the response."""
            # Strategy 1: Parse as-is
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Strip markdown code fences
            if text.startswith("```"):
                lines = text.split("\n")
                # Find where JSON starts and ends
                start_idx = 0
                end_idx = len(lines)
                for i, line in enumerate(lines):
                    if line.startswith("```") or line.startswith("json"):
                        continue
                    if "{" in line or "[" in line:
                        start_idx = i
                        break
                for i in range(len(lines) - 1, start_idx, -1):
                    if lines[i].strip().endswith("}") or lines[i].strip().endswith("]"):
                        end_idx = i + 1
                        break
                json_text = "\n".join(lines[start_idx:end_idx])
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass
            
            # Strategy 3: Find the outermost JSON array
            start = text.find("[")
            if start != -1:
                # Find matching closing bracket
                depth = 0
                end = -1
                for i in range(start, len(text)):
                    if text[i] == "[":
                        depth += 1
                    elif text[i] == "]":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end != -1:
                    json_str = text[start:end]
                    # Clean up common issues: trailing commas before } or ]
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    # Remove any trailing text after the array
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return json.loads(json_str[:10000])  # Try first 10K chars
            return None
        
        conditions = try_parse_json(response_text)
        
        # Post-process: clean up vague entries
        conditions = _post_process_conditions(conditions, combined_text)
        
        if conditions is None:
            raise json.JSONDecodeError("Could not parse JSON", response_text, 0)
        
        if status_callback:
            status_callback(f"Extracted {len(conditions)} conditions from PDF!")
        
        return conditions
        
    except Exception as e:
        if status_callback:
            status_callback(f"Error parsing PDF conditions: {e}")
        return []


def _post_process_conditions(conditions, combined_text):
    """Post-process extracted conditions to ensure they have meaningful data."""
    if not conditions:
        return []
    
    processed = []
    vague_actions = {"general management", "standard treatment", "supportive care", "symptomatic treatment", "observation", "monitoring", "general care"}
    
    for cond in conditions:
        name = cond.get("condition_name", "Unknown")
        symptoms = cond.get("symptoms", [])
        action = cond.get("recommended_action", "")
        
        # Clean up symptoms
        if not symptoms or (isinstance(symptoms, str) and symptoms.strip() in ["", "general symptoms", "various symptoms", "N/A"]):
            # Try to extract symptoms from the action text
            action_lower = action.lower() if action else ""
            if "symptoms" in action_lower or "symptom" in action_lower:
                symptoms = _extract_symptoms_from_text(action)
            elif "management" in action_lower or "treatment" in action_lower:
                symptoms = _extract_symptoms_from_text(action)
            else:
                symptoms = [action] if action else ["general symptoms"]
        elif isinstance(symptoms, str):
            symptoms = [s.strip() for s in symptoms.split(",") if s.strip()]
        
        # Clean up action
        if action and action.strip().lower() in vague_actions:
            action = _generate_action_from_name(name)
        
        processed.append({
            "condition_name": name,
            "symptoms": symptoms,
            "urgency": cond.get("urgency", "Moderate (Consult Doctor)"),
            "recommended_action": action
        })
    
    return processed


def _extract_symptoms_from_text(text):
    """Extract potential symptoms from a text string."""
    if not text:
        return ["general symptoms"]
    
    # Look for comma-separated lists
    parts = [p.strip() for p in text.replace("and", ",").replace("or", ",").split(",") if p.strip()]
    if len(parts) >= 2:
        return parts
    
    # Look for semicolon-separated lists
    parts = [p.strip() for p in text.replace("and", ",").split(";") if p.strip()]
    if len(parts) >= 2:
        return parts
    
    return [text]


def _generate_action_from_name(name):
    """Generate a reasonable action based on the condition name."""
    name_lower = name.lower()
    if "infection" in name_lower or "fever" in name_lower:
        return "Consult a doctor for evaluation and possible medication"
    elif "pain" in name_lower or "headache" in name_lower:
        return "Rest, take pain relievers as needed, and monitor symptoms"
    elif "cancer" in name_lower or "tumor" in name_lower:
        return "Seek specialist consultation for further evaluation"
    elif "diabetes" in name_lower:
        return "Monitor blood sugar levels and follow dietary guidelines"
    elif "hypertension" in name_lower or "blood pressure" in name_lower:
        return "Monitor blood pressure regularly and reduce sodium intake"
    elif "respiratory" in name_lower or "lung" in name_lower or "asthma" in name_lower:
        return "Use breathing exercises and seek medical care if symptoms worsen"
    elif "gastro" in name_lower or "stomach" in name_lower or "intestine" in name_lower:
        return "Maintain hydration and follow a bland diet"
    elif "acute" in name_lower or "emergency" in name_lower:
        return "Seek immediate medical attention"
    else:
        return "Consult a healthcare provider for proper diagnosis and treatment"
