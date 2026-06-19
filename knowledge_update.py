import os
import requests
import chromadb
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
