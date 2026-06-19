<div align="center">
<img src="https://avatars.githubusercontent.com/u/295149021?s=200&v=4" width="160" height="160" style="border-radius: 50%; object-fit: cover;">
<br><br>
<small><i>Vibecoded via antigravity</i></small>
<br>
<small><i>Built for the Atomcamp Hackathon</i></small>
</div>

# Sehatmand 🩺

Sehatmand (Urdu/Hindi for *Healthy*) is a multi-agent health triage and advisory system built with Streamlit, Google Gemini API, and a custom RAG pipeline.

## The Problem

In many rural and underserved areas, the gap between available medical staff and the patient population is staggering — too many patients, too few doctors. A simple headache could mean a week of travel, and by the time a patient sees a doctor, the condition may have progressed. Sehatmand addresses this by letting patients type their symptoms in their own language and receive accurate diagnoses backed by official medical guidelines.

## How It Works

Sehatmand uses a **multi-agent architecture** where each agent specializes in a different part of the diagnostic pipeline. When a user describes their symptoms:

1. The **Triage Agent** categorizes the symptoms, cross-checks them against medical knowledge, and rates severity (Low / Medium / High / Critical).
2. The **Health Advisor Agent** performs vector-based retrieval on the uploaded medical guidelines to find the most relevant sections.
3. The **Guardrail Agent** sanitizes and structures the output, enforcing standard medical disclaimers and flagging cases that need immediate attention.
4. The **UI Formatter Agent** presents the results in a clean, bilingual format (English / Urdu) with actionable next steps.

## Vector-Based RAG with ChromaDB

Sehatmand uses **ChromaDB** as its vector database and **Sentence Transformers** for embedding semantic meaning into the medical guidelines. When a PDF is uploaded:

1. **Text Extraction** — The PDF is parsed using `pypdf`, extracting all text content.
2. **Chunking** — Text is split into overlapping chunks (1000 characters with 200-character overlap) to preserve context across boundaries.
3. **Embedding** — Each chunk is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` Sentence Transformer model, which captures the semantic meaning of medical concepts.
4. **Storage** — Vectors are stored in ChromaDB, enabling fast similarity search. When a patient describes their symptoms, the system embeds their query and finds the most relevant guideline sections in the database.

This approach means Sehatmand doesn't just match keywords — it understands *meaning*. A patient describing "chest pain that gets worse with exertion" will be matched to guidelines about cardiovascular conditions, even if those exact words don't appear.

## Country-Agnostic Upload

While Sehatmand ships with WHO/MOH (Ministry of Health) guidelines for Pakistan, the upload mechanism is fully generic. Any country can:
- Take their own medical guidelines (PDF format)
- Upload them to the system
- The RAG pipeline automatically parses, chunks, and indexes them
- The multi-agent system immediately starts using the new guidelines

This makes Sehatmand adaptable to any healthcare system — just upload the rules and let the agents do the rest.

## Bilingual Support

Sehatmand supports both English and Urdu. A toggle in the sidebar switches the entire UI between languages, and the agents can process and generate responses in either language using the dual Gemini API key architecture.

## Features

- **Triage Agent**: Categorizes client symptoms, cross-checks against medical knowledge, and rates severity.
- **Health Advisor Agent**: Performs RAG queries on official guidelines to offer preventative suggestions and lifestyle tips.
- **Guardrail Agent**: Sanitizes outputs, enforces standard medical disclaimers, and elevates critical cases to immediate emergency status.
- **PDF Upload & Parsing**: Upload any medical PDF and have the system extract, chunk, and index it automatically.
- **Local Triage Conditions**: Dynamic sidebar panel showing current symptom rules and recommended actions.
- **Developer Portal**: Collapsible sidebar with LLM status, embedding model info, RAG controls, and triage inspector.
- **Bilingual UI**: Toggle between English and Urdu with full agent support.
- **Dual API Key Architecture**: Supports two Gemini API keys for load balancing and failover.

## Setup Instructions

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**:
   Edit the `.env` file and replace `your_gemini_api_key_here` with your Google AI Studio API key.
   ```env
   GEMINI_API_KEY=AIzaSy...
   GEMINI_API_KEY_2=AIzaSy...
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Requirements

- Python 3.10+
- Google Gemini API Key (via Google AI Studio)
- OpenCV, pypdf, sentence-transformers, ChromaDB, and other dependencies in `requirements.txt`

## License

MIT
