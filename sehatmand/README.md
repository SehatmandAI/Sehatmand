# Sehatmand 🩺

Sehatmand (Urdu/Hindi for *Healthy*) is a simple multiagent health triage and advisor system built with Streamlit, Google Gemini API, and a custom RAG (Retrieval-Augmented Generation) component.

## Project Structure

```
sehatmand/
├── requirements.txt
├── .gitignore
├── README.md
├── .env
├── knowledge_base.py
├── agents.py
└── app.py
```

## Features

1. **Triage Agent**: Categorizes client symptoms, cross-checks them against a local triage dictionary, and rates severity.
2. **Health Advisor Agent**: Performs RAG queries on official WHO/MOH mock guidelines to offer preventative suggestions and lifestyle tips.
3. **Guardrail Agent**: Sanitizes and structures outputs, enforces standard medical disclaimers, and elevates critical cases (e.g. chest pain) to immediate emergency status.
4. **Interactive UI**: A modern, sleek Streamlit interface designed with teal and slate colors, featuring step-by-step agent execution steps, source text inspector, and clean formatting.

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
   Edit the `.env` file inside `sehatmand/` and replace `your_gemini_api_key_here` with your actual Google AI Studio API key.
   ```env
   GEMINI_API_KEY=AIzaSy...
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
