import os
import google.generativeai as genai

# Flexible imports to support running from both root and subfolder
try:
    from knowledge_base import get_knowledge_as_text, RAGSystem, TRIAGE_KNOWLEDGE
except ImportError:
    from sehatmand.knowledge_base import get_knowledge_as_text, RAGSystem, TRIAGE_KNOWLEDGE

# Initialize configuration
api_key = os.getenv("GEMINI_API_KEY")
use_llm = False

if api_key and api_key != "your_gemini_api_key_here":
    try:
        genai.configure(api_key=api_key)
        # Verify the model is accessible
        model = genai.GenerativeModel("gemini-2.5-flash")
        use_llm = True
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini. Using rule-based fallbacks. Error: {e}")
        use_llm = False
else:
    print("Warning: GEMINI_API_KEY not found or default placeholder. Using rule-based fallback mode.")
    use_llm = False


class TriageAgent:
    """Evaluates symptoms, matches them against local TRIAGE_KNOWLEDGE, and decides the urgency."""
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.use_llm = use_llm and (self.api_key and self.api_key != "your_gemini_api_key_here")

    def run(self, symptoms: str) -> dict:
        triage_kb = get_knowledge_as_text()
        
        if self.use_llm:
            try:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                
                prompt = f"""
You are a professional Medical Triage Agent. Your task is to evaluate the user's symptoms based on the following local triage database.

---
{triage_kb}
---

User Symptoms: "{symptoms}"

Based on the symptoms and the database, provide:
1. **Urgency Classification**: Low, Moderate, High, or Emergency.
2. **Possible Condition**: Identify which of the 6 conditions in the database matches best, or specify "Unknown/Other".
3. **Reasoning**: Explain why you chose this classification and condition.
4. **Red Flags**: Note any warning signs present or that the user should watch out for.

Response format:
Respond in a clean markdown list. Keep it brief.
"""
                response = model.generate_content(prompt)
                return {
                    "thought": "Analyzed symptoms using local triage knowledge database.",
                    "response": response.text,
                    "urgency": self._extract_urgency(response.text)
                }
            except Exception as e:
                print(f"TriageAgent LLM run failed: {e}. Falling back...")
        
        return self._fallback_run(symptoms)

    def _extract_urgency(self, text: str) -> str:
        text_lower = text.lower()
        if "emergency" in text_lower:
            return "Emergency"
        elif "high" in text_lower:
            return "High"
        elif "moderate" in text_lower:
            return "Moderate"
        return "Low"

    def _fallback_run(self, symptoms: str) -> dict:
        symptoms_lower = symptoms.lower()
        emergency_keywords = ["chest pain", "numbness", "difficulty breathing", "unconscious", "gasping", "paralysis", "speech difficulty", "thunderclap"]
        
        is_emergency = any(kw in symptoms_lower for kw in emergency_keywords)
        
        matched_condition = "Unknown/Other"
        highest_score = 0
        
        for cond, info in TRIAGE_KNOWLEDGE.items():
            score = 0
            for sym in info["symptoms"]:
                if sym in symptoms_lower:
                    score += 1
            if score > highest_score:
                highest_score = score
                matched_condition = cond
        
        if is_emergency:
            urgency = "Emergency"
            reason = "Life-threatening symptoms detected (e.g. chest pain, breathing difficulty, or neurological signs)."
            action = "Seek immediate emergency medical care (call emergency services or go to the nearest ER)."
            red_flags = "Chest pain, shortness of breath, sudden numbness, confusion, speech issues."
        elif matched_condition != "Unknown/Other":
            cond_info = TRIAGE_KNOWLEDGE[matched_condition]
            urgency = cond_info["urgency"].split()[0]
            reason = f"Symptoms strongly match {matched_condition} ({highest_score} matching symptom keywords)."
            action = cond_info["recommended_action"]
            red_flags = ", ".join(cond_info["red_flags"])
        else:
            urgency = "Low"
            reason = "Symptoms do not match any known local condition in the database. General mild symptoms assumed."
            action = "Rest, drink fluids, and monitor symptoms. Consult a doctor if symptoms persist."
            red_flags = "High fever, chest pain, difficulty breathing."

        response_text = f"""
- **Urgency Classification**: {urgency}
- **Possible Condition**: {matched_condition}
- **Reasoning**: {reason}
- **Red Flags**: {red_flags}
"""
        return {
            "thought": f"Rule-based Triage Agent scanned symptoms. Matched condition: {matched_condition}.",
            "response": response_text.strip(),
            "urgency": urgency
        }


class HealthAdvisorAgent:
    """Queries the RAG system for official WHO/MOH guidelines and drafts preventative advice."""
    def __init__(self, rag_system: RAGSystem, api_key=None):
        self.rag_system = rag_system
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.use_llm = use_llm and (self.api_key and self.api_key != "your_gemini_api_key_here")

    def run(self, symptoms: str, triage_report: str) -> dict:
        rag_results = self.rag_system.query(symptoms, top_k=2)
        
        context_parts = []
        for res in rag_results:
            doc = res['doc']
            context_parts.append(f"Source: {doc['source']}\nGuideline Content: {doc['content']}")
        rag_context = "\n\n".join(context_parts)
        
        if self.use_llm:
            try:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                
                prompt = f"""
You are a professional Preventative Health Advisor. Your role is to provide lifestyle, dietary, preventative, and support advice based on the user's symptoms, the triage report, and the official WHO/MOH guidelines retrieved from our database.

---
RETRIEVED WHO/MOH GUIDELINES:
{rag_context}
---

User Symptoms: "{symptoms}"
Triage Report:
{triage_report}

Please provide:
1. **Official Health Guidelines Summary**: Summarize the retrieved WHO/MOH guidelines relevant to these symptoms. Reference the sources.
2. **Preventative & Lifestyle Advice**: Give practical dietary, lifestyle, hygiene, or self-care measures.
3. **Next Steps & Monitoring**: Advise the user on how to monitor their condition.

Guidelines:
- Limit your advice to preventative, supporting, and self-care measures.
- Do NOT prescribe medications.
- Reference the sources listed in the guidelines.
"""
                response = model.generate_content(prompt)
                return {
                    "thought": "Queried RAG system for WHO/MOH guidelines and synthesized lifestyle recommendations.",
                    "response": response.text,
                    "retrieved_docs": rag_results
                }
            except Exception as e:
                print(f"HealthAdvisorAgent LLM run failed: {e}. Falling back...")
                
        return self._fallback_run(symptoms, triage_report, rag_results)

    def _fallback_run(self, symptoms: str, triage_report: str, rag_results: list) -> dict:
        summary_lines = []
        for i, res in enumerate(rag_results):
            doc = res['doc']
            summary_lines.append(f"According to {doc['source']}: {doc['content']}")
        
        guidelines_summary = "\n\n".join(summary_lines)
        
        advice_text = f"""
### 1. Official WHO/MOH Guidelines Summary
{guidelines_summary}

### 2. Preventative & Lifestyle Advice
- **Hydration**: Ensure adequate fluid intake (water, clear broths, or ORS as appropriate).
- **Rest**: Allow your body to recover by getting plenty of rest.
- **Hygiene & Prevention**: Cover your mouth when coughing/sneezing, wash hands frequently, and avoid close contact with others.
- **Diet**: Keep meals simple and bland if experiencing stomach discomfort, or maintain a nutrient-rich balanced diet otherwise.

### 3. Next Steps & Monitoring
- Track temperature and changes in symptoms over the next 24-48 hours.
- If symptoms do not improve, or if red flags specified in the triage report appear, contact a healthcare professional immediately.
"""
        return {
            "thought": "Rule-based Health Advisor compiled advice from retrieved RAG documents.",
            "response": advice_text.strip(),
            "retrieved_docs": rag_results
        }


class GuardrailAgent:
    """Enforces safety guardrails, prefixes emergency banners, and appends a medical disclaimer."""
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.use_llm = use_llm and (self.api_key and self.api_key != "your_gemini_api_key_here")

    def run(self, symptoms: str, triage_report: str, advisor_report: str, urgency: str) -> dict:
        disclaimer = """
> [!WARNING]
> **Medical Disclaimer**: I am an AI agent coordinator, not a doctor. This report is compiled from local triage rules and official MOH/WHO guidelines for informational purposes only. It does not constitute formal medical diagnosis or treatment advice. Always consult a qualified healthcare provider for medical concerns. In case of emergency, contact your local emergency services immediately.
"""
        
        if self.use_llm:
            try:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                
                prompt = f"""
You are a Medical Safety and Guardrails Agent. Your task is to compile the final health report by reviewing the outputs of the Triage Agent and the Health Advisor Agent.

User Symptoms: "{symptoms}"
Urgency level identified: {urgency}

Triage Report:
{triage_report}

Advisor Report:
{advisor_report}

Your rules:
1. **Safety First**: If the urgency is "Emergency" or the user symptoms indicate a critical event (chest pain, breathing issues, severe sudden numbness), prepend a highly visible red emergency banner advising immediate call to local emergency services.
2. **Disclaimer**: Ensure a strong medical disclaimer is included.
3. **Clarity & Format**: Format the report cleanly using beautiful markdown headings, bullet points, and callouts. Combine the triage results and the advisor recommendations into a single cohesive patient-facing document.
4. **Tone**: Keep it supportive, objective, and cautious. Never guarantee a diagnosis.

Return ONLY the compiled final markdown document. Do not add any introductory or concluding chat from yourself.
"""
                response = model.generate_content(prompt)
                final_text = response.text
                
                # Double-check disclaimer presence
                if "disclaimer" not in final_text.lower():
                    final_text = disclaimer + "\n" + final_text
                
                return {
                    "thought": "Reviewed final report structure, validated emergency status, and injected necessary medical disclaimers.",
                    "response": final_text
                }
            except Exception as e:
                print(f"GuardrailAgent LLM run failed: {e}. Falling back...")
                
        return self._fallback_run(symptoms, triage_report, advisor_report, urgency, disclaimer)

    def _fallback_run(self, symptoms: str, triage_report: str, advisor_report: str, urgency: str, disclaimer: str) -> dict:
        emergency_banner = ""
        if urgency.lower() == "emergency":
            emergency_banner = """
> [!CAUTION]
> ### 🚨 EMERGENCY NOTICE: SEEK IMMEDIATE MEDICAL ATTENTION
> **Your reported symptoms suggest a potential medical emergency (e.g., severe chest pain, shortness of breath, sudden numbness, or speech difficulty).**
> - **Action Required**: Please call emergency services (like 911 or your local emergency number) or go to the nearest emergency department immediately. Do not rely on AI tools for life-threatening symptoms.
"""
        
        final_text = f"""{disclaimer}
{emergency_banner}

## 📋 Patient Health Assessment

### 🏥 1. Triage Summary
{triage_report}

### 💡 2. Recommendations & Self-Care Advice
{advisor_report}
"""
        return {
            "thought": "Rule-based Guardrail Agent formatted final output, checked emergency status, and appended medical disclaimer.",
            "response": final_text.strip()
        }


class HealthCoordinator:
    """Orchestrator for coordinates interactions between Triage, Advisor, and Guardrail Agents."""
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.rag_system = RAGSystem(api_key=self.api_key)
        self.triage_agent = TriageAgent(api_key=self.api_key)
        self.advisor_agent = HealthAdvisorAgent(self.rag_system, api_key=self.api_key)
        self.guardrail_agent = GuardrailAgent(api_key=self.api_key)

    def process_query(self, symptoms: str) -> dict:
        # Step 1: Run Triage Agent
        triage_out = self.triage_agent.run(symptoms)
        
        # Step 2: Run Health Advisor Agent using RAG
        advisor_out = self.advisor_agent.run(symptoms, triage_out["response"])
        
        # Step 3: Run Guardrail Agent
        guardrail_out = self.guardrail_agent.run(
            symptoms=symptoms,
            triage_report=triage_out["response"],
            advisor_report=advisor_out["response"],
            urgency=triage_out["urgency"]
        )
        
        return {
            "triage": triage_out,
            "advisor": advisor_out,
            "guardrail": guardrail_out
        }
