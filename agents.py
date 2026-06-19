import os
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai

try:
    from openai import OpenAI
except ImportError:
    pass

# Flexible imports to support running from both root and subfolder
try:
    from knowledge_base import get_knowledge_as_text, RAGSystem, TRIAGE_KNOWLEDGE
except ImportError:
    from sehatmand.knowledge_base import get_knowledge_as_text, RAGSystem, TRIAGE_KNOWLEDGE

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

# Initialize configuration
api_key = get_api_key("GEMINI_API_KEY")
use_llm = False

if api_key:
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
        self.api_key = api_key or get_api_key("GEMINI_API_KEY")
        self.use_llm = use_llm and (self.api_key and self.api_key != "your_gemini_api_key_here")

    def run(self, symptoms: str, language: str = "English") -> dict:
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

First, evaluate if the user has provided enough information to make a safe assessment. If the symptoms are too vague (e.g. "I feel sick", "headache" with no duration or context), return:
1. **Urgency Classification**: Insufficient Info
2. **Clarifying Questions**: Provide 1-2 specific questions to ask the user.

Otherwise, based on the symptoms and the database, provide:
1. **Urgency Classification**: Low, Moderate, High, or Emergency.
2. **Possible Condition**: Identify which of the 6 conditions in the database matches best, or specify "Unknown/Other".
3. **Reasoning**: Explain why you chose this classification and condition.
4. **Red Flags**: Note any warning signs present or that the user should watch out for.

Response format:
Respond in a clean markdown list. Keep it brief.

IMPORTANT: You must write the entire thought process, response, reasoning, classifications, and questions strictly in the language: {language}. If {language} is Urdu, translate everything into Urdu.
"""
                response = model.generate_content(prompt)
                thought_text = "سسٹمیٹک طریقے سے مقامی ٹریج ڈیٹا بیس کی مدد سے علامات کا تجزیہ کیا گیا۔" if language == "Urdu" else "Analyzed symptoms using local triage knowledge database."
                return {
                    "thought": thought_text,
                    "response": response.text,
                    "urgency": self._extract_urgency(response.text)
                }
            except Exception as e:
                print(f"TriageAgent LLM run failed: {e}. Falling back...")
        
        return self._fallback_run(symptoms, language)

    def _extract_urgency(self, text: str) -> str:
        text_lower = text.lower()
        if "insufficient" in text_lower or "ناکافی" in text_lower:
            return "Insufficient Info"
        if "emergency" in text_lower or "ہنگامی" in text_lower or "شدید" in text_lower:
            return "Emergency"
        elif "high" in text_lower or "زیادہ" in text_lower or "شدت" in text_lower:
            return "High"
        elif "moderate" in text_lower or "معتدل" in text_lower:
            return "Moderate"
        return "Low"

    def _fallback_run(self, symptoms: str, language: str = "English") -> dict:
        is_urdu = language == "Urdu"
        
        if len(symptoms.split()) < 4:
            if is_urdu:
                return {
                    "thought": "رول پر مبنی ٹریج ایجنٹ نے بہت مختصر ان پٹ کا پتہ لگایا۔ مزید معلومات مانگ رہا ہے۔",
                    "response": "- **شدت کی درجہ بندی**: ناکافی معلومات\n- **وضاحتی سوالات**: کیا آپ براہ کرم مزید تفصیلات فراہم کر سکتے ہیں؟ آپ کو یہ علامات کب سے ہیں، اور کیا آپ کو بخار یا متلی جیسی کوئی دوسری علامات ہیں؟",
                    "urgency": "Insufficient Info"
                }
            else:
                return {
                    "thought": "Rule-based Triage Agent detected very brief input (<4 words). Asking for more information.",
                    "response": "- **Urgency Classification**: Insufficient Info\n- **Clarifying Questions**: Can you please provide more details? How long have you had this symptom, and do you have any other symptoms like fever or nausea?",
                    "urgency": "Insufficient Info"
                }
            
        symptoms_lower = symptoms.lower()
        emergency_keywords = ["chest pain", "numbness", "difficulty breathing", "unconscious", "gasping", "paralysis", "speech difficulty", "thunderclap",
                              "سینے میں درد", "بے ہوش", "سانس لینے میں دشواری", "بولنے میں دشواری"]
        
        is_emergency = any(kw in symptoms_lower for kw in emergency_keywords)
        
        matched_condition = "Unknown/Other"
        highest_score = 0
        
        urdu_conditions = {
            "Cholera": "ہیضہ (Cholera)",
            "COVID-19": "کووڈ-19 (COVID-19)",
            "Dengue Fever": "ڈینگی بخار (Dengue Fever)",
            "Malaria": "ملیریا (Malaria)",
            "Typhoid Fever": "ٹائیفائیڈ بخار (Typhoid Fever)",
            "Influenza (Flu)": "انفلوئنزا / فلو (Influenza)",
            "Unknown/Other": "نامعلوم / دیگر"
        }
        
        for cond, info in TRIAGE_KNOWLEDGE.items():
            score = 0
            for sym in info["symptoms"]:
                if sym in symptoms_lower:
                    score += 1
            if score > highest_score:
                highest_score = score
                matched_condition = cond
        
        if is_emergency:
            urgency = "ہنگامی (Emergency)" if is_urdu else "Emergency"
            reason = "زندگی کے لیے خطرہ پیدا کرنے والی علامات کا پتہ چلا (مثلاً سینے میں درد، سانس لینے میں دشواری، یا اعصابی علامات)۔" if is_urdu else "Life-threatening symptoms detected (e.g. chest pain, breathing difficulty, or neurological signs)."
            action = "فوری طور پر ہنگامی طبی امداد حاصل کریں (ہنگامی خدمات کو کال کریں یا قریبی ایمرجنسی روم میں جائیں)۔" if is_urdu else "Seek immediate emergency medical care (call emergency services or go to the nearest ER)."
            red_flags = "سینے میں درد، سانس کی قلت، اچانک بے حسی، الجھن، بولنے میں دشواری۔" if is_urdu else "Chest pain, shortness of breath, sudden numbness, confusion, speech issues."
        elif matched_condition != "Unknown/Other":
            cond_info = TRIAGE_KNOWLEDGE[matched_condition]
            urgency_en = cond_info["urgency"].split()[0]
            urgency_map = {"Low": "کم", "Moderate": "معتدل", "High": "زیادہ", "Emergency": "ہنگامی"}
            urgency = f"{urgency_map.get(urgency_en, 'کم')} ({urgency_en})" if is_urdu else urgency_en
            
            cond_name = urdu_conditions.get(matched_condition, matched_condition) if is_urdu else matched_condition
            reason = f"علامات {cond_name} سے ملتی جلتی ہیں ({highest_score} مماثل الفاظ)۔" if is_urdu else f"Symptoms strongly match {matched_condition} ({highest_score} matching symptom keywords)."
            action = "براہ کرم فراہم کردہ معلومات کے مطابق ڈاکٹر سے رجوع کریں۔" if is_urdu else cond_info["recommended_action"]
            red_flags = ", ".join(cond_info["red_flags"])
        else:
            urgency = "کم (Low)" if is_urdu else "Low"
            reason = "علامات ڈیٹا بیس میں کسی معلوم مقامی بیماری سے میل نہیں کھاتی ہیں۔ عام معتدل علامات فرض کی گئی ہیں۔" if is_urdu else "Symptoms do not match any known local condition in the database. General mild symptoms assumed."
            action = "آرام کریں، زیادہ مائعات پییں اور علامات کی نگرانی کریں۔ اگر علامات برقرار رہیں تو ڈاکٹر سے مشورہ کریں۔" if is_urdu else "Rest, drink fluids, and monitor symptoms. Consult a doctor if symptoms persist."
            red_flags = "شدید بخار، سینے میں درد، سانس لینے میں دشواری۔" if is_urdu else "High fever, chest pain, difficulty breathing."

        if is_urdu:
            cond_name = urdu_conditions.get(matched_condition, matched_condition)
            response_text = f"""
- **شدت کی درجہ بندی**: {urgency}
- **ممکنہ بیماری**: {cond_name}
- **وجوہات**: {reason}
- **سرخ جھنڈے / اہم علامات**: {red_flags}
"""
            thought_text = f"رول پر مبنی ٹریج ایجنٹ نے علامات کا جائزہ لیا۔ مماثل بیماری: {cond_name}۔"
        else:
            response_text = f"""
- **Urgency Classification**: {urgency}
- **Possible Condition**: {matched_condition}
- **Reasoning**: {reason}
- **Red Flags**: {red_flags}
"""
            thought_text = f"Rule-based Triage Agent scanned symptoms. Matched condition: {matched_condition}."
            
        return {
            "thought": thought_text,
            "response": response_text.strip(),
            "urgency": "Emergency" if is_emergency else ("High" if "High" in urgency or "زیادہ" in urgency else ("Moderate" if "Moderate" in urgency or "معتدل" in urgency else "Low"))
        }


class HealthAdvisorAgent:
    """Queries the RAG system for official WHO/MOH guidelines and drafts preventative advice."""
    def __init__(self, rag_system: RAGSystem, api_key=None):
        self.rag_system = rag_system
        self.api_key = api_key or get_api_key("GEMINI_API_KEY")
        self.use_llm = use_llm and (self.api_key and self.api_key != "your_gemini_api_key_here")

    def run(self, symptoms: str, triage_report: str, language: str = "English") -> dict:
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

IMPORTANT: You must write the entire health advice, summaries, thoughts, and next steps strictly in the language: {language}. If {language} is Urdu, write everything in Urdu (including references).
"""
                response = model.generate_content(prompt)
                thought_text = "ہیلتھ ایڈوائزر نے کروما ڈی بی رہنما خطوط کا مطالعہ کر کے طرزِ زندگی کی تجاویز تیار کیں۔" if language == "Urdu" else "Queried RAG system for WHO/MOH guidelines and synthesized lifestyle recommendations."
                return {
                    "thought": thought_text,
                    "response": response.text,
                    "retrieved_docs": rag_results
                }
            except Exception as e:
                print(f"HealthAdvisorAgent LLM run failed: {e}. Falling back...")
                
        return self._fallback_run(symptoms, triage_report, rag_results, language)

    def _fallback_run(self, symptoms: str, triage_report: str, rag_results: list, language: str = "English") -> dict:
        is_urdu = language == "Urdu"
        summary_lines = []
        for i, res in enumerate(rag_results):
            doc = res['doc']
            if is_urdu:
                summary_lines.append(f"رہنما خطوط کے مطابق ({doc['source']}): {doc['content']}")
            else:
                summary_lines.append(f"According to {doc['source']}: {doc['content']}")
        
        guidelines_summary = "\n\n".join(summary_lines)
        
        if is_urdu:
            advice_text = f"""
### 1. سرکاری WHO/MOH رہنما خطوط کا خلاصہ
{guidelines_summary}

### 2. حفاظتی اور طرز زندگی کے بارے میں مشورہ
- **ہائیڈریشن**: پانی، ہلکی یخنی، یا او آر ایس کا مناسب استعمال یقینی بنائیں۔
- **آرام**: زیادہ سے زیادہ آرام کر کے اپنے جسم کو بحال ہونے دیں۔
- **حفظان صحت اور روک تھام**: کھانستے یا چھینکتے وقت منہ ڈھانپیں، ہاتھ بار بار دھوئیں، اور دوسروں کے ساتھ قریبی رابطے سے گریز کریں۔
- **خوراک**: پیٹ کی خرابی کی صورت میں ہلکی اور سادہ غذا کھائیں، یا متوازن اور غذائیت سے بھرپور خوراک لیں۔

### 3. اگلے اقدامات اور نگرانی
- اگلے 24 سے 48 گھنٹوں کے دوران درجہ حرارت اور علامات میں تبدیلیوں کی نگرانی کریں۔
- اگر علامات میں بہتری نہ آئے، یا اگر خطرے کی علامات ظاہر ہوں، تو فوری طور پر ڈاکٹر سے رابطہ کریں۔
"""
            thought_text = "رول پر مبنی ہیلتھ ایڈوائزر نے RAG دستاویزات سے مشورہ مرتب کیا۔"
        else:
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
            thought_text = "Rule-based Health Advisor compiled advice from retrieved RAG documents."
            
        return {
            "thought": thought_text,
            "response": advice_text.strip(),
            "retrieved_docs": rag_results
        }


class GuardrailAgent:
    """Enforces safety guardrails, prefixes emergency banners, and appends a medical disclaimer."""
    def __init__(self, api_key=None):
        self.api_key = api_key or get_api_key("GEMINI_API_KEY")
        self.use_llm = use_llm and (self.api_key and self.api_key != "your_gemini_api_key_here")

    def run(self, symptoms: str, triage_report: str, advisor_report: str, urgency: str, language: str = "English") -> dict:
        is_urdu = language == "Urdu"
        disclaimer = """
> [!WARNING]
> **طبی ڈس کلیمر (Medical Disclaimer)**: میں ایک خودکار اے آئی ایجنٹ ہوں، کوئی ڈاکٹر نہیں۔ یہ رپورٹ صرف معلوماتی مقاصد کے لیے مقامی ٹریج قواعد اور آفیشل MOH/WHO گائیڈ لائنز کی روشنی میں مرتب کی گئی ہے۔ یہ کسی بھی طرح سے رسمی طبی تشخیص یا علاج کی جگہ نہیں لے سکتی۔ طبی مسائل کے حل کے لیے ہمیشہ مستند معالج سے رجوع کریں۔ ہنگامی حالت میں فوری طور پر قریبی ایمرجنسی سروس سے رابطہ کریں۔
""" if is_urdu else """
> [!WARNING]
> **Medical Disclaimer**: I am an AI agent coordinator, not a doctor. This report is compiled from local triage rules and official MOH/WHO guidelines for informational purposes only. It does not constitute formal medical diagnosis or treatment advice. Always consult a qualified healthcare provider for medical concerns. In case of emergency, contact your local emergency services immediately.
"""
        
        if self.use_llm:
            try:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                
                if is_urdu:
                    prompt = f"""
آپ میڈیکل سیفٹی اور گارڈ ریل ایجنٹ ہیں۔ آپ کا کام ٹریج ایجنٹ اور ہیلتھ ایڈوائزر ایجنٹ کی رپورٹوں کا جائزہ لے کر حتمی صحت کی رپورٹ مرتب کرنا ہے۔

صارف کی علامات: "{symptoms}"
شدت کا درجہ: {urgency}

ٹریج رپورٹ:
{triage_report}

ایڈوائزر رپورٹ:
{advisor_report}

قواعد:
1. **سخت فارمیٹ**: آپ کو اپنی آؤٹ پٹ کو بغیر کسی اضافی گفتگو یا تعارفی پیراگراف کے بالکل اسی طرح ترتیب دینا ہو گا:
- **شدت کا درجہ**: [شدت کا درجہ یہاں درج کریں]
- **علامات**: [صارف کی بتائی گئی علامات کی فہرست بنائیں]
- **ممکنہ تشخیص**: [ممکنہ بیماری یہاں درج کریں]
- **تجویز کردہ اقدامات**: [حفاظتی اقدامات اور اگلے اقدامات کی بلٹ پوائنٹ فہرست بنائیں]

2. **سب سے پہلے حفاظت**: اگر شدت "Emergency" یا ہنگامی ہے، تو یقینی بنائیں کہ آپ کے تجویز کردہ اقدامات ہنگامی خدمات کو کال کرنے کی ضرورت پر زور دیتے ہیں۔
3. **لہجہ**: رپورٹ کو سنجیدہ، معروضی اور محتاط رکھیں۔ کبھی بھی حتمی تشخیص کی ضمانت نہ دیں۔

صرف اردو میں جواب دیں۔
"""
                else:
                    prompt = f"""
You are a Medical Safety and Guardrails Agent. Your task is to compile the final health report by reviewing the outputs of the Triage Agent and the Health Advisor Agent.

User Symptoms: "{symptoms}"
Urgency level identified: {urgency}

Triage Report:
{triage_report}

Advisor Report:
{advisor_report}

Your rules:
1. **Strict Structure**: You MUST format your output EXACTLY as follows, using these exact headings as bullet points. DO NOT add any conversational text, introductory paragraphs, or concluding remarks.
- **Severity of issue**: [Insert Urgency Level]
- **Symptoms**: [List the symptoms the user mentioned]
- **Possible diagnosis**: [Insert Possible Condition]
- **Recommended actions**: [Provide bullet points of the preventative advice and next steps]

2. **Safety First**: If the urgency is "Emergency", ensure your recommended actions strongly emphasize calling local emergency services.
3. **Tone**: Keep it highly structured, objective, and cautious. Never guarantee a diagnosis.

Return ONLY the structured format.
"""
                response = model.generate_content(prompt)
                final_text = response.text
                
                # Double-check disclaimer presence
                if "disclaimer" not in final_text.lower() and "ڈس کلیمر" not in final_text:
                    final_text = disclaimer + "\n" + final_text
                
                thought_text = "حتمی رپورٹ کے ڈھانچے کا جائزہ لیا گیا اور طبی ڈس کلیمر شامل کیا گیا۔" if is_urdu else "Reviewed final report structure, validated emergency status, and injected necessary medical disclaimers."
                return {
                    "thought": thought_text,
                    "response": final_text
                }
            except Exception as e:
                print(f"GuardrailAgent LLM run failed: {e}. Falling back...")
                
        return self._fallback_run(symptoms, triage_report, advisor_report, urgency, disclaimer, language)

    def _fallback_run(self, symptoms: str, triage_report: str, advisor_report: str, urgency: str, disclaimer: str, language: str = "English") -> dict:
        is_urdu = language == "Urdu"
        condition = "نامعلوم" if is_urdu else "Unknown"
        
        possible_keys = ["Possible Condition", "ممکنہ بیماری"]
        for line in triage_report.split('\n'):
            for pk in possible_keys:
                if pk in line:
                    condition = line.split(":", 1)[1].strip()
                    break
        
        if is_urdu:
            final_text = f"""{disclaimer}

- **شدت کا درجہ**: {urgency}
- **علامات**: {symptoms}
- **ممکنہ تشخیص**: {condition}
- **تجویز کردہ اقدامات**: 
{advisor_report}
"""
            thought_text = "رول پر مبنی گارڈ ریل ایجنٹ نے حتمی رپورٹ کو اردو میں ترتیب دیا اور طبی ڈس کلیمر شامل کیا۔"
        else:
            final_text = f"""{disclaimer}

- **Severity of issue**: {urgency.upper()}
- **Symptoms**: {symptoms}
- **Possible diagnosis**: {condition}
- **Recommended actions**: 
{advisor_report}
"""
            thought_text = "Rule-based Guardrail Agent formatted final output strictly into 4 categories and appended medical disclaimer."
            
        return {
            "thought": thought_text,
            "response": final_text.strip()
        }


class UIFormatterAgent:
    """Uses Gemini to format the final output into a strict UI-renderable JSON. Enforces strict anti-hallucination guardrails."""
    def __init__(self, api_key=None):
        self.api_key = api_key or get_api_key("GEMINI_API_KEY_2") or get_api_key("GEMINI_API_KEY")

    def run(self, symptoms: str, triage_report: str, advisor_report: str, urgency: str, language: str = "English") -> dict:
        if not self.api_key:
            return {
                "thought": "Gemini API Key 2 not provided.",
                "response": {"error": "Missing Gemini API Key. The UI Formatter Agent requires a Gemini key to function."},
                "is_json": True
            }
            
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                "gemini-2.5-flash",
                generation_config={"response_mime_type": "application/json"}
            )
            
            prompt = f"""
You are the final Medical Data Formatter. Your task is to output STRICT JSON that will be rendered by the custom HTML UI.
NO GUESSING / ASSUMPTIONS / "MAYBE" / "PERHAPS". You must only use the retrieved information from the Triage and Advisor reports.

User Symptoms: "{symptoms}"
Urgency level identified: {urgency}

Triage Report:
{triage_report}

Advisor Report:
{advisor_report}

Rules:
1. Output MUST be valid JSON with exactly 4 keys: "severity", "symptoms", "diagnosis", "actions".
2. "severity": String translated/written in {language} (e.g. "High", "Emergency" or "ہنگامی").
3. "symptoms": List of strings representing bullet points of symptoms in {language}.
4. "diagnosis": String in {language}. If the disease name is identified in the RAG or Triage reports, you MUST showcase the disease name, even if it is a highly critical emergency. ONLY if the disease is completely unknown/absent from the vector database/reports, you must output exactly the translated equivalent of "Unknown - Physical doctor appointment recommended" in {language} (e.g. "نامعلوم - ڈاکٹر سے رجوع کرنے کی سفارش کی جاتی ہے").
5. "actions": List of strings in {language}. If the diagnosis is unknown, you must add an action explicitly stating "Physical doctor appointment recommended" in {language} accompanied by the urgency.

Return ONLY the raw JSON object.
"""
            response = model.generate_content(prompt)
            raw_json = response.text.strip()
            parsed_json = json.loads(raw_json)
            
            thought_text = "جیمیٹائی یو آئی فارمیٹر ایجنٹ نے رپورٹس کو پڑھ کر اردو میں محفوظ JSON مرتب کیا۔" if language == "Urdu" else "Gemini UI Formatter Agent parsed reports and returned strict JSON."
            return {
                "thought": thought_text,
                "response": parsed_json,
                "is_json": True
            }
        except Exception as e:
            return {
                "thought": f"Gemini UI Formatter Agent failed.",
                "response": {"error": f"Gemini API Error: {str(e)}"},
                "is_json": True
            }


class HealthCoordinator:
    """Orchestrator for coordinates interactions between Triage, Advisor, and Guardrail Agents."""
    def __init__(self, api_key=None, gemini_api_key_2=None):
        self.api_key = api_key or get_api_key("GEMINI_API_KEY")
        self.gemini_api_key_2 = gemini_api_key_2 or get_api_key("GEMINI_API_KEY_2") or self.api_key
        self.rag_system = RAGSystem()
        self.triage_agent = TriageAgent(api_key=self.api_key)
        self.advisor_agent = HealthAdvisorAgent(self.rag_system, api_key=self.api_key)
        self.guardrail_agent = GuardrailAgent(api_key=self.api_key)
        self.ui_agent = UIFormatterAgent(api_key=self.gemini_api_key_2)

    def process_query(self, symptoms: str, language: str = "English") -> dict:
        # Step 1: Run Triage Agent
        triage_out = self.triage_agent.run(symptoms, language=language)
        
        # Check if insufficient info
        if triage_out["urgency"] == "Insufficient Info":
            return {
                "triage": triage_out,
                "advisor": {"thought": "Skipped because more information is needed.", "response": "N/A", "retrieved_docs": []},
                "guardrail": {
                    "thought": "Skipped formatting. Returning triage clarification questions.",
                    "response": f"**Insufficient Information**\n\nThe Triage Agent requires more details to proceed:\n\n{triage_out['response']}"
                }
            }
        
        # Step 2: Run Health Advisor Agent using RAG
        advisor_out = self.advisor_agent.run(symptoms, triage_out["response"], language=language)
        
        # Step 3: Run Guardrail Agent
        guardrail_out = self.guardrail_agent.run(
            symptoms=symptoms,
            triage_report=triage_out["response"],
            advisor_report=advisor_out["response"],
            urgency=triage_out["urgency"],
            language=language
        )
        
        # Step 4: Run UI Formatter Agent (Agent 4)
        ui_out = self.ui_agent.run(
            symptoms=symptoms,
            triage_report=triage_out["response"],
            advisor_report=advisor_out["response"],
            urgency=triage_out["urgency"],
            language=language
        )
        
        return {
            "triage": triage_out,
            "advisor": advisor_out,
            "guardrail": guardrail_out,
            "ui": ui_out
        }
