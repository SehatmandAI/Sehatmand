import streamlit as st
import os
from dotenv import load_dotenv

# Try loading env variables
load_dotenv()

# Flexible imports to support running from both root and subfolder
try:
    from agents import HealthCoordinator
    from knowledge_base import TRIAGE_KNOWLEDGE, check_ollama_status, get_db_document_count
    from knowledge_update import update_knowledge_base
except ImportError:
    from sehatmand.agents import HealthCoordinator
    from sehatmand.knowledge_base import TRIAGE_KNOWLEDGE, check_ollama_status, get_db_document_count
    from sehatmand.knowledge_update import update_knowledge_base

# Page Configuration
st.set_page_config(
    page_title="Sehatmand 🩺 | Multiagent Triage & Advisor [DEMO]",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header & Theme tweaks */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2dd4bf 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Card Design */
    .agent-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .triage-card { border-left: 5px solid #f59e0b; }
    .advisor-card { border-left: 5px solid #3b82f6; }
    .guardrail-card { border-left: 5px solid #10b981; }
    
    .agent-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .triage-title { color: #f59e0b; }
    .advisor-title { color: #3b82f6; }
    .guardrail-title { color: #10b981; }
    
    .agent-thought {
        font-size: 0.9rem;
        font-style: italic;
        color: #64748b;
        margin-bottom: 12px;
        padding-left: 8px;
        border-left: 2px solid #475569;
    }
    
    .agent-content {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .badge-llm {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-fallback {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    /* Source list styling */
    .source-box {
        background-color: #0f172a;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #1e293b;
        margin-top: 10px;
    }
    .source-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #38bdf8;
    }
    .source-content {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    /* Block UI Custom CSS */
    .custom-block {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .block-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 12px;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .severity-high { border-left: 5px solid #ef4444; }
    .severity-moderate { border-left: 5px solid #f59e0b; }
    .severity-low { border-left: 5px solid #10b981; }
    .diagnosis-card { background-color: rgba(56, 189, 248, 0.05); border-left: 5px solid #38bdf8; }
    .symptoms-card { border-left: 5px solid #8b5cf6; }
    .actions-card { border-left: 5px solid #14b8a6; }
    .ul-styled { padding-left: 20px; margin: 0; color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; }
    .ul-styled li { margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/external-flatart-icons-flat-flatarticons/128/external-medical-medical-health-flatart-icons-flat-flatarticons.png", width=80)
st.sidebar.title("Configuration")

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

# API Key handling & Status
gemini_key = get_api_key("GEMINI_API_KEY")
gemini_key_2 = get_api_key("GEMINI_API_KEY_2")

# If gemini_key_2 is missing, we can reuse gemini_key as a fallback
if not gemini_key_2:
    gemini_key_2 = gemini_key

has_gemini = gemini_key != ""
has_gemini_2 = gemini_key_2 != ""

# Initialize coordinator
if has_gemini or has_gemini_2:
    try:
        coordinator = HealthCoordinator(
            api_key=gemini_key if has_gemini else None,
            gemini_api_key_2=gemini_key_2 if has_gemini_2 else None
        )
        is_llm_running = coordinator.triage_agent.use_llm
    except Exception:
        coordinator = HealthCoordinator()
        is_llm_running = False
else:
    coordinator = HealthCoordinator()
    is_llm_running = False

# Sidebar LLM status display
st.sidebar.subheader("LLM Service Status")

if is_llm_running:
    st.sidebar.markdown('🌐 Gemini API Key 1: **Active** 🟢<br><span style="color: #10b981; font-size: 0.85rem; font-weight: 600;">Model: gemini-2.5-flash</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('🌐 Gemini API Key 1: **Inactive** 🟠<br><span style="color: #f59e0b; font-size: 0.85rem; font-weight: 600;">Local fallback active (no-key)</span>', unsafe_allow_html=True)

st.sidebar.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

if has_gemini_2:
    st.sidebar.markdown('🧠 Gemini API Key 2: **Active** 🟢<br><span style="color: #10b981; font-size: 0.85rem; font-weight: 600;">Model: gemini-2.5-flash (Agent 4)</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('🧠 Gemini API Key 2: **Inactive** 🔴<br><span style="color: #ef4444; font-size: 0.85rem; font-weight: 600;">Using Key 1 fallback or Mock</span>', unsafe_allow_html=True)


# --- RAG Database Control Panel ---
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Knowledge Base Control")

db_count = get_db_document_count()
st.sidebar.markdown("🧠 Embeddings: **Sentence-Transformers** (`all-MiniLM-L6-v2`)")
st.sidebar.markdown(f"📄 Documents in DB: **{db_count}**")

# PDF File Uploader (upload only — no local path input)
uploaded_file = st.sidebar.file_uploader("Upload Guidelines PDF", type=["pdf"])

if st.sidebar.button("🔄 Sync Database"):
    status_placeholder = st.sidebar.empty()
    def ui_status_callback(msg):
        status_placeholder.info(msg)
        
    with st.spinner("Processing document and generating embeddings..."):
        try:
            if uploaded_file is not None:
                # Save uploaded file to a writable temp location
                target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_guidelines.pdf")
                with open(target_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            else:
                # No file uploaded — generate mock guidelines document
                target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guidelines.txt")
                
            chunks_added = update_knowledge_base(file_path=target_path, status_callback=ui_status_callback)
            st.sidebar.success(f"✅ {chunks_added} chunks added to ChromaDB.")
        except Exception as e:
            st.sidebar.error(f"Error updating database: {e}")

# Database inspector in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Local Triage Conditions")
conditions_list = list(TRIAGE_KNOWLEDGE.keys())
selected_condition = st.sidebar.selectbox("Inspect Triage Rules:", conditions_list)
if selected_condition:
    cond_data = TRIAGE_KNOWLEDGE[selected_condition]
    st.sidebar.markdown(f"**Symptoms**: {', '.join(cond_data['symptoms'])}")
    st.sidebar.markdown(f"**Urgency**: {cond_data['urgency']}")
    st.sidebar.markdown(f"**Action**: {cond_data['recommended_action']}")

st.sidebar.markdown("---")
st.sidebar.caption("Sehatmand multiagent system v1.0. Designed for hackathons and demonstration purposes.")

# Translation dictionary
TRANSLATIONS = {
    "English": {
        "title": "🩺 Sehatmand Triage & Advisor [DEMO]",
        "subtitle": "A multiagent cooperative healthcare assessment tool using local triage rules and WHO/MOH guidelines.",
        "describe_symptoms": "Describe Your Symptoms",
        "explain_feeling": "Please explain how you are feeling (e.g. onset, duration, and specific symptoms):",
        "placeholder": "Example: I've had a sudden dry cough and muscle aches since yesterday. I also have a mild fever...",
        "choose_template": "Or choose a quick demo template:",
        "temp_cold": "🤒 Common Cold / Flu Symptoms",
        "temp_food": "🤢 Food Poisoning Symptoms",
        "temp_emergency": "🚨 Emergency Warning Signals",
        "run_eval": "Run Multiagent Evaluation",
        "backend_logs": "🤖 Backend Agent Logs",
        "final_report": "📋 Final Safety Assessed Health Report",
        "severity": "Severity Level",
        "diagnosis": "Possible Diagnosis",
        "symptoms_eval": "Symptoms Evaluated",
        "actions": "Recommended Actions",
        "insufficient_info": "⚠️ Insufficient Information Provided",
        "your_answer": "Your Answer:",
        "submit_details": "Submit Details",
        "triage_status": "Agent 1: Triage Agent ({}) evaluating symptoms...",
        "advisor_status": "Agent 2: Health Advisor ({}) querying ChromaDB...",
        "guardrail_status": "Agent 3: Guardrail Agent ({}) verifying safety...",
        "ui_status": "Agent 4: UI Formatter Agent ({}) generating JSON blocks...",
        "triage_expander": "🔍 Agent 1: Triage Agent ({}) Returns:",
        "advisor_expander": "💡 Agent 2: Health Advisor Agent ({}) Returns:",
        "guardrail_expander": "🛡️ Agent 3: Guardrail Agent ({}) Returns:",
        "ui_expander": "✨ Agent 4: UI Formatter Agent ({}) Returns:",
        "rag_retrieved": "RAG Documents Retrieved:",
        "please_enter": "Please enter some symptoms to evaluate.",
        "thought_process": "Thought Process"
    },
    "Urdu": {
        "title": "🩺 صحت مند | ٹریج اور ایڈوائزر [DEMO]",
        "subtitle": "مقامی ٹریج قواعد اور ڈبلیو ایچ او/ایم او ایچ رہنما خطوط کا استعمال کرتے ہوئے ایک کثیر ایجنٹ کوآپریٹو ہیلتھ کیئر تشخیص ٹول۔",
        "describe_symptoms": "اپنی علامات بیان کریں",
        "explain_feeling": "براہ کرم وضاحت کریں کہ آپ کیسا محسوس کر رہے ہیں (مثال کے طور پر شروعات، دورانیہ اور مخصوص علامات):",
        "placeholder": "مثال: مجھے کل سے اچانک خشک کھانسی اور پٹھوں میں درد ہے۔ مجھے ہلکا بخار بھی ہے...",
        "choose_template": "یا ایک فوری ڈیمو ٹیمپلیٹ منتخب کریں:",
        "temp_cold": "🤒 نزلہ زکام / فلو کی علامات",
        "temp_food": "🤢 فوڈ پوائزننگ کی علامات",
        "temp_emergency": "🚨 ہنگامی وارننگ سگنلز",
        "run_eval": "کثیر ایجنٹ تشخیص چلائیں",
        "backend_logs": "🤖 ایجنٹ لاگز (بیک اینڈ)",
        "final_report": "📋 حتمی اور محفوظ طبی رپورٹ",
        "severity": "شدت کا درجہ",
        "diagnosis": "ممکنہ تشخیص",
        "symptoms_eval": "علامات کا جائزہ لیا گیا",
        "actions": "تجویز کردہ اقدامات",
        "insufficient_info": "⚠️ ناکافی معلومات فراہم کی گئیں",
        "your_answer": "آپ کا جواب:",
        "submit_details": "تفصیلات جمع کروائیں",
        "triage_status": "ایجنٹ 1: ٹریج ایجنٹ ({}) علامات کا جائزہ لے رہا ہے...",
        "advisor_status": "ایجنٹ 2: ہیلتھ ایڈوائزر ({}) کروما ڈی بی سے استفسار کر رہا ہے...",
        "guardrail_status": "ایجنٹ 3: گارڈ ریل ایجنٹ ({}) حفاظت کی تصدیق کر رہا ہے...",
        "ui_status": "ایجنٹ 4: یو آئی فارمیٹر ایجنٹ ({}) بلاکس تیار کر رہا ہے...",
        "triage_expander": "🔍 ایجنٹ 1: ٹریج ایجنٹ ({}) کے نتائج:",
        "advisor_expander": "💡 ایجنٹ 2: ہیلتھ ایڈوائزر ایجنٹ ({}) کے نتائج:",
        "guardrail_expander": "🛡️ ایجنٹ 3: گارڈ ریل ایجنٹ ({}) کے نتائج:",
        "ui_expander": "✨ ایجنٹ 4: یو آئی فارمیٹر ایجنٹ ({}) کے نتائج:",
        "rag_retrieved": "حاصل کردہ RAG رہنما خطوط:",
        "please_enter": "براہ کرم تشخیص کے لیے علامات درج کریں۔",
        "thought_process": "سلسلہ خیال (Thought Process)"
    }
}

# Language Selector layout at top right
col_title, col_lang = st.columns([8, 2])

# Handle language selection state
if "lang" not in st.session_state:
    st.session_state["lang"] = "English"

with col_lang:
    selected_lang = st.selectbox(
        "Language / زبان",
        options=["English", "Urdu"],
        key="lang"
    )

t = TRANSLATIONS[selected_lang]

# Dynamic RTL support for Urdu
if selected_lang == "Urdu":
    st.markdown("""
    <style>
        .main-title, .subtitle, p, ul, li, div, h1, h2, h3, h4, h5, h6, .custom-block, .agent-card, label, textarea, button, input {
            direction: rtl !important;
            text-align: right !important;
        }
        .ul-styled {
            padding-right: 20px !important;
            padding-left: 0 !important;
        }
        /* Fix streamlit column formatting under RTL */
        div[data-testid="stColumn"] {
            direction: rtl !important;
        }
        /* Expander headers text-align fix */
        div[data-testid="stExpander"] {
            direction: rtl !important;
            text-align: right !important;
        }
    </style>
    """, unsafe_allow_html=True)

with col_title:
    st.markdown(f'<div class="main-title">{t["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{t["subtitle"]}</div>', unsafe_allow_html=True)

# Symptom Input Area
st.subheader(t["describe_symptoms"])

if "symptoms" not in st.session_state:
    st.session_state["symptoms"] = ""
if "symptoms_run" not in st.session_state:
    st.session_state["symptoms_run"] = False

def on_symptoms_change():
    st.session_state["symptoms_run"] = False

symptoms_input = st.text_area(
    t["explain_feeling"],
    height=120,
    key="symptoms",
    on_change=on_symptoms_change,
    placeholder=t["placeholder"]
)

# Example Symptom suggestions
st.markdown(f"**{t['choose_template']}**")
col_ex1, col_ex2, col_ex3 = st.columns(3)

def set_symptoms(text):
    st.session_state["symptoms"] = text
    st.session_state["symptoms_run"] = True

with col_ex1:
    st.button(t["temp_cold"], on_click=set_symptoms, args=("I have a runny nose, mild sore throat, sneezing, and a low fever that started 2 days ago.",))
with col_ex2:
    st.button(t["temp_food"], on_click=set_symptoms, args=("Feeling severe stomach cramps, nausea, and watery diarrhea since eating street food yesterday.",))
with col_ex3:
    st.button(t["temp_emergency"], on_click=set_symptoms, args=("Sudden onset of severe chest pain that is radiating to my left arm, along with shortness of breath and sweating.",))

st.markdown("<br>", unsafe_allow_html=True)

def on_main_submit():
    st.session_state["symptoms_run"] = True

submit_button = st.button(t["run_eval"], use_container_width=True, on_click=on_main_submit)

if st.session_state.get("symptoms_run"):
    current_symptoms = st.session_state["symptoms"]
    if not current_symptoms.strip():
        st.error(t["please_enter"])
    else:
        st.markdown("---")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Layout for results: Column 1 for Final Output, Column 2 for Agent Collaboration
        col_final, col_agents = st.columns([3, 2])
        
        with col_agents:
            st.subheader(t["backend_logs"])
            triage_box = st.empty()
            advisor_box = st.empty()
            guardrail_box = st.empty()
            ui_box = st.empty()
            
        # Determine active LLMs/engines
        triage_llm = "Gemini: gemini-2.5-flash" if coordinator.triage_agent.use_llm else "Rule-based Fallback"
        advisor_llm = "Gemini: gemini-2.5-flash + ChromaDB RAG" if coordinator.advisor_agent.use_llm else "ChromaDB RAG + Keyword Fallback"
        guardrail_llm = "Gemini: gemini-2.5-flash" if coordinator.guardrail_agent.use_llm else "Rule-based Fallback"
        ui_llm = "Gemini: gemini-2.5-flash (Key 2)" if has_gemini_2 else "Missing Key"

        status_text.markdown("### 🔄 Status: " + t["triage_status"].format(triage_llm))
        triage_res = coordinator.triage_agent.run(current_symptoms, language=selected_lang)
        progress_bar.progress(25)
        
        with triage_box.container():
            with st.expander(t["triage_expander"].format(triage_llm), expanded=True):
                st.caption(f"**{t['thought_process']}**: {triage_res['thought']}")
                st.markdown(triage_res['response'])
                
        if triage_res["urgency"] == "Insufficient Info":
            progress_bar.empty()
            status_text.empty()
            with col_final:
                st.warning(t["insufficient_info"])
                st.markdown(triage_res["response"])
                with st.form("followup_form"):
                    st.text_input(t["your_answer"], key="followup_answer")
                    def submit_followup():
                        ans = st.session_state.get("followup_answer", "")
                        st.session_state["symptoms"] += "\n\nFollow-up Details: " + ans
                        st.session_state["symptoms_run"] = True
                    st.form_submit_button(t["submit_details"], on_click=submit_followup)
        else:
            status_text.markdown("### 🔄 Status: " + t["advisor_status"].format(advisor_llm))
            advisor_res = coordinator.advisor_agent.run(current_symptoms, triage_res["response"], language=selected_lang)
            progress_bar.progress(50)
            
            with advisor_box.container():
                with st.expander(t["advisor_expander"].format(advisor_llm), expanded=True):
                    st.caption(f"**{t['thought_process']}**: {advisor_res['thought']}")
                    st.markdown(advisor_res['response'])
                    retrieved_docs = advisor_res.get("retrieved_docs", [])
                    if retrieved_docs:
                        st.markdown(f"**{t['rag_retrieved']}**")
                        for item in retrieved_docs:
                            doc = item["doc"]
                            st.caption(f"📄 {doc['source']} (Match: {item.get('score', 0.0):.2f})")
                            st.text(doc['content'])
                            
            status_text.markdown("### 🔄 Status: " + t["guardrail_status"].format(guardrail_llm))
            guardrail_res = coordinator.guardrail_agent.run(current_symptoms, triage_res["response"], advisor_res["response"], triage_res["urgency"], language=selected_lang)
            progress_bar.progress(75)
            
            with guardrail_box.container():
                with st.expander(t["guardrail_expander"].format(guardrail_llm), expanded=True):
                    st.caption(f"**{t['thought_process']}**: {guardrail_res['thought']}")
                    st.markdown(guardrail_res['response'])
                    
            status_text.markdown("### 🔄 Status: " + t["ui_status"].format(ui_llm))
            ui_res = coordinator.ui_agent.run(current_symptoms, triage_res["response"], advisor_res["response"], triage_res["urgency"], language=selected_lang)
            progress_bar.progress(100)
            
            with ui_box.container():
                with st.expander(t["ui_expander"].format(ui_llm), expanded=True):
                    st.caption(f"**{t['thought_process']}**: {ui_res.get('thought', '')}")
                    st.json(ui_res.get('response', {}))
                    
            status_text.empty()
            
            with col_final:
                st.subheader(t["final_report"])
                ui_data = ui_res.get("response", {})
                if "error" in ui_data:
                    st.error(ui_data["error"])
                else:
                    severity = ui_data.get("severity", "UNKNOWN").upper()
                    sev_class = "severity-high" if "EMERGENCY" in severity or "HIGH" in severity or "ہنگامی" in severity or "شدید" in severity else "severity-moderate" if "MODERATE" in severity or "معتدل" in severity else "severity-low"
                    
                    st.markdown(f"""
                    <div class="custom-block {sev_class}">
                        <div class="block-header">🚨 {t['severity']}: {severity}</div>
                    </div>
                    
                    <div class="custom-block diagnosis-card">
                        <div class="block-header">🔍 {t['diagnosis']}</div>
                        <p style="color: #cbd5e1; font-size: 1.1rem; font-weight: 600; margin:0;">{ui_data.get('diagnosis', 'Unknown')}</p>
                    </div>
                    
                    <div class="custom-block symptoms-card">
                        <div class="block-header">🤒 {t['symptoms_eval']}</div>
                        <ul class="ul-styled">
                            {''.join([f"<li>{s}</li>" for s in ui_data.get('symptoms', [])])}
                        </ul>
                    </div>
                    
                    <div class="custom-block actions-card">
                        <div class="block-header">💡 {t['actions']}</div>
                        <ul class="ul-styled">
                            {''.join([f"<li>{a}</li>" for a in ui_data.get('actions', [])])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
