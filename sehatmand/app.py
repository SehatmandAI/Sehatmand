import streamlit as st
import os
from dotenv import load_dotenv

# Try loading env variables
load_dotenv()

# Flexible imports to support running from both root and subfolder
try:
    from agents import HealthCoordinator
    from knowledge_base import TRIAGE_KNOWLEDGE
except ImportError:
    from sehatmand.agents import HealthCoordinator
    from sehatmand.knowledge_base import TRIAGE_KNOWLEDGE

# Page Configuration
st.set_page_config(
    page_title="Sehatmand 🩺 | Multiagent Triage & Advisor",
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
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/external-flatart-icons-flat-flatarticons/128/external-medical-medical-health-flatart-icons-flat-flatarticons.png", width=80)
st.sidebar.title("Configuration")

# API Key handling
default_key = os.getenv("GEMINI_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key",
    value=default_key if default_key != "your_gemini_api_key_here" else "",
    type="password",
    help="Enter your Google AI Studio Gemini API key. If left blank, the app will run in fallback rule-based mode."
)

# Set input to environment variable dynamically if provided
active_api_key = api_key_input.strip() if api_key_input else None

# Check status of Coordinator initialization
if active_api_key:
    try:
        coordinator = HealthCoordinator(api_key=active_api_key)
        is_llm_running = coordinator.triage_agent.use_llm
    except Exception:
        coordinator = HealthCoordinator()
        is_llm_running = False
else:
    coordinator = HealthCoordinator()
    is_llm_running = False

# Sidebar status display
if is_llm_running:
    st.sidebar.markdown('<span class="status-badge badge-llm">🟢 Gemini LLM Active</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-badge badge-fallback">🟠 Local Fallback Active</span>', unsafe_allow_html=True)
    st.sidebar.warning("Running without a valid Gemini API key. Complex medical reasoning is mocked using local rules, and RAG retrieves using simple keyword overlaps. Provide a Gemini API key in the field above to enable full AI agent collaboration.")

# Database inspector in Sidebar
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

# Main Application Layout
st.markdown('<div class="main-title">🩺 Sehatmand Triage & Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A multiagent cooperative healthcare assessment tool using local triage rules and WHO/MOH guidelines.</div>', unsafe_allow_html=True)

# Symptom Input Area
st.subheader("Describe Your Symptoms")
symptoms_input = st.text_area(
    "Please explain how you are feeling (e.g. onset, severity, duration, and specific symptoms):",
    height=120,
    placeholder="Example: I've had a sudden dry cough and muscle aches since yesterday. I also have a mild fever..."
)

# Example Symptom suggestions
st.markdown("**Or choose a quick demo template:**")
col_ex1, col_ex2, col_ex3 = st.columns(3)

with col_ex1:
    if st.button("🤒 Common Cold / Flu Symptoms"):
        symptoms_input = "I have a runny nose, mild sore throat, sneezing, and a low fever that started 2 days ago."
        st.rerun()
with col_ex2:
    if st.button("🤢 Food Poisoning Symptoms"):
        symptoms_input = "Feeling severe stomach cramps, nausea, and watery diarrhea since eating street food yesterday."
        st.rerun()
with col_ex3:
    if st.button("🚨 Emergency Warning Signals"):
        symptoms_input = "Sudden onset of severe chest pain that is radiating to my left arm, along with shortness of breath and sweating."
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
submit_button = st.button("Run Multiagent Evaluation", use_container_width=True)

if submit_button or (symptoms_input and "symptoms_run" in st.session_state):
    if not symptoms_input.strip():
        st.error("Please enter some symptoms to evaluate.")
    else:
        with st.spinner("Agents are analyzing your request..."):
            # Run the multiagent coordinator
            results = coordinator.process_query(symptoms_input)
            
            triage_res = results["triage"]
            advisor_res = results["advisor"]
            guardrail_res = results["guardrail"]
            
            # Layout for results: Column 1 for Final Output, Column 2 for Agent Collaboration
            col_final, col_agents = st.columns([3, 2])
            
            with col_final:
                st.subheader("📋 Final Safety Assessed Health Report")
                st.markdown(guardrail_res["response"])
                
            with col_agents:
                st.subheader("🤖 Agent Collaboration & Thought Logs")
                
                # Triage Agent Card
                st.markdown(f"""
                <div class="agent-card triage-card">
                    <div class="agent-title triage-title">🔍 Triage Agent</div>
                    <div class="agent-thought">Thought: {triage_res['thought']}</div>
                    <div class="agent-content">{triage_res['response']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Advisor Agent Card
                advisor_content = advisor_res['response']
                st.markdown(f"""
                <div class="agent-card advisor-card">
                    <div class="agent-title advisor-title">💡 Health Advisor Agent (RAG)</div>
                    <div class="agent-thought">Thought: {advisor_res['thought']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Expand advisor contents
                with st.expander("View Advisor Synthesis", expanded=True):
                    st.markdown(advisor_content)
                
                # RAG Sources Box
                with st.expander("🌐 Retrieved MOH/WHO Reference Guidelines (RAG)", expanded=True):
                    retrieved_docs = advisor_res.get("retrieved_docs", [])
                    if retrieved_docs:
                        for item in retrieved_docs:
                            doc = item["doc"]
                            score = item.get("score", 0.0)
                            method = item.get("method", "RAG")
                            st.markdown(f"""
                            <div class="source-box">
                                <div class="source-title">📄 {doc['source']} <span style="float:right; font-size:0.75rem; color:#10b981;">(Match: {score:.2f} via {method})</span></div>
                                <div class="source-content">{doc['content']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No explicit guideline matched. Standard health tips provided.")
                
                # Guardrail Agent Card
                st.markdown(f"""
                <div class="agent-card guardrail-card">
                    <div class="agent-title guardrail-title">🛡️ Guardrail & Safety Agent</div>
                    <div class="agent-thought">Thought: {guardrail_res['thought']}</div>
                    <div class="agent-content">
                        Ensured medical disclaimers were present, evaluated urgency, and formatted outputs. Urgency status: <strong>{triage_res['urgency']}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        # Persist text input state
        st.session_state["symptoms_run"] = True
