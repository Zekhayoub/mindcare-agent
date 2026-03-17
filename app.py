"""MindCare Agent — Streamlit UI."""

import logging
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from src.config import CONFIG
from src.tools import MindCareTools
from src.strategist import MindCareStrategist
from src.agent import build_agent, invoke_agent, extract_location
from src.analysis import detect_context, determine_intensity, get_emotion_score
from src.carbon import calculate_co2

import folium
from streamlit_folium import st_folium

# ------------------------------------------------------------------
# Bootstrap (runs once)
# ------------------------------------------------------------------

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Cached resource loading
# ------------------------------------------------------------------


@st.cache_resource
def load_tools() -> MindCareTools:
    """Load ML models, advice DB, and RAG vectorstore once."""
    return MindCareTools()


@st.cache_resource
def load_strategist() -> MindCareStrategist:
    """Load the ECO/AGENT routing strategist once."""
    return MindCareStrategist()


@st.cache_resource
def load_agent(_tools: MindCareTools):
    """Build the LangChain agent once."""
    try:
        return build_agent(_tools)
    except ValueError as exc:
        logger.error("Agent build failed: %s", exc)
        return None


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(page_title="MindCare Agent", page_icon="🧠", layout="wide")

EMOTION_COLORS = {
    "Joy": "#FFD700", "Love": "#FF69B4", "Surprise": "#FFA500",
    "Unknown": "#808080", "Fear": "#9370DB", "Sadness": "#1E90FF", "Anger": "#FF4500",
}

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = {k: 0 for k in EMOTION_COLORS}
if "emotion_timeline" not in st.session_state:
    st.session_state.emotion_timeline = []
if "total_co2" not in st.session_state:
    st.session_state.total_co2 = 0.0
if "user_location" not in st.session_state:
    st.session_state.user_location = None

# ------------------------------------------------------------------
# Load resources
# ------------------------------------------------------------------

tools_instance = load_tools()
strategist = load_strategist()
agent_executor = load_agent(tools_instance)

# ------------------------------------------------------------------
# Main chat area
# ------------------------------------------------------------------

st.title("🧠 MindCare")
st.markdown("##### Your intelligent emotional support companion (Eco-Designed)")
st.info(
    "I use a hybrid architecture: local knowledge (Eco Mode) or "
    "Cloud Mistral (Expert Mode) depending on request complexity."
)

# Display history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        role_avatar = "🌱" if getattr(message, "eco_mode", False) else "☁️"
        with st.chat_message("assistant", avatar=role_avatar):
            st.markdown(message.content)

# Input
user_input = st.chat_input("Express how you feel...")

if user_input:
    # Location detection
    detected_loc = extract_location(user_input)
    if detected_loc:
        st.session_state.user_location = detected_loc
        st.toast(f"Location detected: {detected_loc}")

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # ------------------------------------------------------------------
    # Step 1: ML pre-analysis
    # ------------------------------------------------------------------

    try:
        raw_analysis = tools_instance.classify_emotion(user_input)
        detected_emotion = raw_analysis.get("emotion", "unknown").lower()
        confidence = raw_analysis.get("confidence", 0)
        secondary = raw_analysis.get("secondary_emotions", {})
    except Exception:
        detected_emotion, confidence, secondary = "unknown", 0, {}
        raw_analysis = {"emotion": "unknown", "confidence": 0}

    # Update dashboard stats
    cap_emotion = detected_emotion.capitalize()
    if cap_emotion in st.session_state.emotion_log:
        st.session_state.emotion_log[cap_emotion] += 1
    else:
        st.session_state.emotion_log["Unknown"] += 1

    st.session_state.emotion_timeline.append({
        "Step": len(st.session_state.emotion_timeline) + 1,
        "Score": get_emotion_score(detected_emotion),
        "Emotion": cap_emotion,
    })

    # ------------------------------------------------------------------
    # Step 2: Strategy decision
    # ------------------------------------------------------------------

    strategy, reason = strategist.decide_strategy(user_input, raw_analysis)
    use_eco_mode = strategy == "ECO"

    # ------------------------------------------------------------------
    # Step 3: Response generation
    # ------------------------------------------------------------------

    detected_context = detect_context(user_input)
    detected_intensity = determine_intensity(confidence, user_input)

    if use_eco_mode:
        # --- ECO MODE ---
        eco_avatar = "🌱"
        with st.chat_message("assistant", avatar=eco_avatar):
            advice, note = tools_instance.get_advice(
                detected_emotion,
                intensity=detected_intensity,
                context=detected_context,
                confidence=confidence,
            )
            activity_data = tools_instance.get_activity(
                detected_emotion,
                st.session_state.user_location or "Brussels",
            )
            clinical_info = tools_instance.get_clinical_excerpt(detected_emotion)

            ai_response = f"I sense you are feeling **{detected_emotion}** right now.\n\n"
            ai_response += f"**Personalized advice:** {advice}\n"
            if "Technique:" in note:
                technique = note.split("Technique:")[1].split("|")[0].strip()
                ai_response += f"**Technique:** {technique}\n"
            ai_response += f"**Suggested activity:** {activity_data['text']}\n"
            if clinical_info:
                ai_response += f"\n**Expert insight from clinical resources:**\n> *\"{clinical_info}\"*\n"
            if "Source:" in note:
                source = note.split("Source:")[1].strip()
                ai_response += f"\n*Source: {source}*\n"

            st.markdown(ai_response)

        msg = AIMessage(content=ai_response)
        msg.eco_mode = True
        st.session_state.chat_history.append(msg)

        co2_info = calculate_co2(user_input, ai_response, mode="eco")
        mode_label = "ECO (Local + RAG)"
        optimization_msg = (
            f"Context: {detected_context}, Intensity: {detected_intensity} | No LLM call"
        )

    else:
        # --- AGENT MODE ---
        agent_avatar = "☁️"
        with st.chat_message("assistant", avatar=agent_avatar):
            if agent_executor is None:
                ai_response = (
                    "Agent unavailable (API key missing). "
                    "Please check your .env file."
                )
                st.warning(ai_response)
                intermediate_steps = []
            else:
                with st.spinner("The Agent is analyzing your situation..."):
                    response = invoke_agent(
                        executor=agent_executor,
                        user_input=user_input,
                        chat_history=st.session_state.chat_history,
                        user_location=st.session_state.user_location or "Unknown",
                        pre_emotion=detected_emotion,
                        pre_confidence=confidence,
                        pre_intensity=detected_intensity,
                        pre_context=detected_context,
                    )
                    ai_response = response.get("output", "I'm here to help.")
                    intermediate_steps = response.get("intermediate_steps", [])

                    if response.get("quality_warning"):
                        st.warning(f"Response quality note: {response['quality_warning']}")

            st.markdown(ai_response)

        st.session_state.chat_history.append(AIMessage(content=ai_response))

        co2_info = calculate_co2(
            user_input, ai_response, mode="agent",
            intermediate_steps=intermediate_steps,
        )
        tools_used = len(intermediate_steps)
        mode_label = "AGENT (Mistral Cloud)"
        optimization_msg = f"{reason} | Tools used: {tools_used}"

    # ------------------------------------------------------------------
    # Step 4: CO2 tracking and technical expander
    # ------------------------------------------------------------------

    cost = co2_info["total_co2"]
    st.session_state.total_co2 += cost

    with st.expander(f"Technical Analysis & Impact ({mode_label})"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Emotion ML", f"{detected_emotion.upper()}", f"{confidence:.0%}")
        with c2:
            st.metric("Mode", "ECO" if use_eco_mode else "AGENT")
        with c3:
            st.metric(
                "CO2 (query)",
                f"{cost:.5f} g",
                help=(
                    f"±{co2_info['uncertainty']:.5f} g | Source: {co2_info['source']}\n"
                    f"Tokens — In: {co2_info.get('input_tokens', '-')}, "
                    f"Out: {co2_info.get('output_tokens', '-')}, "
                    f"Total: {co2_info.get('total_tokens', '-')}"
                ),
            )
        st.divider()
        st.caption(f"**Strategy:** {optimization_msg}")
        st.caption(f"**Location:** {st.session_state.user_location or 'Not detected'}")

        if secondary:
            st.markdown("**Secondary emotions (>10%):**")
            for emo, score_val in secondary.items():
                col_txt, col_bar = st.columns([1, 3])
                with col_txt:
                    st.write(f"**{emo}**")
                with col_bar:
                    st.progress(float(score_val), text=f"{score_val:.1%}")

# ------------------------------------------------------------------
# Sidebar — Dashboard
# ------------------------------------------------------------------

with st.sidebar:
    st.title("Dashboard")

    if st.button("New Session", type="primary"):
        st.session_state.chat_history = []
        st.session_state.emotion_log = {k: 0 for k in EMOTION_COLORS}
        st.session_state.emotion_timeline = []
        st.session_state.total_co2 = 0.0
        st.session_state.user_location = None
        st.rerun()

    st.divider()

    # Stats
    if sum(st.session_state.emotion_log.values()) > 0:
        dom_emotion = max(st.session_state.emotion_log, key=st.session_state.emotion_log.get)
    else:
        dom_emotion = "-"

    st.subheader("Real-Time Analysis")
    col1, col2 = st.columns(2)
    col1.metric("Messages", len(st.session_state.emotion_timeline))
    col2.metric("Dominant", dom_emotion)
    st.metric("Total Carbon Impact", f"{st.session_state.total_co2:.4f} gCO2")

    st.divider()

    # Session end — activity recommendation
    st.subheader("Summary & Action")
    if st.button("End Session"):
        st.session_state.show_kpi = True

    if st.session_state.get("show_kpi") and dom_emotion != "-":
        current_loc = st.session_state.get("user_location")
        if not current_loc:
            st.warning("Unknown location for recommendation.")
            current_loc = st.text_input("Which city are you in?", placeholder="e.g. London, Paris...")

        if current_loc:

            with st.spinner(f"Searching for an activity in {current_loc}..."):
                rec = tools_instance.get_activity(dom_emotion.lower(), current_loc)

            st.success("**Recommendation for you:**")
            st.write(f"Based on your dominant state (**{dom_emotion}**) in **{current_loc}**:")
            st.write(f"**{rec['text']}**")

            if rec["lat"] and rec["lon"]:
                m = folium.Map(location=[rec["lat"], rec["lon"]], zoom_start=15)
                folium.Marker(
                    [rec["lat"], rec["lon"]],
                    popup=rec["text"],
                    tooltip="MindCare Activity",
                    icon=folium.Icon(color="green", icon="leaf"),
                ).add_to(m)
                st_folium(m, height=250, use_container_width=True)