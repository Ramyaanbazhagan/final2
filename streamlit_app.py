import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Resurrective Neural Persona", layout="wide")

# -------------------------------
# API KEY (PUT YOUR KEY DIRECTLY HERE)
# -------------------------------
genai.configure(api_key="AIzaSyA4yItqQoYqI68aGWvqM9_yPeX9EOMgvUA")

# -------------------------------
# ETHICAL DISCLAIMER
# -------------------------------
st.warning("""
⚠️ This is a research prototype.
This AI does NOT represent a real person.
It does NOT replace human relationships.
It is designed for academic study under Ethical AI principles.
""")

# -------------------------------
# LOAD / CREATE MEMORY FILE
# -------------------------------
MEMORY_FILE = "dataset.json"

if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump({"conversations": []}, f)

@st.cache_resource
def load_memory():
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

memory_data = load_memory()

def save_memory(user, ai):
    memory_data["conversations"].append({
        "user": user,
        "ai": ai
    })
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory_data, f, indent=2)

# -------------------------------
# EMBEDDINGS
# -------------------------------
def flatten_memory(data):
    texts = []
    for item in data["conversations"]:
        texts.append(item["user"])
        texts.append(item["ai"])
    return texts

memory_texts = flatten_memory(memory_data)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

def retrieve_context(query, top_k=3):
    if not memory_texts:
        return []

    embeddings = embed_model.encode(memory_texts, convert_to_tensor=True)
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
    return [memory_texts[i] for i in top_idx]

# -------------------------------
# EMOTION DETECTION
# -------------------------------
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "miss", "lonely", "cry", "hurt"]):
        return "sad"
    if any(w in t for w in ["happy", "love", "excited", "great"]):
        return "happy"
    return "neutral"

# -------------------------------
# VOICE OUTPUT
# -------------------------------
def speak(text, emotion):
    slow = True if emotion == "sad" else False
    tts = gTTS(text, slow=slow)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.title("🧠 Control Panel")

voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)

persona_mode = st.sidebar.radio(
    "Persona Mode",
    ["🧠 Memory Mode", "💬 Casual Talk", "🤍 Emotional Support"]
)

show_reasoning = st.sidebar.checkbox("🧠 Show Emotion Debug")

# -------------------------------
# MAIN UI
# -------------------------------
st.title("🤖 Resurrective Neural Persona")

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://i.imgur.com/8Km9tLL.png", caption="AI Persona")

with col2:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f"🧍 **You:** {msg}")
        else:
            st.markdown(f"🤖 **AI:** {msg}")

# -------------------------------
# USER INPUT
# -------------------------------
st.markdown("---")
user_input = st.text_input("Talk to your companion:")

if st.button("Send"):
    if user_input.strip():

        st.session_state.chat.append(("user", user_input))

        context = retrieve_context(user_input)
        context_text = "\n".join(context)

        # Persona mode behavior
        mode_instruction = ""
        if persona_mode == "🧠 Memory Mode":
            mode_instruction = "Use memory context strongly in response."
        elif persona_mode == "💬 Casual Talk":
            mode_instruction = "Respond casually and short."
        elif persona_mode == "🤍 Emotional Support":
            mode_instruction = "Respond warmly, supportive, empathetic."

        prompt = f"""
You are a synthetic AI persona for academic research.
Never claim to be human.
Never create emotional dependency.
Maintain ethical AI boundaries.

{mode_instruction}

Memory Context:
{context_text}

User: {user_input}
AI:
"""

        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)

        ai_text = response.text.strip()

        emotion = detect_emotion(ai_text)

        st.session_state.chat.append(("ai", ai_text))

        save_memory(user_input, ai_text)

        if voice_on:
            audio_file = speak(ai_text, emotion)
            st.audio(audio_file)

        if show_reasoning:
            st.markdown("### 🧠 Emotion Debug")
            st.write("Detected Emotion:", emotion)
