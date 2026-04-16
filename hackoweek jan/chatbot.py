import streamlit as st
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Institute NLP FAQ Bot", page_icon="🎓", layout="centered")

# ---------------- INTENTS ----------------
INTENT_DATA = {
    "admissions": ["how to apply", "admission process", "join college"],
    "fees": ["what are fees", "tuition cost", "semester charges"],
    "exams": ["exam schedule", "when are exams", "exam dates"],
    "timetable": ["class timetable", "lecture schedule"],
    "hostel": ["hostel facility", "accommodation details"],
    "scholarships": ["scholarship details", "financial aid"]
}

INTENT_RESPONSES = {
    "admissions": "Admissions are open. Apply through the official portal.",
    "fees": "Fees range between ₹50,000 and ₹2,00,000 per semester.",
    "exams": "Exam schedules are published one month before exams.",
    "timetable": "Timetable is available in the student portal.",
    "hostel": "Hostel facilities are available on first-come first-served basis.",
    "scholarships": "Merit-based scholarships up to 50% are available."
}

# ---------------- SYNONYMS ----------------
SYNONYMS = {
    "fees": ["tuition", "payment", "charges"],
    "admissions": ["apply", "enroll", "registration"],
    "hostel": ["dorm", "stay"],
    "exams": ["test", "assessment"],
    "timetable": ["schedule", "routine"],
    "scholarships": ["aid", "financial"]
}

STOPWORDS = {"is", "the", "a", "an", "are", "of", "for", "to", "in", "on", "at", "and", "or"}

# ---------------- PREPROCESS ----------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS]

    normalized = []
    for word in tokens:
        replaced = False
        for key, values in SYNONYMS.items():
            if word in values:
                normalized.append(key)
                replaced = True
                break
        if not replaced:
            normalized.append(word)

    return " ".join(normalized)

# ---------------- ENTITY RECOGNITION ----------------
def extract_entities(text):
    entities = {}

    # Semester (SEM 5 / semester 3)
    sem_match = re.search(r"(sem\s?\d+|semester\s?\d+)", text.lower())
    if sem_match:
        entities["semester"] = sem_match.group()

    # Year (first year, third year)
    year_match = re.search(r"(first|second|third|fourth)\s?year", text.lower())
    if year_match:
        entities["year"] = year_match.group()

    # Course code (CS101, IT202 etc.)
    course_match = re.search(r"\b[A-Z]{2,4}\d{2,3}\b", text)
    if course_match:
        entities["course_code"] = course_match.group()

    # Date pattern (12/05/2026 or 12 May)
    date_match = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]?\d{0,4}\b", text)
    if date_match:
        entities["date"] = date_match.group()

    return entities

# ---------------- TRAIN CLASSIFIER ----------------
train_x = []
train_y = []

for intent, sentences in INTENT_DATA.items():
    for sentence in sentences:
        train_x.append(preprocess(sentence))
        train_y.append(intent)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train_x, train_y)

# ---------------- TF-IDF RETRIEVAL ----------------
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(train_x)

def retrieve_best_answer(user_input):

    processed = preprocess(user_input)
    entities = extract_entities(user_input)

    # 1️⃣ Intent prediction
    predicted_intent = model.predict([processed])[0]

    # 2️⃣ Similarity check
    user_vec = vectorizer.transform([processed])
    similarity = np.dot(faq_vectors, user_vec.T).toarray()
    best_index = np.argmax(similarity)
    matched_intent = train_y[best_index]

    final_intent = predicted_intent if predicted_intent == matched_intent else predicted_intent

    base_response = INTENT_RESPONSES[final_intent]

    # 3️⃣ Add entity-aware dynamic response
    if final_intent == "exams":
        if "semester" in entities:
            base_response += f"\n\nExam details for {entities['semester']} will be announced soon."
        if "year" in entities:
            base_response += f"\nFor {entities['year']} students, check department notice board."
        if "course_code" in entities:
            base_response += f"\nCourse {entities['course_code']} exam date will be updated shortly."

    return final_intent, base_response

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Institute NLP Chatbot", page_icon="🎓", layout="wide")

# ---------- LIGHT THEME CSS ----------
st.markdown("""
<style>

/* Background */
body {
    background-color: #f8fafc;
}

/* Container */
.block-container {
    padding-top: 1rem;
}

/* Header */
.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: #1e293b;
}

.sub-text {
    text-align: center;
    color: #64748b;
    margin-bottom: 20px;
}

/* Chat bubbles */
.user-bubble {
    background-color: #2563eb;
    color: white;
    padding: 12px 16px;
    border-radius: 15px;
    margin: 10px 0;
    width: fit-content;
    max-width: 70%;
    margin-left: auto;
}

.bot-bubble {
    background-color: #e2e8f0;
    color: #1e293b;
    padding: 12px 16px;
    border-radius: 15px;
    margin: 10px 0;
    width: fit-content;
    max-width: 70%;
}

/* Buttons */
.stButton button {
    border-radius: 10px;
    padding: 8px;
    font-weight: 500;
    background-color: #f1f5f9;
    color: #1e293b;
    border: 1px solid #cbd5f5;
}

/* Input box */
.stChatInput > div {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<div class='main-title'>🎓 Institute AI Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Ask about admissions, exams, fees, hostel & more</div>", unsafe_allow_html=True)

# ---------- SESSION ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]

if "context" not in st.session_state:
    st.session_state.context = {"last_intent": None}

# ---------- QUICK BUTTONS ----------
st.markdown("### ⚡ Quick Access")

cols = st.columns(6)
topics = ["Admissions", "Fees", "Exams", "Timetable", "Hostel", "Scholarships"]

for i, topic in enumerate(topics):
    if cols[i].button(topic):
        st.session_state.messages.append({"role": "user", "content": topic})
        intent, response = retrieve_best_answer(topic)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.context["last_intent"] = intent
        st.rerun()

st.markdown("---")

# ---------- CHAT DISPLAY ----------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

# ---------- INPUT ----------
if prompt := st.chat_input("Type your question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    intent, response = retrieve_best_answer(prompt)

    # Multi-turn logic
    if len(prompt.split()) <= 4 and st.session_state.context["last_intent"] == "exams":
        intent = "exams"
        _, response = retrieve_best_answer(prompt)

    st.session_state.context["last_intent"] = intent

    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.context = {"last_intent": None}
        st.rerun()

    st.markdown("---")

    st.markdown("## 🧠 Features")
    st.markdown("""
    ✔ Intent Classification  
    ✔ TF-IDF Matching  
    ✔ Entity Detection  
    ✔ Multi-turn Chat  
    ✔ Smart Responses  
    """)
