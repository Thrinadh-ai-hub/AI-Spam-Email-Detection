
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "spam_model.joblib"

st.set_page_config(
    page_title="AI Spam Email Detector",
    page_icon="📧",
    layout="centered",
)

st.title("📧 AI-Based Spam Email Detection")
st.write("Check whether an email looks like spam or a normal email.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

if not MODEL_FILE.exists():
    st.error("Model not found. Please run python train.py first.")
    st.stop()

model = load_model()

subject = st.text_input("Email Subject")
body = st.text_area("Email Body", height=220)

if st.button("Check Email"):
    email_text = f"Subject: {subject}\n\n{body}".strip()

    if not subject.strip() and not body.strip():
        st.warning("Please enter an email subject or body.")
    else:
     probabilities = model.predict_proba([email_text])[0]
    classes = list(model.classes_)
    spam_index = classes.index("spam")
    spam_probability = probabilities[spam_index]

    if spam_probability >= 0.70:
     prediction = "spam"
    else:
     prediction = "ham"

    st.write("DEBUG prediction:", prediction)
    st.write("DEBUG model:", type(model.named_steps["classifier"]).__name__)
    st.write("DEBUG classes:", model.classes_)

    if hasattr(model, "predict_proba"):
       probabilities = model.predict_proba([email_text])[0]
       st.write(
            "DEBUG probabilities:",
             dict(zip(model.classes_, probabilities))
        ) 
    if prediction == "spam":
            st.error("⚠️ Prediction: SPAM")
    else:
            st.success("✅ Prediction: NORMAL EMAIL (HAM)")

    if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([email_text])[0]
            classes = model.classes_
            spam_index = list(classes).index("spam")
            spam_score = probabilities[spam_index]
            st.metric("Model spam score", f"{spam_score:.1%}")

st.caption(
    "Educational project only. Predictions are not a guarantee "
    "of email safety."
)
