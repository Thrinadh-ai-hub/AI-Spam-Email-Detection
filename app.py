
from pathlib import Path

import joblib
import streamlit as st

# ================= CONFIGURATION =================

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "spam_model.joblib"
THRESHOLD = 0.70

st.set_page_config(
    page_title="SpamGuard | AI Email Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================= CUSTOM CSS =================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 32px;
    border-radius: 20px;
    background: linear-gradient(120deg, #172554, #111827);
    border: 1px solid #334155;
    margin-bottom: 25px;
}

.hero h1 {
    color: #f8fafc;
    font-size: 40px;
    margin-bottom: 10px;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

.panel {
    background: #111827;
    border: 1px solid #334155;
    padding: 24px;
    border-radius: 16px;
}

.stTextInput input,
.stTextArea textarea {
    background-color: #0f172a;
    color: #f8fafc;
    border-radius: 10px;
}

.stButton button,
.stFormSubmitButton button {
    background: #4f46e5;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    min-height: 45px;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #4338ca;
    color: white;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


# ================= SIDEBAR =================

with st.sidebar:
    st.title("🛡️ SpamGuard")
    st.caption("AI Email Security")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📧 Email Scanner", "ℹ️ About"],
    )

    st.divider()
    st.caption("AI Spam Email Detection")
    st.caption("Educational Project")


# ================= HEADER =================

st.markdown("""
<div class="hero">
    <h1>📧 SpamGuard</h1>
    <p>
        AI-powered email spam detection.
        Analyze suspicious messages and review model results.
    </p>
</div>
""", unsafe_allow_html=True)


# ================= DASHBOARD =================

if page == "🏠 Dashboard":

    st.subheader("Welcome to SpamGuard 👋")

    st.write(
        "Your simple AI email analysis dashboard. "
        "Check an email using your trained machine learning model."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="panel">
            <h3>📨</h3>
            <h4>Email Analysis</h4>
            <p>Analyze the subject and message body.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="panel">
            <h3>🤖</h3>
            <h4>Machine Learning</h4>
            <p>Use your existing trained spam classifier.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="panel">
            <h3>📊</h3>
            <h4>Model Results</h4>
            <p>View predicted category and spam score.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.info(
        "Select '📧 Email Scanner' from the sidebar "
        "to analyze an email."
    )


# ================= EMAIL SCANNER =================

elif page == "📧 Email Scanner":

    if not MODEL_FILE.exists():
        st.error(
            "Model not found. Please run python train.py first."
        )
        st.stop()

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    if not hasattr(model, "predict_proba"):
        st.error(
            "Your model does not support predict_proba()."
        )
        st.stop()

    if not hasattr(model, "classes_"):
        st.error("Model class labels are unavailable.")
        st.stop()

    if "email_subject" not in st.session_state:
        st.session_state.email_subject = ""

    if "email_body" not in st.session_state:
        st.session_state.email_body = ""

    left, right = st.columns(
        [1.3, 0.7],
        gap="large",
    )

    with left:
        st.subheader("✉️ Email Scanner")
        st.write("Enter an email to analyze.")

        st.caption("Quick test samples")

        sample1, sample2 = st.columns(2)

        with sample1:
            if st.button(
                "📨 Normal Sample",
                use_container_width=True,
            ):
                st.session_state.email_subject = (
                    "Meeting Reminder"
                )
                st.session_state.email_body = (
                    "Hi team, our project meeting is scheduled "
                    "for tomorrow at 10 AM. Please bring your updates."
                )
                st.rerun()

        with sample2:
            if st.button(
                "⚠️ Spam Sample",
                use_container_width=True,
            ):
                st.session_state.email_subject = (
                    "You won a prize!"
                )
                st.session_state.email_body = (
                    "Congratulations! You have won a free prize. "
                    "Click here to claim your reward now."
                )
                st.rerun()

        with st.form("email_form"):
            subject = st.text_input(
                "Email Subject",
                key="email_subject",
                placeholder="Enter email subject",
            )

            body = st.text_area(
                "Email Body",
                key="email_body",
                height=250,
                placeholder="Paste your email message here...",
            )

            submitted = st.form_submit_button(
                "🔍 Analyze Email",
                use_container_width=True,
            )

    with right:
        st.subheader("🔐 How it works")

        st.markdown("""
        <div class="panel">
            <h4>01 · Enter email</h4>
            <p>Provide the subject and message.</p>
            <hr>
            <h4>02 · AI analysis</h4>
            <p>The trained model analyzes the text.</p>
            <hr>
            <h4>03 · View result</h4>
            <p>Review the category and spam score.</p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.warning(
            "Never click suspicious links or share "
            "passwords or OTPs."
        )

    # ================= ANALYSIS =================

    if submitted:

        if not subject.strip() and not body.strip():
            st.warning(
                "Please enter an email subject or body."
            )

        else:
            email_text = (
                f"Subject: {subject}\n\n{body}"
            ).strip()

            try:
                classes = list(model.classes_)

                if "spam" not in classes:
                    st.error(
                        "The model does not contain a 'spam' class."
                    )
                    st.stop()

                probabilities = model.predict_proba(
                    [email_text]
                )[0]

                spam_index = classes.index("spam")
                spam_score = float(
                    probabilities[spam_index]
                )

                prediction = (
                    "spam"
                    if spam_score >= THRESHOLD
                    else "ham"
                )

                st.divider()
                st.subheader("📊 Analysis Result")

                result_col, score_col = st.columns(2)

                with result_col:
                    if prediction == "spam":
                        st.error("⚠️ LIKELY SPAM")
                        st.write(
                            "Your model classified this email "
                            "as spam."
                        )
                    else:
                        st.success("✅ LIKELY NORMAL EMAIL")
                        st.write(
                            "Your model classified this email "
                            "as normal (ham)."
                        )

                with score_col:
                    st.metric(
                        "Model spam score",
                        f"{spam_score:.1%}",
                    )

                    st.progress(
                        max(0.0, min(1.0, spam_score))
                    )

                    st.caption(
                        "This is a model score, not a guarantee "
                        "of actual spam probability."
                    )

                with st.expander("View analysis details"):
                    st.write(
                        f"**Prediction:** {prediction.upper()}"
                    )
                    st.write(
                        f"**Spam score:** {spam_score:.4f}"
                    )
                    st.write(
                        f"**Decision threshold:** {THRESHOLD:.0%}"
                    )

            except Exception as e:
                st.error(f"Analysis failed: {e}")


# ================= ABOUT =================

elif page == "ℹ️ About":

    st.subheader("About SpamGuard")

    st.markdown("""
    SpamGuard is an AI-based email spam detection project.

    **Features**
    - Email subject and body analysis
    - Machine learning classification
    - Spam score display
    - Quick example emails
    - Interactive Streamlit dashboard

    **Technology**
    - Python
    - Streamlit
    - Scikit-learn
    - Joblib
    """)

    st.info(
        "This project is for educational purposes. "
        "A normal prediction does not guarantee an email is safe."
    )


# ================= FOOTER =================

st.divider()

st.caption(
    "SpamGuard | AI Spam Email Detection | "
    "Educational Project"
)
