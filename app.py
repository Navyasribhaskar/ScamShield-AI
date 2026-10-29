
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from utils import analyze_message


# ============================================
# 1. CONFIGURATION
# ============================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "scamshield_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"

st.set_page_config(
    page_title="ScamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================
# 2. CUSTOM DESIGN
# ============================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #172554, #111827);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #293548;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: #ffffff;
        margin-bottom: 6px;
    }

    .hero p {
        color: #cbd5e1;
        margin-bottom: 0;
    }

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #293548;
        padding: 16px;
        border-radius: 12px;
    }

    div.stButton > button[kind="primary"] {
        background-color: #f43f5e;
        color: white;
        border: none;
        border-radius: 9px;
        min-height: 2.7rem;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #e11d48;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================
# 3. LOAD MODEL AND METRICS
# ============================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)

try:
    metrics = load_metrics()
    metrics_error = None
except Exception as exc:
    metrics = None
    metrics_error = str(exc)


# ============================================
# 4. EXAMPLE MESSAGES
# ============================================

EXAMPLES = {
    "🎁 Prize message":
        "Congratulations! You have won a free gift. "
        "Claim your prize now!",

    "💬 Ordinary message":
        "Hey, are we still meeting at 5 pm today? "
        "Let me know when you reach.",

    "⚠️ Bank warning":
        "URGENT! Your bank account will be blocked today. "
        "Verify your account immediately by clicking this link: "
        "http://example.com",

    "📦 Delivery message":
        "Your parcel delivery is pending. Please check the "
        "delivery status using the official courier website.",
}


def choose_example(example_name):
    st.session_state["message_input"] = EXAMPLES[example_name]
    st.session_state.pop("last_result", None)


# ============================================
# 5. HEADER
# ============================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ ScamShield AI</h1>
        <p>
            Analyze suspicious messages, explore warning signs,
            and understand machine-learning predictions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Educational prototype • NLP • TF-IDF • Logistic Regression"
)


# ============================================
# 6. NAVIGATION
# ============================================

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Message Analyzer", "📊 Model Insights", "ℹ️ About"],
)

st.sidebar.divider()
st.sidebar.write("**System status**")

if model is not None:
    st.sidebar.success("Trained model loaded")
else:
    st.sidebar.error("Trained model unavailable")

if metrics is not None:
    st.sidebar.success("Evaluation metrics loaded")
else:
    st.sidebar.info("Evaluation metrics not saved yet")


# ============================================
# 7. MESSAGE ANALYZER
# ============================================

if page == "🏠 Message Analyzer":

    st.header("📩 Analyze a Message")

    st.write(
        "Enter a message or select an example below. "
        "The model estimates whether the text resembles "
        "spam in its training data."
    )

    if model_error:
        st.error(f"Could not load the model: {model_error}")

    if model is None:
        st.warning(
            "Train the model first using: "
            "python train_model.py"
        )

    st.subheader("Try an example")

    example_columns = st.columns(4)

    for index, (example_name, example_text) in enumerate(
        EXAMPLES.items()
    ):
        with example_columns[index]:
            st.button(
                example_name,
                key=f"example_{index}",
                use_container_width=True,
                on_click=choose_example,
                args=(example_name,),
            )

    message = st.text_area(
        "Your message",
        key="message_input",
        height=150,
        max_chars=5000,
        placeholder="Paste a message here...",
    )

    st.caption(f"{len(message)} / 5000 characters")

    if st.button(
        "🔍 Analyze Message",
        type="primary",
        use_container_width=True,
    ):
        if not message.strip():
            st.warning("Please enter a message first.")

        elif model is None:
            st.error(
                "The model is unavailable. Check the model file "
                "and training environment."
            )

        else:
            try:
                prediction = model.predict([message])[0]
                probabilities = model.predict_proba([message])[0]

                classes = list(model.classes_)
                spam_index = classes.index("spam")
                spam_probability = float(
                    probabilities[spam_index]
                )

                warnings_result = analyze_message(message)

                st.session_state["last_result"] = {
                    "message": message,
                    "prediction": str(prediction),
                    "spam_probability": spam_probability,
                    "warnings": warnings_result["warnings"],
                    "warning_count": warnings_result["warning_count"],
                    "has_url": warnings_result["has_url"],
                }

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("last_result")

    if result:
        st.divider()
        st.header("📊 Analysis Results")

        if result["prediction"] == "spam":
            st.error(
                "⚠️ This message resembles spam in the training data. "
                "Verify it independently."
            )
        else:
            st.success(
                "This message resembles legitimate SMS in the "
                "training data. This does not guarantee safety."
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Estimated Spam Probability",
                f"{result['spam_probability']:.1%}",
            )

        with col2:
            st.metric(
                "Warning Signs",
                result["warning_count"],
            )

        with col3:
            st.metric(
                "Message Length",
                len(result["message"]),
            )

        st.progress(result["spam_probability"])

        st.caption(
            "The probability is a model estimate, not proof of fraud. "
            "It may not be calibrated to real-world scam prevalence."
        )

        left, right = st.columns(2)

        with left:
            st.subheader("🚨 Detected Warning Signs")

            if result["warnings"]:
                for warning in result["warnings"]:
                    st.warning(warning)
            else:
                st.info(
                    "No warning signs were detected by the current "
                    "rules. The message may still be suspicious."
                )

        with right:
            st.subheader("🛡️ Safety Recommendations")

            st.write(
                "- Do not click unexpected links.\n"
                "- Never share OTPs, passwords, or PINs.\n"
                "- Verify urgent requests independently.\n"
                "- Contact organizations through official channels."
            )

        with st.expander("View technical details"):
            st.write("**Predicted class:**", result["prediction"])
            st.write(
                "**Estimated spam probability:**",
                f"{result['spam_probability']:.1%}",
            )
            st.write(
                "**Rule-based URL detection:**",
                "Yes" if result["has_url"] else "No",
            )
            st.write(
                "**ML method:** TF-IDF + Logistic Regression"
            )
            st.caption(
                "Rule-based warnings are separate from the ML model "
                "and do not necessarily explain its prediction."
            )


# ============================================
# 8. MODEL INSIGHTS
# ============================================

elif page == "📊 Model Insights":

    st.header("📊 Model Performance Dashboard")

    st.write(
        "These metrics should come from the saved held-out test "
        "evaluation. They describe performance on that dataset, "
        "not guaranteed real-world scam detection."
    )

    if metrics_error:
        st.error(f"Could not read metrics: {metrics_error}")

    if metrics is None:
        st.warning(
            "No saved evaluation metrics were found. "
            "Run the updated train_model.py to create "
            "models/model_metrics.json."
        )

    else:
        metric_columns = st.columns(4)

        metric_items = [
            ("Accuracy", "accuracy"),
            ("Spam Precision", "spam_precision"),
            ("Spam Recall", "spam_recall"),
            ("Spam F1-score", "spam_f1"),
        ]

        for column, (label, key) in zip(
            metric_columns, metric_items
        ):
            with column:
                value = metrics.get(key)

                if value is not None:
                    st.metric(label, f"{float(value):.2%}")
                else:
                    st.metric(label, "N/A")

        st.divider()
        st.subheader("Confusion Matrix")

        matrix = metrics.get("confusion_matrix")
        label_order = metrics.get(
            "label_order",
            ["legitimate", "spam"],
        )

        if matrix and len(matrix) == 2:
            matrix_df = pd.DataFrame(
                matrix,
                index=[
                    f"Actual {label}" for label in label_order
                ],
                columns=[
                    f"Predicted {label}" for label in label_order
                ],
            )

            st.dataframe(
                matrix_df,
                use_container_width=True,
            )

            st.bar_chart(matrix_df)

            st.caption(
                "Rows represent actual labels; columns represent "
                "predicted labels. Off-diagonal values are errors."
            )
        else:
            st.info(
                "The saved metrics do not contain a valid 2 × 2 "
                "confusion matrix."
            )

        st.divider()

        test_samples = metrics.get("test_samples")

        if test_samples is not None:
            st.metric("Held-out test messages", int(test_samples))

        st.subheader("How to interpret the results")

        st.write(
            "- **Accuracy:** fraction of all test messages classified "
            "correctly.\n"
            "- **Precision:** fraction of predicted spam messages that "
            "were actually spam in the test set.\n"
            "- **Recall:** fraction of actual spam messages detected.\n"
            "- **F1-score:** harmonic mean of precision and recall."
        )

        st.warning(
            "A high test score does not guarantee performance on "
            "modern scams, other languages, or unseen message types."
        )


# ============================================
# 9. ABOUT
# ============================================

elif page == "ℹ️ About":

    st.header("ℹ️ About ScamShield AI")

    st.write(
        "ScamShield AI is an educational NLP prototype that analyzes "
        "SMS-style messages for patterns associated with spam."
    )

    st.subheader("Technology stack")

    st.markdown(
        """
        - **Python:** application logic
        - **Pandas:** dataset preparation
        - **TF-IDF:** text feature extraction
        - **Logistic Regression:** classification
        - **scikit-learn:** training and evaluation
        - **Streamlit:** interactive web interface
        - **Joblib:** saving and loading the model
        """
    )

    st.subheader("Limitations")

    st.write(
        "The model was trained on a general SMS spam dataset. "
        "It has not been established as a reliable detector for every "
        "kind of scam, phishing link, UPI fraud, or regional-language "
        "message. It does not verify senders or URLs."
    )

    st.info(
        "Use this tool as an educational aid. Verify suspicious "
        "messages through official channels and never disclose "
        "sensitive information."
    )


# ============================================
# 10. FOOTER
# ============================================

st.divider()

st.caption(
    "ScamShield AI • Machine Learning + NLP + Rule-Based Analysis"
)
