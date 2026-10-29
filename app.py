
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_PATH}"
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    model_error = None
except Exception as exc:
    model = None
    model_error = f"{type(exc).__name__}: {exc}"

try:
    metrics = load_metrics()
    metrics_error = None
except Exception as exc:
    metrics = None
    metrics_error = f"{type(exc).__name__}: {exc}"


# ============================================
# 4. EXAMPLE MESSAGES
# ============================================

EXAMPLES = {
    "🎁 Prize message": (
        "Congratulations! You have won a free gift. "
        "Claim your prize now!"
    ),
    "💬 Ordinary message": (
        "Hey, are we still meeting at 5 pm today? "
        "Let me know when you reach."
    ),
    "⚠️ Bank warning": (
        "URGENT! Your bank account will be blocked today. "
        "Verify your account immediately by clicking this link: "
        "http://example.com"
    ),
    "📦 Delivery message": (
        "Your parcel delivery is pending. Please check the "
        "delivery status using the official courier website."
    ),
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
st.sidebar.markdown("### System status")


def status_card(title, ready, ready_text="Ready", failed_text="Unavailable"):
    status_text = ready_text if ready else failed_text
    status_color = "#22c55e" if ready else "#f87171"
    symbol = "●" if ready else "●"

    st.sidebar.markdown(
        f"""
        <div style="
            padding: 11px 12px;
            margin: 7px 0;
            border: 1px solid #293548;
            border-radius: 10px;
            background: #111827;
        ">
            <div style="
                font-size: 11px;
                letter-spacing: 0.7px;
                color: #9ca3af;
                margin-bottom: 5px;
            ">{title}</div>
            <div style="
                font-size: 14px;
                font-weight: 600;
                color: {status_color};
            ">{symbol} {status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


status_card("TRAINED MODEL", model is not None)
status_card(
    "BASE EVALUATION",
    metrics is not None,
    ready_text="Ready",
    failed_text="Unavailable",
)

if model_error:
    with st.sidebar.expander("Model loading details"):
        st.code(model_error)

if metrics_error:
    with st.sidebar.expander("Metrics loading details"):
        st.code(metrics_error)


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

    if model is None:
        st.error(
            "The trained model could not be loaded. "
            "Check the model file and the technical details "
            "in the sidebar."
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
                st.error(
                    f"Analysis failed: {type(exc).__name__}: {exc}"
                )

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

            st.markdown(
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
            st.write("**ML method:** TF-IDF + Logistic Regression")
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
        "The original metrics describe the saved held-out test set. "
        "They remain fixed when you analyze individual messages. "
        "Upload a labeled CSV below to evaluate the same trained "
        "model on a different dataset."
    )

    if metrics_error:
        st.error(f"Could not read saved metrics: {metrics_error}")

    st.subheader("Original held-out test evaluation")

    if metrics is None:
        st.warning(
            "No saved evaluation metrics were found. Run "
            "`python train_model.py` to create "
            "`models/model_metrics.json`."
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
                st.metric(
                    label,
                    f"{float(value):.2%}"
                    if value is not None
                    else "N/A",
                )

        matrix = metrics.get("confusion_matrix")
        label_order = metrics.get(
            "label_order",
            ["legitimate", "spam"],
        )

        if matrix and len(matrix) == 2:
            st.subheader("Original confusion matrix")

            matrix_df = pd.DataFrame(
                matrix,
                index=[f"Actual {label}" for label in label_order],
                columns=[f"Predicted {label}" for label in label_order],
            )

            st.dataframe(matrix_df, use_container_width=True)
            st.bar_chart(matrix_df)

        test_samples = metrics.get("test_samples")

        if test_samples is not None:
            st.caption(
                f"Original held-out test messages: {int(test_samples)}"
            )

    # ----------------------------------------
    # Custom dataset evaluation
    # ----------------------------------------

    st.divider()
    st.subheader("🧪 Evaluate a custom labeled dataset")

    st.write(
        "Upload a CSV with a message column and a true-label column. "
        "The existing model will predict the messages, and the app "
        "will compare its predictions with your labels. "
        "This does not retrain the model."
    )

    sample_csv = (
        "message,label\n"
        '"Congratulations! You won a free prize. Claim now!",spam\n'
        '"Hey, are we still meeting at 5 pm today?",ham\n'
        '"URGENT: verify your bank account immediately",spam\n'
        '"I will call you when I reach home",ham\n'
    )

    st.download_button(
        "⬇️ Download sample CSV template",
        data=sample_csv,
        file_name="scamshield_sample_dataset.csv",
        mime="text/csv",
    )

    if model is None:
        st.error(
            "Custom evaluation is unavailable because the trained "
            "model could not be loaded. Check the sidebar details."
        )
    else:
        uploaded_file = st.file_uploader(
            "Choose a labeled CSV file",
            type=["csv"],
            help=(
                "Include message text and the correct label for every "
                "message. For example: message,label"
            ),
            key="custom_evaluation_csv",
        )

        if uploaded_file is not None:

            try:
                custom_df = pd.read_csv(uploaded_file)
            except Exception as exc:
                custom_df = None
                st.error(f"Could not read the CSV file: {exc}")

            if custom_df is not None:

                if custom_df.empty:
                    st.error("The uploaded CSV is empty.")

                elif len(custom_df.columns) < 2:
                    st.error(
                        "The CSV must contain at least two columns: "
                        "message text and true label."
                    )

                else:
                    st.caption(
                        f"Uploaded {len(custom_df):,} rows and "
                        f"{len(custom_df.columns)} columns."
                    )

                    columns = list(custom_df.columns)

                    lower_columns = {
                        str(col).strip().lower(): col
                        for col in columns
                    }

                    message_guess = next(
                        (
                            lower_columns[name]
                            for name in [
                                "message",
                                "text",
                                "sms",
                                "message_text",
                                "content",
                            ]
                            if name in lower_columns
                        ),
                        columns[0],
                    )

                    label_guess = next(
                        (
                            lower_columns[name]
                            for name in [
                                "label",
                                "class",
                                "target",
                                "category",
                                "type",
                            ]
                            if name in lower_columns
                            and lower_columns[name] != message_guess
                        ),
                        next(
                            (
                                col for col in columns
                                if col != message_guess
                            ),
                            columns[1],
                        ),
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        message_column = st.selectbox(
                            "Message text column",
                            options=columns,
                            index=columns.index(message_guess),
                            key="custom_message_column",
                        )

                    available_label_columns = [
                        col for col in columns
                        if col != message_column
                    ]

                    if label_guess not in available_label_columns:
                        label_guess = available_label_columns[0]

                    with col2:
                        label_column = st.selectbox(
                            "True label column",
                            options=available_label_columns,
                            index=available_label_columns.index(
                                label_guess
                            ),
                            key="custom_label_column",
                        )

                    usable_df = custom_df[
                        [message_column, label_column]
                    ].copy()

                    usable_df[message_column] = (
                        usable_df[message_column]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                    )

                    usable_df[label_column] = (
                        usable_df[label_column]
                        .astype("string")
                        .str.strip()
                    )

                    usable_df = usable_df[
                        (usable_df[message_column] != "")
                        & usable_df[label_column].notna()
                        & (usable_df[label_column] != "")
                    ].copy()

                    if usable_df.empty:
                        st.error(
                            "No usable rows remain after removing "
                            "blank messages or labels."
                        )

                    else:
                        label_values = sorted(
                            usable_df[label_column].unique().tolist(),
                            key=str,
                        )

                        if len(label_values) < 2:
                            st.error(
                                "The true-label column must contain "
                                "at least two distinct labels."
                            )

                        else:
                            spam_guess = next(
                                (
                                    value for value in label_values
                                    if str(value).strip().lower()
                                    in [
                                        "spam",
                                        "junk",
                                        "phishing",
                                        "scam",
                                        "1",
                                    ]
                                ),
                                label_values[-1],
                            )

                            spam_label = st.selectbox(
                                "Which label means spam?",
                                options=label_values,
                                index=label_values.index(spam_guess),
                                help=(
                                    "All other label values will be "
                                    "treated as legitimate. Choose the "
                                    "value that means spam."
                                ),
                                key="custom_spam_label",
                            )

                            if len(usable_df) > 20000:
                                st.warning(
                                    "For this demo, upload no more than "
                                    "20,000 usable rows at a time."
                                )

                            elif st.button(
                                "Evaluate uploaded dataset",
                                type="primary",
                                use_container_width=True,
                                key="evaluate_custom_dataset",
                            ):
                                try:
                                    messages = (
                                        usable_df[message_column].tolist()
                                    )

                                    predicted_raw = model.predict(messages)

                                    # Normalize model outputs to two classes.
                                    predicted_labels = [
                                        (
                                            "spam"
                                            if str(value).strip().lower()
                                            == "spam"
                                            else "legitimate"
                                        )
                                        for value in predicted_raw
                                    ]

                                    actual_labels = [
                                        (
                                            "spam"
                                            if value == spam_label
                                            else "legitimate"
                                        )
                                        for value in (
                                            usable_df[label_column].tolist()
                                        )
                                    ]

                                    accuracy = accuracy_score(
                                        actual_labels,
                                        predicted_labels,
                                    )

                                    precision = precision_score(
                                        actual_labels,
                                        predicted_labels,
                                        pos_label="spam",
                                        zero_division=0,
                                    )

                                    recall = recall_score(
                                        actual_labels,
                                        predicted_labels,
                                        pos_label="spam",
                                        zero_division=0,
                                    )

                                    f1 = f1_score(
                                        actual_labels,
                                        predicted_labels,
                                        pos_label="spam",
                                        zero_division=0,
                                    )

                                    cm = confusion_matrix(
                                        actual_labels,
                                        predicted_labels,
                                        labels=["legitimate", "spam"],
                                    )

                                    report = classification_report(
                                        actual_labels,
                                        predicted_labels,
                                        labels=["legitimate", "spam"],
                                        target_names=[
                                            "legitimate",
                                            "spam",
                                        ],
                                        output_dict=True,
                                        zero_division=0,
                                    )

                                    st.session_state[
                                        "custom_eval_results"
                                    ] = {
                                        "accuracy": float(accuracy),
                                        "precision": float(precision),
                                        "recall": float(recall),
                                        "f1": float(f1),
                                        "confusion_matrix": cm.tolist(),
                                        "report": report,
                                        "actual": actual_labels,
                                        "predicted": predicted_labels,
                                        "messages": messages,
                                        "true_original": (
                                            usable_df[label_column]
                                            .astype(str)
                                            .tolist()
                                        ),
                                    }

                                except Exception as exc:
                                    st.error(
                                        "Dataset evaluation failed: "
                                        f"{type(exc).__name__}: {exc}"
                                    )

    # ----------------------------------------
    # Show custom evaluation results
    # ----------------------------------------

    custom_results = st.session_state.get("custom_eval_results")

    if custom_results:
        st.divider()
        st.subheader("📈 Custom dataset evaluation results")

        st.caption(
            "These metrics come from the last successful custom "
            "dataset evaluation. Upload and evaluate another dataset "
            "to replace these results."
        )

        result_columns = st.columns(4)

        result_items = [
            ("Accuracy", "accuracy"),
            ("Spam Precision", "precision"),
            ("Spam Recall", "recall"),
            ("Spam F1-score", "f1"),
        ]

        for column, (label, key) in zip(
            result_columns, result_items
        ):
            with column:
                st.metric(
                    label,
                    f"{custom_results[key]:.2%}",
                )

        st.metric(
            "Evaluated messages",
            len(custom_results["actual"]),
        )

        st.subheader("Custom dataset confusion matrix")

        custom_cm = pd.DataFrame(
            custom_results["confusion_matrix"],
            index=["Actual legitimate", "Actual spam"],
            columns=["Predicted legitimate", "Predicted spam"],
        )

        st.dataframe(custom_cm, use_container_width=True)
        st.bar_chart(custom_cm)

        st.subheader("Classification report")

        report_df = pd.DataFrame(
            custom_results["report"]
        ).transpose()

        st.dataframe(
            report_df.round(3),
            use_container_width=True,
        )

        prediction_report = pd.DataFrame(
            {
                "message": custom_results["messages"],
                "true_label_original": custom_results["true_original"],
                "actual_normalized": custom_results["actual"],
                "predicted_label": custom_results["predicted"],
                "correct": [
                    actual == predicted
                    for actual, predicted in zip(
                        custom_results["actual"],
                        custom_results["predicted"],
                    )
                ],
            }
        )

        st.download_button(
            "⬇️ Download prediction report",
            data=prediction_report.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="scamshield_custom_evaluation_report.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("How to interpret the results")

    st.markdown(
        "- **Accuracy:** fraction of evaluated messages classified correctly.\n"
        "- **Spam precision:** fraction of predicted spam messages that were actually spam.\n"
        "- **Spam recall:** fraction of actual spam messages detected.\n"
        "- **Spam F1-score:** harmonic mean of spam precision and recall."
    )

    st.warning(
        "Custom evaluation is meaningful only when the uploaded labels "
        "are accurate and the dataset is representative. Evaluating on "
        "training data may produce overly optimistic scores. These "
        "results do not guarantee real-world scam-detection performance."
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

