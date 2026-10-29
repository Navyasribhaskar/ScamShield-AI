# 🛡️ ScamShield AI

**An NLP-based SMS spam detection and safety-awareness application.**

ScamShield AI analyzes text messages using machine learning and rule-based checks to help users recognize potentially suspicious messages.

## Features

- **SMS classification:** Predicts whether a message resembles spam or legitimate SMS.
- **NLP pipeline:** Uses TF-IDF feature extraction and Logistic Regression.
- **Warning indicators:** Flags patterns involving urgency, sensitive information, and URLs.
- **Interactive dashboard:** Built with Streamlit.
- **Model evaluation:** Reports accuracy, spam precision, recall, F1-score, and a confusion matrix.
- **Automated tests:** Tests the rule-based message analysis functionality.

## Technology Stack

- Python
- Pandas
- scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Joblib
- Unittest

## Project Structure

```text
ScamShield-AI/
├── app.py
├── train_model.py
├── utils.py
├── test_utils.py
├── requirements.txt
├── README.md
├── data/
└── models/
```

The `data/` and `models/` directories are generated locally during dataset preparation and model training.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/NavyaSreebhaskar/ScamShield-AI.git
cd ScamShield-AI
```

### 2. Create a virtual environment

Windows:

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

The training script downloads the SMS Spam Collection dataset if it is not already available, trains the classifier, evaluates it, and saves the model and evaluation metrics.

An internet connection is required for the initial dataset download.

### 5. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

### 6. Run the tests

```bash
python -m unittest test_utils.py -v
```

## Model Evaluation

The initial held-out test evaluation reported:

| Metric | Result |
|---|---:|
| Accuracy | 97.58% |
| Spam precision | 88.15% |
| Spam recall | 92.97% |
| Spam F1-score | 90.49% |

These figures describe the reported test split of the SMS Spam Collection dataset. They do not establish performance on every type of modern scam or phishing message.

## Limitations

- The model was trained on a general SMS spam dataset.
- Predictions may be incorrect, and estimated probabilities may not be calibrated.
- Rule-based warning indicators are separate from the ML model's prediction.
- The application does not verify message senders, website safety, or whether a message is fraudulent.
- Further testing is required for modern scams, multilingual messages, and real-world deployment.

## Future Improvements

- Explainable model predictions.
- Expanded evaluation on independently labeled scam messages.
- Multilingual message analysis.
- Improved privacy and security controls.
- Deployment as a publicly accessible web application.

## Disclaimer

ScamShield AI is an educational prototype, not a guaranteed fraud detection service. Verify suspicious requests through official channels and never share OTPs, passwords, PINs, or banking credentials.

## Author

Developed as a personal AI/ML project exploring natural language processing, text classification, and digital safety.
