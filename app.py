import streamlit as st
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ---------------- PROJECT PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
REPORT_PATH = BASE_DIR / "reports" / "model_metrics.json"
FIGURES_DIR = BASE_DIR / "reports" / "figures"


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="📰",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

.result-card {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    return None


@st.cache_data
def load_metrics():

    if REPORT_PATH.exists():

        with open(REPORT_PATH, "r") as file:
            return json.load(file)

    return None


model = load_model()
metrics_data = load_metrics()


# ---------------- SIDEBAR ----------------

st.sidebar.title("📰 Fake News AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Detect News",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "An educational machine-learning application "
    "for classifying news articles."
)


# ---------------- HOME PAGE ----------------

if page == "Home":

    st.title("📰 Fake News Detection System")

    st.subheader(
        "Machine Learning Based News Classification"
    )

    st.write(
        "Analyze news articles using Natural Language "
        "Processing and machine-learning algorithms."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    if metrics_data:

        logistic = metrics_data["all_models"].get(
            "Logistic_Regression", {}
        )

        accuracy = logistic.get("accuracy", 0) * 100

        f1 = logistic.get("f1_macro", 0) * 100

        best_model = metrics_data.get(
            "best_model", "Not available"
        )

        col1.metric(
            "Model Accuracy",
            f"{accuracy:.2f}%"
        )

        col2.metric(
            "Macro F1 Score",
            f"{f1:.2f}%"
        )

        col3.metric(
            "Best Model",
            best_model.replace("_", " ")
        )

    else:

        col1.metric("Model Accuracy", "Unavailable")
        col2.metric("Macro F1 Score", "Unavailable")
        col3.metric("Best Model", "Unavailable")

    st.markdown("---")

    st.subheader("Project Features")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        ### 🔍 News Classification

        - Enter a news headline.
        - Paste the complete article.
        - Get a machine-learning prediction.
        - View the model's estimated confidence.
        """)

    with c2:

        st.markdown("""
        ### 📊 Model Analytics

        - Accuracy comparison.
        - Precision and recall.
        - F1-score analysis.
        - Confusion matrix visualization.
        """)

    st.success(
        "The trained machine-learning model is ready."
        if model is not None
        else "Please train the model before using predictions."
    )


# ---------------- NEWS DETECTION PAGE ----------------

elif page == "Detect News":

    st.title("🔍 Detect News")

    st.write(
        "Enter a headline and article to classify the news."
    )

    if model is None:

        st.error(
            "Trained model not found. "
            "Run python src/train_model.py first."
        )

    else:

        title = st.text_input(
            "News Headline",
            placeholder="Enter the news headline..."
        )

        article = st.text_area(
            "News Article",
            height=250,
            placeholder="Paste the news article here..."
        )

        if st.button("Analyze News", type="primary"):

            if not title.strip() and not article.strip():

                st.warning(
                    "Please enter a headline or article."
                )

            else:

                input_text = (
                    title + " " + article
                ).strip()

                prediction = model.predict(
                    [input_text]
                )[0]

                probabilities = model.predict_proba(
                    [input_text]
                )[0]

                confidence = max(probabilities) * 100

                st.markdown("---")

                st.subheader("Analysis Result")

                col1, col2 = st.columns(2)

                with col1:

                    if prediction == 0:

                        st.error(
                            "Prediction: Fake-like"
                        )

                    else:

                        st.success(
                            "Prediction: Real-like"
                        )

                with col2:

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                st.progress(
                    float(confidence / 100)
                )

                st.caption(
                    "Confidence is the model's estimated "
                    "probability for its predicted class."
                )

                st.markdown("---")

                st.subheader("Class Probabilities")

                probability_df = pd.DataFrame({
                    "Category": [
                        "Fake-like",
                        "Real-like"
                    ],
                    "Probability": probabilities * 100
                })

                st.bar_chart(
                    probability_df.set_index("Category")
                )

                st.warning(
                    "This prediction is based on patterns "
                    "learned from the training dataset. "
                    "It is not an independent fact-check."
                )


# ---------------- MODEL PERFORMANCE PAGE ----------------

elif page == "Model Performance":

    st.title("📊 Model Performance Dashboard")

    if metrics_data is None:

        st.warning(
            "Model evaluation report not found."
        )

    else:

        results = metrics_data["all_models"]

        performance_df = pd.DataFrame(
            results
        ).T

        performance_df.index.name = "Model"

        st.subheader("Model Comparison")

        display_df = performance_df.copy()

        display_df.columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]

        st.dataframe(
            (display_df * 100).round(2),
            use_container_width=True
        )

        st.subheader("Accuracy Comparison")

        accuracy_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy": [
                value["accuracy"] * 100
                for value in results.values()
            ]
        })

        st.bar_chart(
            accuracy_df.set_index("Model")
        )

        st.subheader("Evaluation Metrics Comparison")

        st.bar_chart(
            display_df * 100
        )

        st.markdown("---")

        st.subheader("Confusion Matrices")

        col1, col2 = st.columns(2)

        confusion_files = [
            (
                "Logistic Regression",
                "Logistic_Regression_confusion_matrix.png"
            ),
            (
                "Multinomial Naive Bayes",
                "Multinomial_Naive_Bayes_confusion_matrix.png"
            )
        ]

        for index, (name, filename) in enumerate(
            confusion_files
        ):

            image_path = FIGURES_DIR / filename

            if image_path.exists():

                with [col1, col2][index]:

                    st.markdown(f"**{name}**")

                    st.image(
                        str(image_path),
                        use_container_width=True
                    )

            else:

                st.info(
                    f"Confusion matrix unavailable: {name}"
                )

        st.success(
            "Best model: "
            + metrics_data["best_model"].replace("_", " ")
        )


# ---------------- ABOUT PROJECT PAGE ----------------

elif page == "About Project":

    st.title("ℹ️ About This Project")

    st.markdown("""
    ### Fake News Detection Using Machine Learning

    This project uses Natural Language Processing (NLP)
    and supervised machine learning to classify news
    articles based on patterns learned from labelled data.

    ### Technologies Used

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Natural Language Processing
    - TF-IDF Vectorization
    - Streamlit
    - Matplotlib
    - Seaborn

    ### Machine Learning Algorithms

    1. Logistic Regression
    2. Multinomial Naive Bayes

    ### Dataset

    WELFake Dataset.

    ### Feature Extraction

    TF-IDF (Term Frequency–Inverse Document Frequency).

    ### Project Workflow

    Dataset → Text Preprocessing → TF-IDF →
    Model Training → Model Evaluation → Prediction
    """)

    st.markdown("---")

    st.warning("""
    **Disclaimer**

    This application is an educational machine-learning
    project. Its predictions do not establish whether
    information is factually correct. Verify important
    news using reliable independent sources.
    """)


# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "Fake News Detection System | Machine Learning Project"
)