# 📰 Fake News Detection System

### Machine Learning-Based News Classification Using NLP

An educational machine learning project that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) and machine learning algorithms.

The project includes data preprocessing, model training, evaluation, and an interactive Streamlit web application.

---

## 🚀 Project Overview

The Fake News Detection System analyzes news text and predicts whether it is likely to be fake or real based on patterns learned from a labelled dataset.

Two machine learning algorithms are implemented and evaluated:

- Logistic Regression
- Multinomial Naive Bayes

Logistic Regression was selected as the best-performing model based on the evaluation results.

---

## ✨ Features

- **News Classification:** Enter news text and receive a prediction.
- **Text Preprocessing:** Clean and prepare text for machine learning.
- **TF-IDF Vectorization:** Convert text into numerical features.
- **Model Comparison:** Evaluate two machine learning algorithms.
- **Performance Dashboard:** View accuracy, precision, recall, and F1-score.
- **Confusion Matrices:** Visualize model classification performance.
- **Interactive Interface:** Explore the project through a Streamlit application.

---

## 📊 Model Performance

The models were evaluated using a held-out test dataset.

| Evaluation Metric | Logistic Regression | Multinomial Naive Bayes |
|---|---:|---:|
| Accuracy | 94.88% | 88.62% |
| Macro Precision | 94.79% | 88.49% |
| Macro Recall | 94.91% | 88.70% |
| Macro F1-Score | 94.84% | 88.56% |

### 🏆 Best Model: Logistic Regression

Logistic Regression achieved the highest overall performance among the two evaluated models and was selected for the application.

*Note: These results represent the project's held-out test evaluation and should not be interpreted as guaranteed real-world accuracy.*

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| NLTK | Natural Language Processing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Web application |
| Joblib | Model serialization |

---

## 📁 Project Structure

```text
fake-news-detection-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_model.joblib
│
├── reports/
│   ├── Logistic_Regression_report.json
│   ├── Multinomial_Naive_Bayes_report.json
│   ├── model_metrics.json
│   │
│   └── figures/
│       ├── Logistic_Regression_confusion_matrix.png
│       └── Multinomial_Naive_Bayes_confusion_matrix.png
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluation.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/CodeWithRamyaa/fake-news-detection-ml.git
```

### 2. Open the Project Folder

```bash
cd fake-news-detection-ml
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the local address displayed in the terminal, usually:

```text
http://localhost:8501
```

---

## 🔄 Machine Learning Workflow

1. Dataset collection
2. Data cleaning and preprocessing
3. Text normalization
4. TF-IDF feature extraction
5. Model training
6. Model evaluation
7. Best-model selection
8. Model saving
9. Streamlit application deployment

---

## 📈 Application Pages

The application contains four main sections:

- **Home:** Project introduction and overview.
- **Detect News:** Enter news text and view its predicted classification.
- **Model Performance:** Compare model evaluation metrics and visualizations.
- **About Project:** Learn about the methodology, workflow, and limitations.

---

## ⚠️ Limitations

- Predictions depend on patterns learned from the training dataset.
- Unfamiliar topics and writing styles may affect predictions.
- Model confidence does not establish factual correctness.
- The application does not independently verify news against trusted sources.

---

## 📌 Disclaimer

This application is an educational machine-learning project. Its predictions do not establish whether information is factually correct. Verify important news using reliable independent sources.

---

## 👩‍💻 Author

**CodeWithRamyaa**

GitHub: [CodeWithRamyaa](https://github.com/CodeWithRamyaa)

---

⭐ If you find this project interesting, consider starring the repository!