# 📰 Fake News Detection System

### Machine Learning-Based News Classification Using NLP

An educational machine-learning application that classifies news articles using Natural Language Processing (NLP), TF-IDF feature extraction, and machine-learning algorithms.

The application provides an interactive Streamlit dashboard where users can enter a news headline and article to receive a model prediction.

---

## 🚀 Project Overview

The Fake News Detection System explores how machine learning can be used to classify news articles into two categories:

- **Fake:** Classified as fake by the trained model.
- **Real:** Classified as real by the trained model.

The project compares multiple machine-learning algorithms and selects the best-performing model based on Macro F1 Score.

## ✨ Features

- News classification using machine learning.
- Headline and article text input.
- Prediction confidence display.
- TF-IDF text feature extraction.
- Comparison of multiple machine-learning models.
- Accuracy, Precision, Recall and F1-score analysis.
- Confusion matrix visualization.
- Interactive Streamlit dashboard.

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| NLTK | Natural Language Processing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Web application |
| Joblib | Model saving and loading |

## ⚙️ Machine Learning Workflow

Dataset → Text Preprocessing → TF-IDF → Model Training → Model Evaluation → Prediction

### 1. Dataset
The project uses the WELFake news dataset.

### 2. Text Preprocessing
- Text cleaning
- Removal of unnecessary characters
- Combining headline and article text
- Duplicate removal

### 3. Feature Extraction
TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical features for machine-learning models.

### 4. Model Training
The following algorithms are compared:

- Logistic Regression
- Multinomial Naive Bayes

### 5. Model Evaluation
Models are evaluated using:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score
- Confusion Matrix

## 📊 Model Performance

The following results were obtained during model evaluation:

| Metric | Logistic Regression | Multinomial Naive Bayes |
|---|---:|---:|
| Accuracy | 94.88% | 88.62% |
| Macro Precision | 94.79% | 88.49% |
| Macro Recall | 94.91% | 88.70% |
| Macro F1 Score | 94.84% | 88.56% |

**Best Performing Model:** Logistic Regression

*These results are from the project's held-out test split. They should not be interpreted as a guarantee of accuracy on new or real-world news.*

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
│   ├── figures/
│   └── model_metrics.json
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluation.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 💻 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd fake-news-detection-ml
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

Place the WELFake dataset CSV file inside:

```text
data/raw/WELFake_Dataset.csv
```

### Step 5: Train the Models

```bash
python src/train_model.py
```

### Step 6: Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📌 Future Improvements

- Test on independent news datasets.
- Improve model generalization.
- Explore advanced NLP techniques.
- Add additional machine-learning algorithms.
- Improve prediction explanations.
- Develop multilingual news classification.

## ⚠️ Disclaimer

This application is an educational machine-learning project. Its predictions do not establish whether information is factually correct. Verify important news using reliable independent sources.

## 👩‍💻 Author

**CodeWithRamyaa**

Artificial Intelligence and Data Science Student

---

⭐ If you find this project interesting, consider giving the repository a star!