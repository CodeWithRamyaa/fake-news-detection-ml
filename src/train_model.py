import json
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from preprocessing import load_dataset, prepare_dataset
from evaluation import (
    calculate_metrics,
    save_evaluation_results,
    save_confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def train_and_evaluate(sample_size=25000):

    print("Loading dataset...", flush=True)

    df = load_dataset()

    print("Dataset loaded:", len(df), flush=True)

    df = prepare_dataset(df, sample_size)

    print("Dataset prepared:", len(df), flush=True)

    X = df["combined_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Multinomial_Naive_Bayes": MultinomialNB(alpha=0.5)
    }

    results = {}
    best_score = -1
    best_model = None
    best_model_name = None

    for name, classifier in models.items():

        print(f"\nTraining {name}...", flush=True)

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=60000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True
                )
            ),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = calculate_metrics(y_test, predictions)

        results[name] = metrics

        print(f"\nResults for {name}:")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        save_evaluation_results(
            y_test,
            predictions,
            name,
            REPORT_DIR
        )

        save_confusion_matrix(
            y_test,
            predictions,
            name,
            FIGURES_DIR
        )

        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_model = pipeline
            best_model_name = name

    model_path = MODEL_DIR / "best_model.joblib"

    joblib.dump(best_model, model_path)

    with open(REPORT_DIR / "model_metrics.json", "w") as file:
        json.dump({
            "best_model": best_model_name,
            "best_macro_f1": best_score,
            "all_models": results
        }, file, indent=4)

    print("\nTraining completed successfully!")
    print("Best model:", best_model_name)
    print("Best Macro F1:", round(best_score, 4))
    print("Saved model:", model_path)


if __name__ == "__main__":
    train_and_evaluate(sample_size=25000)

    

