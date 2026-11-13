import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        )
    }


def save_evaluation_results(
    y_true,
    y_pred,
    model_name,
    reports_dir
):

    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics = calculate_metrics(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Fake News", "Real News"],
        output_dict=True,
        zero_division=0
    )

    results = {
        "model": model_name,
        "metrics": metrics,
        "classification_report": report
    }

    with open(
        reports_dir / f"{model_name}_report.json",
        "w"
    ) as file:
        json.dump(results, file, indent=4)

    return metrics


def save_confusion_matrix(
    y_true,
    y_pred,
    model_name,
    figures_dir
):

    figures_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake News", "Real News"],
        yticklabels=["Fake News", "Real News"]
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    plt.savefig(
        figures_dir / f"{model_name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()