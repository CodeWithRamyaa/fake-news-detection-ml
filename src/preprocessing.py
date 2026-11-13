import re
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv"
)


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_dataset(path=DATA_PATH):

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    print("Reading dataset...", flush=True)

    df = pd.read_csv(path)

    print("Available columns:", list(df.columns))

    required_columns = {"title", "text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing required columns: {required_columns}"
        )

    df = df[["title", "text", "label"]].copy()

    df = df.dropna(subset=["text", "label"])

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].astype(str)

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    df["combined_text"] = (
        df["title"] + " " + df["text"]
    ).apply(clean_text)

    df = df[df["combined_text"].str.len() > 0]

    df = df.drop_duplicates(
        subset=["combined_text"]
    )

    df = df.reset_index(drop=True)

    print("Dataset loaded successfully!")
    print("Total records:", len(df), flush=True)

    return df


def prepare_dataset(df, sample_size=25000):

    if sample_size > 0 and len(df) > sample_size:

        sampled_groups = []

        for label in df["label"].unique():

            group = df[df["label"] == label]

            proportion = len(group) / len(df)

            number_of_samples = round(
                sample_size * proportion
            )

            sampled_group = group.sample(
                n=number_of_samples,
                random_state=42
            )

            sampled_groups.append(sampled_group)

        df = pd.concat(
            sampled_groups,
            ignore_index=True
        )

        df = df.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

    return df


if __name__ == "__main__":

    dataset = load_dataset()

    print("\nDataset shape:", dataset.shape)

    print("\nLabel distribution:")
    print(dataset["label"].value_counts())

    print("\nPreprocessing completed successfully!")

    