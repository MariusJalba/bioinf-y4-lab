from __future__ import annotations
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

HANDLE = "MariusJalba"

DATA_CSV = Path(f"data/work/{HANDLE}/lab08/expression_matrix_MariusJalba.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200

OUT_DIR = Path(f"labs/08_ml/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CONFUSION = OUT_DIR / f"confusion_rf_{HANDLE}.png"
OUT_REPORT = OUT_DIR / f"classification_report_{HANDLE}.txt"
OUT_FEATIMP = OUT_DIR / f"feature_importance_{HANDLE}.csv"
OUT_CLUSTER_CROSSTAB = OUT_DIR / f"cluster_crosstab_{HANDLE}.csv"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Nu am găsit fișierul de input:\n  {path}\n"
        )


def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, low_memory=False)
    preferred_label = "pam50_+_claudin-low_subtype"
    if preferred_label in df.columns:
        y = df[preferred_label]
        X = df.drop(columns=[preferred_label])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    X_num = X.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] == 0:
        X_num = X.apply(pd.to_numeric, errors="coerce")
    X_num = X_num.fillna(0)
    y = y.astype("object").fillna("UNKNOWN")

    return X_num, y


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    return y_enc, le


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int,
    random_state: int,
) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    out_png: Path,
    out_txt: Path,
) -> None:
    y_pred = model.predict(X_test)

    target_names = list(label_encoder.classes_)
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    out_txt.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(max(5, len(target_names) * 0.6), max(4, len(target_names) * 0.5)))
    sns.heatmap(
        cm,
        annot=False if len(target_names) > 10 else True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Random Forest — confusion matrix")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def compute_feature_importance(
    model: RandomForestClassifier,
    feature_names: pd.Index,
    out_csv: Path,
) -> pd.DataFrame:
    importances = model.feature_importances_
    df_imp = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)
    df_imp.to_csv(out_csv, index=False)
    return df_imp


def run_kmeans_and_crosstab(
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    n_clusters: int,
    out_csv: Path,
) -> None:
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    clusters = kmeans.fit_predict(X.values)

    df = pd.DataFrame(
        {"Label": label_encoder.inverse_transform(y), "Cluster": clusters}
    )
    ctab = pd.crosstab(df["Label"], df["Cluster"])
    ctab.to_csv(out_csv)


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    ensure_exists(DATA_CSV)

    X, y = load_dataset(DATA_CSV)
    y_enc, le = encode_labels(y)

    value_counts = pd.Series(y_enc).value_counts()
    stratify_ok = (value_counts.min() >= 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc if stratify_ok else None,
    )

    rf = train_random_forest(X_train, y_train, N_ESTIMATORS, RANDOM_STATE)
    evaluate_model(rf, X_test, y_test, le, OUT_CONFUSION, OUT_REPORT)

    _feat_imp_df = compute_feature_importance(rf, X.columns, OUT_FEATIMP)

    # Optional KMeans
    n_classes = len(le.classes_)
    if n_classes >= 2:
        run_kmeans_and_crosstab(X, y_enc, le, n_clusters=n_classes, out_csv=OUT_CLUSTER_CROSSTAB)
