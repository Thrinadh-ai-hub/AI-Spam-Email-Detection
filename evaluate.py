
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "emails.csv"

# Load dataset
df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
df["label"] = df["label"].str.lower().str.strip()
df = df[df["label"].isin(["spam", "ham"])]

X = df["text"]
y = df["label"]

# Split: 80% development, 20% final test
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Split development: 60% train, 20% validation overall
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev,
)

print("Training emails:", len(X_train))
print("Validation emails:", len(X_val))
print("Test emails:", len(X_test))

# Train model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)

spam_index = list(model.classes_).index("spam")

# VALIDATION: choose threshold here
val_prob = model.predict_proba(X_val)[:, spam_index]

print("\nVALIDATION RESULTS")

thresholds = np.arange(0.30, 0.91, 0.01)
scores = []

for threshold in thresholds:
    pred = np.where(val_prob >= threshold, "spam", "ham")
    scores.append(
        f1_score(y_val, pred, pos_label="spam", zero_division=0)
    )

best_threshold = float(thresholds[int(np.argmax(scores))])

for threshold in [0.50, 0.60, 0.70, best_threshold]:
    pred = np.where(val_prob >= threshold, "spam", "ham")
    print(
        f"Threshold {threshold:.2f} | "
        f"Accuracy {accuracy_score(y_val, pred):.4f} | "
        f"Spam Precision {precision_score(y_val, pred, pos_label='spam', zero_division=0):.4f} | "
        f"Spam Recall {recall_score(y_val, pred, pos_label='spam', zero_division=0):.4f} | "
        f"Spam F1 {f1_score(y_val, pred, pos_label='spam', zero_division=0):.4f}"
    )

print("\nSelected validation threshold:", round(best_threshold, 2))

# FINAL TEST: evaluate once
test_prob = model.predict_proba(X_test)[:, spam_index]
test_pred = np.where(test_prob >= best_threshold, "spam", "ham")

print("\nFINAL TEST RESULTS")
print("Threshold:", round(best_threshold, 2))
print("Accuracy:", round(accuracy_score(y_test, test_pred), 4))
print("Spam precision:", round(precision_score(y_test, test_pred, pos_label="spam", zero_division=0), 4))
print("Spam recall:", round(recall_score(y_test, test_pred, pos_label="spam", zero_division=0), 4))
print("Spam F1:", round(f1_score(y_test, test_pred, pos_label="spam", zero_division=0), 4))

matrix = confusion_matrix(y_test, test_pred, labels=["ham", "spam"])
print("\nConfusion matrix (rows=actual, columns=predicted):")
print("              Pred HAM  Pred SPAM")
print("Actual HAM:  ", matrix[0])
print("Actual SPAM: ", matrix[1])

tn, fp, fn, tp = matrix.ravel()
print("\nFalse positives:", fp)
print("Missed spam:", fn)
print("True negatives:", tn)
print("Spam correctly detected:", tp)
