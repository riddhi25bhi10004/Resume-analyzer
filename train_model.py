"""
train_model.py
--------------
Trains a TF-IDF + Logistic Regression model on the
"Resume Dataset" from Kaggle to classify resumes by job category.

Kaggle Dataset:
  https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
  (CSV with columns: 'Resume_str' and 'Category')

Usage:
  1. Download the dataset CSV from Kaggle and place it in the same folder.
  2. pip install -r requirements.txt
  3. python train_model.py --data UpdatedResumeDataSet.csv

The trained model and vectorizer are saved as:
  - models/tfidf_vectorizer.pkl
  - models/resume_classifier.pkl
"""

import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from preprocessing import preprocess


# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train resume classifier on Kaggle dataset.")
parser.add_argument(
    "--data",
    type=str,
    default="UpdatedResumeDataSet.csv",
    help="Path to the Kaggle resume CSV file.",
)
parser.add_argument(
    "--test_size",
    type=float,
    default=0.2,
    help="Fraction of data to use for testing (default: 0.2).",
)
parser.add_argument(
    "--max_features",
    type=int,
    default=5000,
    help="Max TF-IDF features (default: 5000).",
)
args = parser.parse_args()


# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"\n📂 Loading dataset: {args.data}")
df = pd.read_csv(args.data)

# Kaggle dataset uses these column names
RESUME_COL = "Resume_str"
LABEL_COL  = "Category"

# Validate columns
assert RESUME_COL in df.columns, f"Column '{RESUME_COL}' not found. Check your CSV."
assert LABEL_COL  in df.columns, f"Column '{LABEL_COL}' not found. Check your CSV."

print(f"✅ Loaded {len(df)} records across {df[LABEL_COL].nunique()} categories.")
print(f"\nCategory distribution:\n{df[LABEL_COL].value_counts().to_string()}\n")


# ── Preprocess ────────────────────────────────────────────────────────────────
print("🔄 Preprocessing resume text (this may take a minute)...")
df["clean_resume"] = df[RESUME_COL].fillna("").apply(preprocess)


# ── Encode labels ─────────────────────────────────────────────────────────────
le = LabelEncoder()
df["label"] = le.fit_transform(df[LABEL_COL])


# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_resume"],
    df["label"],
    test_size=args.test_size,
    random_state=42,
    stratify=df["label"],
)
print(f"📊 Train samples: {len(X_train)} | Test samples: {len(X_test)}")


# ── TF-IDF vectorization ──────────────────────────────────────────────────────
print(f"\n🔢 Fitting TF-IDF vectorizer (max_features={args.max_features})...")
vectorizer = TfidfVectorizer(
    max_features=args.max_features,
    ngram_range=(1, 2),   # unigrams + bigrams
    sublinear_tf=True,    # apply log normalization
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)


# ── Train classifier ──────────────────────────────────────────────────────────
print("🤖 Training Logistic Regression classifier...")
clf = LogisticRegression(max_iter=1000, C=5, solver="lbfgs", multi_class="auto")
clf.fit(X_train_tfidf, y_train)


# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# ── Save model artifacts ──────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/resume_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\n💾 Saved model artifacts to ./models/")
print("   - models/tfidf_vectorizer.pkl")
print("   - models/resume_classifier.pkl")
print("   - models/label_encoder.pkl")
print("\n🚀 Done! Use these in app.py to predict resume categories.")
