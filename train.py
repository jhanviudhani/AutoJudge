import json
import pandas as pd
import joblib
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix

# ======================
# 1. LOAD DATASET
# ======================
print("Loading dataset...")
data = []
with open("data/problems_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
print(f"Loaded {len(df)} problems")

# ======================
# 2. HANDLE MISSING VALUES
# ======================
print("\nHandling missing values...")
text_cols = ['title', 'description', 'input_description', 'output_description']

for col in text_cols:
    df[col] = df[col].fillna("")

# ======================
# 3. COMBINE ALL TEXT
# ======================
print("Combining text fields...")
df['full_text'] = (
    df['title'] + " " +
    df['description'] + " " +
    df['input_description'] + " " +
    df['output_description']
)

# ======================
# 4. DATA ANALYSIS
# ======================
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)

print(f"\nProblem Score Range: {df['problem_score'].min():.2f} - {df['problem_score'].max():.2f}")
print(f"Mean: {df['problem_score'].mean():.2f}, Median: {df['problem_score'].median():.2f}")

print("\nClass Distribution:")
print(df['problem_class'].value_counts())

print("\nScore by Class:")
for cls in df['problem_class'].unique():
    scores = df[df['problem_class'] == cls]['problem_score']
    print(f"{cls}: {scores.min():.2f} - {scores.max():.2f} (mean: {scores.mean():.2f})")

# ======================
# 5. FEATURES & TARGETS
# ======================
X_text = df['full_text']
y_class = df["problem_class"]
y_score = df['problem_score']

# Store score statistics for later use
score_stats = {
    'min': float(df['problem_score'].min()),
    'max': float(df['problem_score'].max()),
    'mean': float(df['problem_score'].mean()),
    'std': float(df['problem_score'].std())
}

# Class score ranges for reference
class_score_ranges = {}
for cls in df['problem_class'].unique():
    scores = df[df['problem_class'] == cls]['problem_score']
    class_score_ranges[cls] = {
        'min': float(scores.min()),
        'max': float(scores.max()),
        'mean': float(scores.mean())
    }

print("\n" + "="*60)

# ======================
# 6. TEXT → NUMBERS (TF-IDF)
# ======================
print("\nVectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,  # Reduced for better generalization
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.8  # Ignore terms that appear in more than 80% of documents
)

X = vectorizer.fit_transform(X_text)
print(f"Feature matrix shape: {X.shape}")

# ======================
# 7. TRAIN-TEST SPLIT
# ======================
print("\nSplitting data...")
X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ======================
# 8. CLASSIFICATION MODEL
# ======================
print("\n" + "="*60)
print("TRAINING CLASSIFICATION MODEL")
print("="*60)

# Try RandomForest (usually better than Naive Bayes)
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_class_train)

class_preds = clf.predict(X_test)
class_acc = accuracy_score(y_class_test, class_preds)

print(f"\n✅ Classification Accuracy: {class_acc:.4f} ({class_acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_class_test, class_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_class_test, class_preds))

# ======================
# 9. REGRESSION MODEL
# ======================
print("\n" + "="*60)
print("TRAINING REGRESSION MODEL")
print("="*60)

# Try RandomForest Regressor (usually better than Linear Regression)
reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
reg.fit(X_train, y_score_train)

score_preds = reg.predict(X_test)
mae = mean_absolute_error(y_score_test, score_preds)
rmse = np.sqrt(np.mean((y_score_test - score_preds) ** 2))

print(f"\n✅ Regression MAE: {mae:.4f}")
print(f"✅ Regression RMSE: {rmse:.4f}")

# Show prediction range
print(f"\nPrediction Statistics:")
print(f"Actual range: {y_score_test.min():.2f} - {y_score_test.max():.2f}")
print(f"Predicted range: {score_preds.min():.2f} - {score_preds.max():.2f}")

# ======================
# 10. SAVE MODELS
# ======================
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

os.makedirs("model", exist_ok=True)

joblib.dump(clf, "model/classifier.pkl")
joblib.dump(reg, "model/regressor.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

# Save metadata for use in Flask app
metadata = {
    'score_stats': score_stats,
    'class_score_ranges': class_score_ranges,
    'vectorizer_features': X.shape[1],
    'training_samples': len(df),
    'class_distribution': df['problem_class'].value_counts().to_dict()
}

with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Saved:")
print("  - model/classifier.pkl")
print("  - model/regressor.pkl")
print("  - model/vectorizer.pkl")
print("  - model/metadata.json")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

print("\n📊 Model Summary:")
print(f"  Classification Accuracy: {class_acc*100:.2f}%")
print(f"  Regression MAE: {mae:.4f}")
print(f"  Score Range: {score_stats['min']:.2f} - {score_stats['max']:.2f}")
print("\n⚠️  IMPORTANT: Update Flask app with this score range!")