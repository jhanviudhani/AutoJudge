from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
import json
import os

app = Flask(__name__)

# ======================
# LOAD MODELS & METADATA
# ======================
print("Loading models...")
vectorizer = joblib.load("model/vectorizer.pkl")
clf = joblib.load("model/classifier.pkl")  # Classification model
reg = joblib.load("model/regressor.pkl")   # Regression model

# Load metadata (score ranges from training)
metadata = {}
if os.path.exists("model/metadata.json"):
    with open("model/metadata.json", "r") as f:
        metadata = json.load(f)
    print("✅ Loaded metadata from training")
    print(f"   Score range: {metadata['score_stats']['min']:.2f} - {metadata['score_stats']['max']:.2f}")
else:
    print("⚠️  No metadata found, using default ranges")
    metadata = {
        'score_stats': {'min': 1.0, 'max': 8.1, 'mean': 4.5, 'std': 2.0},
        'class_score_ranges': {
            'easy': {'min': 1.0, 'max': 3.0, 'mean': 2.0},
            'medium': {'min': 3.0, 'max': 6.0, 'mean': 4.5},
            'hard': {'min': 6.0, 'max': 8.1, 'mean': 7.0}
        }
    }

# Extract score range for normalization
SCORE_MIN = metadata['score_stats']['min']
SCORE_MAX = metadata['score_stats']['max']

print(f"Normalizing scores from [{SCORE_MIN:.2f}, {SCORE_MAX:.2f}] to [0, 100]")

# ======================
# HELPER FUNCTIONS
# ======================
def extract_features(text):
    """Extract additional text features for better prediction"""
    features = {}
    
    # Text length
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    
    # Mathematical complexity indicators
    features['math_symbols'] = len(re.findall(r'[∑∏∫√π±×÷≤≥≠≈∞]', text))
    features['latex_math'] = len(re.findall(r'\$.*?\$', text))
    
    # Algorithm keywords (weighted by difficulty)
    hard_keywords = [
        'dynamic programming', 'dp', 'graph', 'tree', 'recursion', 
        'binary search', 'dfs', 'bfs', 'dijkstra', 'memoization',
        'backtracking', 'greedy', 'optimization', 'complexity',
        'shortest path', 'minimum spanning tree', 'topological sort',
        'segment tree', 'fenwick', 'trie', 'suffix'
    ]
    
    medium_keywords = [
        'array', 'hash', 'sort', 'stack', 'queue', 'string', 
        'prefix', 'suffix', 'iteration', 'two pointer', 'sliding window'
    ]
    
    text_lower = text.lower()
    features['hard_kw_count'] = sum(1 for kw in hard_keywords if kw in text_lower)
    features['medium_kw_count'] = sum(1 for kw in medium_keywords if kw in text_lower)
    
    # Problem size indicators
    features['large_numbers'] = len(re.findall(r'10\^[6-9]|10\^\{[6-9]\}', text))
    features['constraints'] = text.count('≤') + text.count('<=')
    
    return features

def calculate_difficulty_score(text, raw_model_score, features):
    """
    Calculate difficulty score (0-100) based on model prediction + features
    Uses ACTUAL score range from training data
    """
    # Normalize the raw model score to 0-100
    # Using the ACTUAL min/max from training data
    normalized_base = ((raw_model_score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)) * 100
    normalized_base = max(0, min(100, normalized_base))
    
    # Feature-based adjustments (more conservative)
    score_adjustments = 0
    
    # Text complexity
    if features['word_count'] > 300:
        score_adjustments += 8
    elif features['word_count'] > 150:
        score_adjustments += 4
    
    # Mathematical complexity
    if features['math_symbols'] > 5 or features['latex_math'] > 3:
        score_adjustments += 12
    elif features['math_symbols'] > 2:
        score_adjustments += 6
    
    # Algorithm keywords (strong indicator)
    if features['hard_kw_count'] >= 3:
        score_adjustments += 12
    elif features['hard_kw_count'] >= 1:
        score_adjustments += 6
    
    # Large constraint problems
    if features['large_numbers'] > 0:
        score_adjustments += 8
    
    # Combine scores
    final_score = normalized_base + score_adjustments
    
    # Clamp to 0-100
    final_score = max(0, min(100, final_score))
    
    return int(final_score)

def determine_difficulty_class(predicted_class, score, features):
    """
    Determine difficulty class using model prediction + validation
    """
    # Trust the classification model primarily
    primary_class = predicted_class
    
    # But validate with features and score
    # Strong indicators can override if there's clear mismatch
    if features['hard_kw_count'] >= 3 and score > 70:
        # Very strong hard indicators
        if primary_class.lower() == "easy":
            return "Medium"  # At least bump to medium
        return primary_class
    
    if features['hard_kw_count'] == 0 and features['medium_kw_count'] <= 1 and score < 35:
        # Very strong easy indicators
        if primary_class.lower() == "hard":
            return "Medium"  # Don't jump to easy, be conservative
        return primary_class
    
    # Otherwise trust the classifier
    return primary_class

def calculate_confidence(difficulty, score, features, class_proba):
    """
    Calculate confidence based on classifier probability + feature consistency
    """
    # Start with classifier confidence
    base_confidence = float(class_proba)
    
    # Adjust based on score-difficulty alignment
    difficulty_lower = difficulty.lower()
    
    if difficulty_lower == "easy" and score < 30:
        base_confidence = min(0.95, base_confidence + 0.05)
    elif difficulty_lower == "hard" and score > 70:
        base_confidence = min(0.95, base_confidence + 0.05)
    elif difficulty_lower == "medium" and 30 <= score <= 70:
        base_confidence = min(0.90, base_confidence + 0.05)
    else:
        # Score doesn't align perfectly with difficulty
        base_confidence = max(0.60, base_confidence - 0.10)
    
    # Feature consistency boost
    if difficulty_lower == "hard" and features['hard_kw_count'] >= 2:
        base_confidence = min(0.95, base_confidence + 0.05)
    if difficulty_lower == "easy" and features['hard_kw_count'] == 0:
        base_confidence = min(0.95, base_confidence + 0.05)
    
    return round(base_confidence, 2)

# ======================
# ROUTES
# ======================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Combine text fields
        text = (
            data.get("description", "") + " " +
            data.get("input", "") + " " +
            data.get("output", "")
        ).strip()
        
        if not text:
            return jsonify({
                "error": "Please provide problem description"
            }), 400
        
        # Transform text using TF-IDF
        X = vectorizer.transform([text])
        
        # Get predictions from both models
        # Classification
        predicted_class = clf.predict(X)[0]
        class_proba = clf.predict_proba(X)[0]
        class_confidence = float(max(class_proba))
        
        # Regression
        raw_score = float(reg.predict(X)[0])
        
        # Extract features
        features = extract_features(text)
        
        # Calculate difficulty score (0-100)
        score = calculate_difficulty_score(text, raw_score, features)
        
        # Determine final difficulty class
        difficulty = determine_difficulty_class(predicted_class, score, features)
        
        # Calculate confidence
        confidence = calculate_confidence(difficulty, score, features, class_confidence)
        
        # Extract keywords from TF-IDF
        feature_names = vectorizer.get_feature_names_out()
        tfidf_vals = X.toarray()[0]
        top_idx = np.argsort(tfidf_vals)[-8:][::-1]
        keywords = [feature_names[i] for i in top_idx if tfidf_vals[i] > 0][:6]
        
        return jsonify({
            "difficulty": difficulty,
            "score": score,
            "confidence": confidence,
            "keywords": keywords,
            "debug_info": {
                "raw_model_score": round(raw_score, 2),
                "classifier_prediction": predicted_class,
                "classifier_confidence": round(class_confidence, 2),
                "hard_keywords": features['hard_kw_count'],
                "word_count": features['word_count'],
                "score_range": f"{SCORE_MIN:.1f}-{SCORE_MAX:.1f}"
            }
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

@app.route("/stats")
def stats():
    """Return model statistics"""
    return jsonify(metadata)

if __name__ == "__main__":
    app.run()
