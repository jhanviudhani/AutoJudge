# AutoJudge 🧠

AutoJudge is an AI-powered web application that predicts the **difficulty level of programming problems** based on their textual description.

It uses **machine learning (TF-IDF + classification + regression)** to analyze problem statements and provide:
- Difficulty class (Easy / Medium / Hard)
- Relative difficulty score (0–100)
- Confidence level
- Detected keywords

---

## 🚀 Live Demo
🔗 **Live Link:**  
https://autojudge-3vlh.onrender.com

> Note: The app is hosted on Render (Free Tier).  
> It may take ~30–50 seconds to wake up if inactive.

---

## 🛠️ Tech Stack
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask (Python)
- **Machine Learning:** Scikit-learn, TF-IDF
- **Deployment:** Render
- **Version Control:** Git & GitHub

---

## 📊 Features
- AI-based difficulty classification
- Relative difficulty scoring
- Keyword extraction using TF-IDF
- Clean and interactive UI
- Input validation (mandatory problem description)

---

## 🧠 How it Works
1. User enters a programming problem description.
2. Text is vectorized using **TF-IDF**.
3. A trained ML model:
   - Predicts difficulty class
   - Estimates relative difficulty score
4. Keywords are extracted based on TF-IDF weights.
5. Results are displayed visually.

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/jhanviudhani/AutoJudge.git
cd AutoJudge
pip install -r requirements.txt
python app.py
