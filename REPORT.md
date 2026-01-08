AutoJudge – Predicting Programming Problem Difficulty using Machine Learning
1. Introduction

Competitive programming platforms such as Codeforces, CodeChef, LeetCode, and Kattis host thousands of problems with varying difficulty levels. Accurately estimating the difficulty of a programming problem helps learners choose appropriate problems and aids platforms in problem categorization.

AutoJudge is a machine learning–based web application that predicts:

The difficulty class (Easy / Medium / Hard)

A relative difficulty score
based on the textual description of a programming problem.

The system uses Natural Language Processing (NLP) techniques and supervised machine learning models to analyze problem statements and make predictions.

2. Objectives

The main objectives of this project are:

To build an ML model that classifies programming problems into difficulty levels.

To predict a numerical difficulty score using regression.

To design a user-friendly web interface for real-time predictions.

To deploy the application publicly for live usage.

3. Dataset Used

The dataset used in this project is the TaskComplexity Dataset, which contains 4112 programming problems collected from multiple competitive programming platforms.

Each problem includes:

Problem title

Problem description

Input description

Output description

Difficulty class (Easy / Medium / Hard)

Difficulty score

The dataset is stored in JSONL format and serves as labeled training data for supervised learning.

4. Methodology
4.1 Text Processing

Combined problem description, input, and output into a single text field.

Applied TF-IDF Vectorization to convert text into numerical features.

4.2 Machine Learning Models

Two models were trained:

a) Classification Model

Algorithm: Logistic Regression

Task: Predict difficulty class (Easy / Medium / Hard)

b) Regression Model

Algorithm: Linear Regression

Task: Predict a relative difficulty score

4.3 Keyword Extraction

Keywords are extracted from the TF-IDF feature space by selecting terms with the highest weights for a given input.

5. System Architecture

User enters problem description in the web interface

Input is sent to the Flask backend

Text is vectorized using TF-IDF

ML models generate predictions

Results are returned and displayed on the UI

6. Technologies Used

Python

Flask (Backend)

Scikit-learn (ML models)

Pandas & NumPy

HTML, CSS, JavaScript (Frontend)

Gunicorn (Production server)

Render (Deployment)

GitHub (Version control)

7. Deployment

The project is deployed using Render as a public web service.

🔗 Live Demo Link:


https://autojudge-3v1h.onrender.com/

8. Results

The application successfully predicts difficulty class and relative score.

The UI dynamically changes color based on difficulty:

Easy → Green

Medium → Yellow

Hard → Red

Required validation ensures the problem description is mandatory.

9. Limitations

Accuracy depends on the quality and diversity of the dataset.

The difficulty score is relative, not an official platform rating.

NLP-based models may misinterpret very short or ambiguous descriptions.


10. Conclusion

AutoJudge demonstrates how machine learning and NLP can be applied to real-world problems in competitive programming. The project provides a practical tool for difficulty estimation while showcasing full-stack ML deployment.