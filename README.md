
# Project Title

A brief description of what this project does and who it's for

# Spam Classifier

A machine learning web app that classifies a given message (SMS, email, or plain text) as Spam or Not Spam. Built using Natural Language Processing (NLP) and trained on a dataset of labeled messages.

## Overview

This project uses text preprocessing and classification techniques to detect spam messages. It features a web interface built with Streamlit and a Multinomial Naive Bayes model trained on TF-IDF features. The model was selected after evaluating several machine learning algorithms including SVM, Logistic Regression, Decision Tree, KNN, Random Forest, XGBoost, AdaBoost, and ensemble methods like Voting and Stacking classifiers. MultinomialNB was chosen for deployment based on its high accuracy and precision.

## Tech Stack

- Python
- Streamlit
- NLTK
- Scikit-learn
- Pickle (for saving/loading model)
- TF-IDF Vectorizer

## Features

- Input any message or text (SMS, email, etc.)
- Classify it as Spam or Not Spam
- Streamlit web interface for instant predictions
- Lightweight and fast execution using a pre-trained model

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yAnwesha4/spam-classifier-ML-project.git
   cd spam-classifier-ML-project

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

pip install streamlit nltk scikit-learn xgboost

## Download NLTK data

- import nltk
- nltk.download('punkt')
- nltk.download('stopwords')

## Run the application

```bash
streamlit run app.py

