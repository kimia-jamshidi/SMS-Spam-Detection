# SMS Spam Classification

A machine learning project that classifies SMS messages as **spam** or **ham (normal)** using NLP techniques and machine learning models.

## Dataset

The dataset contains SMS messages with two labels:

- `ham` → normal messages
- `spam` → unwanted messages

Labels were encoded as:
ham → 0 spam → 1

## NLP Pipeline

The text data was processed using the following steps:

- Lowercase conversion
- Removing punctuation
- Tokenization
- Stopword removal
- Lemmatization

The final cleaned text was used for feature extraction.

## Feature Extraction

Text was converted into numerical features using:

- **Bag of Words**
- `CountVectorizer`

After vectorization:
5567 messages 8910 features

## Models

Two classification models were tested:

- Multinomial Naive Bayes
- Logistic Regression

## Results

| Model | Accuracy |
|---|---|
| Multinomial Naive Bayes | 97.48% |
| Logistic Regression | 98.47% |

## Evaluation

The Logistic Regression model achieved the best performance.

Classification results:

- Ham detection F1-score: **0.99**
- Spam detection F1-score: **0.94**

The model successfully classified SMS messages while minimizing false spam detection.

## Technologies

- Python
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn