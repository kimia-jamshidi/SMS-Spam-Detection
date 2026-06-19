Project: SMS Spam Detection


Dataset:
SMS spam collection


Reason for selection:
- Appropriate size(5572 rows)
- Binary classification problem
- clean structure
- suitable for beginner NLP project.

rows:5572
columns:2

Features:
-message

Target:
-label

Missing Data:
- None

Class Distribution:
- Ham : 4825(86.6%)
- Spam : 747(13.4%)

Duplicate Records:
- 403 duplicate records were detected.

After analysis:
- 309 duplicate ham messages
- 94 duplicate spam messages

Duplicates were removed to reduce bias caused by repeated messages and improve data quality.



Message Length Analysis

A new feature called message_length was created by calculating the number of characters in each SMS message.

Results

- Average spam message length: 138.67 characters
- Average ham message length: 71.48 characters
- Maximum message length: 910 characters
- Minimum message length: 2 characters

Observations

- Spam messages are significantly longer than ham messages on average.
- The average spam message is almost twice as long as a normal message.
- The longest message in the dataset was a ham message with 910 characters, showing that message length alone cannot perfectly distinguish spam from ham messages.
- The shortest messages contained only 2 characters and were all "Ok" messages.
- Message length appears to be a useful feature for spam detection, but it should be combined with other text-based features for better classification performance.

### Histogram Analysis

The distribution of message lengths is right-skewed. Most messages contain fewer than 200 characters, while a small number of messages are extremely long and appear as outliers.

### Boxplot Analysis

Spam messages tend to be significantly longer than ham messages. The median and average message length of spam messages are noticeably higher. This indicates that message length may be a useful feature for spam classification.

### Conclusion

Message length shows a clear difference between ham and spam messages. Therefore, it can be considered a potentially informative feature for the spam detection model.



## Text Cleaning

A custom function called `remove_punctuation()` was created to preprocess text messages.

The function performs the following operations:

1. Convert all characters to lowercase.
2. Remove punctuation marks.
3. Keep numbers unchanged.

A new column called `message_clean` was created to store the cleaned text.

### Result

- All text is converted to lowercase.
- Punctuation marks are removed.
- Numerical values are preserved.


## Text Preprocessing

Text data cannot be directly used by machine learning models, so several preprocessing steps were applied to convert raw messages into a clean and meaningful format.

The following steps were performed:

### 1. Lowercase Conversion

All messages were converted to lowercase to ensure that words with different capitalization are treated as the same feature.

This prevents duplicate features caused by different letter cases.

---

### 2. Removing Punctuation

Punctuation marks were removed because they usually do not provide useful information for spam classification.

Numbers were kept because they can contain important spam-related information (for example prize amounts, phone numbers, or promotional codes).

---

### 3. Tokenization

Messages were split into individual words (tokens).

Tokenization allows further processing on individual words.

---

### 4. Stopword Removal

Common words with low importance for classification were removed.

Examples of stopwords:
the, a, in, is, and, to

This reduces noise and helps the model focus on more meaningful words.

---

### 5. Lemmatization

Words were converted to their base form to reduce different variations of the same word.

This helps the model recognize related words as the same feature.

---

### 6. Final Text Preparation

After preprocessing, the cleaned tokens were joined back into text format to prepare them for feature extraction.

The final processed text was stored in the `message_final` column.

---

## Feature Extraction

Machine learning models cannot understand text directly, so the processed messages were converted into numerical features using **Bag of Words** with `CountVectorizer`.

The vectorizer created a vocabulary of unique words and transformed each message into a numerical representation.

Dataset shape after vectorization:
(5567, 8910)

Meaning:

- 5567 messages
- 8910 text features (unique words)

---

## Label Encoding

The target labels were converted into numerical values:
ham → 0 spam → 1

This allowed the classification model to work with the target variable.

---

## Train-Test Split

The dataset was divided into training and testing sets:

Training data: 80% Testing data: 20%

Training data was used for learning patterns, while testing data was used to evaluate model performance on unseen messages.


## Model Evaluation

A Multinomial Naive Bayes classifier was trained using the Bag of Words features.

The model achieved an accuracy of:
97.48٪


The classification report shows:

| Class | Precision | Recall | F1-score | Support |
|------|-----------|--------|----------|---------|
| Ham (0) | 0.99 | 0.98 | 0.99 | 966 |
| Spam (1) | 0.87 | 0.95 | 0.91 | 149 |

### Analysis

The model performs very well in detecting normal messages (ham) with an F1-score of 0.99.

For spam messages, the model achieved a recall of 0.95, meaning it successfully detected most spam messages. The precision score of 0.87 shows that some normal messages were incorrectly classified as spam.

Overall, the Multinomial Naive Bayes model provides strong performance for SMS spam classification using Bag of Words features.


## Model Comparison

Two classification models were evaluated:

1. Multinomial Naive Bayes
2. Logistic Regression

### Results

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-score |
|------|----------|----------------|-------------|---------------|
| Multinomial Naive Bayes | 97.48% | 0.87 | 0.95 | 0.91 |
| Logistic Regression | 98.47% | 1.00 | 0.89 | 0.94 |

### Analysis

Both models achieved strong performance for SMS spam classification.

Multinomial Naive Bayes achieved higher recall for spam detection, meaning it detected more spam messages.

Logistic Regression achieved better overall accuracy and higher precision, meaning spam predictions were more reliable with fewer false alarms.

Based on the evaluation results, Logistic Regression provided the best overall performance with an accuracy of 98.47%.


# Conclusion

Both models achieved strong results.

Logistic Regression provided the best overall performance with:

98.47% accuracy


The final pipeline successfully transformed raw SMS messages into numerical features and classified them using machine learning techniques.

---

# Technologies Used

- Python
- Pandas
- Numpy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn