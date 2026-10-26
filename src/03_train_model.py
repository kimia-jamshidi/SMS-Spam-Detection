# Model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import joblib
from pathlib import Path

df = pd.read_csv("../data/processed_sms.csv")

print("Labels in processed data:")
print(df['label'].unique())



df["message_final"] = df["message_final"].fillna("").astype(str)

encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

x_train_text, x_test_text, y_train, y_test = train_test_split(
    df["message_final"],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

cv = CountVectorizer()

x_train = cv.fit_transform(x_train_text)
x_test = cv.transform(x_test_text)

print("Label mapping:", dict(
    zip(encoder.classes_, encoder.transform(encoder.classes_))
))
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)



# Naive Bayes
model = MultinomialNB()

model.fit(x_train, y_train)
y_predict = model.predict(x_test)
accuracy_score(y_test, y_predict)
print(classification_report(y_test,y_predict))


# Logistic Regression
log_model = LogisticRegression()
log_model.fit(x_train, y_train)



y_predict_log = log_model.predict(x_test)
accuracy_score(y_test, y_predict_log)
print(classification_report(y_test,y_predict_log))


# Confusion Matrix
cm = confusion_matrix(y_test, y_predict_log)
cm

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# save trained model and preprocessing objects
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_DIR / "spam_model.joblib")
joblib.dump(cv, MODEL_DIR / "vectorizer.joblib")
joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")

print("Selected model and preprocessing objects saved successfully.")