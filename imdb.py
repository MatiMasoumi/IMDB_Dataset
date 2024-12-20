import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from contractions import fix
import nltk
import unicodedata
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA

import random
import seaborn as sns
import contractions
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
# The AttributeError indicates that there is no method named DataFrame_csv in the pandas module.
# To fix the error, use the correct method for reading a CSV file, which is pandas.read_csv.

import pandas as pd

# Corrected code to read the CSV file:
df = pd.read_csv('IMDB Dataset.csv')
df
df.isnull().sum()
df['review'][80]
sentiment_count=df['sentiment'].value_counts()
print("\nSentiment Label Distribution:")
print(sentiment_count)

# Plot Sentiment
sentiment_count.plot(kind='bar', title='Sentiment Label Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

import nltk

nltk.download('stopwords')
nltk.download('punkt')
def preprocess_text(text):
    text=text.lower()
    text=text.translate(str.maketrans('','',string.punctuation))
    tokens=word_tokenize(text)
    tokens=[word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

df['cleaned_review']=df['review'].apply(preprocess_text)
# Correcting the typo 'fit_trasform' to 'fit_transform'

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
from sklearn.metrics import f1_score, classification_report, accuracy_score

y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f'f1_score: {f1}')
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'ClassificationReport:\n{classification_report(y_test, y_pred)}')
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
# To resolve the NameError, ensure that the function roc_curve is correctly imported before using it. The sklearn.metrics module has both roc_curve and auc available.
# Corrected code below:

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate probabilities based on the trained model (Logistic Regression).
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Ensure to select probabilities for the positive class

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()