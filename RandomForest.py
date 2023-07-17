import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split  # 80%-20%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenizing texts
    tokens = word_tokenize(text)

    # Removing Stopwords from tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Joining tokens back into a single string
    preprocessed = ' '.join(tokens)

    return preprocessed

# Loading the Dataset
data = pd.read_csv("Dataset/news.csv")

# Create feature "Fake"
# Iterate on Label column, 1 if fake, 0 if real
data["fake"] = data["label"].apply(lambda x: 0 if x == "REAL" else 1)
X, y = data["text"], data["fake"]
X = X.apply(preprocess)

# Splitting data
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2)

# print(len(x_train))
# print(len(x_test))
# print(x_test)
# print(x_train)
#print(data)

# Vectorizing text data into numerical data
# Removing stop words appearing in more than 70% of word docs
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

# Vectorizing text data
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

# Train the model
clf = RandomForestClassifier()
clf.fit(x_train_vectorized, y_train)

# Make Predictions
y_pred = clf.predict(x_test_vectorized)

# Measuring Accuracy with different evaluation metrics
score = clf.score(x_test_vectorized, y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: ", score)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)