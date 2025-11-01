# packages
import pandas as pd
import numpy as np
import re #normalize and clean
#tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#reading data to data frame
df = pd.read_csv(r"Data/news.csv")

#drop id column
df_clean = df.drop('Unnamed: 0', axis=1)

#drop null values and empty strings
df_clean = df_clean.dropna()
df = df[(df['title'].str.strip() != '') & (df['text'].str.strip() != '')]
print(df_clean)

# Check label distribution
count = df['label'].value_counts()
percent = df['label'].value_counts(normalize=True) * 100
summary = pd.DataFrame({'Count': count, 'Percentage': percent.round(2)})
print(summary)

# Calculate and compare the average, minimum, and maximum word counts for both fake and real news
df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))
df['text_word_count'] = df['text'].apply(lambda x: len(str(x).split()))

summary = df.groupby('label')[['title_word_count', 'text_word_count']].agg(['mean', 'min', 'max']).round(2)
print(summary)

#cleaning text
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

#cleaning data and tokenizing
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)     # keep only letters
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

#extracting features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text']).toarray()
y = df['label']
#build model

# Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a baseline model (LogisticRegression)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Evaluate model performance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


'''
#Fine-tune with hyperparameters
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 2],
    'max_iter': [200, 300, 500]
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)


# Try different algorithms(Naive Bayes)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
print("Naive Bayes accuracy:", nb.score(X_test, y_test))



# Visualize and interpret results

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()'''

# Save model and vectorizer
import joblib

joblib.dump(model, 'model/fake_news_model.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')


