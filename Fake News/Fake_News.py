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

#Build model
# Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Logistic Regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)




# Evaluate model performance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probabilities for ROC curve

le = LabelEncoder()
y_test_enc = le.fit_transform(y_test)
y_pred_enc = le.transform(y_pred)
# Define 5-Fold cross-validation
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test_enc, y_pred_enc))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Cross-validation accuracies for each fold:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


# Visualize and interpret results
# Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
import matplotlib.pyplot as plt
fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label='REAL')
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", color='blue')
plt.plot([0, 1], [0, 1], 'r--')  # diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid(True)
plt.show()


# Save model and vectorizer
import joblib

joblib.dump(model, 'model/fake_news_model.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')


