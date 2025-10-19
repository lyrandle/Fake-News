# packages
import pandas as pd
import numpy as np
import re #normalize and clean
#tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#reading data to data frame
df = pd.read_csv(r"C:\Users\lyric\source\repos\Fake News\Fake News\Data\news.csv")

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