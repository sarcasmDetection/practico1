# %% markdown
# Aprendizaje Automático y Supervisado

# %%
# import the library thingies
import numpy as np
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm

# import the utils thingies
from utils.concat_dfs import get_df

# %%
# download the language thingies
nltk.download('stopwords')

# %%
# load the data thingies and concatenate them
df = get_df([
    "./dataframes/GEN-sarc-notsarc.csv",
    "./dataframes/RQ-sarc-notsarc.csv",
    "./dataframes/HYP-sarc-notsarc.csv",
])

# %%
# get the tf-idf counter thingy of the data thingies
MIN_NGRAM = 1
MAX_NGRAM = 3
MIN_DF = 5
MAX_DF = 0.2

corpus = df['text'].apply(lambda x: np.str_(x))

vectorizer = TfidfVectorizer(
    ngram_range=(MIN_NGRAM, MAX_NGRAM),
    min_df= MIN_DF,
    max_df= MAX_DF,
    stop_words=stopwords.words('english'),
)

X = vectorizer.fit_transform(corpus)

# %%
# split the vectorizer thingy in train and test thingies
X_train, X_test, y_train, y_test = train_test_split(X, df["class"])

# %%markdown
## Logistic Regression

# %%
# train a classifier thingy
lg_clf = LogisticRegression()
lg_clf.fit(X_train, y_train)

# %%
# predict sarcasm thingies using the classifier thingy
y_pred = lg_clf.predict(X_test)

# %%
# print metrics of the prediction thingy
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# %%markdown
## Super Vector Machine

# %%
# train a classifier thingy
svm_clf = svm.SVC()
svm_clf.fit(X_train, y_train)

# %%
# predict sarcasm thingies using the classifier thingy
y_pred = svm_clf.predict(X_test)

# %%
# print metrics of the prediction thingy
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
