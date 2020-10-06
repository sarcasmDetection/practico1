# %% markdown
#Aprendizaje Automático y Supervisado

# %%
# import the library thingies
import numpy as np
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    TfidfTransformer,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
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
    "../dataframes/GEN-sarc-notsarc.csv",
    "../dataframes/RQ-sarc-notsarc.csv",
    "../dataframes/HYP-sarc-notsarc.csv",
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
    min_df=MIN_DF,
    max_df=MAX_DF,
    stop_words=stopwords.words('english'),
)

X = vectorizer.fit_transform(corpus)

# %%
# split the vectorizer thingy in train and test thingies
X_train, X_test, y_train, y_test = train_test_split(X, df["class"])

# %%markdown
## Pipeline

# %%
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('lg_clf', LogisticRegression()),
])

parameters = {
    'vect__max_df': (0.3, 0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
    'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'lg_clf__max_iter': (100,),
    'lg_clf__penalty': ('l2', 'elasticnet'),
    # 'clf__max_iter': (10, 50, 80),
    # 'svm_clf__max_iter': (20,),
    # 'svm_clf__alpha': (0.00001, 0.000001),
    # 'svm_clf__penalty': ('l2', 'elasticnet'),
    # 'clf__max_iter': (10, 50, 80),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
grid_search.fit(df["text"], df["class"])
grid_search.best_estimator_.get_params()
grid_search.best_estimator_.get_params()
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

confusion_matrix(y_test, y_pred)
# %%markdown
## Support Vector Machine

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

# %%markdown
La medida 'Accuracy' es una forma de medir la frecuencia con la que el algoritmo
clasifica un punto de datos correctamente. En un modelo, se corresponde al
número de puntos de datos predichos correctamente de todos los puntos de datos.
En ambos modelos, obtuvimos un 'Accuracy' que supera el 70%. Esto significa que
nuestro predictor funciona correctamente para decirnos si una frase es
sarcástica, o no, 70 veces de 100. Creemos que es un buen numero para obtener
como resultado, y pensamos seguir modificando parámetros para poder mejorar aún
mas nuestros modelos.

'Precision-Recall' es una medida útil del éxito de la predicción cuando las
clases están muy desequilibradas. En la recuperación de información, 'Precision'
es una medida de la relevancia de los resultados, mientras que 'Recall' es una 
medida de cuántos resultados verdaderamente relevantes se devuelven.
Obtuvimos en nuestros modelos puntajes que superan el 70%. Existe entonces una
alta recuperación y una alta precisión, donde la alta precisión se relaciona con
una tasa baja de falsos positivos y la alta recuperación se relaciona con una
baja tasa de falsos negativos. Los puntajes altos para ambos muestran que el
clasificador está arrojando resultados precisos, así como también, que los
mismos están taggeados correctamente.
