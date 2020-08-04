# %%
import math
import pandas as pd
import spacy
import numpy as np

# %%
nlp = spacy.load('en_core_web_sm')

# %%
df = pd.read_csv("./sarcasm_v2/GEN-sarc-notsarc.csv")

# %%
sarc_df = df["sarc" == df["class"]]["text"]
notsarc_df = df["notsarc" == df["class"]]["text"]

# %%
# tf
def get_doc_tf(text):
    doc = nlp(text)
    doc_tf = {}

    tokens = [
        token for token in doc
        if not (token.is_stop or token.is_punct or token.is_space)
    ]
    token_count = len(tokens)

    for t in tokens:
        doc_tf[t.lower_] = doc_tf.get(t.lower_, 0) + 1

    for t in doc_tf:
        doc_tf[t] = doc_tf[t] / token_count

    return doc_tf

# %%
# idf
def get_docs_idf(texts):
    docs_idf = {}
    doc_count = len(texts)

    for text in texts:
        doc = nlp(text)

        # important to notice this is a set (not an array)
        # and therefore tokens will only appear once
        tokens = {
            token for token in doc
            if not (token.is_stop or token.is_punct or token.is_space)
        }
        token_count = len(tokens)

        for t in tokens:
            docs_idf[t.lower_] = docs_idf.get(t.lower_, 0) + 1

    for token in docs_idf:
        docs_idf[token] = math.log(doc_count/docs_idf[token])

    return docs_idf

# %%
# tf-idf
def get_doc_tfidf(doc_tf, docs_idf):
    doc_tfidf = {
        k: doc_tf[k] * docs_idf[k]
        for k in doc_tf
    }

    return doc_tfidf

# %%
docs_idf = get_docs_idf(sarc_df)

# %%
docs_tf = [get_doc_tf(doc) for doc in sarc_df]

# %%
docs_tfidf = [get_doc_tfidf(doc_tf, docs_idf) for doc_tf in docs_tf]

# %%
# get the tokens that have a tf-idf above average of each document
above_avg_tfidf = []
for tfidf in docs_tfidf:
    for token in tfidf:
        above_avg_tokens = {}
        if tfidf[token] > np.average(list(tfidf.values())):
            above_avg_tokens[token] = tfidf[token]
    # if no token is above average we wont be taking it into account
    if above_avg_tokens != {}:
        above_avg_tfidf.append(above_avg_tokens)

# %%
above_avg_tokens = {}
for doc in above_avg_tfidf:
    for token in doc:
        above_avg_tokens[token] = above_avg_tokens.get(token, 0) + doc[token]

# %%
# sort the tokens that appear the most in the documents
sorted_important_tokens = sorted(
    above_avg_tokens.items(), key=lambda x: x[1], reverse=True)
