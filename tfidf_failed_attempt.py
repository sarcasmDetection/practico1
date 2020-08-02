# %% markdown
# Análisis y Curación sobre DFs de Sarcasmo y No Sarcasmo

# %%
import math

import numpy as np
import pandas as pd
import seaborn as sn
import seaborn as sns
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

# %%
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

# %%
df = pd.read_csv("./sarcasm_v2/GEN-sarc-notsarc.csv")
# %%
sarc_df = df["sarc" == df["class"]]
notsarc_df = df["notsarc" == df["class"]]

# %%
# tf
TF_THRESHOLD = 25
sarc_token_count = {}
sarc_word_count = 0
sarc_token_tf = {}

# idf
IDF_THRESHOLD = 3
sarc_token_doc_occur = {}
sarc_token_idf = {}
sarc_doc_count = len(sarc_df['text'])

# %%
for row in sarc_df['text']:
    text = nlp(row.lower())
    tokens = [
        token for token in text
        if not (token.is_stop or token.is_punct or token.is_space)
    ]
    doc_tokens = set()

    for t in tokens:
        sarc_token_count[t.text] = sarc_token_count.get(t.text, 0) + 1
        doc_tokens.add(t.text)
        sarc_word_count += 1

    for t in doc_tokens:
        sarc_token_doc_occur[t] = sarc_token_doc_occur.get(t, 0) + 1


# %%
# tf
for t in sarc_token_count:
    if sarc_token_count[t] >= TF_THRESHOLD:
        sarc_token_tf[t] = (sarc_token_count[t] / sarc_word_count)

# %%
# idf
for t in sarc_token_doc_occur:
    if sarc_token_doc_occur[t] >= IDF_THRESHOLD:
        sarc_token_idf[t] = math.log(sarc_doc_count/sarc_token_doc_occur[t])

# %%
# tf-idf
sarc_token_tfidf = {
    k: sarc_token_tf[k] * sarc_token_idf[k]
    for k in sarc_token_tf
    if sarc_token_idf.get(k, False)
}

# %%
sorted_tfidf = {
    k: v for k, v in
    sorted(
        sarc_token_tfidf.items(),
        key=lambda item: item[1],
        reverse=True
    )
}

# %%
x = [k for k in sorted_tfidf]
y = [sorted_tfidf[k] for k in sorted_tfidf]

gr = sns.lineplot(x=x, y=y, sort=False,)
_ = gr.set_xticklabels(labels=[k for k in sorted_tfidf], rotation=90)

# %%
v = [sorted_tfidf[k] for k in sorted_tfidf]
np.mean(v)
np.median(v)
