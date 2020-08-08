# %%markdown
# Practico 2: Análisis y Curación

# %%
import math
import spacy
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from collections import Counter

# %%
nlp = spacy.load('en_core_web_sm')

# %% markdown
Definamos variables que serán de utilidad.

# %%
# Por temas de bajo poder de computo vamos a usar solo un dataframe
df = pd.read_csv("./sarcasm_v2/GEN-sarc-notsarc.csv")
# df = pd.read_csv("./sarcasm_v2/HYP-sarc-notsarc.csv")
# df = pd.read_csv("./sarcasm_v2/RQ-sarc-notsarc.csv")

# %%
sarc_df = df["sarc" == df["class"]]["text"]
notsarc_df = df["notsarc" == df["class"]]["text"]

# %% markdown
Funciones útiles

# %%
flatten_list = lambda nested_list: [
    el for sublist in nested_list for el in sublist]

def compare_freq(most_common_freq, cmp_freq,
                 mc_label, cmp_label,
                 mc_color, cmp_color):
    """
    This function compares the frequency of the most common tokens
    of `most_common_freq` with the frequency they have in `cmp_freq`.
    """
    most_common = most_common_freq.most_common(30)

    most_common_words = [x for x, y in most_common]
    most_common_freqs = [y/len(most_common_freq) for x, y in most_common]

    cmp_freq_in_mc = [cmp_freq[x]/len(cmp_freq) for x, y in most_common]

    most_common = sn.lineplot(
        x=most_common_words,
        y=cmp_freq_in_mc,
        label=cmp_label,
        sort=False,
        color=cmp_color
    )
    not_sarc_gr = sn.lineplot(
        x=most_common_words,
        y=most_common_freqs,
        label=mc_label,
        sort=False,
        color=mc_color
    )
    rot_lab_ns = most_common.set_xticklabels(
        labels=most_common_words, rotation=90)

# %%
sarc_token_texts = [nlp(snts.lower()) for snts in sarc_df]
not_sarc_token_texts = [
    nlp(snts.lower()) for snts in notsarc_df]

# %%
sarc_token_list = flatten_list(sarc_token_texts)
not_sarc_token_list = flatten_list(not_sarc_token_texts)

# %% markdown
## Lematización

# %%
sarc_lemm_tokens = [word.lemma_ for word in sarc_token_list]
not_sarc_lemm_tokens = [word.lemma_ for word in not_sarc_token_list]

# %%
sarc_lemm_freq = Counter(sarc_lemm_tokens)
not_sarc_lemm_freq = Counter(not_sarc_lemm_tokens)

# %%
compare_freq(
    sarc_lemm_freq, not_sarc_lemm_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_lemm_freq, sarc_lemm_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %%markdown
Spacy no tiene una herramienta para stemming por lo cual no es posible hacer
ese transformación sobre los tokens.

# %%markdown
## Heatmap

# %%
df0 = pd.read_csv("./sarcasm_v2/GEN-sarc-notsarc.csv")
df1 = pd.read_csv("./sarcasm_v2/HYP-sarc-notsarc.csv")
df2 = pd.read_csv("./sarcasm_v2/RQ-sarc-notsarc.csv")

df_concat = pd.concat([df0, df1, df2], ignore_index=True)

# %%
sarc_df = df_concat["sarc" == df_concat["class"]]
notsarc_df = df_concat["notsarc" == df_concat["class"]]

# %%
sarc_tokens = []
for doc in sarc_df['text']:
    tokenized_doc = nlp(doc)
    for t in tokenized_doc:
        if t.is_stop == False and t.is_punct == False:
            sarc_tokens.append(t.lower_)

# %%
notsarc_tokens = []
for doc in notsarc_df['text']:
    tokenized_doc = nlp(doc)
    for t in tokenized_doc:
        if t.is_stop == False and t.is_punct == False:
            notsarc_tokens.append(t.lower_)

# %%
sarc_token_freq = Counter(sarc_tokens)
not_sarc_token_freq = Counter(notsarc_tokens)

# %%
df_sarc_token = pd.DataFrame.from_dict(sarc_token_freq, orient='index')
df_sarc_token.columns = ['Frequencia Sarcasmo']
df_sarc_token.index.name = 'Termino'


# %%
df_not_sarc = pd.DataFrame.from_dict(not_sarc_token_freq, orient='index')
df_not_sarc.columns = ['Frequencia No Sarcasmo']
df_not_sarc.index.name = 'Termino'

# %%
BINS = pd.IntervalIndex.from_tuples([(1, 5), (6, 30), (31, 9999)],)
df = dict(zip(BINS,["bajo", "medio", "alto"]))

freq_sarc = pd.cut(df_sarc_token['Frequencia Sarcasmo'], BINS).map(df)
freq_not_sarc = pd.cut(df_not_sarc['Frequencia No Sarcasmo'], BINS).map(df)

df_1 = pd.concat([df_sarc_token, freq_sarc, df_not_sarc, freq_not_sarc], axis=1)
df_1.columns = [
    'Frecuencia_Sarcasmo',
    'Frecuencia_categorica_Sarcasmo',
    'Frecuencia_NoSarc',
    'Frecuencia_categorica_NoSarc'
]

# %%
heatmap = pd.crosstab(
    df_1.Frecuencia_categorica_Sarcasmo,
    df_1.Frecuencia_categorica_NoSarc,
    normalize = False
)
sn.heatmap(heatmap, annot=True, fmt="d", linewidths=.5, cmap="YlGnBu")
plt.title("Heatmap de frecuencias")

# %%markdown
Podemos ver en el heatmap que mientras mas frecuentes son las palabras de ambas
categorias menos coinciden entre si.

# %%markdown
# N-GRAMAS

# %%
def get_docs_bigrams(texts):
    """Get all bigrams of the corpus"""
    docs_bigrams = []
    for text in texts:
        doc = nlp(text)
        sent_bigrams = []
        for sent in doc.sents:
            sent_bigrams.append(
                [[sent[ind], sent[ind + 1]] for ind in range(len(sent)-1)]
            )
        docs_bigrams.append(sent_bigrams)
    return docs_bigrams

# %%
bigrams = get_docs_bigrams(sarc_df)
print(bigrams[0])

# %%
def get_docs_trigrams(texts):
    """Get all trigrams of the corpus"""
    docs_trigrams = []
    for text in texts:
        doc = nlp(text)
        sent_trigrams = []
        for sent in doc.sents:
            sent_trigrams.append(
                [
                    [sent[ind], sent[ind + 1], sent[ind + 2]]
                    for ind in range(len(sent)-2)
                ]
            )
        docs_trigrams.append(sent_trigrams)
    return docs_trigrams

# %%
trigrams = get_docs_trigrams(sarc_df)
print(trigrams[0])

# %%markdown
Spacy tiene una funcion para poder ver la relación que hay entre los diferentes
tokens de un documento. Utilizando esta función podemos conseguir bigramas que
aporten más contexto a la relación que tienen las dos palabras y también obviar
bigramas que sean poco relevantes, por ejemplo la puntuación, o relaciones no
definidas (que aparecen con tokens que no son palabras).

Veamos un grafico de este tipo de dependencias en una oración.

# %%
from spacy import displacy

def display_dep(doc):
    """Display token dependencies in a document"""
    displacy.render(
        doc, style="dep")

display_dep(list(nlp(sarc_df[2609]).sents)[0])

# %%
EXCLUDE_DEPS = ["punct", "ccmp", "prep", "pobj", "X", "space", "", "ROOT"]
def get_doc_dep_bigrams(texts):
    """
    Get all relevant bigrams of tokens that may not be next to each other
    """
    docs_bigrams = []
    for text in texts:
        doc = nlp(text)
        sent_trigrams = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in EXCLUDE_DEPS:
                    continue
                sent_trigrams.append([token.text, token.head.text, token.dep_])
            docs_bigrams.append(sent_trigrams)
    return docs_bigrams

# %%
doc_dep_bigrams = get_doc_dep_bigrams(sarc_df)
print(doc_dep_bigrams[0])

# %%markdown
## TF-IDF

# %%
# tf
def get_doc_tf(text):
    """Get dict of token frequency of a document"""
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
    """
    Get Inverse Document Frequency in all documents of a corpus.
    """
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
    """
    Get TF-IDF of all tokens in a document using its TF and corpus IDF.
    """
    doc_tfidf = {
        k: doc_tf[k] * docs_idf[k]
        for k in doc_tf
    }

    return doc_tfidf

# %%
def get_above_avg_tokens(docs_tfidf):
    """
    Filter tokens that have an above average TF-IDF sum.
    """
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

    return above_avg_tfidf

# %%
def get_sum_tfidf(above_avg_tfidf):
    """
    Get sum of TF-IDFs of above average tokens in each document of the corpus
    """
    tokens_tfidf_sum = {}

    for doc in above_avg_tfidf:
        for token in doc:
            curr_token_value = tokens_tfidf_sum.get(token, 0)
            tokens_tfidf_sum[token] = curr_token_value + doc[token]
    # sort important tokens
    sorted_tfidf_sum = sorted(
        tokens_tfidf_sum.items(), key=lambda x: x[1], reverse=True)

    return sorted_tfidf_sum

# %%
def lineplot_graph(tuples, label):
    """Graph a lineplot taking list of tuples (x, y)"""
    x = [x for x, y in tuples]
    y = [y for x, y in tuples]

    graph = sn.lineplot(
        x=x,
        y=y,
        label=label,
        sort=False,
    )
    x_labels_fix = graph.set_xticklabels(
        labels=x, rotation=90)

# %%
sarc_docs_idf = get_docs_idf(sarc_df)
notsarc_docs_idf = get_docs_idf(notsarc_df)

# %%
sarc_docs_tf = [get_doc_tf(doc) for doc in sarc_df]
notsarc_docs_tf = [get_doc_tf(doc) for doc in notsarc_df]

# %%
sarc_docs_tfidf = [
    get_doc_tfidf(doc_tf, sarc_docs_idf)
    for doc_tf in sarc_docs_tf
]
notsarc_docs_tfidf = [
    get_doc_tfidf(doc_tf, notsarc_docs_idf)
    for doc_tf in notsarc_docs_tf
]

# %%
ab_avg_sarc_tokens = get_above_avg_tokens(sarc_docs_tfidf)
ab_avg_notsarc_tokens = get_above_avg_tokens(notsarc_docs_tfidf)

# %%
sarc_tfidf_sum = get_sum_tfidf(ab_avg_sarc_tokens)
notsarc_tfidf_sum = get_sum_tfidf(ab_avg_notsarc_tokens)

# %%
x_sarc = [x for x,y in sarc_tfidf_sum]
y_sarc = [y for x,y in sarc_tfidf_sum]
sarc_label = "SARC TF-IDF"

x_notsarc = [x for x,y in notsarc_tfidf_sum]
y_notsarc = [y for x,y in notsarc_tfidf_sum]
notsarc_label = "NOT SARC TF-IDF"

# %%
lineplot_graph(
    sarc_tfidf_sum[:15], sarc_label,
)

# %%
lineplot_graph(
    notsarc_tfidf_sum[:15], notsarc_label,
)

# %%markdown
Como podemos ver en los graficos, la relación entre las palabras con TF-IDF más
alto de Sarcasmo y No Sarcasmo es practicamente nula. Esto es consistente
con los datos vistos en el heatmap anteriormente.

# %%markdown
## Analisis de Entidades

# %%
def get_corpus_entities(texts):
    """Get dict of entities in the corpus separated by category"""
    docs_entities = {}
    for text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            doc_ent_set = docs_entities.get(ent.label_, set())
            doc_ent_set.add(ent.text)
            docs_entities[ent.label_] = doc_ent_set
    return docs_entities

# %%
corpus_entities = get_corpus_entities(sarc_df)
list(corpus_entities.keys())

# %%
# entity tf
def get_doc_entity_tf(text):
    """Get all entities Term Frequency in a document"""
    doc = nlp(text)

    entities_count = 0
    docs_entities = {}
    doc_tf = {}

    for ent in doc.ents:
        docs_entities[ent.label_] = docs_entities.get(ent.label_, 0) + 1

    for ent in docs_entities:
        entities_count = entities_count + docs_entities.get(ent, 0)

    for e in docs_entities:
        doc_tf[e] = docs_entities[e] / entities_count

    return doc_tf

# %%
# entity idf
def get_docs_entities_idf(texts):
    """Get all entities IDF in the corpus"""
    docs_idf = {}
    doc_count = len(texts)

    for text in texts:
        doc = nlp(text)

        # important to notice this is a set (not an array)
        # and therefore entitites will only appear once
        docs_entities = {
            ent.label_ for ent in doc.ents
        }

        entities_count = len(docs_entities)

        for e in docs_entities:
            docs_idf[e] = docs_idf.get(e, 0) + 1

    for ent in docs_idf:
        docs_idf[ent] = math.log(doc_count/docs_idf[ent])

    return docs_idf

# %%
sarc_docs_entity_idf = get_docs_entities_idf(sarc_df)
notsarc_docs_entity_idf = get_docs_entities_idf(notsarc_df)

# %%
sarc_docs_entity_tf = [get_doc_entity_tf(doc) for doc in sarc_df]
notsarc_docs_entity_tf = [get_doc_entity_tf(doc) for doc in notsarc_df]

# %%
sarc_docs_entity_tfidf = [
    get_doc_tfidf(doc_tf, sarc_docs_entity_idf)
    for doc_tf in sarc_docs_entity_tf
]
notsarc_docs_entity_tfidf = [
    get_doc_tfidf(doc_tf, notsarc_docs_entity_idf)
    for doc_tf in notsarc_docs_entity_tf
]

# %%
ab_avg_sarc_entities = get_above_avg_tokens(sarc_docs_entity_tfidf)
ab_avg_notsarc_entities = get_above_avg_tokens(notsarc_docs_entity_tfidf)

# %%
sarc_entity_tfidf_sum = get_sum_tfidf(ab_avg_sarc_entities)
notsarc_entity_tfidf_sum = get_sum_tfidf(ab_avg_notsarc_entities)

# %%
x_sarc_entity = [x for x,y in sarc_entity_tfidf_sum]
y_sarc_entity = [y for x,y in sarc_entity_tfidf_sum]
sarc_label = "SARC ENTITY TF-IDF"

notsarc_label = "NOT SARC ENTITY TF-IDF"

# %%
lineplot_graph(
    sarc_entity_tfidf_sum[:15], sarc_label,
)
# %%
lineplot_graph(
    notsarc_entity_tfidf_sum[:15], notsarc_label,
)
