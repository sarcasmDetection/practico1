# %%
import math
import pandas as pd
import spacy
import numpy as np
import seaborn as sn

# %%
nlp = spacy.load('en_core_web_sm')

# %%
df = pd.read_csv("../diplodatos2020-deteccion_sarcasmo/sarcasm_v2/sarcasm_v2/GEN-sarc-notsarc.csv")

# %%
sarc_df = df["sarc" == df["class"]]["text"]
notsarc_df = df["notsarc" == df["class"]]["text"]

# %%
# tf
def get_doc_tf(text):

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
# idf
def get_docs_idf(texts):
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
# tf-idf
def get_doc_tfidf(doc_tf, docs_idf):
    doc_tfidf = {
        k: doc_tf[k] * docs_idf[k]
        for k in doc_tf
    }

    return doc_tfidf

# %%
def get_tfidf_sum(entities, docs_tfidf):
    tfidf_sum = 0

    for doc_tfidf in docs_tfidf:
        tfidf_sum += doc_tfidf.get(entitie, 0)

    return tfidf_sum

# %%
def get_above_avg_entities(docs_tfidf):
    # get the entities that have a tf-idf above average of each document
    above_avg_tfidf = []

    for tfidf in docs_tfidf:
        for entitie in tfidf:
            above_avg_entities = {}
            if tfidf[entitie] > np.average(list(tfidf.values())):
                above_avg_entities[entitie] = tfidf[entitie]
        # if no entitie is above average we wont be taking it into account
        if above_avg_entities != {}:
            above_avg_tfidf.append(above_avg_entities)

    return above_avg_tfidf

# %%
def get_sum_tfidf(above_avg_tfidf):
    entities_tfidf_sum = {}

    for doc in above_avg_tfidf:
        for entitie in doc:
            curr_entitie_value = entities_tfidf_sum.get(entitie, 0)
            entities_tfidf_sum[entitie] = curr_entitie_value + doc[entitie]
    # sort important entities
    sorted_tfidf_sum = sorted(
        entities_tfidf_sum.items(), key=lambda x: x[1], reverse=True)

    return sorted_tfidf_sum

# %%
def lineplot_graph(tuples, label):
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
ab_avg_sarc_entities = get_above_avg_entities(sarc_docs_tfidf)
ab_avg_notsarc_entities = get_above_avg_entities(notsarc_docs_tfidf)

# %%
sarc_tfidf_sum = get_sum_tfidf(ab_avg_sarc_entities)
notsarc_tfidf_sum = get_sum_tfidf(ab_avg_notsarc_entities)

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
    notsarc_tfidf_sum[:15], sarc_label,
)
