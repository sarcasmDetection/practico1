# %%
import spacy
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from collections import Counter


# %%
nlp = spacy.load('en_core_web_sm')

# %%
df0 = pd.read_csv("./sarcasm_v2/GEN-sarc-notsarc.csv")
df1 = pd.read_csv("./sarcasm_v2/HYP-sarc-notsarc.csv")
df2 = pd.read_csv("./sarcasm_v2/RQ-sarc-notsarc.csv")

df = pd.concat([df0, df1, df2], ignore_index=True)

# %%
get_token_freq = lambda tokens: Counter(tokens)

# %%
sarc_df = df["sarc" == df["class"]]
notsarc_df = df["notsarc" == df["class"]]

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
sarc_token_freq = get_token_freq(sarc_tokens)
not_sarc_token_freq = get_token_freq(notsarc_tokens)

# %%
df_sarc_token = pd.DataFrame.from_dict(sarc_token_freq, orient='index')
df_sarc_token.columns = ['Frequencia Sarcasmo']
df_sarc_token.index.name = 'Termino'


# %%
df_not_sarc = pd.DataFrame.from_dict(not_sarc_token_freq, orient='index')
df_not_sarc.columns = ['Frequencia No Sarcasmo']
df_not_sarc.index.name = 'Termino'


# %%
BINS = pd.IntervalIndex.from_tuples([(1, 20), (21, 40), (41, 6000)],)
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

df_1


# %%
heatmap = pd.crosstab(
    df_1.Frecuencia_categorica_Sarcasmo,
    df_1.Frecuencia_categorica_NoSarc,
    normalize = False
)
sn.heatmap(heatmap, annot=True, fmt="d", linewidths=.5, cmap="YlGnBu")
plt.title("Heatmap de frecuencias")
