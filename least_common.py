# %% markdown
# Analisis y Visualización sobre DFs de Sarcasmo y No Sarcasmo

# %%
import nltk
import numpy as np
import pandas as pd
import seaborn as sn

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# %% markdown
Voy a unir todos los dataframes para trabajar directamente con
todos los datos con los que disponemos

# %%
df0 = pd.read_csv("./sarcasm_v2/GEN-sarc-notsarc.csv")
df1 = pd.read_csv("./sarcasm_v2/HYP-sarc-notsarc.csv")
df2 = pd.read_csv("./sarcasm_v2/RQ-sarc-notsarc.csv")

df = pd.concat([df0, df1, df2], ignore_index=True)

# %% markdown
## Analisis de palabras menos comunes.

Primero definamos funciones que nos van a servir

# %%
flatten_list = lambda nested_list: [
    el for sublist in nested_list for el in sublist]

lemmatizer = WordNetLemmatizer()
get_lemm_tokens = lambda tokens: [lemmatizer.lemmatize(t) for t in tokens]

get_token_freq = lambda tokens: nltk.FreqDist(tokens)

stop_words = set(stopwords.words('english'))
rm_stop_words = lambda tokens: [t for t in tokens if t not in stop_words]

# %%
def compare_freq(least_common_freq, cmp_freq,
                 lc_label, cmp_label,
                 lc_color, cmp_color):
    """
    This function compares the frequency of the least common tokens
    of `least_common_freq` with the frequency they have in `cmp_freq`.
    """
    # least_common = least_common_freq.least_common()[-10:]

    least_common_words = [x for x in least_common_freq]
    least_common_freqs = [least_common_freq[x]/len(least_common_freq) for x in least_common_freq]

    cmp_freq_in_lc = [cmp_freq[x]/len(cmp_freq) for x in least_common_freq]

    least_common = sn.lineplot(
        x=least_common_words,
        y=cmp_freq_in_lc,
        label=cmp_label,
        sort=False,
        color=cmp_color
    )
    not_sarc_gr = sn.lineplot(
        x=least_common_words,
        y=least_common_freqs,
        label=lc_label,
        sort=False,
        color=lc_color
    )
    rot_lab_ns = least_common.set_xticklabels(
        labels=least_common_words, rotation=90)

# %% markdown
Definamos variables que nos van a servir tambien

# %%
sarc_df = df["sarc" == df["class"]]
not_sarc_df = df["notsarc" == df["class"]]

# %%
sarc_token_texts = [nltk.word_tokenize(snts) for snts in sarc_df['text']]
not_sarc_token_texts = [
    nltk.word_tokenize(snts) for snts in not_sarc_df['text']]

# %%
sarc_token_list = flatten_list(sarc_token_texts)
not_sarc_token_list = flatten_list(not_sarc_token_texts)

# %%
lower_sarc_tokens = [token.lower() for token in sarc_token_list]
lower_not_sarc_tokens = [token.lower() for token in not_sarc_token_list]

# %%
sarc_token_freq = get_token_freq(lower_sarc_tokens)
not_sarc_token_freq = get_token_freq(lower_not_sarc_tokens)

# %% markdown
Veamos los primeros analisis

# %%
sarc_mc = sarc_token_freq.most_common()
sarc_mc, sarc_in, sarc_lc = np.array_split(sarc_mc, 3)
sarc_lc = {x:int(y) for x, y in  sarc_lc}

# %%
not_sarc_mc = sarc_token_freq.most_common()
not_sarc_mc, not_sarc_in, not_sarc_lc = np.array_split(not_sarc_mc, 3)
not_sarc_lc = {x:int(y) for x, y in  not_sarc_lc}
