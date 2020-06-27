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
## Analisis de palabras más comunes.

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
sarc_token_freq.plot(30, cumulative=False)

# %%
not_sarc_token_freq.plot(30, cumulative=False)

# %%
compare_freq(
    sarc_token_freq, not_sarc_token_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_token_freq, sarc_token_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %%
not_sarc_token_freq['!']
sarc_token_freq['!']
len(not_sarc_token_freq)
len(sarc_token_freq)

# %% markdown
### Analisis con lematización

# %%
sarc_lemm_tokens = get_lemm_tokens(sarc_token_list)
not_sarc_lemm_tokens = get_lemm_tokens(not_sarc_token_list)

# %%
sarc_lemm_freq = get_token_freq(lower_sarc_tokens)
not_sarc_lemm_freq = get_token_freq(lower_not_sarc_tokens)

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

# %%
### Analisis con lematización y sin Stopwords

# %%
sarc_lemm_nsw_tokens = rm_stop_words(sarc_lemm_tokens)
not_sarc_lemm_nsw_tokens = rm_stop_words(not_sarc_lemm_tokens)

# %%
sarc_lemm_nsw_freq = get_token_freq(sarc_lemm_nsw_tokens)
not_sarc_lemm_nsw_freq = get_token_freq(not_sarc_lemm_nsw_tokens)

# %%
compare_freq(
    sarc_lemm_nsw_freq, not_sarc_lemm_nsw_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_lemm_nsw_freq, sarc_lemm_nsw_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %% markdown
### Analisis del Uso de Mayusculas en Sarcasmo y No Sarcasmo

# %%
# the "I" token was really messing the graphs
sarc_upper_tokens = [w for w in sarc_token_list if w.isupper() and w != "I"]
not_sarc_upper_tokens = [
    w for w in not_sarc_token_list if w.isupper() and w != "I"]

# %%
"{} upper tokens from {} sarcastic ones".format(
    len(sarc_upper_tokens),
    len(sarc_token_list),
)

# %%
"while there are {} upper tokens from {} non-sarcastic ones".format(
    len(not_sarc_upper_tokens),
    len(not_sarc_token_list)
)

# %%
sarc_upper_freq = get_token_freq(sarc_upper_tokens)
not_sarc_upper_freq = get_token_freq(not_sarc_upper_tokens)

# %%
compare_freq(
    sarc_upper_freq, not_sarc_upper_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_upper_freq, sarc_upper_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %%
sarc_upper_texts = [w.isupper() for w in sarc_df["text"]]
not_sarc_upper_texts = [w.isupper() for w in not_sarc_df["text"]]

# %%
len(sarc_upper_tokens)/len(sarc_token_list)
len(not_sarc_upper_tokens)/len(not_sarc_token_list)

# %% markdown
### Analisis de tipo de palabras

# %%
get_pos_tag = lambda text: nltk.pos_tag(text)

# %%
is_noun = lambda tag: tag == 'NN'
is_adjetive = lambda tag: tag == 'JJ'
is_adverb = lambda tag: tag == 'RB'

# %%
sarc_tagged_tokens = get_pos_tag(sarc_token_list)
not_sarc_tagged_tokens = get_pos_tag(not_sarc_token_list)

# %% markdown
#### Sustantivos

# %%
sarc_nouns = [n for n, t in sarc_tagged_tokens if is_noun(t)]
not_sarc_nouns = [n for n, t in not_sarc_tagged_tokens if is_noun(t)]

# %%
sarc_nouns_freq = get_token_freq(sarc_nouns)
not_sarc_nouns_freq = get_token_freq(not_sarc_nouns)

# %%
compare_freq(
    sarc_nouns_freq, not_sarc_nouns_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_nouns_freq, sarc_nouns_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %%

# %% markdown
#### Adjetivos

# %%
sarc_adj = [n for n, t in sarc_tagged_tokens if is_adjetive(t)]
not_sarc_adj = [n for n, t in not_sarc_tagged_tokens if is_adjetive(t)]

# %%
sarc_adj_freq = get_token_freq(sarc_adj)
not_sarc_adj_freq = get_token_freq(not_sarc_adj)

# %%
compare_freq(
    sarc_adj_freq, not_sarc_adj_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_adj_freq, sarc_adj_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %% markdown
#### Adverbs

# %%
sarc_adv = [n for n, t in sarc_tagged_tokens if is_adverb(t)]
not_sarc_adv = [n for n, t in not_sarc_tagged_tokens if is_adverb(t)]

# %%
sarc_adv_freq = get_token_freq(sarc_adv)
not_sarc_adv_freq = get_token_freq(not_sarc_adv)

# %%
compare_freq(
    sarc_adv_freq, not_sarc_adv_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_adv_freq, sarc_adv_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)
