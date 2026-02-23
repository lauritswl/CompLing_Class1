# %% [markdown]
# # Computational Linguistics Class 3: Spacy and Part-of-Speech Tagging
# In this class, we will work with the `spacy` library to perform Part-of-Speech tagging on a swedish L2-english Learner corpus. The data in its original structure can be found in the USEcorpus folder in the data directory. 
# 
# I have written a small script to read the text files, clean them, and save them as csv files, which we can read into pandas dataframes. You can look at the script "CompLing_F26/notebooks/Class3_preprocess.py" if you are interested in how the data was processed, but we will not go through it in detail in class.

# %% [markdown]
# ## ========== TASK 1 ===========
# First we will look at how to use the spacy library to perform Part-of-Speech tagging on a sentence. 
# You don't need to write any new code for this task, but try to understand what the code does and what the output is.
# Make sure that your spacy library is installed and that you have downloaded the english language model.

# %%
# You can install spacy and the language model using the following commands (or download in the terminal):
# !pip install spacy
# !python -m spacy download en_core_web_sm

# %%
# First we import the spacy library and load the english language model. 
# This will allow us to use spacy's functions for tokenization, lemmatization, and POS tagging.
import spacy
nlp = spacy.load("en_core_web_sm") # This is our natrual language processing object.

# You can use the nlp object to process text and get a spacy Doc object, which contains all the information about the text, including tokens, lemmas, POS tags, etc.
doc = nlp("The sentence, that i have written, is long.")

from spacy import displacy
displacy.render(doc, style="dep", jupyter=True)


# %%
# If you print the doc object, you will see that it contains the original text, as well as a list of tokens. 
# Each token is a spacy Token object, which contains information about the token, such as its text, lemma, POS tag, etc.
# You can see an overview of the attributes of the Token object in the spacy documentation: https://spacy.io/api/token#attributes
print(doc) # The original text of the doc
print(doc[6]) # The 7th token in the doc 
print(doc[6].lemma_) # The lemma of the 7th token

# %%
# The sentence object "doc" is iterable, like a list, which means that we can loop through the tokens in the doc and access their attributes.
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.morph, token.is_alpha, token.is_stop)

# %% [markdown]
# ### Explanation from Spacy documentation:
# Text: The original word text.
# Lemma: The base form of the word.
# POS: The simple UPOS part-of-speech tag.
# Tag: The detailed part-of-speech tag.
# Dep: Syntactic dependency, i.e. the relation between tokens.
# Shape: The word shape – capitalization, punctuation, digits.
# Morph: Morphological features of the token.
# is alpha: Is the token an alpha character?
# is stop: Is the token part of a stop list, i.e. the most common words of the language?

# %%
# This is quite messy, so we can also create a pandas dataframe to display the information in a more structured way.
import pandas as pd
df = pd.DataFrame(
    [(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
      token.shape_, token.morph, token.is_alpha, token.is_stop) for token in doc],
    columns=["text", "lemma", "pos", "tag", "dep", "shape", "morph", "is_alpha", "is_stop"]
)
print(df)

# %%
# If I'm in doubt about what the different attributes of the token mean, I can always check the spacy documentation for more information.
# Or this resource for more indepth linguistic explanations: https://universaldependencies.org/u/feat/index.html
# Or use the spacy.explain() function to get a short explanation of the attribute. For example, if I want to know what the "SCONJ" tag means, I can do:
spacy.explain("SCONJ")

# %% [markdown]
# ## ========== TASK 2 ===========
# The second task consists of loading the USEcorpus dataframe and applying the spacy NLP pipeline on the text.
# The csv files are located in the data/USEcorpus folder, and are called USEcorpus_a2_sentences.csv and USEcorpus_b2_sentences.csv. 
# Read the csv files into pandas dataframes, and display the first few rows to check if they loaded correctly. 
# 
# A lot of the code will be supplied, as spacy can be a bit confusing if you don't have a lot of experience coding.

# %%
# Step 1: Read the csv files into pandas dataframes
import pandas as pd
df_a2 = pd.read_csv("../data/USEcorpus/USEcorpus_a2.csv")
df_a2 = df_a2.dropna(subset=["text"]) # Remove rows where the "text" column is NaN (missing values)
df_b2 = pd.read_csv("../data/USEcorpus/USEcorpus_b2.csv")
df_b2 = df_b2.dropna(subset=["text"])

# Print example sentence from the corpus:
df_b2.head()

# %%
# Step 2: Run spaCy on each essay
# nlp.pipe() is faster than calling nlp() one by one
texts = df_a2["text"].tolist()
doc_ids = df_a2["doc_id"].tolist()

docs = list(nlp.pipe(texts))


# We now have a a list of spacy Doc objects and a list of doc_ids refering to the original sentences in the dataframe. 
print(f"Number of documents: {len(docs)}")
print(f"First doc object: {docs[0][:10]}...")
print(f"Number of doc_ids: {len(doc_ids)}")
print(f"First doc_id: {doc_ids[0]}")

# %% [markdown]
# **For this chunk you need to add the pos, tag, and morph features to the loop:**

# %%
# Step 3: Extract token information from each document 
# We make an empty rows list to store the token information
rows = []


# We zip together the doc_ids and docs lists, and loop through each element:
for doc_id, doc in zip(doc_ids, docs):
    # We loop throug each sentence with "for i in doc.sents", with enumerate to get the sentence index. 
    for sentence_idx, sentence in enumerate(doc.sents):
        # We loop through each token in the sentence and extract the relevant information.
        for token in sentence:
            row = {
                "doc_id":   doc_id,
                "sentence": sentence_idx,
                "token":    token.text,
                "lemma":    token.lemma_,
                "pos":     # Add the part-of-speech tag of the token to the row dictionary  
                "tag":     # Add the detailed part-of-speech tag of the token to the row dictionary
                "morph":   # Add the morphological features of the token to the row dictionary
                }
            rows.append(row) # We append (add) the row to the rows list

print(f"Total tokens extracted: {len(rows)}")

# %%
# ── Step 4: Turn the list of rows into a dataframe ──────────────────────────
pos_df = pd.DataFrame(rows)
print(pos_df.head(10))

# %% [markdown]
# ## ========== TASK 3 ===========
# Now redo the analysis for the B2 sentences, and save the results in a new dataframe called pos_df_b2.
# 
# **Step 1**: Run spaCy on each essay
# 
# **Step 2**: Extract token information from each document
# 
# **Step 3**: Turn the list of rows into a dataframe
# 
# **Step 4**: Turn rows into a dataframe called *pos_df_b2*

# %%


# %% [markdown]
# ## ========== TASK 4 ===========
# We will now use the POS tagged data to do a frequency analysis of the words used in A1 and A2 level sentences.
# In Corpus Linguistics it is important to normalize your frequencies. That is, if we have a corpus of 100.000 tokens and a corpus of 1000 tokens, and both have a verb frequency of 10%. Then by using raw token counts, the first corpus would have a value of 10000, while the second corpus would have a value of 100, even though their frequency is the same. Normalised frequencies are usually given per thousand words or per million words. We will use frequency per thousand words for our analysis. Read more at: http://corpora.lancs.ac.uk/clmtp/2-stat.php

# %%
# Hint: You can use the value_counts() method to get the count of each part-of-speech tag in the "pos" column of the dataframe. 


# Find the frequency of each POS tag in the A2 and B2 dataframes (Count of each POS / Total tokens):


# Multiply the relative frequency of each POS tag by 1000 to get the frequency per 1000 tokens.


# Find the difference between the relative frequencies of each POS tag in the A2 and B2 dataframes.


# This is only a one year difference, so we don't expect to see huge differences in the POS tag frequencies, but we might see some interesting patterns.
# Print the relative frequencies of each POS tag in the A2 and B2 dataframes, and the difference between them.


# %%
# Extra challenge: Load the A1_CEFR level sentences, from the last classes, that I have POS-tagged for you. And compare to the USEcorpus, we see much larger effect here.
# CEFR_A1_POS = pd.read_csv("../data/CEFR_A1_POS.csv")

# %% [markdown]
# ## ========== TASK 5 ===========
# We will now move onto morphology and lemmatization. And see if we can find morphological patterns.
# You can see the documentation for morphological features in spacy here: https://spacy.io/usage/linguistic-features#morphology
# 
# First we will expand our morph column into separate columns for each morphological feature, so that we can analyze them more easily. This is the point of tidy data.
# The morph column contains a spacy MorphAnalysis object, which is a dictionary-like object that contains the morphological features of the token.
# I use a lambda function to convert the MorphAnalysis object to a dictionary, and then apply pd.Series to expand the dictionary into separate columns.
# 
# **Regular function**:
# ```python
# def convert_to_dict(m):
#     return m.to_dict()
# ```
# 
# **Lambda equivalent**:
# ```python
# lambda m: m.to_dict()
# ```
# 
# I then turn it into a pandas series i can add to my dataframe.

# %%
# Convert morph to dict and expand into columns:
morph_expanded = pos_df_a2["morph"].apply(lambda m: m.to_dict()).apply(pd.Series)

# Join back to original dataframe
pos_df_a2 = pd.concat([pos_df_a2, morph_expanded], axis=1).drop(columns="morph")

# Do the same for the B2 dataframe:


# %%
# Find difference in VerbForm between A2 and B2 dataframes:
verbform_count_a2 = pos_df_a2[pos_df_a2["pos"] == "VERB"]["VerbForm"].value_counts() # Read this as "Filter the A2 dataframe to only include rows where the "pos" column is "VERB", then get the "VerbForm" column and count the frequency of each unique value."
verbform_count_b2 = pos_df_b2[pos_df_b2["pos"] == "VERB"]["VerbForm"].value_counts()

# Normalize the frequencies by dividing by the total number of verbs and multiplying by 1000 to get the frequency per 1000 verbs.


# Does this trend hold according to your linguistic intuition? Do you think this is a result of the one year difference or another factor like 'arguement vs counter-arguement essay type'?
# a2. Argumentation. Students argue for or against a statement concerning a topical issue. Formal style. Written in mid-February or early October.
# b2. Argumentation. Students present counter-arguments to views expressed in articles or letters to the editor. Similar in approach and tone to essay a1.

# %%
# Extra challenge: Load the CEFR A1 level sentences to see if the same trend is between that and USEcorpus A2 level sentences.
CEFR_A1_POS = pd.read_csv("../data/CEFR_A1_POS.csv")

# %% [markdown]
# ## ========== EXTRA: TASK 6 ===========
# We will do a t-test, comparing the relative frecuencies of VerbForm=Part in the A2 and B2 dataframes, to see if the difference is statistically significant.
# Our hypothesis is that the extra year of practice with english, results in a higher frequency of complex noun phrases, which often contain participle verbs. 
# So we expect to see a higher relative frequency of VerbForm=Part in the B2 dataframe compared to the A2 dataframe.

# %%
# Step 1: Count VerbForm=Part, Verbs, and total tokens for each document in the A2 and B2 dataframes.
# To get you started, here is how you count the total number of tokens per document:
total_tokens_a2 = pos_df_a2.groupby("doc_id").size()
total_tokens_b2 = pos_df_b2.groupby("doc_id").size()

# Hint: To count only verbs, you need to filter the dataframe before groupby.
# You can filter a dataframe like this: pos_df[pos_df["column"] == "value"].groupby("doc_id").size()
total_verbs_a2 = # YOUR CODE HERE
total_verbs_b2 = # YOUR CODE HERE

# Hint: To count only participle verbs, you need to filter on TWO conditions.
# You can combine two conditions like this: df[(df["col1"] == "val1") & (df["col2"] == "val2")].groupby("doc_id").size()
part_verbs_a2 = # YOUR CODE HERE
part_verbs_b2 = # YOUR CODE HERE

# Step 2: Normalize — divide participle count by total verbs to get a rate per document.
# Hint: Use .fillna(0) to handle documents that have no verbs at all.
rate_a2 = # YOUR CODE HERE
rate_b2 = # YOUR CODE HERE

# %%

# Then perform a t-test to see if the difference in the frequency of the different verb forms is statistically significant between the A2 and B2 dataframes.
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(rate_a2, rate_b2, equal_var=False)
print(f"T-statistic: {t_stat}, P-value: {p_value}")


