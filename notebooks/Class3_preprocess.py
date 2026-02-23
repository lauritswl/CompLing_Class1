# %% Load USEcorpus
# This corpus is structured as individual text files in a folders, so I have written a small script to read all the text files and clean them.
# The cleaned data is saved as a csv file, which we can read into a pandas dataframe.

import re # Regular expressions package for text cleaning
# !pip install pandas
import pandas as pd # Data manipulation package for working with dataframes
from pathlib import Path # Makes it easier to work with file paths and directories

# This functions uses regular expressions to extract the title and body of the text files, and then cleans the text by removing extra whitespace and HTML tags.
def parse_use_file(filepath: Path) -> dict:
    text = filepath.read_text(encoding="latin-1", errors="replace") # Fun fact, since this corpus is old, UTF-8 was not able to read the swedish charachters. 

    title = re.search(r"<title>(.*?)</title>", text, re.DOTALL)
    body  = re.search(r"</title>\s*(.*?)\s*(?:</doc>|$)", text, re.DOTALL)

    clean = lambda s: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()

    return {
        "doc_id": filepath.stem,
        "title":  clean(title.group(1)) if title else "",
        "text":   clean(body.group(1))  if body  else "",
    }

# This function takes a list of directories, finds all the text files in those directories, and applies the parse_use_file function to each file to create a dataframe.
def load_use_corpus(*dirs: str | Path) -> pd.DataFrame:
    files = [f for d in dirs for f in sorted(Path(d).glob("*.txt"))]
    return pd.DataFrame([parse_use_file(f) for f in files])

# Check working directory
df_A2 = load_use_corpus("../data/USEcorpus/a2")
df_B2 = load_use_corpus("../data/USEcorpus/b2")

# Save as CSV files for later use:
df_A2.to_csv("../data/USEcorpus/USEcorpus_a2.csv", index=False)
df_B2.to_csv("../data/USEcorpus/USEcorpus_b2.csv", index=False)

# Explode dataframe by text column, so that each row contains one sentence instead of one document. 
# This will make it easier to perform POS tagging on individual sentences.
df_A2_sentence = df_A2.assign(text=df_A2.text.str.split(".")).explode("text")
df_B2_sentence = df_B2.assign(text=df_B2.text.str.split(".")).explode("text")

# Save sentence-level dataframes as CSV files for later use:
df_A2_sentence.to_csv("../data/USEcorpus/USEcorpus_a2_sentences.csv", index=False)
df_B2_sentence.to_csv("../data/USEcorpus/USEcorpus_b2_sentences.csv", index=False)

# %% Preprocess of POS for CEFR Data
# Read CSV
cefr_df = pd.read_csv("../data/L2_Learner_Data.csv")
# Filter cefr_level == A1
cefr_df = cefr_df[cefr_df["cefr_level"] == "A1"]
import spacy
nlp = spacy.load("en_core_web_sm")
texts = cefr_df["text"].tolist()
doc_ids = cefr_df["title"].tolist()
docs = list(nlp.pipe(texts))

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
                "pos":      token.pos_,   # Add the part-of-speech tag of the token to the row dictionary  
                "tag":      token.tag_,   # Add the detailed part-of-speech tag of the token to the row dictionary
                "morph":    token.morph,  # Add the morphological features of the token to the row dictionary
                }
            rows.append(row) # We append (add) the row to the rows list
# We create a new dataframe from the rows list, which contains all the extracted information for each token in the corpus.
CEFR_A1_POS = pd.DataFrame(rows)

morph_expanded = CEFR_A1_POS["morph"].apply(lambda m: m.to_dict()).apply(pd.Series)
CEFR_A1_POS = pd.concat([CEFR_A1_POS, morph_expanded], axis=1).drop(columns="morph")
CEFR_A1_POS.to_csv("../data/CEFR_A1_POS.csv", index=False)
