# Computational Linguistics Class 2: Pandas and Data Analysis
# In this class, we will work with the Pandas library to analyze the L2 Learner Data. 
# We will read the data into a Pandas DataFrame, and then analyze do a frequency analysis of words used in A1 and A2 level sentences.
# The main goal is to get a feeling of how dataframes work! Feel free to experiment with the data and try out different analyses.

# %%
# Read L2 Learner Data into a Pandas DataFrame, and display the first few rows to check if it loaded correctly.
import pandas as pd
data = pd.read_csv("../data/L2_Learner_Data.csv")
data.head()

# %%
# =========== TASK 1 ===========
# Try figure out what the following code does, and what the output is. 

# data.head()
# data.tail()
# data.shape
# data.columns
# data.info() 
# data['cefr_level'].value_counts()


# =========== TASK 2 ===========
# Now try to get a feeling of the dataframe works by accessing rows and columns. 

# data.iloc[0]
# data.iloc[0:5]
# data.iloc[0]['sentences']
# data.iloc[0, 7]
# data.iloc[0:5, -2:] # The second last and all columns thereafter
# data.iloc[0:5, [0, -2,-1]]
# data.iloc[0:5][["title", "sentences", "cefr_level"]]


# %%
# =========== TASK 3 ===========
# Now create save a cleaned version of the dataframe with only the columns "title", "sentences", "source_name" and "cefr_level".
# It is a good idea to create a new variable for this, so you can always go back to the original dataframe if you need to.
# Remember to check that your new dataframe looks correct by using the head() method to display the first few rows.

cleaned_data = data[["title", "sentences", "source_name", "cefr_level"]]
cleaned_data.tail()  # Display the first few rows of the cleaned dataframe to check if

# %%
# =========== TASK 4 ===========
# Now create two new dataframes, one for A1 level sentences and one for A2 level sentences. You can do this by filtering the original dataframe based on the "cefr_level" column.
# To filter the dataframe, you can use boolean indexing. 
# For example, to create a dataframe with all sentences from the icle500 dataset, you can do:
icle500_data = cleaned_data[cleaned_data["source_name"] == "icle500"]
# cleaned_data["source_name"] == "icle500" is a boolean filter that returns True for rows where the source_name is "icle500" and False for all other rows.

# Now create the A1 and A2 dataframes using similar boolean filters based on the "cefr_level" column. Call the A1 dataframe "a1_data" and the A2 dataframe "a2_data".
a1_data = cleaned_data[cleaned_data["cefr_level"] == "A1"]
a2_data = cleaned_data[cleaned_data["cefr_level"] == "A2"]

# %%
# =========== TASK 5 ===========
# Now you have two dataframes, one for A1 level sentences and one for A2 level sentences.
a1_data["words"] = a1_data["sentences"].str.lower().str.split()  # Split the sentences into lists of words
a1_data.head()  # Display the first few rows of the exploded A1

# %%
# Now use the explode() method to create a new dataframe where each row contains a single word from the sentences.
a1_exploded = a1_data.explode("words")
a1_exploded.head()  # Display the first few rows of the exploded A1 dataframe to check if it worked correctly

# %%
# Now create an a2_exploded dataframe in the same way.
a2_data["words"] = a2_data["sentences"].str.lower().str.split()  # Split the sentences into lists of words
a2_exploded = a2_data.explode("words")
a2_exploded.head()  # Display the first few rows of the exploded A2

# =========== TASK 6 ===========
# Now you have two dataframes, one for A1 level words and one for A2 levels words. You can now do a frequency analysis of the words used in A1 and A2 level sentences.
# Hint: You can use the value_counts() method to get the frequency of each word in the "words" column of each dataframe.
a1_word_freq = a1_exploded["words"].value_counts()
a2_word_freq = a2_exploded["words"].value_counts()



# %%
# =========== TASK 7 ===========
# Get the top 100 most common words in A1 and A2 level sentences, and use sets to find the words that are common to both levels, and the words that are unique to each level.
# Hint: By using set(my_list) you can create a set from a list, and then use set operations like intersection() and difference() to find common and unique elements between two sets.
a1_top100 = set(a1_word_freq.head(100).index)
a2_top100 = set(a2_word_freq.head(100).index)
common_words = a1_top100.intersection(a2_top100)  # Words that are common to both levels
a1_unique = a1_top100.difference(a2_top100)  # Words that are unique to A1 level
a2_unique = a2_top100.difference(a1_top100)  # Words that are unique to A2 level

# %%
# =========== TASK 8 ===========
# This one is hard, but if you make it this far, you can try to find to make a dataframe with the all the words that appear in top 100 of either A1 or A2 level sentences, and then add columns for the frequency of each word in A1 and A2 level sentences.
# Hint: You can use the union() method to get the set of all unique words that appear in the top 100 of either level.
all_top_words = a1_top100.union(a2_top100)  # Set of all unique words in the top 100 of either level
top_words_df = pd.DataFrame(all_top_words, columns=["word"])  # Create a new dataframe with the unique words
top_words_df["a1_freq"] = top_words_df["word"].map(a1_word_freq)  # Map the A1 frequencies to the new dataframe
top_words_df["a2_freq"] = top_words_df["word"].map(a2_word_freq)  # Map the A2 frequencies to the new dataframe
top_words_df.head()  # Display the first few rows of the new dataframe to check if it looks correct



# Can you get filter the words in this dataframe to only include the words that are common to both levels?
common_words_df = top_words_df[top_words_df["word"].isin(common_words)]  # Filter the dataframe to only include common words
# What words have the biggest difference in frequency between A1 and A2 level sentences? You can create a new column that calculates the absolute difference in frequency between the two levels, and then sort the dataframe by this new column to find the words with the biggest difference.
top_words_df["freq_diff"] = abs(top_words_df["a1_freq"] - top_words_df["a2_freq"])  # Calculate the absolute difference in frequency between A1 and A2 levels
top_words_df_sorted = top_words_df.sort_values(by="freq_diff", ascending=False)  # Sort the dataframe by the frequency difference in descending order
top_words_df_sorted.head()  # Display the first few rows of the sorted dataframe to see the words with the biggest difference in

# %%
