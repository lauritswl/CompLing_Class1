# Assignment for Class 1, Computational Linguistics, Aarhus University, Spring 2026
# If you manage to setup a working Python environment quickly, you can start working on this file.
# The task includes testing our very basic hypothesis, that the better you get at a new language, the longer sentences you can produce.
# For this I have collected a small sample of sentences from a open source corpus.

# The sentences are stored in a file called "L2_Learner_Data.csv", which is a comma-separated values file. 
# This "dataframe" contains two important columns: sentences and cefr_level
# Start installing Pandas in your environment, and load the data into a dataframe.

# ============ TASK 1 ===========
# Make sure you can run the file:
print("Hello World!")

# ============ TASK 2 ===========
# Load the data from "data/L2_Learner_Data.csv" into a Pandas dataframe, with the pd.read_csv() function.
import pandas as pd

# ============ TASK 3 ===========
# Filter your dataframe to only include the columns "sentences" and "cefr_level". You can do this by selecting the columns you want from the dataframe.


# ============ TASK 4 ===========
# Write a function that takes a sentence as input and returns the number of words in that sentence. 
# You can use the split() method to split the sentence into words, and then use the len() function to count the number of words.

my_sentence = "This is a sample sentence for testing."
def count_words(sentence):
    # Write your function here. You should store the number of words in the num_words variable


    # =========== Make sure that script can run before defining num_words ======
    try:
        num_words
    except NameError:
        num_words = 0
    return num_words

# Test the function with the sample sentence
num_words = count_words(my_sentence)
print(f"The number of words in the sentence is: {num_words}")

# ============ TASK 5 ===========
# Apply the count_words function to the "sentences" column of your dataframe, and create a new column called "num_words" to store the results. 
# You can use the apply() method, or practice by using a for loop to iterate over the sentences and apply the function to each sentence individually.


# ============ TASK 6 ===========
# Now you have a new column with the number of words in each sentence. You can use this column to analyze the relationship between sentence length and CEFR level. 
# For example, you can group the data by CEFR level and calculate the average number of words for each level using the groupby() method.


# ============ TASK 7 ===========
# If you make it this far, you can try to find the longest word in each sentence, and create a new column called "longest_word_length" to store the length of the longest word.
# You can then text if proficiency also results in longer words.

