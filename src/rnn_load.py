# data processing tools
import os
import sys
import string
import pandas as pd
import numpy as np

# keras module for building LSTM
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

#### HELPER FUNCTIONS

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii", "ignore")
    return txt

def get_sequence_of_tokens(tokenizer, corpus):
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    input_sequences = np.array(input_sequences)  # Convert to numpy array
    return input_sequences

def generate_padded_sequences(input_sequences):
    max_sequence_len = max(len(seq) for seq in input_sequences)
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list],
                                   maxlen=max_sequence_len - 1,
                                   padding="pre")
        predicted = np.argmax(model.predict(token_list), axis=1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text.title()


#### LOAD DATA

# Determine the absolute path to the 'news_data' directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "in", "news_data/")

# append only the headlines of the articles
all_headlines = []
for filename in os.listdir(data_dir):
    if "Articles" in filename:
        article_df = pd.read_csv(data_dir + filename)
        all_headlines.extend(list(article_df["headline"].values))

corpus = [clean_text(x) for x in all_headlines]

#### TOKENIZE

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# use previously defined function to turn text into sequence of tokens
inp_sequences = get_sequence_of_tokens(tokenizer, corpus)

# load the saved model
model = load_model("model", compile=False)

# Get the maximum sequence length
max_sequence_len = max(len(seq) for seq in inp_sequences)

#### GENERATE TEXT

# Get the seed text and amount of words from the command-line argument
seed_text = sys.argv[1]
next_words = int(sys.argv[2])

# use previously defined function to generate text
generated_text = generate_text(seed_text, next_words, model, max_sequence_len, tokenizer)
print(generated_text)