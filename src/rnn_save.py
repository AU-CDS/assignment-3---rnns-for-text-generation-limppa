# data processing tools
import os
import string 
import pandas as pd
import numpy as np
np.random.seed(42)

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential

# surpress warnings to reduce unnecessary output
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#### HELPER FUNCTIONS

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

def get_sequence_of_tokens(tokenizer, corpus):
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

def generate_padded_sequences(input_sequences):
    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # make every sequence the length of the longest on
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 
                        10, 
                        input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model

def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], 
                                    maxlen=max_sequence_len-1, 
                                    padding='pre')
        predicted = np.argmax(model.predict(token_list),
                                            axis=1)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()


#### LOAD DATA

# Determine the absolute path to the 'news_data' directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "in", "news_data/")

# append only the headlines of the articles
all_headlines = []
for filename in os.listdir(data_dir):
    if 'Articles' in filename:
        article_df = pd.read_csv(data_dir + filename)
        all_headlines.extend(list(article_df["headline"].values))

corpus = [clean_text(x) for x in all_headlines]

#### TOKENIZE

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# use previously defined function to turn text into sequence of tokens
inp_sequences = get_sequence_of_tokens(tokenizer, corpus)

# pad input sequences so they are all the same length
predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

#### CREATE MODEL

# use previously defined function to initialize model
# inputs are length of sequences and total size of the vocab
model = create_model(max_sequence_len, total_words)

# train model (this step takes up most of the time)
history = model.fit(predictors,
                    label,
                    epochs=20,  # this can be adjusted as needed
                    batch_size=128,
                    verbose=1)

#### SAVE MODEL
# Create the "models" directory if it doesn't exist
os.makedirs("model", exist_ok=True)
# save model into the "models" folder
tf.keras.models.save_model(
    model, os.path.join("model"), overwrite=True, save_format="tf"
)