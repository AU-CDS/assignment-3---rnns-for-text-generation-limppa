# Assignment 3 - Language Modelling and Text Generation using RNNs

## Contribution
The code in this assignment was developed collaboratively with the help of other students and materials used in class.

## Description of Assignment
For this assignment, I wrote code that trains a text generation model on the headlines of articles by The New York Times from 2017-2018. The goal is to generate new text that follows the same style and structure as the input headlines.

## Data
The dataset used in this assignment is called "New York Times Comments." It contains comments on articles published in The New York Times, as well as the headlines of those articles. The dataset can be found on Kaggle [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

## Methods
The code consists of two scripts:

### rnn_save.py
This script contains helper functions and the main steps for training the text generation model. Here is a summary of its key functionalities:
1. **Data Cleaning**: The `clean_text` function removes punctuation and converts text to lowercase.
2. **Tokenization**: The `get_sequence_of_tokens` function tokenizes the cleaned text into sequences of tokens using a tokenizer.
3. **Padding Sequences**: The `generate_padded_sequences` function pads the input sequences to make them of equal length for model training.
4. **Model Creation**: The `create_model` function initializes an LSTM-based model for text generation.
5. **Data Loading**: The script loads the headlines of the articles from the dataset, cleans the text, and prepares the data for training.
6. **Model Training**: The model is trained on the prepared data using the defined architecture and parameters.
7. **Model Saving**: The trained model is saved in the "model" directory.

### rnn_load.py
This script is used to load the saved model and generate text based on a seed text. Here is a summary of its functionalities:

1. **Data Loading**: The script loads the headlines of the articles from the dataset and prepares the data for text generation.
2. **Model Loading**: The saved model is loaded from the "model" directory.
3. **Text Generation**: The `generate_text` function generates new text given a seed text and the number of words to generate.
4. **Text Output**: The generated text is printed to the console.

## Usage and Reproducibility
You will first need to download the dataset from [here](https://www.kaggle.com/datasets/aashita/nyt-comments) and make sure the .csv files exist inside a folder called `news_data`. That folder should then be placed inside the folder called `in`.

*Note: this code was run on uCloud's Coder Python 1.76.1. Depending on the device you are using, your terminal commands may need to be slightly modified.*

To run the code, first install the requirements by running the following command in terminal: `pip install -r requirements.txt`

Then train save the RNN model by navigating to the root folder and running this command: `python src/rnn_save.py`

Finally, you can load the model and have it generate continuation of any text you input. For example, generate a text that starts with "China" and continues for 6 words by running this command: `python src/rnn_load.py "China" 6`

*Note: you can swap "China" and 6 for any words you want followed by any number.*

## Results
Due to limited computational resources, the results obtained from this assignment were more of a proof-of-concept rather than high-quality outputs. The model was trained for 20 epochs, but you are encouraged to experiment with higher values to achieve better results.
