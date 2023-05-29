# Assignment 3 - Language modelling and text generation using RNNs

For this assignment, I have written code that trains a text generation model on the headlines of articles by The New York Times from 2017-2018. 

*Note: due to technical difficulties, the data used for training - i.e., article headlines - differs from what the original instructions for the assignment outlined. This change was approved by instructor Ross during class.*

## Instructions
*Note: this code was run on uCloud's Coder Python 1.76.1. Depending on the device you are using, your terminal commands may need to be slightly modified.*

To run the code, first install the requirements by running the following command in terminal: `pip install -r requirements.txt`

Then train save the RNN model by navigating to the root folder and running this command: `python src/rnn_save.py`

Finally, you can load the model and have it generate continuation of any text you input. For example, generate a text that starts with "China" and continues for 6 words by running this command: `python src/rnn_load.py "China" 6`
*Note: you can swap "China" and 6 for any words (in quotes) you want followed by any number.