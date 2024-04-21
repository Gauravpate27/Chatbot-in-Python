import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load intents from a JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the pre-trained model and associated data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize the model and load the pre-trained weights
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

def get_response(msg):
    # Tokenize the input message
    sentence = tokenize(msg)
    # Create a bag of words representation of the tokenized sentence
    X = bag_of_words(sentence, all_words)
    # Convert the bag of words to a PyTorch tensor and move it to the appropriate device
    X = torch.from_numpy(X).unsqueeze(0) # Add a batch dimension
    # Make a forward pass through the model
    output = model(X)
    # Get the predicted tag (intent) by finding the index of the maximum value in the output tensor
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    # Choose a random response from the intents based on the predicted tag
    for intent in intents['intents']:
        if tag == intent["tag"]:
            return random.choice(intent['responses'])
    
    # If no appropriate response is found, return a default message
    return "I do not understand..."
