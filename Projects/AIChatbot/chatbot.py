from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import random
import json
import pickle
import numpy as np
import os
print(os.getcwd())
print(os.listdir(os.getcwd()))


lemmatizer = WordNetLemmatizer()
# Get the folder where chatbot.py lives
base_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to this script
intents_path = os.path.join(base_dir, 'intents.json')
words_path = os.path.join(base_dir, 'words.pkl')
classes_path = os.path.join(base_dir, 'classes.pkl')
model_path = os.path.join(base_dir, 'chatbot_model.h5')

# Load files
with open(intents_path, 'r') as f:
    intents = json.load(f)

with open(words_path, 'rb') as f:
    words = pickle.load(f)

with open(classes_path, 'rb') as f:
    classes = pickle.load(f)

model = load_model(model_path)


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result


print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
