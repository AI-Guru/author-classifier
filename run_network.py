import os
from keras.models import load_model
import pickle
import spacy
from nltk.corpus import stopwords
import string
import numpy as np

print("Loading spacy-model...")
spacy_german = spacy.load('de')
german_stopwords = stopwords.words("german")
punctuations = string.punctuation


def main(args=None):
    model_path = "model"
    run_network(model_path)

def run_network(model_path):

    #  Load the model.
    model_name = "network.model"
    model_file_path = os.path.join(model_path, model_name)
    print("Loading model from", model_file_path + "...")
    model = load_model(model_file_path)
    print("Model loaded.")

    # Load model meta-data.
    model_metadata_name = "network.meta"
    model_metadata_file_path = os.path.join(model_path, model_metadata_name)
    print("Loading model-metadata from", model_metadata_file_path + "...")
    model_metadata = pickle.load(open(model_metadata_file_path, "rb" ))
    model_type, class_names = model_metadata
    print("Model-metadata loaded.")

    # TODO Predict...
    sentence = "Gebraucht der Zeit sie eilt so schnell von hinnen!"
    sentence,_ = clean_up_text(sentence)
    print(sentence)
    doc = spacy_german(sentence)
    print(doc)
    vector = doc.vector
    vector = np.expand_dims(vector, axis=2)
    vector = np.expand_dims(vector, axis=0)
    print(vector)
    print(vector.shape)

    prediction = model.predict([vector])[0]
    print(prediction)
    print(prediction_to_string(prediction, class_names))


def clean_up_text(text):
    doc = spacy_german(text)

    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in german_stopwords and tok not in punctuations]

    text = ' '.join(tokens)
    return text, tokens


def prediction_to_string(prediction, class_names):
    prediction_map = {}
    for i in range(len(prediction)):
        p = prediction[i]
        class_name = class_names[i]
        prediction_map[class_name] = p
    return str(prediction_map)

if __name__ == "__main__":
    main()
