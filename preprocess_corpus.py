import os
import glob
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import spacy
from nltk.corpus import stopwords
import string
import numpy as np
import pickle
import shutil

remove_preprocessed_data = True

print("Loading spacy-model...")
spacy_german = spacy.load('de')
german_stopwords = stopwords.words("german")
punctuations = string.punctuation


def main(args=None):
    corpus_path = "corpus"
    preprocessed_data_path = "preprocessed"
    preprocess_corpus(corpus_path, preprocessed_data_path)


def preprocess_corpus(corpus_path="corpus", preprocessed_data_path="preprocessed"):

    # Make sure that there is a proper folder for analysis.
    if os.path.exists(preprocessed_data_path) and remove_preprocessed_data is True:
        shutil.rmtree(preprocessed_data_path)
    if not os.path.exists(preprocessed_data_path):
        os.makedirs(preprocessed_data_path)

    # Get all the classes from filesystem.
    class_names = [element for element in os.listdir(corpus_path) if not os.path.isfile(os.path.join(corpus_path, element))]
    print(class_names)

    # Compute one-hot-encodings.
    print("Computing one-hot-encodings for classes...")
    one_hot_encodings = label_binarize(class_names, classes=class_names)
    for index in range(len(class_names)):
        class_name = class_names[index]
        one_hot_encoding = one_hot_encodings[index]
        print(class_name, "->", one_hot_encoding)

    # Process classes.
    data_in = []
    data_out = []
    for index, class_name in enumerate(class_names):
        print("Processing", class_names)
        one_hot_encoding = one_hot_encodings[index]

        # Get all files.
        glob_path = os.path.join(corpus_path, class_name, "*.body.txt")
        body_file_paths = glob.glob(glob_path)
        body_file_paths = body_file_paths[0:1]

        # For each file split data into paragraphs. Each paragraph is then turned into a document-vector.
        # After that we have all data as input-output.
        for body_file_path in body_file_paths:
            print("Processing file", body_file_path + "...")
            with open(body_file_path) as body_file:
                body = body_file.read()
                paragraphs = split_into_paragraphs(body)
                vectors = [doc.vector for doc in spacy_german.pipe(paragraphs, batch_size=500, n_threads=1)]
                vectors = np.array(vectors)
                data_in.extend(vectors)
                for _ in range(len(vectors)):
                    data_out.append(one_hot_encoding)

                # Double checking.
                if len(data_in) != len(data_out):
                    raise Exception("Inconsistency.", len(data_in), len(data_out))

    # Split into training and text.
    X_train, X_test, y_train, y_test = train_test_split(data_in, data_out, test_size=0.2, random_state=21)
    print('X_train size: {}'.format(len(X_train)))
    print('X_test size: {}'.format(len(X_test)))
    print('y_train size: {}'.format(len(y_train)))
    print('y_test size: {}'.format(len(y_test)))

    # Writing to file.
    preprocessed_data_name = "preprocessed.pickle"
    preprocessed_data_path = os.path.join(preprocessed_data_path, preprocessed_data_name)
    print("Writing preprocessed data to", preprocessed_data_path)
    preprocessed_data = X_train, X_test, y_train, y_test, class_names
    with open(preprocessed_data_path, "wb") as output_file:
        pickle.dump(preprocessed_data, output_file)

    print("Done.")


def split_into_paragraphs(text):
    paragraphs_ret = []

    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        text, tokens = clean_up_text(paragraph)
        if len(tokens) >= 3:
            paragraphs_ret.append(text)

    return paragraphs_ret


def clean_up_text(text):
    doc = spacy_german(text)

    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in german_stopwords and tok not in punctuations]

    text = ' '.join(tokens)
    return text, tokens


if __name__ == "__main__":
    main()
