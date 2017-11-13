import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from pprint import pprint
import json
from wordcloud import WordCloud
import spacy
from nltk.corpus import stopwords
import string
from collections import Counter

remove_analysis = True

print("Loading spacy-model...")
spacy_german = spacy.load('de')
german_stopwords = stopwords.words("german")
punctuations = string.punctuation

# TODO dokumentieren python -m spacy.de.download
# Ansehen TODO https://explosion.ai/blog/german-model
# TODO More output

def main(args=None):

    corpus_path = "corpus"
    analysis_path = "analysis"
    analyze_corpus(corpus_path, analysis_path)


def analyze_corpus(corpus_path="corpus", analysis_path="analysis"):

    print("Analyzing corpus...")

    # Make sure that there is a proper folder for analysis.
    if os.path.exists(analysis_path) and remove_analysis is True:
        shutil.rmtree(analysis_path)
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    # Get all authors from the corpus.
    authors = [element for element in os.listdir(corpus_path) if not os.path.isfile(os.path.join(corpus_path, element))]

    counts = []
    for author in authors:
        all_text, all_tokens, all_count = process_author(corpus_path, author)

        # Render the word cloud.
        word_cloud_name = author + ".wordcloud.png"
        word_cloud_path = os.path.join(analysis_path, word_cloud_name)
        render_word_cloud(all_text, word_cloud_path)

        # Render the word distribution.
        most_frequent_words_name = author + ".most_frequent_words.png"
        most_frequent_words_path = os.path.join(analysis_path, most_frequent_words_name)
        render_most_frequent_words(author, all_tokens, most_frequent_words_path)

        # TODO count words, files, or paragraphs
        counts.append(all_count)

    # Render corpus distribution.
    sns_plot = sns.barplot(x=authors, y=counts)
    corpus_distribution_name = "corpus_distribution.png"
    corpus_distribution_path = os.path.join(analysis_path, corpus_distribution_name)
    sns_plot.get_figure().savefig(corpus_distribution_path)
    print("Written corpus-distribution to", corpus_distribution_path)


def process_author(corpus_path, author):
    all_text = ""
    all_tokens = []
    all_count = 0

    glob_path = os.path.join(corpus_path, author, "*.body.txt")
    body_file_paths = glob.glob(glob_path)
    for body_file_path in body_file_paths:
        print(body_file_path)
        with open(body_file_path) as body_file:
            body = body_file.read()
            text, tokens = clean_up_text(body)
            all_text += text
            all_tokens.extend(tokens)
            all_count += len(tokens)

    return all_text, all_tokens, all_count

def clean_up_text(text):
    doc = spacy_german(text)

    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in german_stopwords and tok not in punctuations]

    text = ' '.join(tokens)
    return text, tokens


def render_word_cloud(all_text, word_cloud_path):
    wordcloud = WordCloud(width=800, height=500,
                      random_state=21, max_font_size=110).generate(all_text)
    figure = plt.figure(figsize=(15, 12))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    #plt.show()
    figure.savefig(word_cloud_path)
    print("Written word-cloud to", word_cloud_path)


def render_most_frequent_words(author, tokens, most_frequent_words_path):

    # Clean up the text.
    #clean_text = [word for word in clean_text if word != '\'s']

    counts = Counter(tokens) # TODO long string or sequence?

    common_words = [word[0] for word in counts.most_common(25)]
    common_counts = [word[1] for word in counts.most_common(25)]

    figure = plt.figure(figsize=(15, 12))
    sns.barplot(x=common_words, y=common_counts)
    plt.title("Most Common Words used by " + author)
    #plt.show()

    figure.savefig(most_frequent_words_path)
    print("Written most frequent words to", most_frequent_words_path)


if __name__ == "__main__":
    main()
