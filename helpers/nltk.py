import nltk
import re
import stop_words  # pip install stop-words
# import morfeusz2  # nie działą
import spacy  # pip install -U spacy  # python -m spacy download pl_core_news_lg
# import gensim  # pip install --upgrade gensim
# from gensim.models import Word2Vec, KeyedVectors, word2vec


def tokenize_text_sentences(text):
    while True:
        try:
            tokenized_text = nltk.tokenize.sent_tokenize(text)
            return tokenized_text
        except LookupError as error:
            resource = re.search("nltk\\.download\\('(.+?)'\\)", str(error)).group(1)
            print(f'Downloading missing resource [{resource}]')
            nltk.download(resource)


def tokenize_text_words(text, to_lower=False, disallowed_chars=[]):
    if to_lower:
        text = text.lower()

    tokenized_words = nltk.tokenize.word_tokenize(text)

    if len(disallowed_chars) > 0:
        filtered_words = [];
        for word in tokenized_words:
            if not any(char in word for char in disallowed_chars):
                filtered_words.append(word)
        tokenized_words = filtered_words

    return tokenized_words


def get_words_frequency(words):
    words_frequency = nltk.probability.FreqDist(words)

    return words_frequency


def get_stopwords_by_lang(language):
    while True:
        try:
            stopwords = set(nltk.corpus.stopwords.words(language))
            return stopwords
        except LookupError as error:
            resource = re.search("nltk\\.download\\('(.+?)'\\)", str(error)).group(1)
            print(f'Downloading missing resource [{resource}]')
            nltk.download(resource)
        except IOError as error:
            stopwords = stop_words.safe_get_stop_words(language)
            return stopwords


def get_stemmed_words(words):
    stemmer = nltk.stem.PorterStemmer()

    return [stemmer.stem(word) for word in words]


def get_lemmatized_english_words(words):
    while True:
        try:
            lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
            return [lemmatizer.lemmatize(word, "v") for word in words]
        except LookupError as error:
            resource = re.search("nltk\\.download\\('(.+?)'\\)", str(error)).group(1)
            print(f'Downloading missing resource [{resource}]')
            nltk.download(resource)


def get_polish_lemmatizer():
    print('Loading lemmatizer database ... ')
    lemmatizer = spacy.load('pl_core_news_lg')
    print('> Lemmatizer database loaded')
    return lemmatizer


def get_lemmatized_polish_words(sentence, lemmatizer=None):
    if lemmatizer is None:
        lemmatizer = get_polish_lemmatizer()
    lemmatized_words = lemmatizer(sentence)
    return lemmatized_words


def get_pos_tagged_words(words):
    while True:
        try:
            pos_tagged_words = nltk.pos_tag(words)
            return pos_tagged_words
        except LookupError as error:
            resource = re.search("nltk\\.download\\('(.+?)'\\)", str(error)).group(1)
            print(f'Downloading missing resource [{resource}]')
            nltk.download(resource)
