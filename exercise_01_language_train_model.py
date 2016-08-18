# -*- coding: utf-8 -*-

"""Build a language detector model

The goal of this exercise is to train a linear classifier on text features
that represent sequences of up to 3 consecutive characters so as to be
recognize natural languages by using the frequencies of short character
sequences as 'fingerprints'.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# Author: Yaroslav Zemlyanuhin <d4rk5eed@yandex.ru>
# License: Simplified BSD

import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from nltk.tokenize import RegexpTokenizer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer

def my_tokenizer(string):
    tokenizer = RegexpTokenizer(r'(\w{1,3})\w*')
    return tokenizer.tokenize(string)

def main(argv):
    # The training data folder must be passed as first argument
    languages_data_folder = argv[1]
    dataset = load_files(languages_data_folder)

    # Split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.5)

    print dataset.target_names

    # TASK: Build a an vectorizer that splits strings into sequence of 1 to 3
    # characters instead of word tokens

    # This one is first tokenizer I've built, prediction gave about 0,9 precision
    # vectorizer = CountVectorizer(tokenizer=my_tokenizer)


    #from sklearn.pipeline import Pipeline
    # clf = Pipeline([('vect', vectorizer),
    #                 ('tfidf', TfidfTransformer()),
    #                 ('clf', Perceptron()),
    # ])

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char',
                                 use_idf=False)

    #TASK: Build a vectorizer / classifier pipeline using the previous analyzer
    #the pipeline instance should stored in a variable named clf
    clf = Pipeline([
        ('vect', vectorizer),
        ('clf', Perceptron()),
    ])

    # TASK: Fit the pipeline on the training set
    clf = clf.fit(docs_train, y_train)

    y_predicted = clf.predict(docs_test)

    # TASK: Predict the outcome on the testing set in a variable named y_predicted

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Plot the confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, y_predicted)
    print conf_matrix

    # Uncomment to show the matrix
    # import pylab as pl
    # pl.matshow(conf_matrix, cmap=pl.cm.jet)
    # pl.show()

    # Predict the result on some short new sentences:
    sentences = [
        u'This is a language detection test.',
        u'Ceci est un test de d\xe9tection de la langue.',
        u'Dies ist ein Test, um die Sprache zu erkennen.',
        u'Это тест для определения языка',
    ]
    predicted = clf.predict(sentences)

    for sent, pred in zip(sentences, predicted):
        print u'The language of "%s" is "%s"' % (unicode(sent), dataset.target_names[pred])

if __name__ == u'__main__':
    main(sys.argv)
