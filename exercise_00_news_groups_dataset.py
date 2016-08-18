# -*- coding: utf-8 -*-

# Author: Yaroslav Zemlyanuhin <d4rk5eed@yandex.ru>
# License: Simplified BSD

import sys

from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn import metrics

def main():

    #comp.graphics
    # comp.os.ms-windows.misc
    # comp.sys.ibm.pc.hardware
    # comp.sys.mac.hardware
    # comp.windows.x	rec.autos
    # rec.motorcycles
    # rec.sport.baseball
    # rec.sport.hockey	sci.crypt
    # sci.electronics
    # sci.med
    # sci.space
    # misc.forsale
    # talk.politics.misc
    # talk.politics.guns
    # talk.politics.mideast
    # talk.religion.misc
    # alt.atheism
    # soc.religion.christian

    categories = ['alt.atheism', 'soc.religion.christian', 'talk.religion.misc',
                  'sci.med', 'talk.politics.mideast',
                  'talk.politics.misc']

    dataset = fetch_20newsgroups(subset='train',
                                 categories=categories, shuffle=True, random_state=42)

    # Split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.5)

    print dataset.target_names


    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, n_iter=5, random_state=42)),
    ])

    text_clfclf = text_clf.fit(docs_train, y_train)

    y_predicted = text_clf.predict(docs_test)

    # TASK: Predict the outcome on the testing set in a variable named y_predicted

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Plot the confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, y_predicted)
    print conf_matrix

    # Predict the result on some short new sentences:
    sentences = [
        u'Turkey attacks: Blasts targeting police and soldiers kill at least 12',
        u'And Moses and Aaron did all these wonders before Pharaoh: and the LORD hardened Pharaoh\'s heart, so that he would not let the children  go out of his land.',
    ]
    predicted = text_clf.predict(sentences)

    for sent, pred in zip(sentences, predicted):
        print u'The language of "%s" is "%s"' % (unicode(sent), dataset.target_names[pred])

if __name__ == u'__main__':
    main()
