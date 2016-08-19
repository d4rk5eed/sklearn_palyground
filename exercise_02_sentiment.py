"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess wether the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics

def main(argv):
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    movie_reviews_data_folder = argv[1]
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print "n_samples: %d" % len(dataset.data)

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    vectorizer = TfidfVectorizer()


    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # # from sklearn.grid_search import GridSearchCV
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': [0.9, 0.95],
        'vect__min_df': [0.05, 3],
    }

    # Fit the pipeline on the training set using grid search for the parameters
    clf = Pipeline([
        ('vect', vectorizer),
        ('clf', LinearSVC(C=1000)),
    ])

    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(docs_train, y_train)


    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print "%s: %r" % (param_name, best_parameters[param_name])

    print "score: %s" % score
    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted

    y_predicted = gs_clf.predict(docs_test)
    # Print the classification report
    print metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names)

    # Print and plot the confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, y_predicted)
    print conf_matrix
    # import matplotlib.pyplot as plt
    # plt.matshow(cm)
    # plt.show()

if __name__ == "__main__":
    main(sys.argv)
