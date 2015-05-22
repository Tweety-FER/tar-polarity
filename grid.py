from __future__ import print_function
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, make_scorer

#Number of folds
k = 5

X_train = np.loadtxt('features/train/classification.txt')
y_train = np.loadtxt('features/train/y.txt')[:,0]

X_test = np.loadtxt('features/test/classification.txt')
y_test = np.loadtxt('features/test/y.txt')[:,0]

X_train = scale(X_train)
X_test = scale(X_test)

N = X_train.shape[0]

# ---------- GRID SETUP ----------- #
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
                     'C': [5, 10, 25, 50, 72, 100, 250, 500, 750, 1000, 1250, 1500]}]#,
                    #{'kernel': ['linear'], 'C': [10, 50, 100, 500, 1000]}]

# Ignore linear for now, too slow

scores = [('f1',make_scorer(f1_score))]

for score in scores:
    print("# Tuning hyper-parameters for %s" % score[0])
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=k,
                       scoring=score[1])
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
