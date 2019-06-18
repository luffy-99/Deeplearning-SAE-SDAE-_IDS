import time
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import LinearSVC
from imblearn.metrics import (geometric_mean_score, make_index_balanced_accuracy)
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class DataSet(object):
    def __init__(self, features, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert features.shape[0] == labels.shape[0], (
                    "images.shape: %s labels.shape: %s" % (features.shape,
                                                           labels.shape))
            self._num_examples = features.shape[0]
        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(38)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]


def read_data_sets(X_train, Y_train, X_test, Y_test):
    class DataSets(object):
        pass

    data_sets = DataSets()
    data_sets.train = DataSet(X_train, Y_train)
    data_sets.test = DataSet(X_test, Y_test)
    return data_sets


"Normalize training and testing sets"


def normalize_data(train_X, test_X, scale):
    if ((scale == "standard") | (scale == "maxabs") | (scale == "minmax")):
        if (scale == "standard"):
            scaler = preprocessing.StandardScaler()
        elif (scale == "maxabs"):
            scaler = preprocessing.MaxAbsScaler()
        elif (scale == "minmax"):
            scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)
    else:
        print("No scaler")
    return train_X, test_X

def KNeighbors(X_tr, Y_tr, X_te, Y_te):
    if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)

    clf = KNeighborsClassifier(n_neighbors=1).fit(X_tr, Y_tr)

    start = time.time()
    y_pred = clf.predict(X_te)
    end = time.time()
    elapsed = (end - start) / float(len(X_te))
    acc = accuracy_score(Y_te, y_pred)
    fpr_vot, tpr_vot, _ = roc_curve(Y_te, y_pred, pos_label=1, drop_intermediate=False)
    roc_auc_vot = auc(fpr_vot, tpr_vot)
    print('K-Neighbors')
    print('The auc is {}'.format(roc_auc_vot))
    return roc_auc_vot, elapsed

def Logistic(X_tr, Y_tr, X_te, Y_te):
    if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_tr, Y_tr)

    start = time.time()
    y_pred = clf.predict(X_te)
    end = time.time()
    elapsed = (end - start) / float(len(X_te))
    acc = accuracy_score(Y_te, y_pred)
    fpr_vot, tpr_vot, _ = roc_curve(Y_te, y_pred, pos_label=1, drop_intermediate=False)
    roc_auc_vot = auc(fpr_vot, tpr_vot)
    print('Logistic:')
    print('The auc is {}'.format(roc_auc_vot))
    return roc_auc_vot, elapsed

def svm(X_tr, Y_tr, X_te, Y_te):
    # bw = (len(X_tr)/2.0)**0.5        #default value in One-class SVM
    # gamma = 1/(2*bw*bw)
    X_tr, X_te = normalize_data(X_tr, X_te, "minmax")
    if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)
    # parameters =  [{'kernel': ['rbf'], 'gamma': [1e-3],
    #                 'C': [1]}]
    # {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # svc = svm.SVC()
    # clf = GridSearchCV(svc, parameters,cv= 5)
    # clf = SVC (gamma = gamma)
    clf = LinearSVC(random_state=0)
    clf.fit(X_tr, Y_tr)
    start = time.time()
    y_pred = clf.predict(X_te)
    end = time.time()
    elapsed = (end - start) / float(len(X_te))
    acc = accuracy_score(Y_te, y_pred)
    fpr_vot, tpr_vot, _ = roc_curve(Y_te, y_pred, pos_label=1, drop_intermediate=False)
    roc_auc_vot = auc(fpr_vot, tpr_vot)
    cmat = classification_report_imbalanced(Y_te, y_pred)
    print("SVM")
    geo = geometric_mean_score(Y_te, y_pred)
    f1 = f1_score(Y_te, y_pred, average='macro')
    print('The auc is {} '.format(roc_auc_vot))
    return roc_auc_vot, elapsed


def rf(X_tr, Y_tr, X_te, Y_te):
    X_tr, X_te = normalize_data(X_tr, X_te, "minmax")
    if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)

    clf = RandomForestClassifier(n_estimators=100, max_depth=80, random_state=0)
    clf.fit(X_tr, Y_tr)
    start = time.time()
    y_pred = clf.predict(X_te)
    end = time.time()
    elapsed = (end - start) / float(len(X_te))
    acc = accuracy_score(Y_te, y_pred)
    fpr_vot, tpr_vot, _ = roc_curve(Y_te, y_pred, pos_label=1, drop_intermediate=False)
    roc_auc_vot = auc(fpr_vot, tpr_vot)
    cmat = classification_report(Y_te, y_pred)
    print("Random forest")

    print('The auc is {} '.format(roc_auc_vot))
    return roc_auc_vot, elapsed


def decisiontree(X_tr, Y_tr, X_te, Y_te):
    # X_tr, X_te = normalize_data(X_tr, X_te, "minmax")
    if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)
    param_grid = {'max_depth': np.arange(3, 6)}

    tree = GridSearchCV(DecisionTreeClassifier(), param_grid)

    tree.fit(X_tr, Y_tr)
    start = time.time()
    y_pred = tree.predict(X_te)
    end = time.time()
    elapsed = (end - start) / float(len(X_te))
    acc = accuracy_score(Y_te, y_pred)
    fpr_vot, tpr_vot, _ = roc_curve(Y_te, y_pred, pos_label=1, drop_intermediate=False)
    roc_auc_vot = auc(fpr_vot, tpr_vot)
    cmat = classification_report_imbalanced(Y_te, y_pred)
    print("Decision tree")
    # print (cmat)

    geo = geometric_mean_score(Y_te, y_pred)
    f1 = f1_score(Y_te, y_pred, average='micro')

    print('The auc is {} '.format(roc_auc_vot))
    return roc_auc_vot, elapsed


def naive_baves(X_tr, Y_tr, X_te, Y_te):
    if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)
    clf = GaussianNB()
    clf.fit(X_tr, Y_tr)
    start = time.time()
    y_pred = clf.predict(X_te)
    end = time.time()
    time_about = end - start
    acc = accuracy_score(Y_te, y_pred)
    fpr_vot, tpr_vot, _ = roc_curve(Y_te, y_pred, pos_label=1, drop_intermediate=False)
    roc_auc_vot = auc(fpr_vot, tpr_vot)
    # cmat = classification_report_imbalanced(Y_te, y_pred)
    print("Naive Baves")
    print('The auc is {} '.format(roc_auc_vot))
    return roc_auc_vot, time_about
