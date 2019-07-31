import os, sys
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy
import pandas as pd
import numpy as np
import pdb

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


SEED = 10


def vis_data(X, y):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="Set3",s=20)

    fig.show()


def load_data(fn):
    data = pd.read_csv(fn, dtype=np.float64).to_numpy()
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def load_tag_fea(fn):
    data = pd.read_csv(fn, dtype=np.float64).to_numpy()
    return data

def model_training(X, y, params, linear=True): # normalized X and y
    # training 
    if linear:
        clf = LinearSVC(C=params["C"], fit_intercept=False, random_state=SEED)
    else:
        clf = SVC(C=params["C"], gamma=params["gamma"], random_state=SEED)

    clf.fit(X, y)

    # testing
    y_test = clf.predict(X)

    # accuracy
    print ("The classification accuracy: ", round(clf.score(X, y_test), 4))

    # classification report
    print ("The classification report:\n", classification_report(y, y_test))

    return clf.coef_, clf.intercept_
    


def model_select(X, y, linear=True):
    '''
        input: training data
        output: best params
    '''
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=SEED)
    if linear:
        param = {'C': range(1, 200, 1)}
        random_search = RandomizedSearchCV(LinearSVC(fit_intercept=False, random_state=SEED), param, cv=5, n_iter=10)
    else:
        param = {'C': range(1, 200, 1), 'gamma': scipy.stats.expon()}
        random_search = RandomizedSearchCV(SCV(random_state=SEED), param, cv=5, n_iter=10)

    random_search.fit(train_x, train_y)

    print ("The accuracy on testing data: %.3f"%(random_search.score(test_x, test_y)))

    print ("The best parameters are: {}".format(random_search.best_params_))
    print ("The best score in cross validation: %.3f"%(random_search.best_score_))
    return random_search.best_params_


def main():
    data_path = "../../data/dataset_top50tag"
    fn_training_data = os.path.join(data_path, "training.csv")
    fn_tag_fea = os.path.join(data_path, "tag_features.csv")
    fn_tag_score = os.path.join(data_path, "tag_score.txt")
    X_, y = load_data(fn_training_data)

    # pdb.set_trace()
    # vis_data(X_, y)

    linear = True

    # normalize
    scaler = preprocessing.StandardScaler().fit(X_)
    X = scaler.transform(X_)

    # vis data
    #vis_data(X, y)
    #pdb.set_trace()

    best_params = model_select(X, y, linear)

    #pdb.set_trace()
    coef, intercept = model_training(X, y, best_params, linear)

    tag_feas_raw = load_tag_fea(fn_tag_fea)
    tag_feas = scaler.transform(tag_feas_raw)

    tag_score_raw = get_tag_score(coef, tag_feas_raw)
    tag_score = get_tag_score(coef, tag_feas)

    

    #pdb.set_trace()
    #show_bars(tag_score, tag_score_raw)
    np.savetxt(fn_tag_score, tag_score, fmt='%.5f')
    print ("generate tag score over !")



def get_tag_score(coef, feas):
    return np.sum(coef*feas, axis=1)

def show_bars(y_norm, y_raw):
    fig, axs = plt.subplots(2, 1)
    x = np.arange(len(y_norm))
    axs[0].bar(x, y_norm)
    axs[1].bar(x, y_raw)
    plt.show()

    
if __name__ == '__main__':
    main()
