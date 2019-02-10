import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import itertools
import timeit
import os
%matplotlib inline

#Read data in and replace last column, drop ID column
df_default = pd.read_csv("defaultCreditCards.csv")
df_default.columns = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'y']
df_default.drop('ID', axis=1, inplace=True)

#Function to scale numeric features
def scaleDefaultData(df, scale=False, encode = False):
    if scale == True:
        numCols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        df_num = df[numCols]
        df_stand = (df_num-df_num.min()) / (df_num.max()-df_num.min())
        df_cat = df.drop(numCols, axis=1)
        df = pd.concat([df_stand, df_cat], axis=1)
        if encode == True:
            col_1hot =  ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0',
                           'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
            df_1hot = df[col_1hot].astype('category')
            df_1hot = pd.get_dummies(df_1hot).astype('int')
            df_others = df.drop(col_1hot, axis=1)
            df = pd.concat([df_1hot, df_others], axis=1)
    return df

df_default_scaled = scaleDefaultData(df_default, scale=True, encode=True)

#Load in wine quality dataset
df_wine = pd.read_csv('wineQuality.txt', sep=';')

#If wine rating > 5, set quality = 1, else 0.
df_wine['quality'] = df_wine['quality'].apply(lambda x: 1 if x > 5 else 0)

#Function to scale wine numeric features
def scaleWineData(df):
    numCols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
    df_num = df[numCols]
    df_stand = (df_num-df_num.min()) / (df_num.max()-df_num.min())
    df_cat = df.drop(numCols, axis=1)
    df = pd.concat([df_stand, df_cat], axis=1)
    return df

df_wine_scaled = scaleWineData(df_wine)

#Split both datasets into x and y.
def splitData(df_default, df_wine):
    X1 = np.array(df_default.values[:, 0:-1])
    y1 = np.array(df_default.values[:, -1])
    X2 = np.array(df_wine.values[:, 0:-1])
    y2 = np.array(df_wine.values[:, -1])

    return X1, y1, X2, y2

#Train test split
def trainTestSplit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
    return X_train, X_test, y_train, y_test


#Split both scaled datasets into X and Y.
X_default, y_default, X_wine, y_wine = splitData(df_default_scaled, df_wine_scaled)

#---Create train and test sets for both datasets---#

#Default
X_default_train, X_default_test, y_default_train, y_default_test = trainTestSplit(X_default, y_default)

#Wine
X_wine_train, X_wine_test, y_wine_train, y_wine_test = trainTestSplit(X_wine, y_wine)

#--Helper functions --##

def plot_learning_curve(clf, X, y, title="Insert Title"):

    n = len(y)
    train_mean = []; train_std = [] #model performance score (f1)
    cv_mean = []; cv_std = [] #model performance score (f1)
    fit_mean = []; fit_std = [] #model fit/training time
    pred_mean = []; pred_std = [] #model test/prediction times
    train_sizes=(np.linspace(.05, 1.0, 20)*n).astype('int')

    for i in train_sizes:
        print(i)
        idx = np.random.randint(X.shape[0], size=i)
        X_subset = X[idx,:]
        y_subset = y[idx]
        scores = cross_validate(clf, X_subset, y_subset, cv=10, scoring='f1', n_jobs=-1, return_train_score=True)

        train_mean.append(np.mean(scores['train_score'])); train_std.append(np.std(scores['train_score']))
        cv_mean.append(np.mean(scores['test_score'])); cv_std.append(np.std(scores['test_score']))
        fit_mean.append(np.mean(scores['fit_time'])); fit_std.append(np.std(scores['fit_time']))
        pred_mean.append(np.mean(scores['score_time'])); pred_std.append(np.std(scores['score_time']))

    train_mean = np.array(train_mean); train_std = np.array(train_std)
    cv_mean = np.array(cv_mean); cv_std = np.array(cv_std)
    fit_mean = np.array(fit_mean); fit_std = np.array(fit_std)
    pred_mean = np.array(pred_mean); pred_std = np.array(pred_std)

    plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title)
    plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title)

    return train_sizes, train_mean, fit_mean, pred_mean


def plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title):

    plt.figure()
    plt.title("Learning Curve: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.fill_between(train_sizes, train_mean - 2*train_std, train_mean + 2*train_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, cv_mean - 2*cv_std, cv_mean + 2*cv_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    plt.show()


def plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title):

    plt.figure()
    plt.title("Modeling Time: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(train_sizes, fit_mean - 2*fit_std, fit_mean + 2*fit_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, pred_mean - 2*pred_std, pred_mean + 2*pred_std, alpha=0.1, color="r")
    plt.plot(train_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(train_sizes, pred_std, 'o-', color="r", label="Prediction Time (s)")
    plt.legend(loc="best")
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def final_classifier_evaluation(clf,X_train, X_test, y_train, y_test):

    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    start_time = timeit.default_timer()
    y_pred = clf.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time

    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   "+"{:.5f}".format(training_time))
    print("Model Prediction Time (s): "+"{:.5f}\n".format(pred_time))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"     AUC:       "+"{:.2f}".format(auc))
    print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0","1"], title='Confusion Matrix')
    plt.show()

from sklearn.neighbors import KNeighborsClassifier

def hyperKNN(X_train, y_train, X_test, y_test, title):

    f1_test = []
    f1_train = []
    klist = np.linspace(1,250,25).astype('int')
    for i in klist:
        print(i)
        clf = KNeighborsClassifier(n_neighbors=i,n_jobs=-1, metric='euclidean')
        clf.fit(X_train,y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))

    plt.plot(klist, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(klist, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Neighbors')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

hyperKNN(X_wine_train, y_wine_train, X_wine_test, y_wine_test,title="Model Complexity Curve for kNN (Wine Data)\nHyperparameter : No. Neighbors")
estimator_wine = KNeighborsClassifier(n_neighbors=1, n_jobs=-1, metric='euclidean')
train_samp_wine, kNN_train_score_wine, kNN_fit_time_wine, kNN_pred_time_wine = plot_learning_curve(estimator_wine, X_wine_train, y_wine_train,title="kNN Phishing Data")
final_classifier_evaluation(estimator_wine, X_wine_train, X_wine_test, y_wine_train, y_wine_test)

hyperKNN(X_default_train, y_default_train, X_default_test, y_default_test,title="Model Complexity Curve for kNN (Default Data)\nHyperparameter : No. Neighbors")
estimator_default = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
train_samp_default, kNN_train_score_default, kNN_fit_time_default, kNN_pred_time_default = plot_learning_curve(estimator_default, X_default_train, y_default_train,title="kNN Default Data")
final_classifier_evaluation(estimator_default, X_default_train, X_default_test, y_default_train, y_default_test)

from sklearn.tree import DecisionTreeClassifier

def hyperTree(X_train, y_train, X_test, y_test, title):

    f1_test = []
    f1_train = []
    max_depth = list(range(1,31))
    for i in max_depth:
        print(i)
        clf = DecisionTreeClassifier(max_depth=i, random_state=100, min_samples_leaf=1, criterion='entropy')
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))

    plt.plot(max_depth, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(max_depth, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Max Tree Depth')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def TreeGridSearchCV(start_leaf_n, end_leaf_n, X_train, y_train):
    #parameters to search:
    #20 values of min_samples leaf from 0.5% sample to 5% of the training data
    #20 values of max_depth from 1, 20
    param_grid = {'min_samples_leaf':np.linspace(start_leaf_n,end_leaf_n,20).round().astype('int'), 'max_depth':np.arange(1,20)}

    tree = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid=param_grid, cv=10)
    tree.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(tree.best_params_)
    return tree.best_params_['max_depth'], tree.best_params_['min_samples_leaf']

hyperTree(X_wine_train, y_wine_train, X_wine_test, y_wine_test,title="Model Complexity Curve for Wine Tree (Wine Data)\nHyperparameter : Tree Max Depth")
start_leaf_n = round(0.005*len(X_wine_train))
end_leaf_n = round(0.05*len(X_wine_train)) #leaf nodes of size [0.5%, 5% will be tested]
max_depth, min_samples_leaf = TreeGridSearchCV(start_leaf_n,end_leaf_n,X_wine_train,y_wine_train)
estimator_wine = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=100, criterion='entropy')
train_samp_wine, DT_train_score_wine, DT_fit_time_wine, DT_pred_time_wine = plot_learning_curve(estimator_wine, X_wine_train, y_wine_train,title="Decision Tree Wine Data")
final_classifier_evaluation(estimator_wine, X_wine_train, X_wine_test, y_wine_train, y_wine_test)

hyperTree(X_default_train, y_default_train, X_default_test, y_default_test,title="Model Complexity Curve for Default Tree (Default Data)\nHyperparameter : Tree Max Depth")
start_leaf_n = round(0.005*len(X_default_train))
end_leaf_n = round(0.05*len(X_default_train)) #leaf nodes of size [0.5%, 5% will be tested]
max_depth, min_samples_leaf = TreeGridSearchCV(start_leaf_n,end_leaf_n,X_default_train,y_default_train)
estimator_default = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=100, criterion='entropy')
train_samp_default, DT_train_score_default, DT_fit_time_default, DT_pred_time_default = plot_learning_curve(estimator_default, X_default_train, y_default_train,title="Decision Tree Wine Data")
final_classifier_evaluation(estimator_default, X_default_train, X_default_test, y_default_train, y_default_test)

from sklearn.neural_network import MLPClassifier

def hyperNN(X_train, y_train, X_test, y_test, title):

    f1_test = []
    f1_train = []
    hlist = np.linspace(1,150,30).astype('int')
    for i in hlist:
            clf = MLPClassifier(hidden_layer_sizes=(i,), solver='adam', activation='logistic',
                                learning_rate_init=0.05, random_state=100)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))

    plt.plot(hlist, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(hlist, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Hidden Units')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def NNGridSearchCV(X_train, y_train):
    #parameters to search:
    #number of hidden units
    #learning_rate
    h_units = [5, 10, 20, 30, 40, 50, 75, 100]
    learning_rates = [0.01, 0.05, .1]
    param_grid = {'hidden_layer_sizes': h_units, 'learning_rate_init': learning_rates}

    net = GridSearchCV(estimator = MLPClassifier(solver='adam',activation='logistic',random_state=100),
                       param_grid=param_grid, cv=10, n_jobs=-1)
    net.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(net.best_params_)
    return net.best_params_['hidden_layer_sizes'], net.best_params_['learning_rate_init']


hyperNN(X_wine_train, y_wine_train, X_wine_test, y_wine_test,title="Model Complexity Curve for NN (Wine Data)\nHyperparameter : No. Hidden Units")
h_units, learn_rate = NNGridSearchCV(X_wine_train, y_wine_train)
estimator_wine = MLPClassifier(hidden_layer_sizes=(h_units,), solver='adam', activation='logistic',
                               learning_rate_init=learn_rate, random_state=100)
train_samp_wine, NN_train_score_wine, NN_fit_time_wine, NN_pred_time_wine = plot_learning_curve(estimator_wine, X_wine_train, y_wine_train,title="Neural Net Wine Data")
final_classifier_evaluation(estimator_wine, X_wine_train, X_wine_test, y_wine_train, y_wine_test)


hyperNN(X_default_train, y_default_train, X_default_test, y_default_test,title="Model Complexity Curve for NN (Default Data)\nHyperparameter : No. Hidden Units")
h_units, learn_rate = NNGridSearchCV(X_default_train, y_default_train)
estimator_default = MLPClassifier(hidden_layer_sizes=(h_units,), solver='adam', activation='logistic',
                               learning_rate_init=learn_rate, random_state=100)
train_samp_default, NN_train_score_default, NN_fit_time_default, NN_pred_time_default = plot_learning_curve(estimator_default, X_default_train, y_default_train,title="Neural Net Banking Data")
final_classifier_evaluation(estimator_default, X_default_train, X_default_test, y_default_train, y_default_test)

from sklearn.svm import LinearSVC

def hyperSVM(X_train, y_train, X_test, y_test, title):

    f1_test = []
    f1_train = []
    C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    for i in C:
            clf = LinearSVC(C=i, random_state=100)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))

    xvals = [i for i in range(-5, 3)]
    plt.plot(xvals, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(xvals, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Kernel Function')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def SVMGridSearchCV(X_train, y_train):
    #parameters to search:
    #penalty parameter, C
    #
    Cs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    loss = ['hinge', 'squared_hinge']
    param_grid = {'C': Cs, 'loss': loss}

    clf = GridSearchCV(estimator = LinearSVC(random_state=100),
                       param_grid=param_grid, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(clf.best_params_)
    return clf.best_params_['C'], clf.best_params_['loss']

hyperSVM(X_wine_train, y_wine_train, X_wine_test, y_wine_test,title="Model Complexity Curve for SVM (Wine Data)\nHyperparameter : Kernel Function")
C_val, loss_val = SVMGridSearchCV(X_wine_train, y_wine_train)
estimator_wine = LinearSVC(C=C_val, loss=loss_val, random_state=100)
train_samp_wine, SVM_train_score_wine, SVM_fit_time_wine, SVM_pred_time_wine = plot_learning_curve(estimator_wine, X_wine_train, y_wine_train,title="SVM Wine Data")
final_classifier_evaluation(estimator_wine, X_wine_train, X_wine_test, y_wine_train, y_wine_test)

hyperSVM(X_default_train, y_default_train, X_default_test, y_default_test,title="Model Complexity Curve for SVM (Default Data)\nHyperparameter : Kernel Function")
C_val, gamma_val = SVMGridSearchCV(X_default_train, y_default_train)
estimator_default = LinearSVC(C=C_val, loss=loss_val, random_state=100)
train_samp_default, SVM_train_score_default, SVM_fit_time_default, SVM_pred_time_default = plot_learning_curve(estimator_default, X_default_train, y_default_train,title="SVM Default Data")
final_classifier_evaluation(estimator_default, X_default_train, X_default_test, y_default_train, y_default_test)

from sklearn.ensemble import GradientBoostingClassifier

def hyperBoost(X_train, y_train, X_test, y_test, max_depth, min_samples_leaf, title):

    f1_test = []
    f1_train = []
    n_estimators = np.linspace(1,250,40).astype('int')
    for i in n_estimators:
            clf = GradientBoostingClassifier(n_estimators=i, max_depth=int(max_depth/2),
                                             min_samples_leaf=int(min_samples_leaf/2), random_state=100,)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))

    plt.plot(n_estimators, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(n_estimators, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Estimators')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def BoostedGridSearchCV(start_leaf_n, end_leaf_n, X_train, y_train):
    #parameters to search:
    #n_estimators, learning_rate, max_depth, min_samples_leaf
    param_grid = {'min_samples_leaf': np.linspace(start_leaf_n,end_leaf_n,3).round().astype('int'),
                  'max_depth': np.arange(1,4),
                  'n_estimators': np.linspace(10,100,3).round().astype('int'),
                  'learning_rate': np.linspace(.001,.1,3)}

    boost = GridSearchCV(estimator = GradientBoostingClassifier(), param_grid=param_grid, cv=5, n_jobs=-1)
    boost.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(boost.best_params_)
    return boost.best_params_['max_depth'], boost.best_params_['min_samples_leaf'], boost.best_params_['n_estimators'], boost.best_params_['learning_rate']


hyperBoost(X_wine_train, y_wine_train, X_wine_test, y_wine_test, 3, 50, title="Model Complexity Curve for Boosted Tree (Wine Data)\nHyperparameter : No. Estimators")
start_leaf_n = round(0.005*len(X_wine_train))
end_leaf_n = round(0.05*len(X_wine_train)) #leaf nodes of size [0.5%, 5% will be tested]
max_depth, min_samples_leaf, n_est, learn_rate = BoostedGridSearchCV(start_leaf_n,end_leaf_n,X_wine_train,y_wine_train)
estimator_wine = GradientBoostingClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                              n_estimators=n_est, learning_rate=learn_rate, random_state=100)
train_samp_wine, BT_train_score_wine, BT_fit_time_wine, BT_pred_time_wine = plot_learning_curve(estimator_wine, X_wine_train, y_wine_train,title="Boosted Tree Wine Data")
final_classifier_evaluation(estimator_wine, X_wine_train, X_wine_test, y_wine_train, y_wine_test)

hyperBoost(X_default_train, y_default_train, X_default_test, y_default_test, 3, 50, title="Model Complexity Curve for Boosted Tree (Default Data)\nHyperparameter : No. Estimators")
start_leaf_n = round(0.005*len(X_default_train))
end_leaf_n = round(0.05*len(X_default_train)) #leaf nodes of size [0.5%, 5% will be tested]

max_depth, min_samples_leaf, n_est, learn_rate = BoostedGridSearchCV(start_leaf_n,end_leaf_n,X_default_train,y_default_train)

estimator_default = GradientBoostingClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                              n_estimators=n_est, learning_rate=learn_rate, random_state=100)
train_samp_default, BT_train_score_default, BT_fit_time_default, BT_pred_time_default = plot_learning_curve(estimator_default, X_default_train, y_default_train,title="Boosted Tree Banking Data")
final_classifier_evaluation(estimator_default, X_default_train, X_default_test, y_default_train, y_default_test)

def compare_fit_time(n,NNtime, SMVtime, kNNtime, DTtime, BTtime, title):

    plt.figure()
    plt.title("Model Training Times: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Training Time (s)")
    plt.plot(n, NNtime, '-', color="b", label="Neural Network")
    plt.plot(n, SMVtime, '-', color="r", label="SVM")
    plt.plot(n, kNNtime, '-', color="g", label="kNN")
    plt.plot(n, DTtime, '-', color="m", label="Decision Tree")
    plt.plot(n, BTtime, '-', color="k", label="Boosted Tree")
    plt.legend(loc="best")
    plt.show()

def compare_pred_time(n,NNpred, SMVpred, kNNpred, DTpred, BTpred, title):

    plt.figure()
    plt.title("Model Prediction Times: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Prediction Time (s)")
    plt.plot(n, NNpred, '-', color="b", label="Neural Network")
    plt.plot(n, SMVpred, '-', color="r", label="SVM")
    plt.plot(n, kNNpred, '-', color="g", label="kNN")
    plt.plot(n, DTpred, '-', color="m", label="Decision Tree")
    plt.plot(n, BTpred, '-', color="k", label="Boosted Tree")
    plt.legend(loc="best")
    plt.show()


def compare_learn_time(n,NNlearn, SMVlearn, kNNlearn, DTlearn, BTlearn, title):

    plt.figure()
    plt.title("Model Learning Rates: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.plot(n, NNlearn, '-', color="b", label="Neural Network")
    plt.plot(n, SMVlearn, '-', color="r", label="SVM")
    plt.plot(n, kNNlearn, '-', color="g", label="kNN")
    plt.plot(n, DTlearn, '-', color="m", label="Decision Tree")
    plt.plot(n, BTlearn, '-', color="k", label="Boosted Tree")
    plt.legend(loc="best")
    plt.show()

compare_fit_time(train_samp_wine, NN_fit_time_wine, SVM_fit_time_wine, kNN_fit_time_wine,
                 DT_fit_time_wine, BT_fit_time_wine, 'Wine Dataset')
compare_pred_time(train_samp_wine, NN_pred_time_wine, SVM_pred_time_wine, kNN_pred_time_wine,
                 DT_pred_time_wine, BT_pred_time_wine, 'Wine Dataset')
compare_learn_time(train_samp_wine, NN_train_score_wine, SVM_train_score_wine, kNN_train_score_wine,
                 DT_train_score_wine, BT_train_score_wine, 'Wine Dataset')



compare_fit_time(train_samp_default, NN_fit_time_default, SVM_fit_time_default, kNN_fit_time_default,
                 DT_fit_time_default, BT_fit_time_default, 'Default Dataset')
compare_pred_time(train_samp_default, NN_pred_time_default, SVM_pred_time_default, kNN_pred_time_default,
                 DT_pred_time_default, BT_pred_time_default, 'Default Dataset')
compare_learn_time(train_samp_default, NN_train_score_default, SVM_train_score_default, kNN_train_score_default,
                 DT_train_score_default, BT_train_score_default, 'Default Dataset')
