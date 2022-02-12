#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    # deal with cases where the denominator could be 0
    if np.sum(C) = 0:
        return 0
    else:
        return np.trace(C)/np.sum(C)

def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    assert len(C) == 4
    recall_=[]
    for i in range(len(C)):
        recall_.append(C[i,i]/np.sum(C[i,:]))
    return recall_

def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precision_=[]
    assert len(C) == 4
    for i in range(len(C)):
        precision_.append(C[i,i]/np.sum(C[:,i]))
    return precision_

SDG = SGDClassifier()
GNB = GaussianNB()
RFC = RandomForestClassifier(n_estimators=10, max_depth=5)
MLP = MLPClassifier(alpha=0.05)
ABC = AdaBoostClassifier()
listModels = [SDG, GNB, RFC, MLP, ABC]

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    print('TODO Section 3.1')

    iBest = None
    best_acc = 0
    
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        #     outf.write(f'\tAccuracy: {acc:.4f}\n')
        #     outf.write(f'\tRecall: {[round(item, 4) for item in recall]}\n')
        #     outf.write(f'\tPrecision: {[round(item, 4) for item in precision]}\n')
        #     outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

        for i, model in enumerate(listModels):
            classifier_name = model.__class__.__name__
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_test_pred)
            accuracies = accuracy(conf_matrix)
            precisions = precision(conf_matrix)
            recalls = recall(conf_matrix)

            outf.write(f'Results for {classifier_name}:\n')  
            outf.write(f'\tAccuracy: {accuracies:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recalls]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precisions]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

            if accuracies > best_acc:
                iBest = i 
                best_acc = accuracies
    return iBest +1
        
def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print('TODO Section 3.2')
    iBest = iBest-1 # indexing
    best_model = models[iBest]
    dataset_sizes = [1000, 5000, 10000, 20000]
    accuracies = []

    for size in dataset_sizes:
        index_ = np.random.choice(a  = X_train.shape[0], size = size)
        X_sub = X_train[index_]
        y_sub = y_train[index_]

        best.fit(X_sub, y_sub)
        y_pred_test = best.predict(X_test)
        C = confusion_matrix(y_test, y_pred_test)
        accuracies.append(accuracy(C))
        if s==1000:
            X_1k = X_sub
            y_1k = y_sub

    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        #     outf.write(f'{num_train}: {acc:.4f}\n'))
        for i, size in enumerate(dataset_sizes):
            outf.write(f'{size}: {accuracies:.4f}\n'))

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')
    best_model = models[i-1]
    p_value = []
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        for k in [5,50]:
            sel = SelectKBest(f_classif, k=k)
            X_new = sel.fit_transform(X_train, y_train) # 1k rows from 3.2
            index_ = sel.p_values_
            index_ = index_.argsort()
            p_values = sel.pvalues_[index_]

            # for each number of features k_feat, write the p-values for
            # that number of features:
            outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')
            ix_32k = sel.pvalues_.argsort()[:k]
            outf.write(f'{k} best features: {ix_32k}\n')

            if k==5:
                # 1000
                X_new5 = sel.fit_transform(X_1k, y_1k)
                best_model.fit(X_new5, y_1k)
                y_predict5 = best_model.predict(sel.transform(X_test))
                C = confusion_matrix(y_test, y_predict5)
                accuracy_1k = accuracy(C)
                outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')

                ix_1k = sel.pvalues_.argsort()[:k]
                print('ix_1k', ix_1k)

                # 32,000
                X_new32 = sel.fit_transform(X_train, y_train)
                best_model.fit(X_new32, y_train)
                y_predict32 =best.model(sel.transform(X_test))
                C = confusion_matrix(y_test, y_predict32)
                accuracy_32k = accuracy(C)
                outf.write(f'Accuracy for full dataset: {accuracy_32k:.4f}\n')

                ix_32k = sel.pvalues_.argsort()[:k]
                print('ix_32k', ix_32k)

                feature_intersection = set(ix_1k) & set(ix_32k)
                outf.write(f'Chosen feature intersection: {feature_intersection}\n')
                outf.write(f'Top-5 at higher: {ix_32k}\n')
        pass


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, x_train), axis= 0)
    k =5 
    kfold = Kfold(k, shuffle = True, random_state=401)
    accuracy_models = []
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        #     outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')
        for i, model in enumerate(models):
            accuracy_i = []
            for test, train in kf.split(X):
                X_train = X[train]
                X_test = X[test]
                y_train = y[train]
                y_test = y[test]
                model.fit(X_train, y_train)
                y_test_pred = mode.predict(X_test)
                C = confusion_matrix(y_test, y_test_pred)
                accuracy_i.append(accuracy(C))
            accuracy_models.append(accuracy_i)
        mean_acc = []
        for acc in accuracy_models:
            mean_acc.append(np.mean(acc))

        


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    # create output dir if it doesn't exist 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # TODO: load data and split into train and test. (-)
    in_file = np.load(args.input)
    data = in_file[in_file.files[0]]
    
    feats = data[:,:-1]
    label = data[:,-1].ravel()
    X_train, y_train, X_test, y_test = train_test_split(feats, label, test_size = 0.2, 
                                                        shuffle=True, random_state=401)
    # TODO : complete each classification experiment, in sequence. (-)
    iBest = class31(args.output_dir, x_train, x_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, x_train, x_test, y_train, y_test, iBest)
    class33(args.output_dir, x_train, x_test, y_train, y_test, iBest, X_1k, y_1k)
    # ALL DATA?@55
    class34(args.output_dir, x_train, x_test, y_train, y_test, iBest)

