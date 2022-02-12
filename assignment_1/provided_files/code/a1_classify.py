#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

## Kopal Garg, 1003063221
#!/usr/bin/env python

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  
from scipy import stats

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    # deal with cases where the denominator could be 0
    if np.sum(C) == 0:
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
    # keep track of best model index
    iBest = None
    # keep track of best model acc
    best_acc = 0
    
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:

        for i, model in enumerate(listModels):
            # get classifier name
            classifier_name = model.__class__.__name__
            # fit on full training set
            model.fit(X_train, y_train)
            # predictions on full test set
            y_test_pred = model.predict(X_test)
            # confusion matrix using test pred  y and test y
            conf_matrix = confusion_matrix(y_test, y_test_pred)
            # accuracy, precision and recall from conf matrix
            accuracies = accuracy(conf_matrix)
            precisions = precision(conf_matrix)
            recalls = recall(conf_matrix)

            outf.write(f'Results for {classifier_name}:\n')  
            outf.write(f'\tAccuracy: {accuracies:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recalls]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precisions]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
            # if curr acc > best acc, replace best acc with curr acc
            if accuracies > best_acc:
                iBest = i 
                best_acc = accuracies
            # to keep track of training progress
            print("Trained", classifier_name)
            print(i, accuracies)

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
    # get the best index from 3.1. 
    iBest = iBest-1 # indexing differently so subtract 1
    # get the best model using the index
    best_model = listModels[iBest]
    # different training set sizes to try
    dataset_sizes = [1000, 5000, 10000, 20000]
    accuracies = []
    # loop over all different training sizes 
    for size in dataset_sizes:
        # index random ints of size size
        index_ = np.random.choice(a  = X_train.shape[0], size = size)
        # use these indices to subset the training set
        X_sub = X_train[index_]
        y_sub = y_train[index_]
        # fit the best model on subset of the training set
        best_model.fit(X_sub, y_sub)
        # make predictions on full test set
        y_pred_test = best_model.predict(X_test)
        # get conf matrix of y test pred and y test
        C = confusion_matrix(y_test, y_pred_test)
        # compute accuracy
        accuracies.append(accuracy(C))
        # if size of the set if 1000, set the sets we're going to return to the current training subset
        if size==1000:
            X_1k = X_sub
            y_1k = y_sub
        
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        #     outf.write(f'{num_train}: {acc:.4f}\n'))
        for i, size in enumerate(dataset_sizes):
            outf.write(f'{size}: {accuracies[i]:.4f}\n')

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
    best_model = listModels[i-1]
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment this, so it writes them to outf.
        # for each number of features k_feat, write the p-values for that number of features:
        for k_feat in [5,50]:
            # best k features
            sel = SelectKBest(score_func=f_classif, k = k_feat)
            # fit and transform the full train 32k set
            X_new  = sel.fit_transform(X_train, y_train)
            # pvalue of the features 
            p_values = sel.pvalues_
            # arg sort the p values and using indices get the pvalues of top k features 
            p_values = p_values[p_values.argsort()] # top k p-values
            # output to the main file
            outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')

        accuracies = [] # [0] would be 1k and [1] would be for 32k
        
        for i in ['1k','full']:
            if i == '1k': X = X_1k; y = y_1k
            else: X = X_train; y = y_train
            # best 5 features
            sel = SelectKBest(score_func=f_classif, k = 5)
            # fit and transform the 1k and 32k sets
            X_new = sel.fit_transform(X, y)
            # transform the full test set
            X_test_new = sel.transform(X_test)
            # fit the best model from part 1
            best_model.fit(X_new, y)
            # predict the y value using the transformed test set
            y_test_pred = best_model.predict(X_test_new)
            # build confusion matrix using test pred y and test y values
            conf_matrix = confusion_matrix(y_test, y_test_pred)
            # get the acc from the conf matrix
            accuracies.append(accuracy(conf_matrix))
        accuracy_1k =accuracies[0]; accuracy_full = accuracies[1]
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')

        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        # feature intersection
        k_feat = 5
        ix_1k = SelectKBest(score_func = f_classif,
                            k = k_feat).fit(X_1k, y_1k).get_support(indices= True)
        ix_full = SelectKBest(score_func = f_classif,
                            k = k_feat).fit(X_train, y_train).get_support(indices= True)
        feature_intersection = set(ix_1k) & set(ix_full)
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        # top k = 5 feature indices extracted for full dataset
        top_5 = set(ix_full) # write as a set 
        outf.write(f'Top-5 at higher: {top_5}\n')
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
    
    # since using the full dataset, vertically stack the train and test sets that we initially did a split on
    X = np.vstack((X_train, X_test))
    # do the same for the lables (making sure they're stacked in the same order)
    y = np.concatenate((y_train, y_test), axis= 0)
    # k
    k=6
    # do a random split but set the random state to 401 for reproducibility
    kfold = KFold(k, shuffle = True, random_state=401)
    # matrix of acuracies  (col=classifier, row=fold)
    accuracy_models = np.zeros((k, len(listModels)))
    
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        #     outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')
        for i, model in enumerate(listModels):
            # taking random kfold splits 
            j=0 # j is fold, i is model
            for test, train in kfold.split(X):
                X_train = X[train];X_test = X[test]
                y_train = y[train];y_test = y[test]
                # fit model on the training split
                model.fit(X_train, y_train)
                # test predictions on the testing split
                y_test_pred = model.predict(X_test)
                # create a conf matrix, and get the accuracy
                C = confusion_matrix(y_test, y_test_pred)
                acc = accuracy(C)
                accuracy_models[j,i] = acc
                j+=1 # increment for split
            # compute the mean acc using the top list we appened all acc to
        
        for j in accuracy_models:
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in j]}\n')
        import pdb; pdb.set_trace()
        
        # p-value calculations
        # S = stats.ttest_rel(a, b)
        p_values = []
        iBest =i-1
        for ind, model in enumerate(listModels):
            if not listModels[iBest]==model:
                S = stats.ttest_rel(accuracy_models[:, ind], accuracy_models[:, iBest])
                p_values.append(S.pvalue)
        outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')
        pass
        

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
    X_train, X_test, y_train, y_test = train_test_split(feats, label, test_size = 0.2, 
                                                        shuffle=True, random_state=401)
    # TODO : complete each classification experiment, in sequence. (-)
    #iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    #print(iBest)
    iBest = 5
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)

    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)

