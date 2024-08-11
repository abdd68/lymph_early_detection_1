import os 
import time
import timeit
import pandas as pd 
import numpy as np
import warnings

from copy import deepcopy

def bool_reverse(int_value):
    '''
    reverse 0 and 1
    Parameters
    ----------
    int_value: int, input bin
    Returns
    -------
    _: int, reversed bin
    '''
    return (int_value == 0).astype(np.int16)

def in_list(target, search_list, where=False):
    '''
    check if an array of feature is in a specific list or not
    Parameters
    ----------
    target: ndarray of int
    search_list: list of ndarray of int, OPEN or CLOSE
    where: bool, if return an index or bool
    Returns
    -------
    _: bool, if the target is contained in search list
    _: int, index of the target is contained in search list
    '''
    if not where:

        for item in search_list:
            if (target == item).all():
                return True

        return False

    else:

        for item_idx, item in enumerate(search_list):
            if (target == item).all():
                return item_idx

        return -1

def best_first_search_mg(X, y, patience, estimator, evaluator, num_repeated=1, verbose=False, mega_step=False, eps=1e-6):
    '''more greedy best first search feature selection
    Parameters
    ----------
    X: ndarray, data
    y: ndarray, label
    patience: int, patience for termination, will terminate the searching if no significant improvement after consecutive #patience iteration
    estimator: sklearn estimator, with parameters initilaized
    evaluator: function to evaludate return accuracy
    num_repeated: int, number of repeated cv
    Returns
    -------
    best_set_acc: float, best accuracy for validation
    best_set_acc_tr: float corresponding training accuracy
    best_set: ndarray of int {0,1}, the binary selected feature
    '''

    # threshold to be considered as significant improvement
    # eps = 1e-5 # 1e-4
    # get total feature vector dimension 
    _ , num_features = X.shape
    # 1. put init state on open list, close list = empty, best = initial
    # TODO for Junbo: Change the list to priority queue 
    open_list = []
    open_acc_list = []
    # TODO for Junbo: Change the close list to hashset or hashmap
    close_list = [] # init the close
    close_acc_list = [] # init the close accuracy
    # push the root vertex to queue
    open_list.append(np.zeros([num_features]).astype(np.int16)) # init the open
    open_acc_list.append(0) # init open accuracy
    # record of current best feature
    best_set = np.zeros([num_features]).astype(np.int16) # init the best
    best_set_acc = 0
    # cross score
    current_compound_ts = 0
    # current accuracy 
    current_acc = 0
    # start searching 
    k = 0
    while 1: 
        start_time = time.time()
        # print(k, mega_step)
        if (current_compound_ts > current_acc + eps) and mega_step:
            # if the compound is better than current
            current = current_compound
            current_acc = current_compound_ts
        else:
            # 2. let v = argmax(w in open)(open), get state from open with maximal f(w)
            current_idx = np.argmax(open_acc_list) # get the best v in open
            current = open_list[current_idx] # get v
            current_acc = open_acc_list[current_idx] # get v_acc

            # 3 remove v from open, add v to close
            _ = open_list.pop(current_idx) # remove v from open
            _ = open_acc_list.pop(current_idx) # remove v_acc from open_acc

        close_list.append(current) # add v to close
        close_acc_list.append(current_acc) # add v acc to close_acc

        # 4. If current acc - eps > best acc, best = current
        if current_acc > best_set_acc + eps:
            k = 0 # reset the patience counter
            # best is updated by current!
            best_set = current
            best_set_acc = current_acc

            print('local best = {0:.4f}, features = {1}'.format(best_set_acc, best_set.nonzero()[0])) # adding
        else:
            k += 1 # else counter + 1

        if k >= patience:

            history = {'features' : close_list, 'accuracy': close_acc_list}

            return best_set_acc, 0, best_set, history

        # 5. expand the child of current

        local_list = []
        local_acc_list = []

        for f, feature_state in enumerate(current):

            current_temp = deepcopy(current)
            current_temp[f] = bool_reverse(feature_state) # get the child of current

            # 6. for child not in open or close, evaluate and add to open

            if (not in_list(current_temp, open_list)) and (not in_list(current_temp, close_list)):

                current_temp_idx = current_temp.nonzero()[0]

                if current_temp_idx.shape[0] == 0:
                    current_temp_ts, current_temp_tr = (-1e8, -1e8)
                else:
                    current_temp_ts, current_temp_tr = evaluator(X[:, current_temp_idx], y, num_repeated, estimator)
                    print(current_temp_idx)
                    print("accuracy: {}".format(current_temp_ts))
                local_list.append(current_temp)
                local_acc_list.append(current_temp_ts)

                open_list.append(current_temp)
                open_acc_list.append(current_temp_ts)

            else:
                pass

        local_sort = np.argsort(local_acc_list)
        # calculate the current compound list
        current_compound = local_list[local_sort[-1]] + local_list[local_sort[-2]] - current

        if (not in_list(current_compound, open_list)) and (not in_list(current_compound, close_list)):

            # if white, add to current list
            # if np.sum(current_compound) == np.sum(current):
            current_compound_idx = current_compound.nonzero()[0]
            current_compound_ts, _ = evaluator(X[:, current_compound_idx], y, num_repeated, estimator)
            print(current_compound_idx)
            print("accuracy: {}".format(current_compound_ts)) 
            if not mega_step:
                open_list.append(current_compound)
                open_acc_list.append(current_compound_ts) 
            pass
        # 6. if best set/acc change in last k expansion, goto 2
        print("K: {}".format(k))
        end_time = time.time()
        print("iteration time=", end_time - start_time)