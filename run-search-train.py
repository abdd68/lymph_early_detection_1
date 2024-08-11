import os
import pandas as pd 
import numpy as np

from expert_tree import get_expert_tree_results, Expert_Tree
from wrapper import best_first_search_mg

import argparse


'''Define Argument Parser'''
parser = argparse.ArgumentParser()

parser.add_argument('--DataPth',  type=str, default='./data/result_data/split_train_Oct21_stratified.csv', 
                    help='path to data directory')
parser.add_argument('--estimator',  type=str, default='gbt', 
                    help='which estimator to use')  
parser.add_argument('--learning-rate',  type=float, default = 0.1,
                    help='learning rate for gbt/xgb')
parser.add_argument('--max-depth',  type=int, default = 2,
                    help='max-depth for gbt/xgb')
parser.add_argument('--n-estimators',  type=int, default = 70,
                    help='n-estimators for gbt/xgb')
parser.add_argument('--patience',  type=int, default = 100,
                    help='patients for best first search') 
parser.add_argument('--num-repeated',  type=int, default = 1,
                    help='num-repeated for CV') 
parser.add_argument('--mega-step',  type=int, default = 1,
                    help='if use mega step') 
parser.add_argument('--shuffle',  type=int, default = 1,
                    help='if shuffle the data') 
parser.add_argument('--eps',  type=float, default = 1e-6,
                    help='eps for wrapper')
args = parser.parse_args()

assert args.estimator in ['xgb', 'gbt'], f"{args.estimator} not recognized, please select in {['xgb', 'gbt']}"
                    


'''generate random key for shuffling'''
random_state = (None, np.random.randint(0, high=2000, size=None, dtype=int))[args.shuffle]
shuffle = args.shuffle > 0
print(f'shuffle: {shuffle}, random_state: {random_state}')


'''Read in data'''
# read in dataset 
DATA_PATH = args.DataPth
data = pd.read_csv(DATA_PATH)
data = data.drop(columns=['Unnamed: 0', 'Username'])

print('DATAPATH', DATA_PATH)

# print shape and columns
print(f"data shape: {data.iloc[:,:-1].shape}")
print("columns:")
print(data.iloc[:,:-1].columns.values)

'''Generate X and y'''
X = data.iloc[:,:-1].values
y = data['3class_label'].values
print(f"X shape: {X.shape}")
print(f"y length: {len(y)}")
print(np.unique(y, return_counts=True))
data = data.iloc[:,:-1]


'''Define Evaluator'''
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate
def cross_validate_custom(X, y, num_repeated, estimator):
    n_splits = 8
    if num_repeated > 1:
        print("num_repeated is not 1")
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=num_repeated, random_state=random_state)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        #     scores = cross_validate(estimator, X, y, scoring='accuracy', n_jobs=-1, cv=skf, verbose=0, 
        #                             return_estimator=True, return_train_score=True)
    w = [0.15913545, 0.54254953, 0.29831502]
    test_scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train, y_train, sample_weight=[w[i] for i in y_train])
        score = estimator.score(X_test, y_test, sample_weight=[w[i] for i in y_test])
        test_scores.append(score)
    # return np.mean(scores['test_score']), np.mean(scores['train_score'])
    # print(test_scores)
    return np.mean(test_scores), 0

'''Define Estimator'''
if args.estimator == 'gbt':
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier
else:
    from xgboost import XGBClassifier
    model = XGBClassifier

params = {'learning_rate': args.learning_rate, 'max_depth': args.max_depth, 'n_estimators': args.n_estimators}
estimator = model(**params)

'''Run Search'''  

evaluator = cross_validate_custom
print(f"patience for searching: {args.patience}")
print(f"eps for searching: {args.eps}")
print(f"num of repeat for CV: {args.num_repeated}")
print("estimator:")
print(estimator)
print(f"X shape: {X.shape}")
print(f"y shape: {len(y)}")
print(f"if mega step: {args.mega_step}")

best_feature_set_accu, _, best_feature_set, record = best_first_search_mg(X, y, args.patience, estimator, evaluator, \
                     num_repeated=args.num_repeated, verbose=False, mega_step=args.mega_step, eps=args.eps)

print("best feature set accuracy")
print(best_feature_set_accu)
print("selected one hot")
print(best_feature_set)
print("selected feature: {}".format(data.columns.values[best_feature_set != 0]))


'''Get Result with Wrapper'''
#1) accu + std
X_selected = X[:, best_feature_set.nonzero()[0]]
print(f'X_selected shape: {X_selected.shape}')
n_splits = 8
skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
scores = cross_validate(estimator, X_selected, y, scoring='accuracy', n_jobs=-1, cv=skf, verbose=0, return_estimator=True, return_train_score=True)
print("With Wrapper:")
print("test  mean %f, test std: %f, train mean %f, train std %f" % (np.mean(scores['test_score']), np.std(scores['test_score']), np.mean(scores['train_score']), np.std(scores['train_score'])) )

# #2) feature importance:
# feature_importance_arrays_norm = np.sum([x.feature_importances_ for x in scores['estimator']], axis=0) / len(scores['estimator'])
# feat_names = data.columns[best_feature_set.nonzero()[0]].values
# print("sorted, feature name; importance")
# assert len(feat_names) == len(feature_importance_arrays_norm), "importance and feature number not equal"
# feature_weight_pair = sorted(zip(feat_names, feature_importance_arrays_norm), key=lambda pair : pair[1], reverse=True)
# print(feature_weight_pair)

'''Get Result without Wrapper'''
#1) accu + std
n_splits = 8
skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
scores = cross_validate(estimator, X, y, scoring='accuracy', n_jobs=-1, cv=skf, verbose=0, return_estimator=True, return_train_score=True)
print("Without Wrapper:")
print("test  mean %f, test std: %f, train mean %f, train std %f" % (np.mean(scores['test_score']), np.std(scores['test_score']), np.mean(scores['train_score']), np.std(scores['train_score'])) )
# #2) feature importance:
# feature_importance_arrays_norm = np.sum([x.feature_importances_ for x in scores['estimator']], axis=0) / len(scores['estimator'])
# feat_names = data.columns.values
# print("sorted, feature name; importance")
# assert len(feat_names) == len(feature_importance_arrays_norm), "importance and feature number not equal"
# feature_weight_pair = sorted(zip(feat_names, feature_importance_arrays_norm), key=lambda pair : pair[1], reverse=True)
# print(feature_weight_pair)


                    



    

