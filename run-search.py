import os
import pandas as pd 
import numpy as np

from expert_tree import get_expert_tree_results, Expert_Tree
from wrapper import best_first_search_mg

import argparse
import warnings
import pickle
warnings.filterwarnings("ignore")
'''Define Argument Parser'''
parser = argparse.ArgumentParser()

parser.add_argument('--DataPth',  type=str, default='./data/feature_selection_preprocessed_data.csv', 
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
parser.add_argument('--replace3by1',  type=int, default = 1,
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

delete_rows = ['A001', 'A003', 'A004', 'A005', 'A015', 'A016',
               'A018', 'A025', 'A026', 'A029', 'A031', 'A032',
               'A035', 'A036', 'A038', 'A042', 'A046', 'A055',
               'T010', 'T013', 'T014', 'T016', 'T018', 'T019',
               'T021', 'T026', 'T029', 'T030', 'T036', 'T038',
               'T040', 'T043', 'T044', 'T054', 'T055', 'T056',
               'T057'] 

# read in dataset 
DATA_PATH = args.DataPth
data = pd.read_csv(DATA_PATH)

# drop list of subjects
data = data[~data.Username.isin(delete_rows)]
# drop data within a 6 months
data = data[data.TIME_LAPSE >=0.5]
data = data.drop(columns=['Username', 'Unnamed: 0'] if 'feature_selection' in DATA_PATH else 'Username')

# add log of time elapsed
data['TIME_LAPSE_Log'] = np.log(data['TIME_LAPSE'])

# the data_pre contain '#DEV/0!' or "" to make the coloumn string instead of float
# and these dirty element would cause error in .astype operation
# here convert "" or  '#DEV/0!' to nan
data = data.apply(lambda x: pd.to_numeric(x, errors='coerce'))
# drop rows containing nan
data = data.dropna()

# print shape and columns
print(f"data shape: {data.shape}")
print("columns:")
print(data.columns.values)

'''Generate Label'''

# generate 5 labels 
# for i, row in data.iterrows():
#    tag = Expert_Tree(row).run()

# generate 3 labels
labels_3, labelNames_3, flags_3 = get_expert_tree_results(data, is_Kinect = False, class_number = 3)
# generate 2 labels 
# labels_2, labelNames_2, flags_2 = get_expert_tree_results(data, three_class=False)
                    
'''Drop LVC and TIME ELAPSE'''

# drop LVC and time time elapse
data = data.drop(columns=['LVC', 'TIME_LAPSE', 'fluid_total'])
if args.replace3by1:
    data = data.drop(columns=['SLNB_Removed_LN', 'ALND_Removed_LN', 'SLNB_ALND_Removed'])
else:
    data = data.drop(columns=['Number_nodes'])
print('Drop LVC and TIME ELAPSE')
# print shape and columns
print(f"data shape: {data.shape}")
print("columns:")
print(data.columns.values)
'''Define Evaluator'''

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate

def cross_validate_custom(X, y, num_repeated, estimator):
    n_splits = 8
    if num_repeated > 1:
        print("num_repeated is not 1")
        skf = RepeatedStratifiedKFold(n_splits=n_splits, num_repeated=num_repeated)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores = cross_validate(estimator, X, y, scoring='accuracy', n_jobs=-1, cv=skf, verbose=0, 
                            return_estimator=True, return_train_score=True)
    return np.mean(scores['test_score']), np.mean(scores['train_score'])

'''Define Estimator'''
if args.estimator == 'gbt':
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier
else:
    from xgboost import XGBClassifier
    model = XGBClassifier

params = {'learning_rate': args.learning_rate, 'max_depth': args.max_depth, 'n_estimators': args.n_estimators}
estimator = model(**params)

'''Generate Data for model and Search'''

# get features for model  
X = data.values
y = labels_3
print(f"X shape: {X.shape}")
print(f"y length: {len(y)}")

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
print(f"selected feature: {str(data.columns.values[best_feature_set != 0]).replace(' ',', ')}")
'''Get Result with Wrapper'''
#1) accu + std
X_selected = X[:, best_feature_set.nonzero()[0]]
print(f'X_selected shape: {X_selected.shape}')
n_splits = 8
skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
scores = cross_validate(estimator, X_selected, y, scoring='accuracy', n_jobs=-1, cv=skf, verbose=0, return_estimator=True, return_train_score=True)
print("With Wrapper:")
print("test  mean %f, test std: %f, train mean %f, train std %f" % (np.mean(scores['test_score']), np.std(scores['test_score']), np.mean(scores['train_score']), np.std(scores['train_score'])) )

#2) feature importance:
feature_importance_arrays_norm = np.sum([x.feature_importances_ for x in scores['estimator']], axis=0) / len(scores['estimator'])
feat_names = data.columns[best_feature_set.nonzero()[0]].values
print("sorted, feature name; importance")
assert len(feat_names) == len(feature_importance_arrays_norm), "importance and feature number not equal"
feature_weight_pair = sorted(zip(feat_names, feature_importance_arrays_norm), key=lambda pair : pair[1], reverse=True)
print(feature_weight_pair)

with open('models/GBT.pkl', 'wb') as f:
    pickle.dump(scores['estimator'][0], f)

print("Model saved to 'models/GBT.pkl'")

'''Get Result without Wrapper'''
#1) accu + std
n_splits = 8
skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
scores = cross_validate(estimator, X, y, scoring='accuracy', n_jobs=-1, cv=skf, verbose=0, return_estimator=True, return_train_score=True)
print("Without Wrapper:")
print("test  mean %f, test std: %f, train mean %f, train std %f" % (np.mean(scores['test_score']), np.std(scores['test_score']), np.mean(scores['train_score']), np.std(scores['train_score'])) )
#2) feature importance:
feature_importance_arrays_norm = np.sum([x.feature_importances_ for x in scores['estimator']], axis=0) / len(scores['estimator'])
feat_names = data.columns.values
print("sorted, feature name; importance")
assert len(feat_names) == len(feature_importance_arrays_norm), "importance and feature number not equal"
feature_weight_pair = sorted(zip(feat_names, feature_importance_arrays_norm), key=lambda pair : pair[1], reverse=True)
print(feature_weight_pair)





    

