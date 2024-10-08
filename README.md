# Lymphedema-Detection-and-Prediction-for-Breast-Cancer-Survivors
This repository is the demonstration to early-detect the Lymphedema for breast cancer survivers. 

To implement the result please follow the steps:

**Step 1:** Create the conda environment using:

```python
conda env create -f python37.yaml
```
**Step 2:** To reimplement three methods in the paper, i.e. Gradient Boosting Tree (including eXtreme Gradient Boosting), Decision Tree and Artificial Neural Network, follow the instructions:

## Gradient Boosting Tree (GBT)

```python
python run-search.py --DataPth ./data/feature_selection_preprocessed_data.csv --estimator gbt --learning-rate 0.1 \
--max-depth 2 --n-estimators 70 --patience 50              
```

## eXtreme Gradient Boosting (XGB)

```python
python run-search.py --DataPth ./data/feature_selection_preprocessed_data.csv --estimator xgb --learning-rate 0.1 \
--max-depth 2 --n-estimators 70 --patience 50          
```

## Decision Tree (DT)

Please directly run **DT.ipynb**.

## Artificial Neural Network (ANN)

Please directly run **MLP.ipynb**.

**Additional Infomation:** We offer a software based on the gbt method of this repository, for patients to early-detect the Lymphedema in a convenient and promotive manner. Please visit our [Lymphedema detection website](https://optimallymph.org/) to access the software.