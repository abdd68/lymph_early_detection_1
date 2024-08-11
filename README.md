# Lymphedema-Detection-and-Prediction-for-Breast-Cancer-Survivors
This repository is the demonstration to early-detect the Lymphedema for breast cancer survivers. 

To implement the result, first run:

```python
conda env create -f python37.yaml
```

Example: 

```python
python run-search.py --DataPth ./data/feature_selection_preprocessed_data.csv --estimator gbt --learning-rate 0.1 \
--max-depth 2 --n-estimators 70 --patience 5              
```