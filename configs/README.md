# Config Settings

## Sampling Strategy

    1. default is `none`, model uses data as-is and try to predict
    2. strategy select as `svm`, data will be passes to SVM-SMOTE to make it more balanced and then model will predict on balance data.
    3. strategy select as `gan` data will be passes to GAN model to make it more balanced and then model will predict on balance data.

## Feature Selection Strategy

    1. default is `none`, model will be used all the feature from dataset.
    2. strategy selected as `lasso`, LassoCV will be used to extract important features from the dataset and model will use only extracted features to predict.

## Classification Strategy

    1. default is `rfc` - RandomForestClassifier, 
    2. `xgb` - XGBClassifier
    3. `lgbm` - LightGBM
    4. `cnn` - 1D-CNN

### Example of default config

    ```JSON
       {
          "sampling" : "none",
          "selection" : "none",
          "classification": "rfc",
          "explanation": {
              "lime" : true,
              "lime_plot": false,
              "shap": true,
              "shap_bar_plot": false
          }
       }   
    ```
