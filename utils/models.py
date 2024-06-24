from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.metrics import classification_report , r2_score, accuracy_score

import numpy as np


## Train a Random Forest Classifier 
def train_and_predict_using_random_forest_classifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42) 
    model.fit(X_train, y_train) 
    
    print("Evaluation for RandomForestsClassifier")
    # r2-score for train data
    y_pred_train = model.predict(X_train)
    print("r2 score on train data: ", r2_score(y_train, y_pred_train))
                
    # Evaluate the model 
    y_pred = model.predict(X_test) 
    print("r2 score on test data: ", r2_score(y_test, y_pred))
    print('The accuracy of the model on test data:\t',accuracy_score(y_true=y_test,y_pred=y_pred))
    print("classification report on test data:\n", classification_report(y_true=y_test, y_pred=y_pred))
    return model 

# Train a XGBoost Classifier 
def train_and_predict_using_xgb_classifier(X_train, X_test, y_train, y_test):
    model = XGBClassifier(n_estimators=100, random_state=42) 
    model.fit(X_train, y_train) 
    
    print("Evaluation for XGBoostClassifier")
    # r2-score for train data
    y_pred_train = model.predict(X_train)
    print("r2 score on train data: ", r2_score(y_train, y_pred_train))
                
    # Evaluate the model 
    y_pred = model.predict(X_test) 
    print("r2 score on test data: ", r2_score(y_test, y_pred))
    print('The accuracy of the model on test data:\t',accuracy_score(y_true=y_test,y_pred=y_pred))
    print("classification report on test data:\n", classification_report(y_true=y_test, y_pred=y_pred))
    return model 

## Train a LightGBM Classifier 
def train_and_predict_using_lightgbm_classifier(X_train, X_test, y_train, y_test):
    ## specify your configurations as a dict
    lgb_params = {
        'task': 'train',
        'boosting_type': 'goss',
        'objective': 'binary',
        'metric':'binary_logloss',
        # 'metric': {'l2', 'auc'},
        'num_leaves': 50,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbose': None,
        'num_iteration':100,
        'num_threads':7,
        'max_depth':12,
        'min_data_in_leaf':100,
        'alpha':0.5
    }

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test)
    # training the lightgbm model
    model = lgb.train(lgb_params,lgb_train,num_boost_round=20,valid_sets=lgb_eval)
    
    print("Evaluation for LightGBM")
    # r2-score for train data
    y_pred_train = model.predict(X_train)
    print("r2 score on train data: ", r2_score(y_train, y_pred_train))
                
    # Evaluate the model 
    y_pred = model.predict(X_test) 
    print("r2 score on test data: ", r2_score(y_test, y_pred))
    # print('The accuracy of the model on test data:\t',accuracy_score(y_true=y_test,y_pred=y_pred))
    # print("classification report on test data:\n", classification_report(y_true=y_test, y_pred=y_pred))
    return model 

# TODO: TRY to use 1D-CNN as classifier 
# def train_and_predict_using_one_d_cnn(X_train, X_test, y_train, y_test):

# LightGBM directly returns probability for class 1 by default 
def lightgbm_prediction_probability(model, data):
    return np.array(list(zip(1-model.predict(data),model.predict(data))))

# RandomForestClassifier/ XGBoostClassifier directly returns probability for class 1 by default 
def classifier_prediction_probability(model, data):
    return model.predict_proba(data).astype(float)