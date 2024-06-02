import numpy as np

# from sklearn.cluster import MiniBatchKMeans as MiniKM
from imblearn.over_sampling import SVMSMOTE, SMOTENC
from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import SelectFromModel 
from sklearn.ensemble import RandomForestClassifier 
import lightgbm as lgb
from sklearn.metrics import classification_report , r2_score, accuracy_score

import shap

from lime import lime_tabular

import matplotlib.pyplot as plt

from utils import constants, preprocess, utils

# this is required as LIME requires class probabilities in case of classification

# LightGBM directly returns probability for class 1 by default 
# def prob(data):
#     return np.array(list(zip(1-model.predict(data),model.predict(data))))

# RFClassifier directly returns probability for class 1 by default 
def prob(data):
    return model.predict_proba(data).astype(float)

X_train, X_test, y_train, y_test = preprocess.get_train_and_test_data()

# K-Means SMOTE
# clf = MiniKM(n_clusters = 100, random_state = 1234)
# clf = MiniKM(n_clusters = 10, random_state = 1234)
# smote = KMeansSMOTE(random_state = 1234,k_neighbors=10,
                            # kmeans_estimator = clf,cluster_balance_threshold = "auto")
smote = SVMSMOTE(random_state = 42, sampling_strategy=0.85, k_neighbors=10)
X_resampled_train, y_resampled_train = smote.fit_resample(X_train, y_train)

# Fit the model in LASSO
lassoreg = LassoCV(cv=10, max_iter=5000, tol=1e-2) #(alpha=0.01, max_iter=1000)
# lassoreg = Lasso(alpha=0.02, max_iter=1000)
lassoreg.fit(X_resampled_train,y_resampled_train.values.ravel())

# Feature selection 
sfm = SelectFromModel(lassoreg, prefit=True) 
X_train_selected = sfm.transform(X_resampled_train) 
X_test_selected = sfm.transform(X_test) 

selected_feature_indices = np.where(sfm.get_support())[0] 
selected_features = X_resampled_train.columns[selected_feature_indices] 
coefficients = lassoreg.coef_ 
print("Selected Features:", selected_features) 
print("Feature Coefficients:", coefficients) 

# Train a Random Forest Classifier using the selected features 
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train_selected, y_resampled_train) 

# # specify your configurations as a dict
# lgb_params = {
#     'task': 'train',
#     'boosting_type': 'goss',
#     'objective': 'binary',
#     'metric':'binary_logloss',
#     'metric': {'l2', 'auc'},
#     'num_leaves': 50,
#     'learning_rate': 0.1,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'verbose': None,
#     'num_iteration':100,
#     'num_threads':7,
#     'max_depth':12,
#     'min_data_in_leaf':100,
#     'alpha':0.5}

# # create dataset for lightgbm
# lgb_train = lgb.Dataset(X_train_selected, y_resampled_train)
# lgb_eval = lgb.Dataset(X_test_selected, y_test)
# # training the lightgbm model
# model = lgb.train(lgb_params,lgb_train,num_boost_round=20,valid_sets=lgb_eval)


# r2-score for train data
y_pred_train = model.predict(X_train_selected)
print("r2 score on train data: ", r2_score(y_resampled_train, y_pred_train))
               
# Evaluate the model 
y_pred = model.predict(X_test_selected) 
print("r2 score on test data: ", r2_score(y_test, y_pred))
print('The accuracy of the Random Forests model is :\t',accuracy_score(y_true=y_test,y_pred=y_pred))
print("classfication report on test data:\n", classification_report(y_true=y_test, y_pred=y_pred)) 

# LIME Instance
# training_data=np.array(X_resampled_train)
training_data=np.array(X_train_selected)

# feature_names = X_resampled_train.columns,
feature_names = np.array(selected_features)
explainer = lime_tabular.LimeTabularExplainer(
    training_data=training_data,
    feature_names=feature_names,
    training_labels=y_train,
    class_names=["POSITIVE", "NEGETIVE"],
    mode="classification",
    kernel_width=10,
    verbose=True,)

# asking for explanation for LIME model
i = 10
print(X_test_selected[i])
print(y_test[i])
# choosen_instance = X_test.iloc[i]
choosen_instance = X_test_selected[i]

# predict_fn=model.predict
predict_fn=prob
exp = explainer.explain_instance(
        data_row=choosen_instance, 
        predict_fn=predict_fn)
        # num_features=len(selected_features))

print(exp.as_list())
# exp.show_in_notebook(show_all=True)
exp.as_pyplot_figure()
exp.save_to_file(constants.LIME_EXPLANATION_HTML)
# exp.visualize_instance_html(show_table=True)

# Initialize SHAP
shapExplainer = shap.Explainer(model=model, feature_names=feature_names)

# Evaluate SHAP values for entire test set
shap_values = shapExplainer.shap_values(X_test_selected)
# shap_values = utils.global_shap_importance(model=model, X=X_test_selected)

# Plot SHAP summry 
shap.summary_plot(shap_values, feature_names=feature_names)
# shap.summary_plot(shap_values[1], X_test_selected, feature_names=feature_names)
# shap.plots.force(shapExplainer.expected_value[0], shap_values[0][0,:], X_test_selected[0, :], matplotlib = True, feature_names=feature_names)

plt.show() 
