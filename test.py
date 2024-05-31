import numpy as np

# from sklearn.cluster import MiniBatchKMeans as MiniKM
from imblearn.over_sampling import SVMSMOTE, SMOTENC
from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import SelectFromModel 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report , r2_score, accuracy_score

from lime import lime_tabular

import matplotlib.pyplot as plt

from utils import preprocess

# def predict_fn_rf(x):
#     return model.predict_proba(x).astype(float)

X_train, X_test, y_train, y_test = preprocess.get_train_and_test_data()

# K-Means SMOTE
# clf = MiniKM(n_clusters = 100, random_state = 1234)
# clf = MiniKM(n_clusters = 10, random_state = 1234)
# smote = KMeansSMOTE(random_state = 1234,k_neighbors=10,
                            # kmeans_estimator = clf,cluster_balance_threshold = "auto")
smote = SVMSMOTE(random_state = 42, sampling_strategy=0.85, k_neighbors=10)
X_resampled_train, y_resampled_train = smote.fit_resample(X_train, y_train)


# Fit the model in LASSO
# lassoreg = LassoCV(cv=5, max_iter=100) #(alpha=0.01, max_iter=1000)
lassoreg = Lasso(alpha=0.02, max_iter=1000)
lassoreg.fit(X_resampled_train,y_resampled_train.values.ravel())

# Feature selection 
sfm = SelectFromModel(lassoreg, prefit=True) 
X_train_selected = sfm.transform(X_resampled_train) 
X_test_selected = sfm.transform(X_test) 

selected_feature_indices = np.where(sfm.get_support())[0] 
selected_features = X_test.columns[selected_feature_indices] 
coefficients = lassoreg.coef_ 
print("Selected Features:", selected_features) 
print("Feature Coefficients:", coefficients) 

# Train a Random Forest Classifier using the selected features 
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train_selected, y_resampled_train) 

# r2-score for train data
y_pred_train = model.predict(X_train_selected)
print("r2 score on test data: ", r2_score(y_resampled_train, y_pred_train))
               
# Evaluate the model 
y_pred = model.predict(X_test_selected) 
print('The accuracy of the Random Forests model is :\t',accuracy_score(y_pred,y_test))
print("classfication report on test data:\n", classification_report(y_test, y_pred)) 

# LIME Instance
# explainer = lime_tabular.LimeTabularExplainer(
#     X,feature_names = X_train.columns,class_names=['Will Die','Will Survive'],kernel_width=5)
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_selected),
    feature_names=selected_features,
    # training_labels=y_test["SC2_PCR"],
    class_names=["POS", "NEG"],
    mode="classification",
    kernel_width=5,
    verbose=True,)

# asking for explanation for LIME model
i = 10
# print(X_test_selected[i])
choosen_instance = X_test_selected[i]

exp = explainer.explain_instance(
        data_row=choosen_instance, 
        predict_fn=model.predict_proba,
        num_features=10)
        # num_features=len(selected_features))

print(exp.as_list())
# exp.show_in_notebook(show_all=False)
exp.as_pyplot_figure()
# exp.visualize_instance_html(show_table=True)
plt.show() 
