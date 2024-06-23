from random import randint
import numpy as np

import shap

import matplotlib.pyplot as plt

from utils import explanations, models, preprocess, feature_selection as fs, sampling

from configs import config
from utils.constants import (
    SHAP_EXPLANATION_GLOBAL_PLOT,
    SHAP_EXPLANATION_BAR_PLOT,
    SHAP_EXPLANATION_CLUSTER_PLOT,
    SHAP_EXPLANATION_SCATTER_PLOT
)



# load data preprocess data
data = None
data = preprocess.load_data()


# load and parse config.json
cfg = config.load_config()

# check for over sampling flag and try to balance data
match cfg.sampling:
    case "svm":
        # balance data using SVM
        X, y = sampling.sample_data_with_svm_smote(X=data.drop("SC2_PCR", axis=1), y=data["SC2_PCR"])
    case "gan":
        # balance data using GAN
        X, y = sampling.sample_data_with_svm_smote(X=data.drop("SC2_PCR", axis=1), y=data["SC2_PCR"])
    case "none":
        # do not try to balance, use as is
        X = data.drop("SC2_PCR", axis=1)
        y = data["SC2_PCR"]
    
# check for feature selection flag and try to select features
match cfg.selection:
    case "lasso":
        # Select features using LassoCV
        X_selected, y_selected, selected_features = fs.select_feature_using_lasso_cv(X, y)
    case "none":
        #  use all features present in data
        X_selected, y_selected, selected_features = X, y, X.columns
    
X_train, X_test, y_train, y_test = preprocess.get_train_and_test_split(X=X_selected, y=y_selected)

match cfg.classification:
    case "rfc":
        model = models.train_and_predict_using_random_forest_classifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        predict = models.classifier_prediction_probability
    case "lgbm":
        model = models.train_and_predict_using_lightgbm_classifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        predict = models.lightgbm_prediction_probability
    case "xgb":
        model = models.train_and_predict_using_xgb_classifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        predict = models.classifier_prediction_probability
        
def predict_fn(data):
    return predict(model=model, data=data)
        
if cfg.explanation.lime:
    i = randint(0, len(X_test) - 1)
    # print(X_test[i])
    # print(np.array(y_test)[i])
    # chosen_instance = X_test.iloc[i]
    chosen_instance = X_test[i]
    explanations.lime_explanations(feature_names=selected_features, X_train=X_train, selected_instance=chosen_instance, 
                                   predict_fn=predict_fn, py_plot=cfg.explanation.lime_plot)
    
if cfg.explanation.shap:
    shap_values=explanations.shap_explanations(model=model, feature_names=selected_features, X_test=X_test)
    
    # Plot SHAP summary 
    fig = shap.summary_plot(shap_values, feature_names=selected_features, show=False)
    plt.savefig(SHAP_EXPLANATION_GLOBAL_PLOT, dpi=300, bbox_inches='tight')
    plt.close(fig=fig)
    
    if cfg.explanation.shap_bar_plot:
        fig = shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(SHAP_EXPLANATION_BAR_PLOT, dpi=300, bbox_inches='tight')
        plt.close(fig=fig)
        
    i = randint(0, len(selected_features) - 1)
    selected_feature = selected_features[i]
    # print(shap_values)
    # print(selected_feature)
    
    # not working
    # shap.dependence_plot(X_test[selected_feature] , shap_values, X_test)
    # shap.summary_plot(shap_values[0], X_test)
    
    # not working
    # gender = (
    #     X['gender']
    #     .apply(lambda sex: 'Women' if sex == 0 else 'Men')
    #     .values
    # )
    # shap.plots.bar(shap_values.cohorts(gender).abs.mean(axis=0))
    
    if cfg.explanation.shap_cluster:
        clustering = shap.utils.hclust(X_test, y_test, random_state=42)
        fig = shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=0.5, show=False)
        plt.savefig(SHAP_EXPLANATION_CLUSTER_PLOT, dpi=300, bbox_inches='tight')
        plt.close(fig=fig)
    
    if cfg.explanation.shap_scatter:
        fig = shap.plots.scatter(shap_values[:, selected_feature], show=False)
        plt.savefig(SHAP_EXPLANATION_SCATTER_PLOT, dpi=300, bbox_inches='tight')