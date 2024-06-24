from lime import lime_tabular

import shap

import matplotlib.pyplot as plt

from random import randint

from utils.constants import (
    LIME_EXPLANATION_HTML,
    SHAP_EXPLANATION_GLOBAL_PLOT,
    SHAP_EXPLANATION_BAR_PLOT,
    SHAP_EXPLANATION_CLUSTER_PLOT,
    SHAP_EXPLANATION_SCATTER_PLOT
)


def lime_explanations(feature_names, X_train, selected_instance, predict_fn, py_plot):
    # LIME Instance
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        training_labels="SC2_PCR",
        class_names=["POSITIVE", "NEGATIVE"],
        mode="classification",
        kernel_width=10,
        random_state=42,
        verbose=True,)

    # asking for explanation for LIME model
    num_features = len(feature_names)
    if num_features > 10:
        num_features = 10
    exp = explainer.explain_instance(
            data_row=selected_instance, 
            predict_fn=predict_fn,
            num_features=num_features)

    # print("LIME Explainer: ", exp.as_list())
    
    # exp.show_in_notebook(show_all=True)
    fig = None
    if py_plot:
        fig = exp.as_pyplot_figure()
    exp.save_to_file(LIME_EXPLANATION_HTML)
    # exp.visualize_instance_html(show_table=True)
    return fig, exp.as_list()
    

# https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html
# https://medium.com/@akarabaev96/are-you-using-the-shap-library-to-its-full-potential-bb45efce2c9
def shap_explanations(model, feature_names, X_test, y_test, shap_bar_plot, shap_cluster, shap_scatter):
    # Initialize SHAP
    shapExplainer = shap.Explainer(model=model, feature_names=feature_names)

    # Evaluate SHAP values for entire test set
    shap_values = shapExplainer(X_test)

    # print("SHAP Explainer: ", shap_values)
    # Plot SHAP summary 
    fig = shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.savefig(SHAP_EXPLANATION_GLOBAL_PLOT, dpi=300, bbox_inches='tight')
    plt.close(fig=fig)
    
    if shap_bar_plot:
        fig = shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(SHAP_EXPLANATION_BAR_PLOT, dpi=300, bbox_inches='tight')
        plt.close(fig=fig)
        
    i = randint(0, len(feature_names) - 1)
    selected_feature = feature_names[i]
    
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
    
    # do not create cluster explanations for more than 10 features
    if len(feature_names) > 10:
        shap_cluster = False
    if shap_cluster:
        clustering = shap.utils.hclust(X_test, y_test, random_state=42)
        fig = shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=0.5, show=False)
        plt.savefig(SHAP_EXPLANATION_CLUSTER_PLOT, dpi=300, bbox_inches='tight')
        plt.close(fig=fig)
    
    if shap_scatter:
        fig = shap.plots.scatter(shap_values[:, selected_feature], show=False)
        plt.savefig(SHAP_EXPLANATION_SCATTER_PLOT, dpi=300, bbox_inches='tight')
        
    return shap_values