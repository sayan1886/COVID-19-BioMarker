from lime import lime_tabular

import shap

from utils import constants


def lime_explanations(feature_names, X_train, selected_instance, predict_fn, py_plot):
    # LIME Instance
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        training_labels="SC2_PCR",
        class_names=["POSITIVE", "NEGATIVE"],
        mode="classification",
        kernel_width=10,
        verbose=True,)

    # asking for explanation for LIME model
    exp = explainer.explain_instance(
            data_row=selected_instance, 
            predict_fn=predict_fn,
            num_features=len(feature_names))

    # print("LIME Explainer: ", exp.as_list())
    
    # exp.show_in_notebook(show_all=True)
    fig = None
    if py_plot:
        fig = exp.as_pyplot_figure()
    exp.save_to_file(constants.LIME_EXPLANATION_HTML)
    # exp.visualize_instance_html(show_table=True)
    return fig
    

# https://medium.com/@akarabaev96/are-you-using-the-shap-library-to-its-full-potential-bb45efce2c9
def shap_explanations(model, feature_names, X_test):
    # Initialize SHAP
    shapExplainer = shap.Explainer(model=model, feature_names=feature_names)

    # Evaluate SHAP values for entire test set
    shap_values = shapExplainer(X_test)

    # print("SHAP Explainer: ", shap_values)
    return shap_values