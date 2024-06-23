import numpy as np

from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

# TODO: try use other feature selection methods for better comparison 
# TODO: try with all the feature and present result
# Compare with multiple feature selections algorithm 
# try to take union of all the various selected feature from different algos and compare


# Fit the model in LASSO
def select_feature_using_lasso_cv(X, y):
    lasso_regression = LassoCV(cv=10, max_iter=5000, tol=1e-2) #(alpha=0.01, max_iter=1000)
    # lasso_regression = Lasso(alpha=0.02, max_iter=1000)
    lasso_regression.fit(X,y)

    # Feature selection from Lasso 
    sfm = SelectFromModel(lasso_regression, prefit=True) 
    X_selected = sfm.transform(X) 

    selected_feature_indices = np.where(sfm.get_support())[0] 
    selected_features = X.columns[selected_feature_indices] 
    coefficients = lasso_regression.coef_ 
    print("Selected Features:", selected_features) 
    print("Feature Coefficients:", coefficients) 
    
    return X_selected, y, selected_features