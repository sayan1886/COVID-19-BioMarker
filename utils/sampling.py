from imblearn.over_sampling import SVMSMOTE

def sample_data_with_svm_smote(X, y):
    smote = SVMSMOTE(random_state = 42, sampling_strategy=0.85, k_neighbors=10)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
