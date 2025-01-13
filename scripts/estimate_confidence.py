import numpy as np

def estimate_confidence_intervals(model, X_val, alpha=0.95):
    """
    Estimate the confidence intervals for predictions from a RandomForestRegressor.
    
    """
    # Get predictions from each tree in the random forest
    all_tree_predictions = np.array([tree.predict(X_val) for tree in model.named_steps['rf'].estimators_])
    
    # Calculate the mean and standard deviation of the predictions
    mean_prediction = np.mean(all_tree_predictions, axis=0)
    std_prediction = np.std(all_tree_predictions, axis=0)
    
    # Calculate the z-score for the desired confidence level
    z = 1.96  # For a 95% confidence interval
    
    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean_prediction - z * std_prediction
    upper_bound = mean_prediction + z * std_prediction
    
    return lower_bound, upper_bound