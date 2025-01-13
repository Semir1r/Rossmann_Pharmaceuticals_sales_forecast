import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from the RandomForestRegressor model.
    """
    # Extract feature importance values
    feature_importance = model.named_steps['rf'].feature_importances_
    
    # Sort the feature importance values and their corresponding feature names
    indices = np.argsort(feature_importance)[::-1]
    
    # Plot the feature importance
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importance from Random Forest")
    plt.bar(range(len(feature_names)), feature_importance[indices], align="center")
    plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=90)
    plt.tight_layout()
    plt.show()