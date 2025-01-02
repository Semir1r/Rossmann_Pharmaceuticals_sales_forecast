import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler to log to a file
file_handler = logging.FileHandler('sales_analysis.log', mode='w')  # 'w' to overwrite the file each run
file_handler.setLevel(logging.INFO)

# Create a stream handler to log to the notebook console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Define a common formatter for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Add the formatter to both handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Logger setup complete. Logging to both file and console.")

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a column using the IQR method.
    Returns a boolean mask indicating where the outliers are.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outlier_mask

def remove_outliers(df, numerical_columns):
    """
    Remove outliers from the DataFrame for specified numerical columns.
    """
    logger.info("Starting outlier detection and removal.")
    for column in numerical_columns:
        logger.info(f"Checking outliers for column: {column}")
        outlier_mask = detect_outliers_iqr(df, column)
        num_outliers = outlier_mask.sum()
        logger.info(f"Found {num_outliers} outliers in column {column}.")
        df = df[~outlier_mask]
        logger.info(f"Outliers removed from column {column}.")
    return df

# Example usage
def process_datasets(train_df, test_df, store_df):
    """
    Process train, test, and store datasets by checking and removing outliers.
    """
    # List of numerical columns where outliers should be checked
    numerical_columns_train = ['Sales', 'Customers']  # Add more columns as needed
    numerical_columns_test = []  # Add numerical columns if any
    numerical_columns_store = ['CompetitionDistance']  # Add more columns as needed

    # Remove outliers from the datasets
    logger.info("Processing train dataset.")
    train_cleaned = remove_outliers(train_df, numerical_columns_train)
    
    logger.info("Processing test dataset.")
    test_cleaned = remove_outliers(test_df, numerical_columns_test)  # Update if you have numeric columns

    logger.info("Processing store dataset.")
    store_cleaned = remove_outliers(store_df, numerical_columns_store)

    return train_cleaned, test_cleaned, store_cleaned

def check_promotion_distribution(train_df, test_df, promotion_column):
    """
    Check the distribution of promotions between training and test sets.
    
    """
    logger.info("Checking promotion distribution between training and test sets.")
    # Check for missing values in the promotion column
    if train_df[promotion_column].isnull().sum() > 0 or test_df[promotion_column].isnull().sum() > 0:
        logger.warning("Missing values found in the promotion column. Consider handling them before analysis.")
    logger.info("Plotting the promotion distribution for the training and test sets.")
    # Plot the distribution of promotions in both datasets
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training set promotion distribution
    sns.histplot(train_df[promotion_column], kde=False, ax=axes[0], color="blue")
    axes[0].set_title('Promotion Distribution in Training Set')
    axes[0].set_xlabel('Promotion')
    axes[0].set_ylabel('Frequency')

    # Test set promotion distribution
    sns.histplot(test_df[promotion_column], kde=False, ax=axes[1], color="green")
    axes[1].set_title('Promotion Distribution in Test Set')
    axes[1].set_xlabel('Promotion')
    axes[1].set_ylabel('Frequency')

    # Display the plots
    plt.tight_layout()
    plt.show()

    # Statistical comparison (optional)
    logger.info("Calculating normalized promotion distribution for both datasets.")
    train_promo_dist = train_df[promotion_column].value_counts(normalize=True)
    test_promo_dist = test_df[promotion_column].value_counts(normalize=True)
    logger.info(f"Training Set Promotion Distribution (Normalized):\n{train_promo_dist}")
    logger.info(f"Test Set Promotion Distribution (Normalized):\n{test_promo_dist}")

   
    
    # Calculate percentage difference between the two distributions
    promo_diff = abs(train_promo_dist - test_promo_dist).fillna(0)
    logger.info(f"Percentage Difference in Promotion Distribution:\n{promo_diff}")