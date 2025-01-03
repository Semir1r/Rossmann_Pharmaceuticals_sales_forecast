import logging
import pandas as pd

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler to log to a file
file_handler = logging.FileHandler('sales_analysis.log', mode='w')
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

def merge_train_store(train_file, store_file):
    """
    Merge the train and store datasets on the 'Store' column.
    """
    logger.info("Merging the train and store datasets on the 'Store' column.")


    # Perform the merge operation on the 'Store' column
    merged_df = pd.merge(train_file, store_file, how='inner', on='Store')
    logger.info(f"Merge completed. The merged dataset has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")
    # Return the merged DataFrame
    return merged_df