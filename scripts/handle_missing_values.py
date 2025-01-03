import logging

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler to log to a file
file_handler = logging.FileHandler('sales_analysis.log')
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

def clean_missing_values(merged_df):
    """
    Cleans the missing values in the dataset according to the specified strategies.

    """
    # Fill missing values in 'CompetitionDistance' with the median
    merged_df['CompetitionDistance'].fillna(merged_df['CompetitionDistance'].median(), inplace=True)

    # Fill missing values in 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear' with 0
    merged_df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    merged_df['CompetitionOpenSinceYear'].fillna(0, inplace=True)

    # Fill missing values in 'Promo2SinceWeek' and 'Promo2SinceYear' with 0
    merged_df['Promo2SinceWeek'].fillna(0, inplace=True)
    merged_df['Promo2SinceYear'].fillna(0, inplace=True)

    # Fill missing values in 'PromoInterval' with 'None'
    merged_df['PromoInterval'].fillna('None', inplace=True)

    # Return the cleaned DataFrame
    return merged_df