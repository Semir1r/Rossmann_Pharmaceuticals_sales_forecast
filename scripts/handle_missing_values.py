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