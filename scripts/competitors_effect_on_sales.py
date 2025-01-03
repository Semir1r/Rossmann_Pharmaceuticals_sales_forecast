import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def analyze_new_competitors_effect_on_sales(merged_df, sales_column, competition_distance_column, date_column, store_column):
    """
    Analyze the effect of the opening of new competitors on store sales by checking stores with initially 'NA' 
    in CompetitionDistance and later getting values.
    """

    # Ensure the date column is in datetime format
    merged_df[date_column] = pd.to_datetime(merged_df[date_column])

    # 1. Identify stores with initially 'NA' in CompetitionDistance and later have a valid value
    na_competitors = merged_df[merged_df[competition_distance_column].isna()]
    new_competitors = merged_df[merged_df[competition_distance_column].notna()]

    # Merge based on Store and Date to track when a competitor entered
    stores_with_new_competitors = new_competitors[store_column].unique()

    # Filter data for stores that had 'NA' for competitors initially
    affected_stores = merged_df[merged_df[store_column].isin(stores_with_new_competitors)]

    # 2. Analyze sales before and after the competitor entered the market
    # Assuming that after the competitor distance is no longer 'NA', a competitor opened
    affected_stores['CompetitorEntered'] = affected_stores[competition_distance_column].notna()

    # Create a 'Before' and 'After' column for competitor entry
    affected_stores['TimePeriod'] = ['Before Competitor' if x is None else 'After Competitor' for x in affected_stores[competition_distance_column]]

    # 3. Visualize sales before and after competitor entry
    plt.figure(figsize=(10, 6))
    sns.barplot(x='TimePeriod', y=sales_column, data=affected_stores)
    plt.title('Sales Before and After Competitor Entry')
    plt.xlabel('Competitor Entry Period')
    plt.ylabel('Sales')
    plt.show()

    # 4. Plot a time series of sales for affected stores
    affected_stores.set_index(date_column, inplace=True)
    affected_stores.groupby('TimePeriod')[sales_column].plot(legend=True)
    plt.title('Sales Trends Before and After Competitor Entry')
    plt.ylabel('Sales')
    plt.xlabel('Date')
    plt.show()

    # 5. Summary statistics
    competitor_effect_summary = affected_stores.groupby('TimePeriod')[sales_column].agg(['mean', 'median', 'std', 'count'])
    print("\nSales Summary Before and After Competitor Entry:\n", competitor_effect_summary)