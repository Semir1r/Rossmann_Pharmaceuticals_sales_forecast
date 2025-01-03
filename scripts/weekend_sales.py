import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def analyze_weekday_open_stores_and_weekend_sales(merged_df, sales_column, date_column, store_column):
    """
    Identify stores open on all weekdays (Monday to Friday) and analyze how that affects their sales on weekends.
    
    """

    # Ensure the date column is in datetime format
    merged_df[date_column] = pd.to_datetime(merged_df[date_column])

    # Add a day of the week column based on the Date
    merged_df['DayOfWeek'] = merged_df[date_column].dt.day_name()

    # Filter for stores open on all weekdays
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekday_data = merged_df[merged_df['DayOfWeek'].isin(weekdays) & (merged_df['Open'] == 1)]
    
    # Group by store to count how many weekdays each store is open
    weekday_open_counts = weekday_data.groupby(store_column)['DayOfWeek'].nunique()
    
    # Identify stores that are open all 5 weekdays
    stores_open_all_weekdays = weekday_open_counts[weekday_open_counts == 5].index
    print(f"Number of stores open on all weekdays: {len(stores_open_all_weekdays)}")

    # Now, analyze weekend sales for these stores
    weekend_days = ['Saturday', 'Sunday']
    weekend_data = merged_df[(merged_df[store_column].isin(stores_open_all_weekdays)) & 
                             (merged_df['DayOfWeek'].isin(weekend_days)) & 
                             (merged_df['Open'] == 1)]

     # 1. Visualize weekend sales (Saturday vs Sunday)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='DayOfWeek', y=sales_column, data=weekend_data)
    plt.title('Weekend Sales for Stores Open on All Weekdays')
    plt.xlabel('Weekend Day')
    plt.ylabel('Sales')
    plt.show()

    # 2. Summary statistics for weekend sales
    weekend_sales_summary = weekend_data.groupby('DayOfWeek')[sales_column].agg(['mean', 'median', 'std', 'count'])
    print("\nWeekend Sales Summary for Stores Open on All Weekdays:\n", weekend_sales_summary)

    # Optional: Compare weekend sales for stores that are open vs closed during weekdays
    non_weekday_open_stores = merged_df[~merged_df[store_column].isin(stores_open_all_weekdays)]
    non_weekday_weekend_data = non_weekday_open_stores[non_weekday_open_stores['DayOfWeek'].isin(weekend_days) & (non_weekday_open_stores['Open'] == 1)]

    # 3. Visualize weekend sales for stores not open all weekdays
    plt.figure(figsize=(10, 6))
    sns.barplot(x='DayOfWeek', y=sales_column, data=non_weekday_weekend_data)
    plt.title('Weekend Sales for Stores Not Open on All Weekdays')
    plt.xlabel('Weekend Day')
    plt.ylabel('Sales')
    plt.show()

    # 4. Summary statistics for stores not open all weekdays
    non_weekday_weekend_sales_summary = non_weekday_weekend_data.groupby('DayOfWeek')[sales_column].agg(['mean', 'median', 'std', 'count'])
    print("\nWeekend Sales Summary for Stores Not Open on All Weekdays:\n", non_weekday_weekend_sales_summary)