import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def analyze_store_opening_closing_behavior(merged_df, sales_column, customers_column, open_column, date_column):
    """
    Analyze trends of customer behavior during store opening and closing times.
    """

    # Convert date_column to datetime format if needed
    merged_df[date_column] = pd.to_datetime(merged_df[date_column])

    # 1. Compare Sales and Customers when Store is Open vs Closed
    plt.figure(figsize=(10, 6))
    sns.barplot(x=open_column, y=sales_column, data=merged_df)
    plt.title('Sales When Store is Open (1) vs Closed (0)')
    plt.xlabel('Store Open (1) vs Closed (0)')
    plt.ylabel('Sales')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=open_column, y=customers_column, data=merged_df)
    plt.title('Number of Customers When Store is Open (1) vs Closed (0)')
    plt.xlabel('Store Open (1) vs Closed (0)')
    plt.ylabel('Number of Customers')
    plt.show()

    # 2. Analyze day-wise trends for customer behavior during store opening and closing
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=merged_df[date_column].dt.day_name(), y=customers_column, hue=open_column, data=merged_df, ci=None)
    plt.title('Customer Behavior by Day of Week (Open vs Closed)')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Customers')
    plt.show()

    # 3. Analyze store sales by day-wise trends
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=merged_df[date_column].dt.day_name(), y=sales_column, hue=open_column, data=merged_df, ci=None)
    plt.title('Sales by Day of Week (Open vs Closed)')
    plt.xlabel('Day of the Week')
    plt.ylabel('Sales')
    plt.show()

    # Summary statistics of sales and customer behavior when open vs closed
    open_closed_summary = merged_df.groupby(open_column)[[sales_column, customers_column]].agg(['mean', 'median', 'std', 'count'])
    print("\nStore Opening and Closing Summary (Sales and Customers):\n", open_closed_summary)