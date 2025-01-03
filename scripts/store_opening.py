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

   # 1. Facet Grid for Customer Behavior by Day of the Week
    g = sns.FacetGrid(merged_df, col=open_column, height=5, aspect=1.5, sharey=True)
    g.map(sns.lineplot, merged_df[date_column].dt.day_name(), customers_column, ci=None)
    g.set_titles("{col_name}")
    g.set_axis_labels("Day of the Week", "Number of Customers")
    g.fig.suptitle("Customer Behavior by Day of Week (Open vs Closed)", y=1.02)
    plt.show()

    # 2. Facet Grid for Sales by Day of the Week
    g = sns.FacetGrid(merged_df, col=open_column, height=5, aspect=1.5, sharey=True)
    g.map(sns.lineplot, merged_df[date_column].dt.day_name(), sales_column, ci=None)
    g.set_titles("{col_name}")
    g.set_axis_labels("Day of the Week", "Sales")
    g.fig.suptitle("Sales by Day of Week (Open vs Closed)", y=1.02)
    plt.show()


    # Summary statistics of sales and customer behavior when open vs closed
    open_closed_summary = merged_df.groupby(open_column)[[sales_column, customers_column]].agg(['mean', 'median', 'std', 'count'])
    print("\nStore Opening and Closing Summary (Sales and Customers):\n", open_closed_summary)