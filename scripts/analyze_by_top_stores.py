import seaborn as sns
import matplotlib.pyplot as plt

def analyze_promo_by_top_stores(merged_df, sales_column, customers_column, promo_column, store_column, store_type_column, top_n=10):
    """
    Analyze the effect of promotions on sales and customers by store and store type,
    limiting the analysis to the top N stores based on sales.
    """

    # Group by store and analyze sales and customer changes with/without promotions
    promo_sales_by_store = merged_df.groupby([store_column, promo_column])[[sales_column, customers_column]].mean().unstack()

    # Get top N stores based on average sales (with and without promo)
    top_stores = promo_sales_by_store[sales_column].mean(axis=1).nlargest(top_n).index

    # Filter the dataset for only the top N stores
    filtered_promo_sales_by_store = promo_sales_by_store.loc[top_stores]

    # Visualize promo effectiveness by store for the top N stores
    ax = filtered_promo_sales_by_store.plot(kind='bar', figsize=(12, 6), title=f'Average Sales and Customers for Top {top_n} Stores (Promo vs Non-Promo)')
    plt.xlabel('Store')
    plt.ylabel('Average Sales/Customers')
    plt.legend(['Sales (No Promo)', 'Sales (Promo)', 'Customers (No Promo)', 'Customers (Promo)'])
    
    # Rotate the x-axis labels to avoid overlap
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

    # Analyze the impact of promotions across store types
    plt.figure(figsize=(10, 6))
    sns.barplot(x=store_type_column, y=sales_column, hue=promo_column, data=merged_df)
    plt.title('Sales by Store Type (Promo vs Non-Promo)')
    plt.xlabel('Store Type')
    plt.ylabel('Sales')
    plt.show()

    # Summary statistics of promo effects by store
    promo_effectiveness_summary = merged_df.groupby([store_column, promo_column])[[sales_column, customers_column]].agg(['mean', 'std', 'count'])
    print(f"\nPromo Effectiveness Summary for Top {top_n} Stores (Sales and Customers):\n", promo_effectiveness_summary)
