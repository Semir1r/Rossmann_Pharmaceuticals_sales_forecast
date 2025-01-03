import matplotlib.pyplot as plt
import seaborn as sns


def analyze_competitor_distance_effect_on_sales(merged_df, sales_column, competition_distance_column, store_type_column):
    """
    Analyze how the distance to the nearest competitor affects sales and how this varies in city centers.
    """

    # 1. Correlation between competition distance and sales
    correlation = merged_df[[sales_column, competition_distance_column]].corr().iloc[0, 1]
    print(f"Correlation between {competition_distance_column} and {sales_column}: {correlation:.4f}")

    # 2. Visualize how sales are distributed across different ranges of competition distance
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='DistanceBin', y=sales_column, data=merged_df)
    plt.title('Sales Distribution by Competition Distance')
    plt.xlabel('Competition Distance (meters)')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=store_type_column, y=sales_column, data=merged_df, hue=competition_distance_column)
    sns.stripplot(x=store_type_column, y=sales_column, data=merged_df, hue=competition_distance_column, jitter=True, size=2, alpha=0.5, dodge=True)
    plt.title('Sales by Store Location and Competition Distance')
    plt.xlabel('Store Location')
    plt.ylabel('Sales')
    plt.show()

    # Summary statistics for sales based on competition distance
    distance_sales_summary = merged_df.groupby([competition_distance_column, store_type_column])[sales_column].agg(['mean', 'median', 'std', 'count'])
    print("\nSales Summary by Competition Distance and Store Location:\n", distance_sales_summary)