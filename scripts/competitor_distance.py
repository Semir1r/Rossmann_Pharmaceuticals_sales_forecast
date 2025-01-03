import matplotlib.pyplot as plt
import seaborn as sns
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


def analyze_competitor_distance_effect_on_sales(merged_df, sales_column, competition_distance_column, store_type_column):
    """
    Analyze how the distance to the nearest competitor affects sales and how this varies in city centers.
    """

    # 1. Correlation between competition distance and sales
    correlation = merged_df[[sales_column, competition_distance_column]].corr().iloc[0, 1]
    print(f"Correlation between {competition_distance_column} and {sales_column}: {correlation:.4f}")

    # 2. Violin plot for competition distance vs sales
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=store_type_column, y=sales_column, hue=competition_distance_column, 
                   data=merged_df, split=True, palette="muted")
    plt.title('Sales Distribution by Store Location and Competition Distance')
    plt.xlabel('Store Location')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.legend(title="Competition Distance", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # 3. Facet grid for sales by competition distance and store type
    g = sns.FacetGrid(merged_df, col=store_type_column, height=5, aspect=1.5, sharey=True)
    g.map(sns.scatterplot, competition_distance_column, sales_column, alpha=0.5)
    g.add_legend()
    g.set_axis_labels("Competition Distance (meters)", "Sales")
    g.fig.suptitle("Sales vs Competition Distance by Store Location", y=1.02)
    plt.show()

    # 4. Heatmap for aggregated sales data
    sales_summary = merged_df.groupby([competition_distance_column, store_type_column])[sales_column].mean().unstack()
    plt.figure(figsize=(10, 6))
    sns.heatmap(sales_summary, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Average Sales'})
    plt.title("Average Sales by Competition Distance and Store Location")
    plt.xlabel("Store Location")
    plt.ylabel("Competition Distance")
    plt.show()

    # Summary statistics for sales based on competition distance
    distance_sales_summary = merged_df.groupby([competition_distance_column, store_type_column])[sales_column].agg(['mean', 'median', 'std', 'count'])
    print("\nSales Summary by Competition Distance and Store Location:\n", distance_sales_summary)
