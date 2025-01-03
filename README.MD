# Rossmann Pharmaceuticals Sales Analysis

## Business Need
Rossmann Pharmaceuticals aims to improve its sales forecasting capabilities for its stores across various cities. Currently, store managers rely on personal judgment to predict sales. The finance team has identified key factors influencing sales, including:
- Promotions
- Competition
- School and state holidays
- Seasonality
- Locality

Your role as a Machine Learning Engineer is to develop and deliver an end-to-end product to accurately predict sales six weeks ahead. This tool will empower analysts in the finance team with actionable insights for planning and strategy.

## Data Description
The dataset contains the following fields:

- **Id**: Unique identifier for a (Store, Date) tuple in the test set.
- **Store**: Unique ID for each store.
- **Sales**: Daily turnover (target variable).
- **Customers**: Number of customers on a given day.
- **Open**: Indicates whether the store was open (1) or closed (0).
- **StateHoliday**: State holiday indicator (`a`: public, `b`: Easter, `c`: Christmas, `0`: None).
- **SchoolHoliday**: Indicates if the date was affected by school closures.
- **StoreType**: Differentiates between four store models (`a`, `b`, `c`, `d`).
- **Assortment**: Describes assortment level (`a`: basic, `b`: extra, `c`: extended).
- **CompetitionDistance**: Distance to the nearest competitor store (in meters).
- **CompetitionOpenSince[Month/Year]**: Month and year the nearest competitor opened.
- **Promo**: Indicates if a promo was running on a given day.
- **Promo2**: Continuous promotion participation indicator.
- **Promo2Since[Year/Week]**: Year and calendar week when Promo2 started.
- **PromoInterval**: Months when Promo2 promotions begin anew.

For the full dataset, visit [Kaggle: Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data).

## Task 1: Exploration of Customer Purchasing Behavior
### Objective
Perform an exploratory data analysis (EDA) to understand customer purchasing behavior across stores, focusing on factors influencing sales. The analysis involves cleaning the data, handling outliers, and visualizing insights.

### Questions to Answer
1. Are promotions distributed similarly between training and test sets?
2. How do sales behave before, during, and after holidays?
3. What seasonal trends (e.g., Christmas, Easter) affect purchasing behavior?
4. What is the correlation between sales and the number of customers?
5. How do promotions affect sales and customer numbers?
6. Could promotions be deployed more effectively? Identify stores to prioritize.
7. What are customer behavior trends during store opening/closing times?
8. Which stores are open on all weekdays, and how does this impact weekend sales?
9. How does the assortment type affect sales?
10. How does the distance to the nearest competitor affect sales?
11. What happens when new competitors open near a store?

### Deliverables
1. **Exploratory Analysis Notebook**:
   - Handle missing values and outliers using pipelines.
   - Use visualizations (e.g., histograms, box plots, line charts) to communicate findings.
   - Provide summary tables where applicable.
2. **Logging**:
   - Use the Python `logging` library to log steps, ensuring traceability and reproducibility.
   - Log file should capture key milestones and results.

## Tools and Frameworks
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Logging**: Python's `logging` library for tracking and documenting progress.

## How to Use
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


This `README.md` provides a comprehensive overview of the task, objectives, data, and expected outcomes. Let me know if you need further adjustments!
