# promo_analysis.py
import matplotlib.pyplot as plt

def analyze_promo_impact(cleaned_data):
    """
    Analyze the impact of promotions on sales and customer behavior.

    Parameters:
        cleaned_data (DataFrame): The cleaned dataset containing sales, customers, and promo information.
    """
    if 'Promo' in cleaned_data.columns and 'Customers' in cleaned_data.columns:
        # Promo vs Non-Promo Days: Average Sales and Customer Count
        promo_effect = cleaned_data.groupby('Promo').agg({
            'Sales': 'mean',
            'Customers': 'mean'
        }).rename(columns={'Sales': 'Avg Sales', 'Customers': 'Avg Customers'})

        print("Promo vs Non-Promo Average Sales and Customers:")
        print(promo_effect)

        # Sales Per Customer: Promo vs Non-Promo
        cleaned_data['SalesPerCustomer'] = cleaned_data['Sales'] / cleaned_data['Customers']
        sales_per_customer = cleaned_data.groupby('Promo')['SalesPerCustomer'].mean()
        print("Sales per Customer on Promo vs Non-Promo Days:")
        print(sales_per_customer)

        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Average Sales
        promo_effect['Avg Sales'].plot(kind='bar', ax=axes[0], color=['skyblue', 'orange'], title='Average Sales')
        axes[0].set_xlabel('Promo Status')
        axes[0].set_ylabel('Sales')

        # Average Customers
        promo_effect['Avg Customers'].plot(kind='bar', ax=axes[1], color=['skyblue', 'orange'], title='Average Customers')
        axes[1].set_xlabel('Promo Status')
        axes[1].set_ylabel('Customers')

        # Sales per Customer
        sales_per_customer.plot(kind='bar', ax=axes[2], color=['skyblue', 'orange'], title='Sales per Customer')
        axes[2].set_xlabel('Promo Status')
        axes[2].set_ylabel('Sales per Customer')

        plt.tight_layout()
        plt.show()
    else:
        print("Required columns ('Promo', 'Customers') are missing in the dataset.")
