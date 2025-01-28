# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Basic EDA
print("Customers Data:")
print(customers.info())
print("\nProducts Data:")
print(products.info())
print("\nTransactions Data:")
print(transactions.info())

# Merge datasets for analysis
merged_data = pd.merge(transactions, customers, on='CustomerID')
merged_data = pd.merge(merged_data, products, on='ProductID')

# Business Insights
# 1. Most popular product categories
plt.figure(figsize=(10, 6))
sns.countplot(y='Category', data=merged_data, order=merged_data['Category'].value_counts().index)
plt.title('Most Popular Product Categories')
plt.show()

# 2. Customer distribution by region
plt.figure(figsize=(10, 6))
sns.countplot(y='Region', data=customers, order=customers['Region'].value_counts().index)
plt.title('Customer Distribution by Region')
plt.show()

# 3. Total sales over time
merged_data['TransactionDate'] = pd.to_datetime(merged_data['TransactionDate'])
merged_data.set_index('TransactionDate', inplace=True)
monthly_sales = merged_data['TotalValue'].resample('M').sum()
plt.figure(figsize=(10, 6))
monthly_sales.plot()
plt.title('Monthly Sales Over Time')
plt.show()

# 4. Average transaction value by region
avg_transaction_value = merged_data.groupby('Region')['TotalValue'].mean()
plt.figure(figsize=(10, 6))
avg_transaction_value.plot(kind='bar')
plt.title('Average Transaction Value by Region')
plt.show()

# 5. Customer signups over time
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
customers.set_index('SignupDate', inplace=True)
monthly_signups = customers.resample('M').size()
plt.figure(figsize=(10, 6))
monthly_signups.plot()
plt.title('Monthly Customer Signups Over Time')
plt.show()
