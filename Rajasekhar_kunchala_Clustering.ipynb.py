# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt

# Load datasets
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')

# Merge datasets
merged_data = pd.merge(transactions, customers, on='CustomerID')

# Create customer profiles
customer_profiles = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'Region': 'first'
}).reset_index()

# One-hot encode categorical variables
customer_profiles = pd.get_dummies(customer_profiles, columns=['Region'])

# Normalize features
scaler = StandardScaler()
customer_profiles_scaled = scaler.fit_transform(customer_profiles.drop('CustomerID', axis=1))

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customer_profiles['Cluster'] = kmeans.fit_predict(customer_profiles_scaled)

# Calculate DB Index
db_index = davies_bouldin_score(customer_profiles_scaled, customer_profiles['Cluster'])
print(f'DB Index: {db_index}')

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(customer_profiles_scaled[:, 0], customer_profiles_scaled[:, 1], c=customer_profiles['Cluster'], cmap='viridis')
plt.title('Customer Segmentation Clusters')
plt.show()
