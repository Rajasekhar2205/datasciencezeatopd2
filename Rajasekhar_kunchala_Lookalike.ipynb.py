# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

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

# Calculate similarity matrix
similarity_matrix = cosine_similarity(customer_profiles_scaled)

# Function to get top 3 similar customers
def get_top_3_lookalikes(customer_id, similarity_matrix, customer_profiles):
    customer_index = customer_profiles[customer_profiles['CustomerID'] == customer_id].index[0]
    similarity_scores = list(enumerate(similarity_matrix[customer_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_3 = similarity_scores[1:4]  # Exclude the customer itself
    return [(customer_profiles.iloc[i[0]]['CustomerID'], i[1]) for i in top_3]

# Generate lookalike recommendations for the first 20 customers
lookalike_recommendations = {}
for customer_id in customer_profiles['CustomerID'][:20]:
    lookalike_recommendations[customer_id] = get_top_3_lookalikes(customer_id, similarity_matrix, customer_profiles)

# Save recommendations to CSV
lookalike_df = pd.DataFrame(lookalike_recommendations).T
lookalike_df.to_csv('FirstName_LastName_Lookalike.csv')
