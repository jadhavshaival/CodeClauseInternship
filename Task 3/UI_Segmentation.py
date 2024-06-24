import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the data
customer_data = pd.read_csv('C:/Users/jadha/Desktop/Codeclause_Internship/Task 3(Crop Disease Identification)/Dataset.csv')  # Replace with your dataset path

# Prepare data for clustering (if needed)
X = customer_data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values

# Load the saved model
kmeans = joblib.load('C:/Users/jadha/Desktop/Codeclause_Internship/Task 3(Crop Disease Identification)/kmeans_model.pkl')

# Streamlit UI
st.title("Customer Segmentation using K-Means Clustering")
st.sidebar.title("Input Features")
age = st.sidebar.slider("Age", int(customer_data["Age"].min()), int(customer_data["Age"].max()), int(customer_data["Age"].mean()))
annual_income = st.sidebar.slider("Annual Income (k$)", int(customer_data["Annual Income (k$)"].min()), int(customer_data["Annual Income (k$)"].max()), int(customer_data["Annual Income (k$)"].mean()))
spending_score = st.sidebar.slider("Spending Score (1-100)", int(customer_data["Spending Score (1-100)"].min()), int(customer_data["Spending Score (1-100)"].max()), int(customer_data["Spending Score (1-100)"].mean()))

input_data = np.array([[age, annual_income, spending_score]])
predicted_cluster = kmeans.predict(input_data)[0]

st.write(f"The customer belongs to Cluster: {predicted_cluster + 1}")

# Plotting the clusters
plt.figure(figsize=(6, 6))
colors = ['green', 'red', 'yellow', 'violet', 'blue']
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

Y = kmeans.predict(X)  # Predicted clusters for all data points

for i in range(5):
    plt.scatter(X[Y == i, 1], X[Y == i, 2], s=50, c=colors[i], label=labels[i])

plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=100, c='cyan', label='Centroids')
plt.scatter(input_data[0, 1], input_data[0, 2], s=200, c='black', label='Your Data', marker='x')

plt.title('Customer Groups')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
st.pyplot(plt)

