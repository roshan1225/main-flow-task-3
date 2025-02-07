Skip to main content
main flow task 3.ipynb
main flow task 3.ipynb_
Table of contents

[1]
7s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

[2]
0s
df = pd.read_csv("Mall_Customers.csv")
df

Next steps:

[3]
0s
sh = df.shape
print(f'The number of rows in dataset is {sh[0]}')
print(f'The number of columns in dataset is {sh[1]}')
The number of rows in dataset is 200
The number of columns in dataset is 5

[4]
0s
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 5 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   CustomerID              200 non-null    int64 
 1   Genre                   200 non-null    object
 2   Age                     200 non-null    int64 
 3   Annual Income (k$)      200 non-null    int64 
 4   Spending Score (1-100)  200 non-null    int64 
dtypes: int64(4), object(1)
memory usage: 7.9+ KB

[5]
0s
df.describe()


[6]
0s
# Data cleaning
missing_data = df.isnull().sum()
missing_data


[7]
0s
df.drop_duplicates(inplace = True)

[8]
0s
df.shape # Hence no duplicate found
(200, 5)

[9]
0s
# Data Preprocessing
# Standarize the data
features = ['Age','Annual Income (k$)', 'Spending Score (1-100)']

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

df_scaled.head()


Next steps:

[10]
0s
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df_scaled['Genre'] = label_encoder.fit_transform(df_scaled['Genre'])
df_scaled.head()

Next steps:

[11]
0s
# Data Visualization
import warnings
warnings.filterwarnings("ignore")

[12]
1s
# Elbow Method to find the optimal number of clusters
inertia = []
k_range = range(1,11)

for k in k_range:
    kmeans = KMeans(n_clusters=k,random_state = 42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

#plot the elbow method graph
plt.figure(figsize=(8,5))
plt.plot(k_range, inertia, marker = 'o')
plt.title("Elbow Method for Optimal k")
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.show()


[13]
0s
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k,random_state = 42)
cluster_labels = kmeans.fit_predict(df_scaled)
df_scaled['Cluster'] = cluster_labels
df_scaled

Next steps:

[14]
1s
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
df_scaled_pca = pca.fit_transform(df_scaled[['Age','Annual Income (k$)', 'Spending Score (1-100)']])
plt.figure(figsize=(8,6))
plt.scatter(df_scaled_pca[:,0],df_scaled_pca[:,1],
            c=df_scaled['Cluster'],cmap='tab10', edgecolors='k')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Visualization of Clusters using PCA')
plt.colorbar(label='Cluster')
plt.show()


[15]
3s
df_clustered = df.copy()
df_clustered['Cluster'] = df_scaled['Cluster']

sns.pairplot(df_clustered,hue='Cluster',palette='tab10',diag_kind='kde')
plt.show()


[16]
0s
centroids = kmeans.cluster_centers_

plt.figure(figsize=(8,6))
plt.scatter(df_scaled['Annual Income (k$)'], df_scaled['Spending Score (1-100)'],
            c=df_scaled['Cluster'],cmap='tab10', edgecolors='k',alpha=0.6)
plt.scatter(centroids[:,1],centroids[:,2],c='red',marker='X', s=200, label='Centroids')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.title('Cluster Visualization with centroids')
plt.show()


Colab paid products - Cancel contracts here
