# 📊 Mall Customer Segmentation

Welcome to the **Mall Customer Segmentation** project! This project focuses on analyzing mall customer data to group them into segments based on their behavior.

## 📚 Project Overview

Customer segmentation involves dividing customers into groups based on common characteristics. In this project, we use clustering algorithms to segment mall customers by analyzing their spending patterns and income levels.

## 📦 Dataset

The dataset contains the following columns:

- **CustomerID**: Unique ID for each customer
- **Gender**: Gender of the customer
- **Age**: Age of the customer
- **Annual Income (k\$)**: Annual income in thousands of dollars
- **Spending Score (1-100)**: Score assigned by the mall based on customer spending behavior

## 🚀 Project Setup

1. **Clone the repository:**

```bash
git clone <repository_url>
```

2. **Navigate to the project directory:**

```bash
cd Mall_Customer_Segmentation
```

3. **Install required libraries:**

```bash
pip install -r requirements.txt
```

4. **Run the Jupyter notebook:**

```bash
jupyter notebook Mall_Customer_Segmentation.ipynb
```

## 📈 Code Explanation

### Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```

### Load Dataset

```python
df = pd.read_csv('Mall_Customers.csv')
df.head()
```

### Data Preprocessing

```python
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
```

### Using K-Means Clustering

```python
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```

### Visualizing the Elbow Curve

```python
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

### Final Clustering

```python
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=69)
y_kmeans = kmeans.fit_predict(X)
```

### 📊 Results

The customers are segmented into 6 groups:

- **Cluster 1**: High income, low spending
- **Cluster 2**: Average income, average spending
- **Cluster 3**: Low income, high spending
- **Cluster 4**: High income, high spending
- **Cluster 5**: Low income, low spending
- **Cluster 6**: Moderate income, moderate spending

## ✨ Future Enhancements

- Try other clustering algorithms.
- Integrate customer demographics for more refined segments.

## 🤝 Contributing

Contributions are welcome! Please create an issue or pull request for any improvements.

## 📧 Contact

For questions or collaboration, feel free to reach out!

---

⭐ **Happy Coding!** ⭐

