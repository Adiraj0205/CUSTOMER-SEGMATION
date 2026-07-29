# AI-ML Assignment – 7
## Customer Segmentation using K-Means Clustering and PCA

### Objective
The objective of this assignment is to segment mall customers into different groups using **K-Means Clustering** based on customer characteristics, annual income and spending behavior. **Principal Component Analysis (PCA)** is then applied to reduce the standardized feature space to two principal components for visualization.

### Dataset
**Mall Customer Segmentation Dataset**

Kaggle:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

The dataset should **not** be uploaded to this GitHub repository unless its license explicitly allows redistribution.

### Project Structure

```text
Assignment-7-Customer-Segmentation/
│
├── Assignment-7.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── Mall_Customers.csv        # Download manually; do not commit if license does not allow it
│
└── outputs/
    ├── elbow_curve.png
    ├── customer_clusters.png
    ├── pca_clusters.png
    ├── elbow_values.csv
    ├── customer_cluster_results.csv
    ├── cluster_summary.csv
    └── pca_results.csv
```

### Libraries Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

### Methodology

1. Load the Mall Customer Segmentation Dataset using Pandas.
2. Display the first five records.
3. Identify numerical and categorical features.
4. Display dataset information and summary statistics.
5. Check for missing values.
6. Remove the unnecessary `CustomerID` column.
7. Encode the categorical `Gender` feature.
8. Select:
   - Age
   - Annual Income (k$)
   - Spending Score (1-100)
9. Standardize the numerical features using `StandardScaler`.
10. Use the Elbow Method to evaluate different values of K.
11. Train the K-Means model with the selected K.
12. Assign a cluster label to every customer.
13. Apply PCA to reduce the standardized data to two principal components.
14. Generate:
   - Elbow Curve
   - Customer Cluster Scatter Plot
   - PCA Cluster Visualization
15. Generate a cluster summary for business interpretation.

### Results

The Elbow Method is used to inspect the reduction in K-Means inertia as K increases. For the standard Mall Customers dataset, **K = 5** is selected for this assignment.

The model creates five customer segments. The exact average characteristics of each segment are generated automatically in:

```text
outputs/cluster_summary.csv
```

The generated visualizations are:

```text
outputs/elbow_curve.png
outputs/customer_clusters.png
outputs/pca_clusters.png
```

The PCA output also reports the proportion of variance explained by the first two principal components.

### Observations

1. The elbow curve indicates that **5 clusters** provide a suitable segmentation for the standard Mall Customers dataset.
2. PCA reduces the standardized multi-feature data into two principal components, making the customer groups easier to visualize in a 2D graph.
3. The identified clusters represent customers with different combinations of age, annual income and spending score.
4. The `cluster_summary.csv` file can be used to identify groups such as high-income/high-spending customers, low-income/low-spending customers and other intermediate segments.

### Conclusion
Customer segmentation using K-Means clustering can divide mall customers into meaningful groups according to their demographic and spending characteristics. In this project, customer data was cleaned, standardized and divided into five clusters after examining the Elbow Method. PCA was then used to transform the standardized features into two principal components, making the cluster structure easier to visualize. These segments can help mall management design targeted marketing campaigns, personalized offers, loyalty programs and customer retention strategies. For example, high-income and high-spending customers may be targeted with premium offers, while low-spending groups may receive promotional incentives. A limitation of K-Means is that the result depends on the selected number of clusters and can be sensitive to initialization and outliers. PCA's major advantage is that it reduces dimensionality while retaining as much variance as possible, simplifying visualization and analysis.

### How to Run

#### 1. Install Python
Use Python 3.10 or newer.

#### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Download the dataset
Download the CSV from the Kaggle link above.

Rename it to:

```text
Mall_Customers.csv
```

Place it here:

```text
data/Mall_Customers.csv
```

#### 5. Run the assignment

```bash
python Assignment-7.py
```

#### 6. Check the results

The program creates the graphs and CSV results inside the `outputs` folder.

### GitHub Submission

Create a **public** GitHub repository and upload:

- `Assignment-7.py`
- `README.md`
- `requirements.txt`
- `.gitignore`

Do **not** upload the dataset unless its license explicitly allows redistribution.

Keep the repository public until evaluation is completed.

### Submission Checklist

- [ ] `Assignment-7.py` uploaded
- [ ] `README.md` uploaded
- [ ] `requirements.txt` uploaded
- [ ] `.gitignore` uploaded
- [ ] Dataset link included in README
- [ ] Repository is public
- [ ] Code runs successfully
- [ ] Elbow Curve generated
- [ ] Customer cluster scatter plot generated
- [ ] PCA visualization generated
- [ ] 3–4 observations included
- [ ] Conclusion included
- [ ] GitHub repository link copied
- [ ] Google Form submitted before the deadline
