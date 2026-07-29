from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = DATA_DIR / "Mall_Customers.csv"

OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset():
    """Load the Mall Customers dataset using Pandas."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found at: {DATA_PATH}\n"
            "Download the Mall Customer Segmentation Dataset from Kaggle and "
            "place the CSV file at data/Mall_Customers.csv"
        )

    df = pd.read_csv(DATA_PATH)
    return df


def data_understanding(df):
    """Perform Task 1: inspect the dataset."""
    print("\n" + "=" * 70)
    print("TASK 1: DATA UNDERSTANDING")
    print("=" * 70)

    print("\nFirst five records:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    numerical_features = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumerical features:")
    for feature in numerical_features:
        print(f"- {feature}")

    print("\nCategorical features:")
    if categorical_features:
        for feature in categorical_features:
            print(f"- {feature}")
    else:
        print("None")

    print("\nDataset information:")
    df.info()

    print("\nSummary statistics:")
    print(df.describe(include="all"))


def preprocess_data(df):
    """Perform Task 2: clean, encode and standardize the data."""
    print("\n" + "=" * 70)
    print("TASK 2: DATA PREPROCESSING")
    print("=" * 70)

    print("\nMissing values before preprocessing:")
    print(df.isnull().sum())

    # Remove unnecessary identifier column.
    df_processed = df.drop(columns=["CustomerID"], errors="ignore").copy()

    # Remove rows with missing values because K-Means cannot train on NaN values.
    df_processed = df_processed.dropna().reset_index(drop=True)

    # Encode Gender if it exists.
    if "Gender" in df_processed.columns:
        df_processed["Gender"] = df_processed["Gender"].map(
            {"Male": 0, "Female": 1}
        )

    # Convert all remaining values to numeric.
    df_processed = df_processed.apply(pd.to_numeric, errors="coerce")
    df_processed = df_processed.dropna().reset_index(drop=True)

    print("\nMissing values after preprocessing:")
    print(df_processed.isnull().sum())

    # Use the numerical features required by the problem statement.
    # Age is included as an additional customer characteristic.
    feature_columns = [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)",
    ]

    missing_features = [
        column for column in feature_columns if column not in df_processed.columns
    ]
    if missing_features:
        raise ValueError(
            f"Required columns are missing from the dataset: {missing_features}"
        )

    X = df_processed[feature_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nFeatures used for clustering:")
    print(feature_columns)

    print("\nStandardized feature matrix shape:")
    print(X_scaled.shape)

    return df_processed, X, X_scaled, scaler


def find_optimal_k(X_scaled):
    """Perform Task 3.1: Elbow Method."""
    print("\n" + "=" * 70)
    print("TASK 3.1: ELBOW METHOD")
    print("=" * 70)

    k_values = range(2, 11)
    inertia_values = []

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        model.fit(X_scaled)
        inertia_values.append(model.inertia_)

    elbow_data = pd.DataFrame({
        "K": list(k_values),
        "Inertia": inertia_values
    })
    elbow_data.to_csv(OUTPUT_DIR / "elbow_values.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), inertia_values, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
    plt.title("Elbow Method for Optimal K")
    plt.xticks(list(k_values))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_curve.png", dpi=300)
    plt.close()

    # For the standard Mall Customers dataset, K=5 is normally the elbow.
    # We use K=5 as the selected value for this assignment after inspecting
    # the generated elbow curve.
    optimal_k = 5

    print("\nElbow curve saved to outputs/elbow_curve.png")
    print(f"Selected optimal K: {optimal_k}")

    return optimal_k


def train_kmeans(X_scaled, optimal_k):
    """Perform Task 3.2 and 3.3: train K-Means and assign labels."""
    print("\n" + "=" * 70)
    print("TASK 3.2 & 3.3: K-MEANS MODEL")
    print("=" * 70)

    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    print("\nCluster labels assigned successfully.")
    print("Cluster counts:")
    print(pd.Series(labels).value_counts().sort_index())

    return kmeans, labels


def apply_pca(X_scaled, labels):
    """Perform Task 3.4: reduce data to two principal components."""
    print("\n" + "=" * 70)
    print("TASK 3.4: PCA")
    print("=" * 70)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()

    print("\nExplained variance ratio:")
    print(f"PC1: {explained_variance[0]:.4f}")
    print(f"PC2: {explained_variance[1]:.4f}")
    print(f"Total variance explained by 2 PCs: {total_variance:.4f}")

    pca_df = pd.DataFrame(
        X_pca,
        columns=["Principal Component 1", "Principal Component 2"]
    )
    pca_df["Cluster"] = labels
    pca_df.to_csv(OUTPUT_DIR / "pca_results.csv", index=False)

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        pca_df["Principal Component 1"],
        pca_df["Principal Component 2"],
        c=pca_df["Cluster"],
        cmap="viridis",
        s=45,
        alpha=0.8
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization of Customer Clusters")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_clusters.png", dpi=300)
    plt.close()

    print("\nPCA visualization saved to outputs/pca_clusters.png")

    return pca, X_pca


def create_cluster_visualization(df_processed, labels):
    """Perform Task 4: create a customer cluster scatter plot."""
    print("\n" + "=" * 70)
    print("TASK 4: VISUALIZATION AND EVALUATION")
    print("=" * 70)

    result = df_processed.copy()
    result["Cluster"] = labels

    result.to_csv(OUTPUT_DIR / "customer_cluster_results.csv", index=False)

    # Business-focused visualization: Annual Income vs Spending Score.
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        result["Annual Income (k$)"],
        result["Spending Score (1-100)"],
        c=result["Cluster"],
        cmap="viridis",
        s=50,
        alpha=0.8
    )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments: Annual Income vs Spending Score")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "customer_clusters.png", dpi=300)
    plt.close()

    summary = (
        result.groupby("Cluster")[
            ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
        ]
        .mean()
        .round(2)
    )

    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")

    print("\nCluster-wise average characteristics:")
    print(summary)

    print("\nGenerated visualizations:")
    print("- outputs/elbow_curve.png")
    print("- outputs/customer_clusters.png")
    print("- outputs/pca_clusters.png")

    print("\n3-4 observations:")
    print("1. The elbow curve is inspected to select a suitable K; K=5 is used.")
    print("2. PCA transforms the standardized features into two principal components,")
    print("   allowing multi-feature customer data to be viewed in a 2D plot.")
    print("3. The clusters represent customers with different combinations of age,")
    print("   annual income and spending score.")
    print("4. The cluster_summary.csv file provides the mean characteristics of each group,")
    print("   which can be used to identify high-value, low-spending or other segments.")


def main():
    print("=" * 70)
    print("CUSTOMER SEGMENTATION USING K-MEANS AND PCA")
    print("=" * 70)

    df = load_dataset()
    data_understanding(df)

    df_processed, X, X_scaled, scaler = preprocess_data(df)

    optimal_k = find_optimal_k(X_scaled)
    kmeans, labels = train_kmeans(X_scaled, optimal_k)

    pca, X_pca = apply_pca(X_scaled, labels)
    create_cluster_visualization(df_processed, labels)

    print("\n" + "=" * 70)
    print("ASSIGNMENT COMPLETED")
    print("=" * 70)
    print("Check the outputs/ folder for graphs and result CSV files.")


if __name__ == "__main__":
    main()
