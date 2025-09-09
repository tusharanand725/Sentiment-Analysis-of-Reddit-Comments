import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import warnings
warnings.filterwarnings('ignore')

class RedditCommentClustering:
    """
    K-Means clustering implementation for Reddit comments sentiment analysis
    using TF-IDF matrix to classify comments into distinct clusters based on document similarity
    """
    
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_labels = None
        self.silhouette_score = None
        self.inertia = None
        self.cluster_centers = None
        
    def find_optimal_clusters(self, tfidf_matrix, max_clusters=15):
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        """
        print("Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(tfidf_matrix, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(tfidf_matrix, cluster_labels))
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Elbow curve
        axes[0].plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method for Optimal k')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette score
        axes[1].plot(k_range, silhouette_scores, marker='s', color='orange', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')
        axes[1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz score
        axes[2].plot(k_range, calinski_scores, marker='^', color='green', linewidth=2, markersize=8)
        axes[2].set_xlabel('Number of Clusters (k)')
        axes[2].set_ylabel('Calinski-Harabasz Score')
        axes[2].set_title('Calinski-Harabasz Index')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cluster_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal k based on silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        
        return {
            'k_range': k_range,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'optimal_k': optimal_k
        }
    
    def fit_kmeans(self, tfidf_matrix):
        """
        Fit K-means clustering on TF-IDF matrix
        """
        print(f"Fitting K-means with {self.n_clusters} clusters...")
        
        # Initialize and fit K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        # Fit the model and get cluster labels
        self.cluster_labels = self.kmeans.fit_predict(tfidf_matrix)
        self.cluster_centers = self.kmeans.cluster_centers_
        self.inertia = self.kmeans.inertia_
        
        # Calculate clustering metrics
        self.silhouette_score = silhouette_score(tfidf_matrix, self.cluster_labels)
        self.calinski_score = calinski_harabasz_score(tfidf_matrix, self.cluster_labels)
        
        print(f"Clustering completed!")
        print(f"Inertia: {self.inertia:.2f}")
        print(f"Silhouette Score: {self.silhouette_score:.3f}")
        print(f"Calinski-Harabasz Score: {self.calinski_score:.2f}")
        
        return self.cluster_labels
    
    def analyze_clusters(self, tfidf_matrix, feature_names, original_comments=None, top_terms=10):
        """
        Analyze and interpret the clusters
        """
        print("\nAnalyzing clusters...")
        
        # Get cluster centers
        cluster_centers = self.cluster_centers
        
        # Analyze top terms for each cluster
        cluster_analysis = {}
        
        for i in range(self.n_clusters):
            # Get top terms for this cluster
            center = cluster_centers[i]
            top_indices = center.argsort()[-top_terms:][::-1]
            top_terms_list = [feature_names[idx] for idx in top_indices]
            top_scores = [center[idx] for idx in top_indices]
            
            cluster_analysis[i] = {
                'top_terms': top_terms_list,
                'top_scores': top_scores,
                'size': np.sum(self.cluster_labels == i),
                'percentage': (np.sum(self.cluster_labels == i) / len(self.cluster_labels)) * 100
            }
        
        # Print cluster analysis
        print(f"\nCluster Analysis (Top {top_terms} terms per cluster):")
        print("=" * 80)
        
        for cluster_id, analysis in cluster_analysis.items():
            print(f"\nCluster {cluster_id} ({analysis['size']} comments, {analysis['percentage']:.1f}%):")
            print("Top terms:", ", ".join(analysis['top_terms']))
            
            # Show sample comments if available
            if original_comments is not None:
                cluster_mask = self.cluster_labels == cluster_id
                sample_comments = original_comments[cluster_mask].head(2)
                print("Sample comments:")
                for idx, comment in enumerate(sample_comments, 1):
                    print(f"  {idx}. {comment[:100]}...")
            print("-" * 60)
        
        return cluster_analysis
    
    def visualize_clusters(self, tfidf_matrix, method='pca', sample_size=1000):
        """
        Visualize clusters using dimensionality reduction
        """
        print(f"Visualizing clusters using {method.upper()}...")
        
        # Sample data if too large
        if tfidf_matrix.shape[0] > sample_size:
            indices = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
            matrix_sample = tfidf_matrix[indices]
            labels_sample = self.cluster_labels[indices]
        else:
            matrix_sample = tfidf_matrix
            labels_sample = self.cluster_labels
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
            reduced_data = reducer.fit_transform(matrix_sample.toarray() if hasattr(matrix_sample, 'toarray') else matrix_sample)
            title = f'K-means Clusters Visualization (PCA) - {self.n_clusters} Clusters'
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
            reduced_data = reducer.fit_transform(matrix_sample.toarray() if hasattr(matrix_sample, 'toarray') else matrix_sample)
            title = f'K-means Clusters Visualization (t-SNE) - {self.n_clusters} Clusters'
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            reduced_data[:, 0], 
            reduced_data[:, 1], 
            c=labels_sample, 
            cmap='tab10', 
            alpha=0.7,
            s=50
        )
        
        plt.colorbar(scatter, label='Cluster')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        
        # Add cluster centers if using PCA
        if method.lower() == 'pca':
            centers_reduced = reducer.transform(self.cluster_centers)
            plt.scatter(
                centers_reduced[:, 0], 
                centers_reduced[:, 1], 
                marker='x', 
                s=300, 
                linewidths=3, 
                color='red',
                label='Centroids'
            )
            plt.legend()
        
        plt.savefig(f'cluster_visualization_{method}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def cluster_distribution_analysis(self, sentiment_labels=None):
        """
        Analyze the distribution of clusters and their relationship with sentiment
        """
        print("\nCluster Distribution Analysis:")
        print("=" * 50)
        
        # Basic cluster distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        cluster_dist = pd.DataFrame({
            'Cluster': unique,
            'Count': counts,
            'Percentage': (counts / len(self.cluster_labels)) * 100
        })
        
        print("Cluster Distribution:")
        print(cluster_dist.to_string(index=False))
        
        # Visualize cluster distribution
        plt.figure(figsize=(12, 5))
        
        # Pie chart
        plt.subplot(1, 2, 1)
        plt.pie(counts, labels=[f'Cluster {i}' for i in unique], autopct='%1.1f%%', startangle=90)
        plt.title('Cluster Distribution')
        
        # Bar chart
        plt.subplot(1, 2, 2)
        bars = plt.bar(unique, counts, color='skyblue', alpha=0.7)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Comments')
        plt.title('Comments per Cluster')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Sentiment analysis if labels provided
        if sentiment_labels is not None:
            print("\nCluster-Sentiment Relationship:")
            sentiment_cluster_analysis = pd.crosstab(
                self.cluster_labels, 
                sentiment_labels, 
                margins=True
            )
            print(sentiment_cluster_analysis)
            
            # Visualize sentiment distribution across clusters
            sentiment_cluster_df = pd.crosstab(self.cluster_labels, sentiment_labels)
            sentiment_cluster_df.plot(kind='bar', stacked=True, figsize=(12, 6))
            plt.title('Sentiment Distribution Across Clusters')
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of Comments')
            plt.legend(title='Sentiment')
            plt.xticks(rotation=0)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('cluster_sentiment_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return cluster_dist, sentiment_cluster_analysis
        
        return cluster_dist
    
    def save_clustering_model(self, filepath):
        """Save the clustering model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'cluster_labels': self.cluster_labels,
                'n_clusters': self.n_clusters,
                'silhouette_score': self.silhouette_score,
                'inertia': self.inertia
            }, f)
        print(f"Clustering model saved to {filepath}")
    
    def load_clustering_model(self, filepath):
        """Load a saved clustering model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.cluster_labels = data['cluster_labels']
            self.n_clusters = data['n_clusters']
            self.silhouette_score = data['silhouette_score']
            self.inertia = data['inertia']
        print(f"Clustering model loaded from {filepath}")

def main():
    """
    Main function demonstrating the clustering pipeline
    """
    print("Reddit Comment Clustering - Demo")
    print("=" * 50)
    
    # Load preprocessed data (assuming it exists from preprocessing step)
    try:
        # Load TF-IDF matrix and other data
        processed_df = pd.read_csv('processed_comments.csv')
        feature_matrix = pd.read_csv('processed_features.csv')
        
        # Load preprocessor to get TF-IDF matrix
        with open('reddit_preprocessor.pkl', 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        # Get TF-IDF features (assuming they're the first part of feature matrix)
        tfidf_features = feature_matrix.iloc[:, :-12]  # Exclude the last 12 numerical features
        feature_names = tfidf_features.columns.tolist()
        
        print(f"Loaded data: {len(processed_df)} comments")
        print(f"TF-IDF matrix shape: {tfidf_features.shape}")
        
        # Initialize clustering
        clusterer = RedditCommentClustering(n_clusters=10)
        
        # Find optimal number of clusters
        optimization_results = clusterer.find_optimal_clusters(tfidf_features.values)
        
        # Fit K-means clustering
        cluster_labels = clusterer.fit_kmeans(tfidf_features.values)
        
        # Analyze clusters
        cluster_analysis = clusterer.analyze_clusters(
            tfidf_features.values,
            feature_names,
            processed_df['comment'].values if 'comment' in processed_df.columns else None
        )
        
        # Visualize clusters
        clusterer.visualize_clusters(tfidf_features.values, method='pca')
        clusterer.visualize_clusters(tfidf_features.values, method='tsne')
        
        # Analyze cluster distribution
        sentiment_labels = processed_df['sentiment'].values if 'sentiment' in processed_df.columns else None
        distribution_analysis = clusterer.cluster_distribution_analysis(sentiment_labels)
        
        # Save the clustering model
        clusterer.save_clustering_model('kmeans_clustering_model.pkl')
        
        # Save cluster labels
        processed_df['cluster'] = cluster_labels
        processed_df.to_csv('comments_with_clusters.csv', index=False)
        
        print("\nClustering analysis completed successfully!")
        print("Results saved to various files and visualizations.")
        
    except FileNotFoundError as e:
        print(f"Required file not found: {e}")
        print("Please run data_preprocessing.py first to generate the required data files.")
        
        # Create sample data for demonstration
        print("\nCreating sample data for demonstration...")
        from data_preprocessing import load_sample_data, RedditCommentPreprocessor
        
        # Load and preprocess sample data
        df = load_sample_data()
        preprocessor = RedditCommentPreprocessor()
        result = preprocessor.preprocess_dataset(df)
        
        # Extract TF-IDF features
        tfidf_df = result['feature_matrix'].iloc[:, :-12]  # Exclude numerical features
        feature_names = tfidf_df.columns.tolist()
        
        # Run clustering on sample data
        clusterer = RedditCommentClustering(n_clusters=3)  # Fewer clusters for small sample
        cluster_labels = clusterer.fit_kmeans(tfidf_df.values)
        
        # Analyze results
        cluster_analysis = clusterer.analyze_clusters(
            tfidf_df.values,
            feature_names,
            df['comment'].values
        )
        
        print("Sample clustering analysis completed!")

if __name__ == "__main__":
    main()
