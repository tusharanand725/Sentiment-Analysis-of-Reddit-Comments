#!/usr/bin/env python3
"""
Reddit Sentiment Analysis Pipeline
Complete end-to-end machine learning pipeline for sentiment analysis of Reddit comments

This script integrates:
1. Data preprocessing with TF-IDF and feature engineering
2. K-Means clustering for document similarity analysis
3. Optimized KNN classification with hyperparameter tuning

Author: [Your Name]
Date: July 2025
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_preprocessing import RedditCommentPreprocessor, load_sample_data
from kmeans_clustering import RedditCommentClustering
from knn_classifier import OptimizedKNNClassifier

class RedditSentimentAnalysisPipeline:
    """
    Complete pipeline for Reddit comment sentiment analysis
    """
    
    def __init__(self, random_state=42, verbose=True):
        self.random_state = random_state
        self.verbose = verbose
        self.preprocessor = RedditCommentPreprocessor()
        self.clusterer = RedditCommentClustering(random_state=random_state)
        self.classifier = OptimizedKNNClassifier(random_state=random_state)
        
        # Pipeline results storage
        self.results = {
            'preprocessing': {},
            'clustering': {},
            'classification': {},
            'pipeline_metrics': {}
        }
        
    def log(self, message):
        """Logging utility"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def load_data(self, data_path=None, sample_data=False):
        """
        Load Reddit comments data
        """
        self.log("Loading data...")
        
        if sample_data or data_path is None:
            self.log("Classification results saved to files")
        
        self.log(f"Classification completed in {self.results['classification']['processing_time']:.2f} seconds")
        return evaluation_results, importance_df
    
    def generate_pipeline_report(self):
        """
        Generate comprehensive pipeline report
        """
        self.log("Generating pipeline report...")
        
        report = []
        report.append("Reddit Sentiment Analysis Pipeline Report")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Preprocessing results
        if 'preprocessing' in self.results:
            report.append("1. DATA PREPROCESSING")
            report.append("-" * 30)
            prep_results = self.results['preprocessing']
            report.append(f"Original comments: {prep_results['original_comments']:,}")
            report.append(f"Feature matrix shape: {prep_results['feature_matrix_shape']}")
            report.append(f"TF-IDF matrix shape: {prep_results['tfidf_matrix_shape']}")
            report.append(f"Processing time: {prep_results['processing_time']:.2f} seconds")
            report.append("")
        
        # Clustering results
        if 'clustering' in self.results:
            report.append("2. K-MEANS CLUSTERING")
            report.append("-" * 30)
            clust_results = self.results['clustering']
            report.append(f"Number of clusters: {clust_results['n_clusters']}")
            report.append(f"Silhouette score: {clust_results['silhouette_score']:.4f}")
            report.append(f"Inertia: {clust_results['inertia']:.2f}")
            report.append(f"Suggested optimal k: {clust_results['optimal_k_suggested']}")
            report.append(f"Processing time: {clust_results['processing_time']:.2f} seconds")
            report.append("")
        
        # Classification results
        if 'classification' in self.results:
            report.append("3. KNN CLASSIFICATION")
            report.append("-" * 30)
            class_results = self.results['classification']
            report.append(f"Best parameters: {class_results['best_parameters']}")
            report.append(f"Cross-validation F1-score: {class_results['cv_f1_score']:.4f}")
            report.append(f"Test accuracy: {class_results['test_accuracy']:.4f}")
            report.append(f"Test precision: {class_results['test_precision']:.4f}")
            report.append(f"Test recall: {class_results['test_recall']:.4f}")
            report.append(f"Test F1-score: {class_results['test_f1_score']:.4f}")
            report.append(f"Processing time: {class_results['processing_time']:.2f} seconds")
            report.append("")
            report.append("Top 10 Important Features:")
            for i, feature in enumerate(class_results['top_features'], 1):
                report.append(f"  {i}. {feature}")
            report.append("")
        
        # Pipeline metrics
        total_time = sum([
            self.results.get('preprocessing', {}).get('processing_time', 0),
            self.results.get('clustering', {}).get('processing_time', 0),
            self.results.get('classification', {}).get('processing_time', 0)
        ])
        
        report.append("4. OVERALL PIPELINE METRICS")
        report.append("-" * 30)
        report.append(f"Total processing time: {total_time:.2f} seconds")
        report.append(f"Average processing time per comment: {total_time/self.results['preprocessing']['original_comments']*1000:.2f} ms")
        report.append("")
        
        # Technical specifications
        report.append("5. TECHNICAL SPECIFICATIONS")
        report.append("-" * 30)
        report.append("Preprocessing:")
        report.append("  - TF-IDF vectorization with n-grams (1,2)")
        report.append("  - Advanced feature engineering (text statistics, punctuation)")
        report.append("  - Text cleaning and normalization")
        report.append("  - Lemmatization and stop word removal")
        report.append("")
        report.append("Clustering:")
        report.append("  - K-Means algorithm with multiple initializations")
        report.append("  - Silhouette analysis for cluster validation")
        report.append("  - PCA visualization for cluster interpretation")
        report.append("")
        report.append("Classification:")
        report.append("  - K-Nearest Neighbors with hyperparameter tuning")
        report.append("  - Grid search with cross-validation")
        report.append("  - Comprehensive evaluation metrics")
        report.append("  - Feature importance analysis")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('pipeline_report.txt', 'w') as f:
            f.write(report_text)
        
        self.log("Pipeline report saved to 'pipeline_report.txt'")
        
        if self.verbose:
            print("\n" + report_text)
        
        return report_text
    
    def run_complete_pipeline(self, data_path=None, n_clusters=10, perform_tuning=True, sample_data=False):
        """
        Run the complete end-to-end pipeline
        """
        pipeline_start_time = time.time()
        self.log("Starting complete Reddit sentiment analysis pipeline")
        
        try:
            # Step 1: Load data
            df = self.load_data(data_path, sample_data)
            
            # Step 2: Preprocessing
            preprocessing_result = self.run_preprocessing(df)
            
            # Step 3: Clustering
            cluster_labels, cluster_analysis = self.run_clustering(
                preprocessing_result, n_clusters
            )
            
            # Step 4: Classification
            evaluation_results, importance_df = self.run_classification(
                preprocessing_result, perform_tuning
            )
            
            # Step 5: Generate report
            total_pipeline_time = time.time() - pipeline_start_time
            self.results['pipeline_metrics']['total_time'] = total_pipeline_time
            
            report = self.generate_pipeline_report()
            
            self.log(f"Complete pipeline finished in {total_pipeline_time:.2f} seconds")
            self.log("All results saved to files and visualizations generated")
            
            return {
                'preprocessing': preprocessing_result,
                'clustering': (cluster_labels, cluster_analysis),
                'classification': (evaluation_results, importance_df),
                'report': report
            }
            
        except Exception as e:
            self.log(f"Pipeline error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_new_comments(self, comments_list, load_models=True):
        """
        Predict sentiment for new comments using trained models
        """
        if load_models:
            try:
                # Load saved models
                self.preprocessor.load_preprocessor('reddit_preprocessor.pkl')
                self.classifier.load_model('optimized_knn_model.pkl')
                self.log("Models loaded successfully")
            except Exception as e:
                self.log(f"Error loading models: {e}")
                return None
        
        # Create DataFrame for new comments
        new_df = pd.DataFrame({'comment': comments_list})
        
        # Preprocess new comments
        self.log("Preprocessing new comments...")
        preprocessing_result = self.preprocessor.preprocess_dataset(new_df)
        
        # Make predictions
        self.log("Making predictions...")
        predictions = self.classifier.predict_new_comments(
            preprocessing_result['feature_matrix'].values
        )
        
        # Combine results
        results = []
        for i, comment in enumerate(comments_list):
            result = {
                'comment': comment,
                'predicted_sentiment': predictions[i]['prediction'],
                'confidence': predictions[i]['confidence'],
                'probabilities': predictions[i]['probabilities']
            }
            results.append(result)
        
        return results

def create_project_structure():
    """
    Create project directory structure
    """
    directories = [
        'data',
        'models',
        'results',
        'visualizations',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Project directory structure created")

def main():
    """
    Main execution function
    """
    print("Reddit Sentiment Analysis - Complete Pipeline")
    print("=" * 80)
    print("This pipeline implements:")
    print("• TF-IDF and advanced feature engineering for preprocessing")
    print("• K-Means clustering for document similarity analysis")
    print("• Optimized KNN classification with hyperparameter tuning")
    print("=" * 80)
    
    # Create project structure
    create_project_structure()
    
    # Configuration
    CONFIG = {
        'n_clusters': 10,
        'perform_hyperparameter_tuning': True,
        'use_sample_data': True,  # Set to False if you have real data
        'data_path': None,  # Path to your CSV file
        'random_state': 42
    }
    
    # Initialize pipeline
    pipeline = RedditSentimentAnalysisPipeline(
        random_state=CONFIG['random_state'],
        verbose=True
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        data_path=CONFIG['data_path'],
        n_clusters=CONFIG['n_clusters'],
        perform_tuning=CONFIG['perform_hyperparameter_tuning'],
        sample_data=CONFIG['use_sample_data']
    )
    
    if results:
        print("\n🎉 Pipeline completed successfully!")
        print("\nGenerated files:")
        print("📊 Data files:")
        print("  • processed_comments.csv - Cleaned and processed comments")
        print("  • processed_features.csv - Feature matrix for ML")
        print("  • comments_with_clusters.csv - Comments with cluster assignments")
        print("  • feature_importance.csv - Feature importance analysis")
        
        print("\n🤖 Model files:")
        print("  • reddit_preprocessor.pkl - Preprocessing pipeline")
        print("  • kmeans_clustering_model.pkl - Clustering model")
        print("  • optimized_knn_model.pkl - Classification model")
        
        print("\n📈 Visualizations:")
        print("  • cluster_optimization.png - Optimal cluster analysis")
        print("  • cluster_visualization_pca.png - PCA cluster visualization")
        print("  • cluster_distribution.png - Cluster distribution charts")
        print("  • confusion_matrix.png - Classification confusion matrix")
        print("  • hyperparameter_analysis.png - KNN tuning results")
        print("  • feature_importance.png - Top important features")
        
        print("\n📋 Reports:")
        print("  • pipeline_report.txt - Complete analysis report")
        
        # Demonstration of prediction on new comments
        print("\n" + "="*50)
        print("DEMONSTRATION: Predicting new comments")
        print("="*50)
        
        new_comments = [
            "This is absolutely amazing! I love it so much!",
            "This is terrible and I hate everything about it.",
            "Not sure how I feel about this, it's okay I guess.",
            "Thanks for sharing this awesome content!",
            "This is boring and pointless."
        ]
        
        predictions = pipeline.predict_new_comments(new_comments, load_models=False)
        
        if predictions:
            for pred in predictions:
                print(f"\nComment: '{pred['comment'][:50]}...'")
                print(f"Predicted Sentiment: {pred['predicted_sentiment']}")
                print(f"Confidence: {pred['confidence']:.3f}")
        
    else:
        print("\n❌ Pipeline failed. Check the logs for details.")

if __name__ == "__main__":
    main()("Using sample data for demonstration")
            df = load_sample_data()
            
            # Expand sample data for better analysis
            expanded_data = []
            for _ in range(20):  # Create more samples
                expanded_data.append(df)
            df = pd.concat(expanded_data, ignore_index=True)
            
            # Add some noise to make it more realistic
            df['comment'] = df['comment'] + " " + np.random.choice(
                ['Great!', 'Hmm...', 'Interesting.', 'Not sure.', 'Cool!'], 
                size=len(df)
            )
        else:
            self.log(f"Loading data from {data_path}")
            try:
                df = pd.read_csv(data_path)
            except Exception as e:
                self.log(f"Error loading data: {e}")
                self.log("Falling back to sample data")
                df = load_sample_data()
        
        self.log(f"Loaded {len(df)} comments")
        return df
    
    def run_preprocessing(self, df, save_results=True):
        """
        Run the preprocessing pipeline
        """
        self.log("Starting preprocessing phase...")
        start_time = time.time()
        
        # Preprocess the data
        preprocessing_result = self.preprocessor.preprocess_dataset(df)
        
        # Store results
        self.results['preprocessing'] = {
            'original_comments': len(df),
            'feature_matrix_shape': preprocessing_result['feature_matrix'].shape,
            'tfidf_matrix_shape': preprocessing_result['tfidf_matrix'].shape,
            'processing_time': time.time() - start_time
        }
        
        if save_results:
            # Save preprocessed data
            preprocessing_result['processed_df'].to_csv('processed_comments.csv', index=False)
            preprocessing_result['feature_matrix'].to_csv('processed_features.csv', index=False)
            self.preprocessor.save_preprocessor('reddit_preprocessor.pkl')
            
            self.log("Preprocessing results saved to files")
        
        self.log(f"Preprocessing completed in {self.results['preprocessing']['processing_time']:.2f} seconds")
        return preprocessing_result
    
    def run_clustering(self, preprocessing_result, n_clusters=10, save_results=True):
        """
        Run the clustering analysis
        """
        self.log("Starting clustering phase...")
        start_time = time.time()
        
        # Set number of clusters
        self.clusterer.n_clusters = n_clusters
        
        # Extract TF-IDF features for clustering
        feature_matrix = preprocessing_result['feature_matrix']
        # Assume TF-IDF features are the first columns (excluding numerical features)
        n_numerical_features = 12  # Based on preprocessing
        tfidf_features = feature_matrix.iloc[:, :-n_numerical_features]
        
        # Find optimal clusters
        optimization_results = self.clusterer.find_optimal_clusters(
            tfidf_features.values, max_clusters=min(15, len(tfidf_features)//2)
        )
        
        # Fit clustering
        cluster_labels = self.clusterer.fit_kmeans(tfidf_features.values)
        
        # Analyze clusters
        cluster_analysis = self.clusterer.analyze_clusters(
            tfidf_features.values,
            tfidf_features.columns.tolist(),
            preprocessing_result['processed_df']['comment'].values
        )
        
        # Visualize clusters
        self.clusterer.visualize_clusters(tfidf_features.values, method='pca')
        
        # Distribution analysis
        sentiment_labels = preprocessing_result['processed_df']['sentiment'].values
        distribution_analysis = self.clusterer.cluster_distribution_analysis(sentiment_labels)
        
        # Store results
        self.results['clustering'] = {
            'n_clusters': n_clusters,
            'silhouette_score': self.clusterer.silhouette_score,
            'inertia': self.clusterer.inertia,
            'optimal_k_suggested': optimization_results['optimal_k'],
            'cluster_sizes': distribution_analysis['Count'].tolist(),
            'processing_time': time.time() - start_time
        }
        
        if save_results:
            # Save clustering results
            preprocessing_result['processed_df']['cluster'] = cluster_labels
            preprocessing_result['processed_df'].to_csv('comments_with_clusters.csv', index=False)
            self.clusterer.save_clustering_model('kmeans_clustering_model.pkl')
            
            self.log("Clustering results saved to files")
        
        self.log(f"Clustering completed in {self.results['clustering']['processing_time']:.2f} seconds")
        return cluster_labels, cluster_analysis
    
    def run_classification(self, preprocessing_result, perform_tuning=True, save_results=True):
        """
        Run the classification pipeline
        """
        self.log("Starting classification phase...")
        start_time = time.time()
        
        # Prepare data
        feature_matrix = preprocessing_result['feature_matrix']
        target_labels = preprocessing_result['processed_df']['sentiment']
        
        self.classifier.feature_names = feature_matrix.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = self.classifier.prepare_data(
            feature_matrix.values, target_labels.values
        )
        
        # Hyperparameter tuning
        if perform_tuning:
            self.log("Performing hyperparameter tuning...")
            grid_search = self.classifier.hyperparameter_tuning(X_train, y_train)
        else:
            self.log("Skipping hyperparameter tuning, using default parameters")
            self.classifier.knn_model = KNeighborsClassifier(
                n_neighbors=5, weights='distance', metric='minkowski'
            )
        
        # Train model
        self.classifier.train_model(X_train, y_train)
        
        # Evaluate model
        evaluation_results = self.classifier.evaluate_model(X_test, y_test, detailed=True)
        
        # Feature importance
        importance_df = self.classifier.feature_importance_analysis(X_train, y_train)
        
        # Store results
        self.results['classification'] = {
            'best_parameters': self.classifier.best_params,
            'cv_f1_score': self.classifier.training_scores.get('cv_mean', 0),
            'test_accuracy': evaluation_results['accuracy'],
            'test_precision': evaluation_results['precision'],
            'test_recall': evaluation_results['recall'],
            'test_f1_score': evaluation_results['f1_score'],
            'top_features': importance_df.head(10)['Feature'].tolist(),
            'processing_time': time.time() - start_time
        }
        
        if save_results:
            # Save classification results
            self.classifier.save_model('optimized_knn_model.pkl')
            importance_df.to_csv('feature_importance.csv', index=False)
            
            self.log