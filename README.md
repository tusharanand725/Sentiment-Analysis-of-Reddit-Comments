# Sentiment-Analysis-of-Reddit-Comments
Reddit Sentiment Analysis Project
A comprehensive machine learning pipeline for analyzing sentiment in Reddit comments using advanced NLP techniques, clustering, and optimized classification algorithms.
🎯 Project Overview
This project implements a complete end-to-end sentiment analysis system that:

Preprocesses Reddit comments using TF-IDF and advanced feature engineering techniques
Clusters comments into 10 distinct groups using K-Means clustering based on document similarity
Classifies sentiment using an optimized KNN classifier with hyperparameter tuning and comprehensive evaluation

🚀 Key Features
1. Advanced Data Preprocessing

TF-IDF Vectorization: Converts text to numerical features with n-gram support
Feature Engineering: Extracts text statistics, punctuation patterns, and Reddit-specific features
Text Cleaning: Comprehensive cleaning including URL removal, normalization, and noise reduction
Lemmatization & Stopword Removal: Advanced NLP preprocessing for better feature quality

2. Intelligent Clustering Analysis

K-Means Clustering: Groups comments into 10 clusters based on document similarity
Cluster Optimization: Automated optimal cluster detection using silhouette analysis
Visualization: PCA and t-SNE visualizations for cluster interpretation
Cluster Analysis: Detailed analysis of top terms and characteristics per cluster

3. Optimized Classification

KNN Classifier: Hyperparameter-tuned K-Nearest Neighbors algorithm
Grid Search: Comprehensive hyperparameter optimization with cross-validation
Performance Metrics: Detailed evaluation using precision, recall, F1-score, and confusion matrices
Feature Importance: Analysis of most influential features for classification

📊 Performance Metrics
The pipeline provides comprehensive evaluation including:

Accuracy, Precision, Recall, F1-Score: Standard classification metrics
Cross-validation: 5-fold stratified cross-validation for robust performance estimation
Confusion Matrix: Detailed classification results visualization
ROC Curves: Multi-class ROC analysis for model performance assessment
