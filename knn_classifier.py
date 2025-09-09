import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

class OptimizedKNNClassifier:
    """
    Optimized KNN-based text classifier for Reddit comment sentiment analysis
    with hyperparameter tuning and comprehensive evaluation metrics
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.knn_model = None
        self.label_encoder = LabelEncoder()
        self.best_params = None
        self.feature_names = None
        self.training_scores = {}
        
    def prepare_data(self, feature_matrix, target_labels, test_size=0.2):
        """
        Prepare data for training and testing
        """
        print("Preparing data for classification...")
        
        # Encode target labels if they are strings
        if target_labels.dtype == 'object':
            encoded_labels = self.label_encoder.fit_transform(target_labels)
        else:
            encoded_labels = target_labels
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, 
            encoded_labels, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=encoded_labels
        )
        
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of classes: {len(np.unique(encoded_labels))}")
        
        return X_train, X_test, y_train, y_test
    
    def hyperparameter_tuning(self, X_train, y_train, cv_folds=5):
        """
        Perform comprehensive hyperparameter tuning for KNN classifier
        """
        print("Starting hyperparameter tuning...")
        
        # Define parameter grid for GridSearch
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]  # for minkowski metric
        }
        
        # Initialize KNN classifier
        knn = KNeighborsClassifier()
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Perform grid search
        print("Performing grid search with cross-validation...")
        grid_search = GridSearchCV(
            knn,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store results
        self.best_params = grid_search.best_params_
        self.knn_model = grid_search.best_estimator_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")
        
        # Analyze hyperparameter importance
        self.analyze_hyperparameters(grid_search)
        
        return grid_search
    
    def analyze_hyperparameters(self, grid_search):
        """
        Analyze the impact of different hyperparameters
        """
        print("\nAnalyzing hyperparameter performance...")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Plot performance vs k (n_neighbors)
        plt.figure(figsize=(15, 10))
        
        # K vs Performance
        plt.subplot(2, 3, 1)
        k_values = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        k_scores = []
        for k in k_values:
            k_mask = results_df['param_n_neighbors'] == k
            if k_mask.any():
                k_scores.append(results_df[k_mask]['mean_test_score'].max())
            else:
                k_scores.append(0)
        
        plt.plot(k_values, k_scores, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('F1-Score')
        plt.title('KNN Performance vs K Value')
        plt.grid(True, alpha=0.3)
        
        # Weights comparison
        plt.subplot(2, 3, 2)
        weights_performance = results_df.groupby('param_weights')['mean_test_score'].mean()
        weights_performance.plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.xlabel('Weights')
        plt.ylabel('Average F1-Score')
        plt.title('Performance by Weight Type')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Distance metrics comparison
        plt.subplot(2, 3, 3)
        metrics_performance = results_df.groupby('param_metric')['mean_test_score'].mean()
        metrics_performance.plot(kind='bar', color=['lightgreen', 'orange', 'pink'])
        plt.xlabel('Distance Metric')
        plt.ylabel('Average F1-Score')
        plt.title('Performance by Distance Metric')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Algorithm comparison
        plt.subplot(2, 3, 4)
        algorithm_performance = results_df.groupby('param_algorithm')['mean_test_score'].mean()
        algorithm_performance.plot(kind='bar', color=['gold', 'lightblue', 'lightcyan', 'plum'])
        plt.xlabel('Algorithm')
        plt.ylabel('Average F1-Score')
        plt.title('Performance by Algorithm')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Cross-validation score distribution
        plt.subplot(2, 3, 5)
        plt.hist(results_df['mean_test_score'], bins=20, alpha=0.7, color='lightsteelblue')
        plt.axvline(grid_search.best_score_, color='red', linestyle='--', linewidth=2, label='Best Score')
        plt.xlabel('F1-Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of CV Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Top 10 configurations
        plt.subplot(2, 3, 6)
        top_10 = results_df.nlargest(10, 'mean_test_score')
        plt.barh(range(10), top_10['mean_test_score'], color='lightseagreen')
        plt.ylabel('Configuration Rank')
        plt.xlabel('F1-Score')
        plt.title('Top 10 Configurations')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_model(self, X_train, y_train):
        """
        Train the KNN model with optimized parameters
        """
        if self.knn_model is None:
            print("No optimized model found. Using default parameters.")
            self.knn_model = KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='minkowski'
            )
        
        print("Training optimized KNN model...")
        self.knn_model.fit(X_train, y_train)
        
        # Perform cross-validation to assess model stability
        cv_scores = cross_val_score(
            self.knn_model, 
            X_train, 
            y_train, 
            cv=5, 
            scoring='f1_weighted'
        )
        
        self.training_scores = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"Cross-validation F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.knn_model
    
    def evaluate_model(self, X_test, y_test, detailed=True):
        """
        Comprehensive model evaluation with precision, recall, and F1-score
        """
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.knn_model.predict(X_test)
        y_pred_proba = self.knn_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store evaluation results
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print("\nModel Performance Metrics:")
        print("=" * 50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        if detailed:
            self.detailed_evaluation(y_test, y_pred, y_pred_proba)
        
        return evaluation_results
    
    def detailed_evaluation(self, y_test, y_pred, y_pred_proba):
        """
        Detailed evaluation with confusion matrix and classification report
        """
        # Classification report
        print("\nDetailed Classification Report:")
        print("=" * 60)
        
        # Get class names
        if hasattr(self.label_encoder, 'classes_'):
            class_names = self.label_encoder.classes_
        else:
            class_names = [f'Class_{i}' for i in range(len(np.unique(y_test)))]
        
        report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
        print(report)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        
        metrics_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        print("\nPer-Class Metrics:")
        print(metrics_df.to_string(index=False))
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred, class_names)
        
        # ROC curves for multiclass (if applicable)
        if len(class_names) <= 5:  # Only for reasonable number of classes
            self.plot_roc_curves(y_test, y_pred_proba, class_names)
    
    def plot_confusion_matrix(self, y_test, y_pred, class_names):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Raw Counts)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_test, y_pred_proba, class_names):
        """
        Plot ROC curves for multiclass classification
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
        n_classes = y_test_bin.shape[1]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(12, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multiclass Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self, X_train, y_train, top_k=20):
        """
        Analyze feature importance using permutation importance
        """
        from sklearn.inspection import permutation_importance
        
        print(f"Analyzing top {top_k} most important features...")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.knn_model, X_train, y_train,
            n_repeats=5, random_state=self.random_state
        )
        
        # Create feature importance DataFrame
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_k)
        
        plt.barh(range(top_k), top_features['Importance'], 
                xerr=top_features['Std'], alpha=0.7, color='skyblue')
        plt.yticks(range(top_k), top_features['Feature'])
        plt.xlabel('Permutation Importance')
        plt.title(f'Top {top_k} Most Important Features')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nTop {top_k} Most Important Features:")
        print(top_features.to_string(index=False))
        
        return importance_df
    
    def predict_new_comments(self, new_features):
        """
        Predict sentiment for new comments
        """
        predictions = self.knn_model.predict(new_features)
        probabilities = self.knn_model.predict_proba(new_features)
        
        # Decode predictions if label encoder was used
        if hasattr(self.label_encoder, 'classes_'):
            predicted_labels = self.label_encoder.inverse_transform(predictions)
            class_names = self.label_encoder.classes_
        else:
            predicted_labels = predictions
            class_names = [f'Class_{i}' for i in range(probabilities.shape[1])]
        
        results = []
        for i, (pred, prob) in enumerate(zip(predicted_labels, probabilities)):
            result = {
                'prediction': pred,
                'confidence': prob.max(),
                'probabilities': dict(zip(class_names, prob))
            }
            results.append(result)
        
        return results
    
    def save_model(self, filepath):
        """Save the trained model and associated components"""
        model_data = {
            'knn_model': self.knn_model,
            'label_encoder': self.label_encoder,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'training_scores': self.training_scores
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.knn_model = model_data['knn_model']
        self.label_encoder = model_data['label_encoder']
        self.best_params = model_data['best_params']
        self.feature_names = model_data.get('feature_names')
        self.training_scores = model_data.get('training_scores', {})
        
        print(f"Model loaded from {filepath}")

def main():
    """
    Main function demonstrating the KNN classification pipeline
    """
    print("Optimized KNN Classifier for Reddit Sentiment Analysis")
    print("=" * 60)
    
    try:
        # Load preprocessed data
        processed_df = pd.read_csv('processed_comments.csv')
        feature_matrix = pd.read_csv('processed_features.csv')
        
        print(f"Loaded dataset with {len(processed_df)} comments")
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Initialize classifier
        knn_classifier = OptimizedKNNClassifier()
        knn_classifier.feature_names = feature_matrix.columns.tolist()
        
        # Prepare data
        X_train, X_test, y_train, y_test = knn_classifier.prepare_data(
            feature_matrix.values,
            processed_df['sentiment'].values
        )
        
        # Hyperparameter tuning
        grid_search = knn_classifier.hyperparameter_tuning(X_train, y_train)
        
        # Train the model
        knn_classifier.train_model(X_train, y_train)
        
        # Evaluate the model
        evaluation_results = knn_classifier.evaluate_model(X_test, y_test, detailed=True)
        
        # Feature importance analysis
        importance_df = knn_classifier.feature_importance_analysis(X_train, y_train)
        
        # Save the model
        knn_classifier.save_model('optimized_knn_model.pkl')
        
        # Save evaluation results
        results_summary = {
            'best_parameters': knn_classifier.best_params,
            'training_cv_score': knn_classifier.training_scores,
            'test_performance': {
                'accuracy': evaluation_results['accuracy'],
                'precision': evaluation_results['precision'],
                'recall': evaluation_results['recall'],
                'f1_score': evaluation_results['f1_score']
            }
        }
        
        # Save results to file
        with open('knn_results_summary.txt', 'w') as f:
            f.write("KNN Classifier Results Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best Parameters: {results_summary['best_parameters']}\n\n")
            f.write("Training Performance:\n")
            f.write(f"CV F1-Score: {results_summary['training_cv_score']['cv_mean']:.4f}\n\n")
            f.write("Test Performance:\n")
            for metric, value in results_summary['test_performance'].items():
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        print("\nKNN classification pipeline completed successfully!")
        print("Results saved to files and visualizations generated.")
        
    except FileNotFoundError as e:
        print(f"Required file not found: {e}")
        print("Please run data_preprocessing.py first.")
        
        # Create sample data for demonstration
        print("\nCreating sample data for demonstration...")
        from data_preprocessing import load_sample_data, RedditCommentPreprocessor
        
        # Load and preprocess sample data
        df = load_sample_data()
        preprocessor = RedditCommentPreprocessor()
        result = preprocessor.preprocess_dataset(df)
        
        # Initialize classifier
        knn_classifier = OptimizedKNNClassifier()
        knn_classifier.feature_names = result['feature_matrix'].columns.tolist()
        
        # Prepare and train on sample data
        X_train, X_test, y_train, y_test = knn_classifier.prepare_data(
            result['feature_matrix'].values,
            result['processed_df']['sentiment'].values
        )
        
        # Train with default parameters (skip tuning for small sample)
        knn_classifier.knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.train_model(X_train, y_train)
        
        # Evaluate
        evaluation_results = knn_classifier.evaluate_model(X_test, y_test)
        
        print("Sample KNN classification completed!")

if __name__ == "__main__":
    main()