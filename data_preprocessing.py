import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class RedditCommentPreprocessor:
    """
    A comprehensive preprocessing pipeline for Reddit comments
    implementing TF-IDF and advanced feature engineering techniques
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def clean_text(self, text):
        """
        Clean and normalize text data
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove Reddit-specific patterns
        text = re.sub(r'/u/\w+|/r/\w+', '', text)  # Remove usernames and subreddit mentions
        text = re.sub(r'\[deleted\]|\[removed\]', '', text)  # Remove deleted/removed content
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize and lemmatize text
        """
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def extract_text_features(self, df):
        """
        Extract advanced text features for sentiment analysis
        """
        features = pd.DataFrame()
        
        # Basic text statistics
        features['comment_length'] = df['cleaned_comment'].str.len()
        features['word_count'] = df['cleaned_comment'].str.split().str.len()
        features['avg_word_length'] = features['comment_length'] / features['word_count']
        features['avg_word_length'].fillna(0, inplace=True)
        
        # Sentiment-related features
        features['exclamation_count'] = df['comment'].str.count('!')
        features['question_count'] = df['comment'].str.count('\?')
        features['caps_ratio'] = df['comment'].str.count('[A-Z]') / df['comment'].str.len()
        features['caps_ratio'].fillna(0, inplace=True)
        
        # Punctuation features
        features['comma_count'] = df['comment'].str.count(',')
        features['period_count'] = df['comment'].str.count('\.')
        features['semicolon_count'] = df['comment'].str.count(';')
        
        # Reddit-specific features
        features['has_edit'] = df['comment'].str.contains('edit:', case=False, na=False).astype(int)
        features['has_update'] = df['comment'].str.contains('update:', case=False, na=False).astype(int)
        features['has_thanks'] = df['comment'].str.contains('thank', case=False, na=False).astype(int)
        
        return features
    
    def apply_tfidf(self, texts, max_features=5000, ngram_range=(1, 2)):
        """
        Apply TF-IDF vectorization to text data
        """
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95,
                strip_accents='unicode',
                lowercase=True,
                token_pattern=r'\w{1,}'
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        # Convert to DataFrame for easier handling
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        
        return tfidf_df, tfidf_matrix
    
    def preprocess_dataset(self, df, text_column='comment', target_column='sentiment'):
        """
        Complete preprocessing pipeline for Reddit comments dataset
        """
        print("Starting data preprocessing...")
        
        # Create a copy of the dataframe
        processed_df = df.copy()
        
        # Clean text data
        print("Cleaning text data...")
        processed_df['cleaned_comment'] = processed_df[text_column].apply(self.clean_text)
        
        # Tokenize and lemmatize
        print("Tokenizing and lemmatizing...")
        processed_df['processed_comment'] = processed_df['cleaned_comment'].apply(
            self.tokenize_and_lemmatize
        )
        
        # Extract advanced features
        print("Extracting text features...")
        text_features = self.extract_text_features(processed_df)
        
        # Apply TF-IDF
        print("Applying TF-IDF vectorization...")
        tfidf_df, tfidf_matrix = self.apply_tfidf(processed_df['processed_comment'])
        
        # Combine all features
        print("Combining features...")
        # Scale numerical features
        scaled_features = pd.DataFrame(
            self.scaler.fit_transform(text_features),
            columns=text_features.columns,
            index=text_features.index
        )
        
        # Combine TF-IDF features with scaled numerical features
        final_features = pd.concat([tfidf_df, scaled_features], axis=1)
        
        # Store feature names for later use
        self.feature_names = final_features.columns.tolist()
        
        print(f"Preprocessing complete! Final feature matrix shape: {final_features.shape}")
        
        # Prepare output dictionary
        result = {
            'processed_df': processed_df,
            'feature_matrix': final_features,
            'tfidf_matrix': tfidf_matrix,
            'target': processed_df[target_column] if target_column in processed_df.columns else None
        }
        
        return result
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor for future use"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
    
    def load_preprocessor(self, filepath):
        """Load a saved preprocessor"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.tfidf_vectorizer = data['tfidf_vectorizer']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']

def load_sample_data():
    """
    Generate sample Reddit comments data for testing
    """
    sample_data = {
        'comment': [
            "This is amazing! I love this subreddit so much!",
            "This is terrible. I hate everything about this post.",
            "Not sure how I feel about this. Mixed emotions here.",
            "Great job OP! Thanks for sharing this awesome content.",
            "This is boring and pointless. Why did you even post this?",
            "Interesting perspective. I never thought about it that way.",
            "EDIT: Thanks for the gold! This community is the best!",
            "Downvoted. This doesn't contribute anything to the discussion.",
            "Amazing work! Can't wait to see more updates on this project.",
            "This is confusing. Can someone explain what's happening here?"
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 
                     'neutral', 'positive', 'negative', 'positive', 'neutral'],
        'subreddit': ['r/MachineLearning', 'r/technology', 'r/AskReddit', 
                     'r/programming', 'r/mildlyinfuriating', 'r/explainlikeimfive',
                     'r/datascience', 'r/unpopularopinion', 'r/todayilearned', 'r/NoStupidQuestions']
    }
    
    return pd.DataFrame(sample_data)

if __name__ == "__main__":
    # Example usage
    print("Reddit Comment Preprocessor - Demo")
    print("=" * 50)
    
    # Load sample data
    df = load_sample_data()
    print(f"Loaded sample dataset with {len(df)} comments")
    
    # Initialize preprocessor
    preprocessor = RedditCommentPreprocessor()
    
    # Preprocess the data
    result = preprocessor.preprocess_dataset(df)
    
    # Display results
    print("\nPreprocessing Results:")
    print(f"Original comments: {len(df)}")
    print(f"Feature matrix shape: {result['feature_matrix'].shape}")
    print(f"TF-IDF matrix shape: {result['tfidf_matrix'].shape}")
    
    # Save preprocessor
    preprocessor.save_preprocessor('reddit_preprocessor.pkl')
    print("\nPreprocessor saved to 'reddit_preprocessor.pkl'")
    
    # Save processed data
    result['feature_matrix'].to_csv('processed_features.csv', index=False)
    result['processed_df'].to_csv('processed_comments.csv', index=False)
    print("Processed data saved to CSV files")
