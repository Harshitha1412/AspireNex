import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load dataset
def load_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    print("Dataset loaded successfully.")
    return df

# Data Preprocessing
def preprocess_data(df):
    print("Preprocessing data...")
    df['label'] = df['label'].map({'ham': 'legitimate', 'spam': 'spam'})
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test

# Feature Extraction
def extract_features(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Model Training and Evaluation
def train_and_evaluate_model(X_train_tfidf, X_test_tfidf, y_train, y_test, model, model_name):
    print(f"\nTraining {model_name} model...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=['legitimate', 'spam']))
    print(f"Confusion Matrix for {model_name}:")
    print(confusion_matrix(y_test, y_pred, labels=['legitimate', 'spam']))

def main():
    # Load the dataset
    df = load_data('spam.csv')  # Ensure this file is in the same directory as your script

    # Explore the dataset
    print("\nDataset information:")
    print(df.info())
    print("\nClass distribution:")
    print(df['label'].value_counts())

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Extract features using TF-IDF
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)

    # Train and evaluate Naive Bayes
    nb_model = MultinomialNB()
    train_and_evaluate_model(X_train_tfidf, X_test_tfidf, y_train, y_test, nb_model, "Naive Bayes")

    # Train and evaluate Logistic Regression
    lr_model = LogisticRegression(max_iter=10000)
    train_and_evaluate_model(X_train_tfidf, X_test_tfidf, y_train, y_test, lr_model, "Logistic Regression")

    # Train and evaluate Support Vector Machine
    svm_model = SVC(kernel='linear')
    train_and_evaluate_model(X_train_tfidf, X_test_tfidf, y_train, y_test, svm_model, "Support Vector Machine")

if __name__ == '__main__':
    main()
