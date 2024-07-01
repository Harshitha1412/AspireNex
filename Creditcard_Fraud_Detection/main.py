import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load dataset
def load_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print("Dataset loaded successfully.")
    return df

# Data Preprocessing
def preprocess_data(df):
    print("Preprocessing data...")
    # Map labels to 'legitimate' and 'fraud'
    df['Class'] = df['Class'].map({0: 'legitimate', 1: 'fraud'})

    # Scale the 'Amount' feature
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    # Drop the 'Time' column as it is not relevant for the analysis
    df.drop(columns=['Time'], inplace=True)

    # Split the data into features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test

# Model Training and Evaluation
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name):
    print(f"\nTraining {model_name} model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=['legitimate', 'fraud']))
    print(f"Confusion Matrix for {model_name}:")
    print(confusion_matrix(y_test, y_pred, labels=['legitimate', 'fraud']))

def main():
    # Load the dataset
    df = load_data('creditcard.csv')  # Ensure this file is in the same directory as your script

    # Explore the dataset
    print("\nDataset information:")
    print(df.info())
    print("\nClass distribution:")
    print(df['Class'].value_counts())

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train and evaluate Logistic Regression
    lr_model = LogisticRegression(max_iter=10000)
    train_and_evaluate_model(X_train, X_test, y_train, y_test, lr_model, "Logistic Regression")

    # Train and evaluate Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    train_and_evaluate_model(X_train, X_test, y_train, y_test, dt_model, "Decision Tree")

    # Train and evaluate Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    train_and_evaluate_model(X_train, X_test, y_train, y_test, rf_model, "Random Forest")

if __name__ == '__main__':
    main()
