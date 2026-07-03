import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, classification_report

def main():
    print("Loading datasets...")
    df_train = pd.read_csv(os.path.join('data', 'train.csv'))
    df_val = pd.read_csv(os.path.join('data', 'validate.csv'))
    df_test = pd.read_csv(os.path.join('data', 'test.csv'))
    
    df_train['text'] = df_train['text'].fillna('')
    df_val['text'] = df_val['text'].fillna('')
    df_test['text'] = df_test['text'].fillna('')

    X_train, y_train = df_train['text'], df_train['label']
    X_val, y_val = df_val['text'], df_val['label']
    X_test, y_test = df_test['text'], df_test['label']

    print(f"Training on {len(X_train)} samples...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=50000)),
        ('clf', MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)
    print("Training complete!")

    print("\n--- Validation Results ---")
    y_val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred)
    print(f"Validation Accuracy:  {val_acc:.4f}")
    print(f"Validation Precision: {val_prec:.4f}")

    print("\n--- Final Test Results ---")
    y_test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=["Safe (0)", "Spam (1)"]))

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'spam_classifier_nb.joblib')
    joblib.dump(pipeline, model_path)
    print(f"\nModel successfully saved to: {model_path}")

if __name__ == "__main__":
    main()
