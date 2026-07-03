import os
import pandas as pd
from datasets import load_dataset

def main():
    os.makedirs('data', exist_ok=True)
    
    print("Downloading the dataset from Hugging Face...")
    try:
        dataset = load_dataset("SetFit/enron_spam")
        df_train = dataset['train'].to_pandas()
        df_test = dataset['test'].to_pandas()
        
        train_path = os.path.join('data', 'train.csv')
        test_path = os.path.join('data', 'test.csv')
        
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        
        print(f"Successfully downloaded and saved data!")
        print(f"Train data: {train_path} ({len(df_train)} rows)")
        print(f"Test data: {test_path} ({len(df_test)} rows)")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == '__main__':
    main()
