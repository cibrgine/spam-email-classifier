import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    print("Loading data...")
    train_path = os.path.join('data', 'train.csv')
    test_path = os.path.join('data', 'test.csv')
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    
    df_all = df_all.dropna(subset=['text', 'label'])
    print(f"Total dataset size after combining and cleaning: {len(df_all)} rows")
    
    print("Splitting data into 70/15/15...")
    train_df, temp_df = train_test_split(
        df_all, test_size=0.3, random_state=42, stratify=df_all['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )
    
    train_df.to_csv(os.path.join('data', 'train.csv'), index=False)
    val_df.to_csv(os.path.join('data', 'validate.csv'), index=False)
    test_df.to_csv(os.path.join('data', 'test.csv'), index=False)
    
    print("Data successfully split and saved!")
    print(f"- Train set:      {len(train_df)} rows ({len(train_df)/len(df_all)*100:.1f}%)")
    print(f"- Validation set: {len(val_df)} rows ({len(val_df)/len(df_all)*100:.1f}%)")
    print(f"- Test set:       {len(test_df)} rows ({len(test_df)/len(df_all)*100:.1f}%)")

if __name__ == "__main__":
    main()
