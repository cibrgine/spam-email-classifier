import pandas as pd
import os
import torch
import numpy as np
from datasets import Dataset, disable_progress_bar
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    logging as hf_logging
)
import evaluate

hf_logging.set_verbosity_error()
disable_progress_bar()


def main():
    print("Checking for GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("WARNING: No GPU detected! Training on CPU will take a very long time.")

    print("Loading tokenizer and model...")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    print("Loading data...")
    df_train = pd.read_csv(os.path.join('data', 'train.csv'))
    df_val = pd.read_csv(os.path.join('data', 'validate.csv'))
    
    df_train['text'] = df_train['text'].fillna('')
    df_val['text'] = df_val['text'].fillna('')

    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=False, max_length=512)

    print("Tokenizing data...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metric_acc = evaluate.load("accuracy")
    metric_prec = evaluate.load("precision")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        acc = metric_acc.compute(predictions=predictions, references=labels)
        prec = metric_prec.compute(predictions=predictions, references=labels)
        
        return {
            "accuracy": acc["accuracy"],
            "precision": prec["precision"],
        }

    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2, 
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True if device == "cuda" else False, 
        logging_dir='./logs',
        logging_steps=100,
        disable_tqdm=True, 
        report_to="none", 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS:")
    print("="*50)
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("="*50 + "\n")

    print("Saving final model...")
    model_save_path = os.path.join('models', 'spam_classifier_distilbert')
    trainer.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
