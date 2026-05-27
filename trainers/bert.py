import os
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

class BertSentimentTrainer:
    def __init__(self, model_name="bert-base-cased", text_col="clean_text", target_col="voted_up", epochs=3, batch_size=16):
        self.model_name = model_name
        self.text_col = text_col
        self.target_col = target_col
        self.epochs = epochs
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[BERT] Używane urządzenie: {self.device.upper()}")

    def _prepare_dataset(self, df):

        df_temp = df.copy()

        label_map = {
            True: 1, False: 0, 
            'True': 1, 'False': 0, 
            1: 1, 0: 0, 
            'Positive': 1, 'Negative': 0
        }
        df_temp['label'] = df_temp[self.target_col].map(label_map).astype(int)

        dataset = Dataset.from_pandas(df_temp[[self.text_col, 'label']])

        def tokenize_fn(batch):
            return self.tokenizer(batch[self.text_col], truncation=True, max_length=128)

        return dataset.map(tokenize_fn, batched=True)

    def train_and_evaluate(self, df_train, df_val, df_test):
        print("\n- Przygotowanie danych dla modelu BERT")
        train_dataset = self._prepare_dataset(df_train)
        val_dataset = self._prepare_dataset(df_val)
        test_dataset = self._prepare_dataset(df_test)

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        model.to(self.device)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro")
            return {"accuracy": acc, "f1": f1}

        training_args = TrainingArguments(
            output_dir="./model_res/results_bert",
            learning_rate=2e-5,                 
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            eval_strategy="epoch",       
            save_strategy="epoch",        
            load_best_model_at_end=True,         
            metric_for_best_model="f1",      
            logging_dir="./logs_bert",
            logging_steps=50,
            report_to="none"                     
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer, 
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
        )

        print("\n[BERT] Rozpoczynanie Fine-tuningu.")
        trainer.train()

        print("\n[BERT] Ewaluacja.")
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids

        # Raport klasyfikacji w konsoli
        print(f"\n{15*'='} Wyniki dla: BERT {15*'='}")
        print(classification_report(labels, preds, target_names=["Negative", "Positive"]))
        print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
        print(60*'=')

        cm = confusion_matrix(labels, preds)
        os.makedirs('./stats_pics', exist_ok=True)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )

        plt.title(f"Confusion Matrix – {self.model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        
        save_path = f'./stats_pics/macierz_pomylek_{self.model_name.replace("/", "_")}.png'
        plt.savefig(save_path)
        print(f"\n[BERT] Zapisano macierz pomyłek do: {save_path}")

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro")
        }
        
        return model, metrics

    def save_model(self, model, output_dir):
        if model is None:
            print(f"[{self.model_name}] [ERR] Brak modelu do zapisania")
            return
        os.makedirs(output_dir, exist_ok=True)
        # Zapisuje wagi modelu, konfigurację i tokenizator
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[{self.model_name}] Pomyślnie zapisano model do folderu: {output_dir}")

    def load_model(self, input_dir):
        if not os.path.exists(input_dir):
            print(f"[{self.model_name}] [ERR] Folder {input_dir} nie istnieje")
            return None
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"[{self.model_name}] Wczytywanie modelu z dysku")
        self.tokenizer = AutoTokenizer.from_pretrained(input_dir)
        model = AutoModelForSequenceClassification.from_pretrained(input_dir)
        model.to(self.device)
        print(f"[{self.model_name}] Pomyślnie wczytano model")
        return model