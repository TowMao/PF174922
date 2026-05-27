import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

class Baseline:
    def __init__(self, text_col='clean_text', target_col='voted_up'):
        self.text_col = text_col
        self.target_col = target_col
        self.classifiers = {
            "SVM (LinearSVC)": LinearSVC(max_iter=5000, random_state=42),
            "Naive Bayes": MultinomialNB(alpha=0.1),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
        }

        self.tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        # Zmienne do przechowywania najlepszego modelu po treningu
        self.best_model = None
        self.best_model_name = None

    def trening_ewaluacja(self, df_train, df_val, df_test):
        X_train = df_train[self.text_col].fillna('').astype(str)
        y_train = df_train[self.target_col].astype(str)

        X_val = df_val[self.text_col].fillna('').astype(str)
        y_val = df_val[self.target_col].astype(str)

        X_test = df_test[self.text_col].fillna('').astype(str)
        y_test = df_test[self.target_col].astype(str)

        print("\nWektoryzacja tekstów (TF-IDF)...")
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_val_tfidf = self.tfidf.transform(X_val)
        X_test_tfidf = self.tfidf.transform(X_test)

        classic_results = {}

        for name, clf in self.classifiers.items():
            print(f"\n{'='*50}")
            print(f"Trening: {name}")
            print(f"{'='*50}")

            clf.fit(X_train_tfidf, y_train)
            y_val_pred = clf.predict(X_val_tfidf)

            acc = accuracy_score(y_val, y_val_pred)
            f1 = f1_score(y_val, y_val_pred, average="weighted")

            classic_results[name] = {
                "model": clf,
                "val_accuracy": acc,
                "val_f1": f1,
                "val_predictions": y_val_pred,
            }

            print(f"Accuracy (val): {acc:.4f}")
            print(f"F1-score (val): {f1:.4f}")
            print(f"\nRaport klasyfikacji (val):")
            print(
                classification_report(
                    y_val, y_val_pred, target_names=["Negative", "Positive"]
                )
            )

        best_classic_name = max(classic_results, key=lambda x: classic_results[x]["val_f1"])
        best_classic = classic_results[best_classic_name]

        # Zapisanie najlepszego modelu wewnątrz instancji klasy
        self.best_model = best_classic["model"]
        self.best_model_name = best_classic_name

        print(f"\nNajlepszy klasyfikator klasyczny: {self.best_model_name}")
        print(f"F1-score (val): {best_classic['val_f1']:.4f}")

        y_test_pred_classic = self.best_model.predict(X_test_tfidf)
        classic_test_acc = accuracy_score(y_test, y_test_pred_classic)
        classic_test_f1 = f1_score(y_test, y_test_pred_classic, average="weighted")

        print(f"\nWyniki na zbiorze testowym ({self.best_model_name}):")
        print(f"Accuracy: {classic_test_acc:.4f}")
        print(f"F1-score: {classic_test_f1:.4f}")
        print(f"\nRaport klasyfikacji (test):")
        print(
            classification_report(
                y_test, y_test_pred_classic, target_names=["Negative", "Positive"]
            )
        )
        cm_classic = confusion_matrix(y_test, y_test_pred_classic)

        os.makedirs('./stats_pics', exist_ok=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_classic,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.title(f"Confusion Matrix – {self.best_model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        
        save_path = f'./stats_pics/macierz_pomylek_{self.best_model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
        plt.savefig(save_path)
        print(f"\nZapisano macierz pomyłek jako: {save_path}")


        metrics = {
            "accuracy": classic_test_acc,
            "f1": classic_test_f1
        }
        
        return self.best_model, metrics

    def predict(self, texts):

        if self.best_model is None:
            raise ValueError("Model nie został wytrenowany! Najpierw odpal 'trening_ewaluacja'.")

        # Ujednolicenie wejścia do listy
        if isinstance(texts, str):
            texts = [texts]

        # Wektoryzacja nowych tekstów za pomocą dopasowanego już TF-IDF
        X_tfidf = self.tfidf.transform(texts)
        
        # Predykcja
        preds = self.best_model.predict(X_tfidf)
        
        # Tłumaczenie wyników modelu na jednolity standard Positive/Negative
        results = []
        for p in preds:
            if str(p).lower() in ['true', '1', '1.0', 'positive']:
                results.append("Positive")
            else:
                results.append("Negative")

        # Jeśli wrzuciliśmy jeden tekst, zwracamy jeden string. Jak listę, to listę wyników.
        return results[0] if len(results) == 1 else results

    def save_model(self, path="./saved_models/baseline.pkl"):
        if self.best_model is None:
            print("[Baseline] Błąd: Nie ma wytrenowanego modelu do zapisania!")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Zapisujemy w jednym pliku model, jego nazwę i nasz wektoryzator TF-IDF
        with open(path, 'wb') as f:
            pickle.dump({'model': self.best_model, 'name': self.best_model_name, 'tfidf': self.tfidf}, f)
        print(f"[Baseline] Pomyślnie zapisano model {self.best_model_name} do {path}")

    def load_model(self, path="./saved_models/baseline.pkl"):
        if not os.path.exists(path):
            print(f"[Baseline] Błąd: Plik {path} nie istnieje!")
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.best_model = data['model']
            self.best_model_name = data['name']
            self.tfidf = data['tfidf']
        print(f"[Baseline] Pomyślnie wczytano model {self.best_model_name} z {path}")
        return True