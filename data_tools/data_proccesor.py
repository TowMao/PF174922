import pandas as pd
import re
import matplotlib.pyplot as plt

class DataProcessor:

    def __init__(self, min_words=3):
        self.min_words = min_words

    @staticmethod
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        
        # Usuwanie linków
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Usuwanie znaczników([b], [h1], [spoiler])
        text = re.sub(r'\[.*?\]', '', text)
        
        # Usuwanie ASCII artów i dziwnych znaków, ZOSTAWIANIE interpunkcji
        text = re.sub(r'[^\w\s\.\,\!\?\'\"\-\:]', ' ', text)
        
        # Usuwanie wielokrotnych spacji
        text = re.sub(r'\s+', ' ', text).strip()

        # Usuwanie cyrylicy
        text = re.sub(r'[\u0400-\u04FF\u0500-\u052F]', ' ', text)
        
        return text

    def process_data(self, df):

        print(f"Początkowy rozmiar: {df.shape}")

        if 'text' not in df.columns:
            raise ValueError("Brak kolumny 'text' w dostarczonym zbiorze danych!")

        print("Czyszczenie danych:")

        # Usuwanie NaN
        print(f"- Usuwanie wartości NaN")
        df = df.dropna(subset=['text'])

        # Usuwanie dupikatów
        df = df.drop_duplicates(subset=['text']).copy()
        print(f"Rozmiar data frame'u: {df.shape}\n")

        # Regex
        print(f"- Regex")
        df['clean_text'] = df['text'].apply(self.clean_text)
        print(f"Rozmiar data frame'u: {df.shape}\n")

        # Usuwanie pustych wierszy
        print(f"- Usuwanie pustych wierszy")
        df = df[df['clean_text'].str.len() > 0].copy()
        print(f"Rozmiar data frame'u: {df.shape}\n")
        
        # Usuwanie za krótkich recenzji
        print(f"- Usuwanie zbyt krótkich recenzji")
        df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
        df = df[df['word_count'] >= self.min_words].copy()
        print(f"Rozmiar data frame'u: {df.shape}\n")

        print(f"Rozmiar po czyszczeniu ({self.min_words} słów): {df.shape}")
        print("Przykładowe recenzje ze zbioru:\n")

        for i, row in enumerate(df.head(3).itertuples(), 1):

            print(f"{row.clean_text:^20}\n")
            
        return df