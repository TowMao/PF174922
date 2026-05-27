import matplotlib.pyplot as plt
from wordcloud import WordCloud

class DataAnalyzer:
    def __init__(self, sentyment='voted_up', tekst='clean_text', typ_danych='clean'):
        self.sentyment = sentyment
        self.typ_danych = typ_danych
        self.tekst = tekst  # Kolumna z tekstem (np. 'clean_text' lub 'text')
        self.label_map = {
            True: 'Pozytywny', False: 'Negatywny',
            'True': 'Pozytywny', 'False': 'Negatywny',
            1: 'Pozytywny', 0: 'Negatywny'
        }

    def rozklad_sentymentu(self, df):
        if self.sentyment not in df.columns:
            print(f"Błąd: Brak kolumny '{self.sentyment}'. Dostępne kolumny to: {df.columns.tolist()}")
            return

        # Obliczenia statystyczne
        counts = df[self.sentyment].value_counts()
        total = len(df)

        # Mapowanie indeksów
        counts.index = counts.index.map(lambda x: self.label_map.get(x, str(x)))

        print(f"{20*'='} STATYSTYKI SENTYMENTU {20*'='}\n")
        print(f"Łączna liczba recenzji:     {total}")
        for label, count in counts.items():
            percent = (count / total) * 100
            print(f" - Sentyment '{label:>1}': {count:^3} ({percent:.2f}%)")
        print(63*'=')

        # Generowanie wykresów
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        labels_str = counts.index.tolist()
        colors = ['#2ca02c' if val == 'Pozytywny' else '#d62728' for val in labels_str]

        # Bar Chart
        axes[0].bar(labels_str, counts.values, color=colors, edgecolor='black', alpha=0.8)
        axes[0].set_title('Rozkład liczbowy recenzji')
        axes[0].set_ylabel('Liczba recenzji')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Pie Chart
        axes[1].pie(counts.values, labels=labels_str, autopct='%1.1f%%', colors=colors, startangle=140, 
                    wedgeprops={'edgecolor': 'black'})
        axes[1].set_title('Rozkład procentowy')

        plt.tight_layout()
        plt.savefig(f"./stats_pics/sentyment_{self.typ_danych}.png", dpi=150)
        print("Zapisano wykres sentymentu jako 'sentyment.png'\n")

    def dlugosc_sentyment(self, df):
        if self.sentyment not in df.columns:
            print(f"Błąd: Brak kolumny '{self.sentyment}'.")
            return
        if self.tekst not in df.columns:
            print(f"Błąd: Brak kolumny tekstowej '{self.tekst}' w danych!")
            return

        df_temp = df.copy()
        df_temp['sentyment'] = df_temp[self.sentyment].map(lambda x: self.label_map.get(x, str(x)))

        df_temp['word_count'] = df_temp[self.tekst].fillna('').astype(str).apply(lambda x: len(x.split()))

        stats = df_temp.groupby('sentyment')['word_count'].agg(['mean', 'median', 'std', 'min', 'max'])

        print(f"\n{10*'='} STATYSTYKI DŁUGOŚCI RECENZJI ({self.tekst}) {10*'='}")
        for sentiment, row in stats.iterrows():
            print(f"\nSentyment: \"{sentiment}\"")
            print(f" - Średnia liczba słów: {row['mean']:.2f}")
            print(f" - Mediana (środkowa):  {row['median']:.0f} słów")
            print(f" - Odchylenie std:      {row['std']:.2f}")
            print(f" - Najkrótsza recenzja: {row['min']:.0f} słów")
            print(f" - Najdłuższa recenzja: {row['max']:.0f} słów")
        print(60*'=')

        plt.figure(figsize=(10, 6))

        pos_lengths = df_temp[df_temp['sentyment'] == 'Pozytywny']['word_count']
        neg_lengths = df_temp[df_temp['sentyment'] == 'Negatywny']['word_count']

        # Wyznaczamy limit 95% percentyla, aby wykres był czytelny (odrzucamy anomalie)
        max_limit = int(df_temp['word_count'].quantile(0.95))

        plt.hist(pos_lengths, bins=30, range=(3, max_limit), alpha=0.6, color='#2ca02c', label='Pozytywne', edgecolor='black')
        plt.hist(neg_lengths, bins=30, range=(3, max_limit), alpha=0.6, color='#d62728', label='Negatywne', edgecolor='black')

        plt.title(f'Rozkład długości recenzji ({self.tekst}) w zależności od sentymentu')
        plt.xlabel('Liczba słów w recenzji')
        plt.ylabel('Liczba recenzji')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"./stats_pics/długość_by_sentyment_{self.typ_danych}.png", dpi=150)
        print("Zapisano wykres długości jako 'długość_by_sentyment.png'\n")

    def generuj_chmury(self, df):
        if self.sentyment not in df.columns:
            print(f"Błąd: Brak kolumny '{self.sentyment}'.")
            return
        if self.tekst not in df.columns:
            print(f"Błąd: Brak kolumny tekstowej '{self.tekst}' w danych!")
            return

        df_temp = df.copy()
        df_temp['sentyment'] = df_temp[self.sentyment].map(lambda x: self.label_map.get(x, str(x)))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, (label, title) in enumerate([("Negatywny", "Negatywne"), ("Pozytywny", "Pozytywne")]):
            recenzja = df_temp[df_temp["sentyment"] == label][self.tekst].fillna('').astype(str)
            text = " ".join(recenzja)
            
            if not text.strip():
                text = "pusty"

            wc = WordCloud(
                width=800, 
                height=400, 
                background_color="white", 
                max_words=100
            ).generate(text)

            axes[idx].imshow(wc, interpolation="bilinear")
            axes[idx].set_title(f"Chmura słów – recenzje {title} ({self.tekst})", fontsize=14)
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(f"./stats_pics/chmura_slow_{self.typ_danych}.png", dpi=150)
        print("Wykres Zapisano jako chmura_slow.png")