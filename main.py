from steam_scraper import SteamScraper
from data_tools import DataAnalyzer, DataProcessor, DataSplitter
from trainers import Baseline, BertSentimentTrainer, RobertaSentimentTrainer
from quick import quick_setup
import re
import pandas as pd
import torch
import urllib.request
import matplotlib.pyplot as plt
import os

def generate_model_comparison(results):

    print(f"\n{18*'='} PODSUMOWANIE {18*'='}\n")
    print(f"{'Model Name':<25} | {'Test Accuracy':<15} | {'Test F1-Score':<15}")
    print("-" * 62)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} | {metrics['accuracy']:<15.4f} | {metrics['f1']:<15.4f}")
    print(62*'=')
    
    df_comp = pd.DataFrame(results).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df_comp.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'], edgecolor='black', alpha=0.8)
    
    ax.set_title("Porównanie na datasecie testowym", fontsize=14, fontweight='bold')
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15, fontsize=10)
    plt.legend(["Accuracy", "F1-Score (Macro)"], loc="lower right")
    
    plt.tight_layout()
    os.makedirs('./stats_pics', exist_ok=True)
    plt.savefig("./stats_pics/model_comparison.png", dpi=150)
    print("\nWykres zapisano jako './stats_pics/model_comparison.png'\n")

def download_data(app_id=573100, number=1000):
    scraper = SteamScraper(app_id=app_id, num_reviews=number)
    scraper.fetch_reviews()
    os.makedirs('./data', exist_ok=True)
    path = f'./data/steam_reviews_{app_id}_{number}.csv'
    scraper.save_to_csv(path)
    
    df = scraper.get_reviews_df()
    if 'review' in df.columns and 'text' not in df.columns:
        df = df.rename(columns={'review': 'text'})
    return df

def predict_sentiment(text, model, tokenizer, device):
    if model is None:
        return "Brak modelu"
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    pred_idx = torch.argmax(outputs.logits, dim=1).item()
    return "Positive" if pred_idx == 1 else "Negative"

def handle_single_review(processor, baseline, bert_trainer, bert_model, roberta_trainer, roberta_model):
    if baseline.best_model is None and bert_model is None and roberta_model is None:
        print("\n[BŁĄD] Żaden model nie jest załadowany. Użyj opcji 3 (Trening) lub opcji 4 (Wczytaj).")
        return
        
    recenzja = input("\nWklej tekst recenzji (po angielsku): ")
    clean_text = processor.clean_text_for_roberta(recenzja) if hasattr(processor, 'clean_text_for_roberta') else recenzja
    print(f"\nTekst po wyczyszczeniu: {clean_text}\n")
    
    if baseline.best_model is not None:
        print(f"---> Wynik Baseline ({baseline.best_model_name}): {baseline.predict(clean_text)}")
    if bert_model is not None:
        print(f"---> Wynik BERT:           {predict_sentiment(clean_text, bert_model, bert_trainer.tokenizer, bert_trainer.device)}")
    if roberta_model is not None:
        print(f"---> Wynik RoBERTa:        {predict_sentiment(clean_text, roberta_model, roberta_trainer.tokenizer, roberta_trainer.device)}")

def handle_multiple_reviews(processor, baseline, roberta_trainer, roberta_model):
    if baseline.best_model is None and roberta_model is None:
        print("\n[ERR] Żaden model nie jest załadowany. Użyj opcji 3 (Trening) lub opcji 4 (Wczytaj).")
        return

    app_id = input("\nPodaj app_id gry do przeanalizowania: ")
    n = int(input("Podaj liczbę recenzji do pobrania: "))
    
    df_nowe = download_data(int(app_id), n)
    df_nowe_czyste = processor.process_data(df_nowe)
    print("\nKlasyfikacja w toku...")
    
    cols_to_show = ['clean_text']
    
    use_roberta = roberta_model is not None
    predictions_to_summarize = []
    
    if baseline.best_model is not None:
        base_preds = baseline.predict(df_nowe_czyste['clean_text'].tolist())
        df_nowe_czyste['Baseline_pred'] = base_preds
        cols_to_show.append('Baseline_pred')
        print(f"Baseline -> Pozytywne: {base_preds.count('Positive')} | Negatywne: {base_preds.count('Negative')}")
        if not use_roberta:
            predictions_to_summarize = base_preds

    if roberta_model is not None:
        rob_preds = [predict_sentiment(text, roberta_model, roberta_trainer.tokenizer, roberta_trainer.device) for text in df_nowe_czyste['clean_text']]
        df_nowe_czyste['RoBERTa_pred'] = rob_preds
        cols_to_show.append('RoBERTa_pred')
        print(f"RoBERTa  -> Pozytywne: {rob_preds.count('Positive')} | Negatywne: {rob_preds.count('Negative')}")
        predictions_to_summarize = rob_preds
    
    total = len(predictions_to_summarize)
    if total > 0:
        pos_count = predictions_to_summarize.count("Positive")
        pos_pct = (pos_count / total) * 100
        
        print(f"\n{15*'='} Podsumowanie modelu {15*'='}")
        
        if pos_pct >= 90:
            print(">>> Przytłoczająco pozytywne")
        elif pos_pct >= 70:
            print(">>> W większości pozytywne")
        elif pos_pct >= 40:
            print(">>> Mieszane")
        elif pos_pct >= 20:
            print(">>> W większości negatywne")
        else:
            print(">>> Przytłaczająco negatywne")

        print(f"{56*'='}\n")
    
    print("Przykładowe predykcje:")
    print(df_nowe_czyste[cols_to_show].sample(min(5, len(df_nowe_czyste))))

def handle_training(splitter, processor, analyzer, baseline, bert_trainer, bert_model, roberta_trainer, roberta_model):
    print("\n=== PRZYGOTOWANIE DANYCH DO TRENINGU ===")
    app_id_input = input("Podaj app_id gry (wciśnij Enter na pusto, aby użyć domyślnego pliku): ").strip()
    
    if not app_id_input:
        print("Wczytywanie domyślnego zbioru")
        try:
            df = pd.read_csv("./data/combined_test_set.csv")
            if 'review' in df.columns and 'text' not in df.columns:
                df = df.rename(columns={'review': 'text'})
        except:
            print("Plik nie istnieje, pobieranie")
            df = download_data(573100, 7000)
    else:
        app_id = int(app_id_input)
        n = int(input("Podaj liczbę recenzji do pobrania: "))
        print("Pobieranie danych...")
        df = download_data(app_id, n)

    # Analiza wybranego zbioru
    print("\nGenerowanie wykresów statystycznych")
    analyzer.rozklad_sentymentu(df)
    analyzer.dlugosc_sentyment(df)
    analyzer.generuj_chmury(df)

    # Czyszczenie
    print("\nCzyszczenie i podział danych")
    df_czysty = processor.process_data(df)
    df_train, df_val, df_test = splitter.split(df_czysty, train_ratio=0.6, val_ratio=0.15, test_ratio=0.25)
    
    wybor = input("\nWybierz co chcesz wytrenować:\n1) Baseline (SVM/NB)\n2) BERT\n3) RoBERTa\n4) Wszystkie po kolei\nWybór: ")
    
    results = {}
    
    # Odbieramy model i jego metryki
    if wybor in ['1', '4']:
        best_base_model, base_metrics = baseline.trening_ewaluacja(df_train, df_val, df_test)
        results[f"Baseline ({baseline.best_model_name})"] = base_metrics
        
    if wybor in ['2', '4']:
        bert_model, bert_metrics = bert_trainer.train_and_evaluate(df_train, df_val, df_test)
        results["BERT"] = bert_metrics
        
    if wybor in ['3', '4']:
        roberta_model, roberta_metrics = roberta_trainer.train_and_evaluate(df_train, df_val, df_test)
        results["RoBERTa"] = roberta_metrics
        
    if results:
        generate_model_comparison(results)
        
    return bert_model, roberta_model

def handle_management(baseline, bert_trainer, bert_model, roberta_trainer, roberta_model):
    sub = input("\nZarządzanie modelami:\n1) ZAPISZ obecne modele na dysk\n2) WCZYTAJ gotowe modele z dysku\nWybór: ")
    if sub == '1':
        print("\nZapisywanie")
        if baseline.best_model is not None: baseline.save_model("./saved_models/baseline")
        if bert_model is not None: bert_trainer.save_model(bert_model, "./saved_models/bert")
        if roberta_model is not None: roberta_trainer.save_model(roberta_model, "./saved_models/roberta")
        print("Zakończono zapisywanie.")
    elif sub == '2':
        print("\nWczytywanie")
        if os.path.exists("./saved_models/baseline.pkl"):
            baseline.load_model()
        if os.path.exists("./saved_models/bert"):
            bert_model = bert_trainer.load_model("./saved_models/bert")
        if os.path.exists("./saved_models/roberta"):
            roberta_model = roberta_trainer.load_model("./saved_models/roberta")

    else:
        print("Zły wybór.")
        
    return bert_model, roberta_model

def main():
    print(60*"=")
    analyzer = DataAnalyzer(sentyment='voted_up', tekst='text')
    processor = DataProcessor(min_words=3)
    splitter = DataSplitter(target_col='voted_up')
    
    baseline = Baseline(text_col='clean_text', target_col='voted_up')
    bert_trainer = BertSentimentTrainer(
        model_name="bert-base-cased", text_col='clean_text', target_col='voted_up', epochs=2, batch_size=16
    )
    bert_model = None

    roberta_trainer = RobertaSentimentTrainer(
        model_name="roberta-base", text_col='clean_text', target_col='voted_up', epochs=2, batch_size=16
    )
    roberta_model = None

    while True:
        action = input("\n" + 40*"=" + "\nMENU GŁÓWNE:\n1) Przetwórz jedną recenzję\n2) Przetwórz N recenzji gry\n3) Trenuj / Doszkól modele na zbiorze\n4) Zarządzaj modelami (Zapisz / Wczytaj)\n5) Zakończ program\nWybór: ")
        
        match action:
            case '1':
                handle_single_review(processor, baseline, bert_trainer, bert_model, roberta_trainer, roberta_model)
            case '2':
                handle_multiple_reviews(processor, baseline, roberta_trainer, roberta_model)
            case '3':
                bert_model, roberta_model = handle_training(
                    splitter, processor, analyzer, baseline, bert_trainer, bert_model, roberta_trainer, roberta_model
                )
            case '4':
                bert_model, roberta_model = handle_management(
                    baseline, bert_trainer, bert_model, roberta_trainer, roberta_model
                )
            case '5':
                print("\nZamykanie programu")
                break
            case _:
                print("\nNie ma takiej opcji, spróbuj ponownie.")

if __name__ == "__main__":
    main()