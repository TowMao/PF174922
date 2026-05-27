# Analiza sentymentu recenzji gier w języku angielskim i porównanie modeli BERT i roBERTa

#### Autor: Krzysztof Stopyra

Opis: Projekt poddaje analizie sentymentu recenzje społeczności wybranej przez użytkownika gry komputerowej dostępnej na platformie Steam. Porównanie jakości dwóch modeli

Źródło danych: Pobierane poprzez scrapper recenzje z platformy Steam.

## CZĘŚĆ A - Dane i przetwarzanie:

Po podaniu przez użytkownika *app_id* gry oraz liczby, scrapper pobiera określoną liczbę recenzji a następnie przeprowadza oczyszczanie danych ze zbyt krótkich i języków nie będących napisanych w alfabecie łacińskim.

Za oczyszczenie danych odpowiada klasa Data_processor która:
- Usuwa duplikaty
- Usuwa wartości NaN
- Używa regexa do usuwania artefaktów (np. ASCII art, copypasty) które mogą zanieczyszczać dane
-  Usuwa inne alfabety niż łacińskie

Za wygenerowanie statystyk odpowiada klasa Data_analyzer:
- Tworzy chmurę słów
- Wykres długości recenzji w zależności od sentymentu
- Wykresy ilości recenzji w zalezności od sentymentu

Klasa data_splitter tworzy 3 zbiory, treningowy, walidacyjny i testowy.\
Pozwala on na fine-tuning modelu.

Wszystkie klasy związane z przetwarzaniem danych znajdują się w folderze data_tools.\
Wykresy wygenerowane przez narzędzia zapisywane są w folderze stats_pics.\
Dane testowe dotyczyły 4 gier z kategorii RTS (*Battlefleet Gothic Armada 2*, *Total War Shogun 2*, *Men of War Assault Squad 2* i *Command & Conquer 2 Red Alert*).

## CZĘŚĆ B - Modele:

Zaimplementowane modele to:
- **TF-IDF + SVM + Naive Bayes** (Baseline)
- **BERT**
- **roBERTa**

Motywacja:
- TF-IDF + SVM + Naive Bayes - Wykorzystane jako baseline model który pozwoli porównać pozostałe 
- BERT - Klasyczny model pretrenowany na języku angielskim
- roBERTa - model przygotowany brzez META, pretrenowany na częściej występującym w internecie języku

Modele BERT i roBERTa są używane w wersji 'cased' w celu możliwości wykrycia sentymentu bazowanego w zależności od występowania tekstów pisanych caps-lockiem.

Same modele są zapisywane w folderze saved_models, wyniki szkoleń w folderze model./

Fine-tuning został przeprowadzony na 7278 recenzjach, podział na zbiory wyglądał następująco:
- trening   -  4366 rekordów
- walidacja -  1092 rekordów
- testowy   -  1820 rekordów

## CZĘŚĆ C - Analiza i ewaluacja

Analiza: \
Modele zostały pretrenowane na tym samym zbiorze danych, ich statystyki wyglądały w sposób następujący:

```
================== PODSUMOWANIE ==============================

Model Name                | Test Accuracy   | Test F1-Score  
--------------------------------------------------------------
Baseline (Naive Bayes)    | 0.9077          | 0.9018         
BERT                      | 0.9374          | 0.8945         
RoBERTa                   | 0.9352          | 0.8933         
==============================================================
```


Wnioski: \
Mimo że 3 różne modele zostały przetrenowane na tych samych danych testowych, wzrost procentowy na metrykach Accuracy i F1-score jest niewielki pomiędzy tymi modelami. Zaskakująco wysoko uplasowało się klasyczne podejście oparte na Naive Bayes. 
W założeniach model roBERTa powinien poradzić sobie lepiej niż BERT lub podejścia klasyczne nie oparte na transformerach, prawdopodobnie wynika to z niezbalansowanego lub za małego datasetu. 

Porównanie modeli:

BERT:
- Pre-trening na małym zbiorze danych
- Tokenizer WordPiece
- Next Sentence Prediction
- Mniejsza elastyczność

roBERTa:
- Pre-trening na dużym zbiorze danych
- Tokenizer BSP
- Brak NSP
- Łatwiejsza nauka

Możliwe ścieżki poprawy:
- więcej epok treningu modeli BERT i roBERTa
- poprawa ilości i jakości danych
- lepszy podział datasetu treningowego w kwestii ilości recenzji pozytywnych i negatywnych

## CZĘŚĆ D - Jak uruchomić program / Działanie programu

Uruchomienie:
1. Pobrać projekt z Githuba
1. Zainstalować biblioteki za pomocą: \
```pip install -r requirements.txt```
1. Utworzyć foldery
```mkdir data saved_models model_res stats_pics```
3. Uruchomić plik *main.py*
4. Wybrać w menu najpierw opcję numer 3, jesli nie ma ścieżki *./data/combined_test_set.csv* to program sam pobierze i ją stworzy, wystarczy zostawić pustą linijkę i zostawić program aby przetrenował wybrane modele
5. Wybrać dowolną opcję z menu i postępować zgodnie z wyświetlanymi poleceniami

Z menu głównego można wybrać 4 opcje:
- Przetwarzanie jednej recenzji
    - Po wpisaniu tekstu, program przetworzy przetrenowanymi wcześniej modelami tekst wpisany przez użytkownika, a następnie poda sentyment
- Przetwarzanie wielu recenzji
    - Po podaniu app_id i liczby recenzji, program pobiera je, a następnie poda ogólny trend
- Trening modeli
    - Pozwala na przetrenowanie wybranych modeli na wybranym przez użytkownika datasecie
- Zapis/Odczyt modeli
    - Zapisuje lub odczytuje przygotowane modele


## CZĘŚĆ F - Użyte technologie i narzędzia

- Gemini - klasa SteamScrapper, rozwiązywanie niektórych problemów technicznych, części kodu generujące wykresy, argumenty treningu
- CUDA - wsparcie obliczeń dotyczących fine-tuningu
- biblioteki Python wymienione w pliku requirements.txt
