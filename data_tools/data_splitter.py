from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, target_col='voted_up', random_state=42):
        self.target_col = target_col
        self.random_state = random_state

    def split(self, df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):

        if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
            raise ValueError("Proporcje podziału (train, val, test) muszą sumować się do 1.0!")

        temp_ratio = val_ratio + test_ratio

        print(f"\n{15*'='} PODZIAŁ DANYCH (TRAIN / VAL / TEST) {15*'='}")
        
        df_train, df_temp = train_test_split(
            df, 
            test_size=temp_ratio, 
            random_state=self.random_state,
            stratify=df[self.target_col]
        )

        proportion_test = test_ratio / temp_ratio

        df_val, df_test = train_test_split(
            df_temp, 
            test_size=proportion_test, 
            random_state=self.random_state,
            stratify=df_temp[self.target_col]
        )

        print(f"Rozmiar zbioru Treningowego (Train): {len(df_train)} rekordów ({train_ratio*100:.0f}%)")
        print(f"Rozmiar zbioru Walidacyjnego (Val):  {len(df_val)} rekordów ({val_ratio*100:.0f}%)")
        print(f"Rozmiar zbioru Testowego (Test):     {len(df_test)} rekordów ({test_ratio*100:.0f}%)")
        print(59*'=')

        return df_train, df_val, df_test