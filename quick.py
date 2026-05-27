from steam_scraper import SteamScraper
import pandas as pd
from data_tools import DataProcessor
import os

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

def quick_setup():

    proc = DataProcessor(min_words=3)


    test_ids = [573100, 2229850, 201270, 244450]
    test_dfs = []


    for id in test_ids:
        test_dfs.append(download_data(id, 3000))

    df = pd.concat(test_dfs, ignore_index=True)
    df = proc.process_data(df)

    df.to_csv("./data/combined_test_set.csv", index=False)

    return df

