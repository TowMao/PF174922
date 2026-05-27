import requests
import pandas as pd
import time

class SteamScraper:
    def __init__(self, app_id, language='english', num_reviews=1000):
        self.app_id = app_id
        self.language = language
        self.num_reviews = num_reviews
        self.base_url = f"https://store.steampowered.com/appreviews/{app_id}"
        self.reviews = []

    def fetch_reviews(self):
        cursor = '*'
        while len(self.reviews) < self.num_reviews:
            params = {
                'json': 1,
                'cursor': cursor,
                'filter': 'recent',
                'language': self.language,
                'review_type': 'all',
                'purchase_type': 'all',
                'num_per_page': 100
            }
            response = requests.get(self.base_url, params=params)
            if response.status_code != 200:
                print(f"Error fetching reviews: {response.status_code}")
                break
            data = response.json()
            if 'reviews' not in data or not data['reviews']:
                break
            self.reviews.extend(data['reviews'])
            cursor = data.get('cursor', '*')
            if cursor == '*':
                break
            time.sleep(1)  # Respect rate limits
        self.reviews = self.reviews[:self.num_reviews]

    def save_to_csv(self, filepath):
        if not self.reviews:
            print("No reviews to save.")
            return
        df = pd.DataFrame(self.reviews)
        df.rename(columns={'review': 'text'}, inplace=True)
        columns = ['recommendationid', 'author', 'text', 'timestamp_created', 'timestamp_updated', 'voted_up', 'votes_up', 'votes_funny', 'weighted_vote_score', 'comment_count', 'steam_purchase', 'received_for_free', 'written_during_early_access']
        df = df[columns]
        df.to_csv(filepath, index=False)
        print(f"Pobrano {len(df)} recenzji do {filepath}")

    def get_reviews_df(self):
        return pd.DataFrame(self.reviews)