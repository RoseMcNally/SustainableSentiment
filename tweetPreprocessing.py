import os
import pandas as pd
import numpy as np
import regex as re


class TweetPreprocessor:

    company_names = ['@NEC', 'Ericsson', 'Samsung', '@Apple', 'Microsoft', '@Cisco', 'Qualcomm', 'Fujitsu', 'Sony',
                     'Hitachi', 'Toshiba', '@LGUS', 'Lenovo', 'Pegatron', 'Foxconn', '@QuantaQCT', 'Huawei', 'ZTE',
                     'IBM', 'Dell', '@Intel', 'Siemens', 'Asus', 'Wistron', 'Compal', 'Panasonic', 'Nokia']

    keywords = ['sustainable', 'sustainability', 'poverty', 'hunger', 'health', 'education', 'gender', 'women',
                'clean water', 'sanitation', 'clean energy', 'economic growth', 'employment', 'innovation',
                'inequality', 'inequalities', 'local communities', 'climate', 'marine life', 'sea life',
                'deforestation', 'desertification', 'biodiversity', 'justice', 'global peace', 'sdg', 'sdgs']

    minimum_tweets = 5000
    sample_size = 5000
    seed = 42

    def __init__(self, input_folder_path: str):
        self.path = input_folder_path
        self.data = None

    def collate(self):
        if not self.data:
            print("Reading files...")
            self.group()
            print("Dropping duplicates...")
            self.drop_duplicates()
            print("Dropping tweets not containing company names...")
            self.drop_missing(self.company_names)
            print("Dropping tweets not containing keywords...")
            self.drop_missing(self.keywords)
            print("Done!")
        else:
            pass

    def clean(self):
        if isinstance(self.data, pd.DataFrame):
            print("Removing companies with fewer than 5000 tweets...")
            self.drop_small_companies()
            print("Formatting tweet text...")
            self.format_tweet_text()
            print("Done!")
        else:
            pass

    def save_csv(self, output_file_path: str):
        if isinstance(self.data, pd.DataFrame):
            self.data.to_csv(output_file_path, index=False)
        else:
            pass

    def save_pickle(self, output_file_path:str):
        if isinstance(self.data, pd.DataFrame):
            self.data.to_pickle(output_file_path)
        else:
            pass

    def save_subset_csv(self, output_file_path: str):
        if isinstance(self.data, pd.DataFrame):
            subset = self.get_subset()
            subset.to_csv(output_file_path, index=False)
        else:
            pass

    def group(self):
        data_list = list()
        for file in os.listdir(self.path):
            if file.endswith(".csv"):
                try:
                    file_df = pd.read_csv(self.path + "/" + file)
                    data_list.append(file_df)
                except pd.errors.EmptyDataError:
                    pass
        self.data = pd.concat(data_list)

    def drop_duplicates(self):
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data.drop_duplicates("Tweet_ID")
        else:
            pass

    def drop_missing(self, words):
        if isinstance(self.data, pd.DataFrame):
            pattern = '|'.join(words)
            self.data = self.data[self.data["Text"].str.contains(pattern, flags=re.IGNORECASE, regex=True)]
        else:
            pass

    def get_subset(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data.sample(self.sample_size, random_state=self.seed)
        else:
            return None

    def drop_small_companies(self):
        if isinstance(self.data, pd.DataFrame):
            companies_to_keep = []
            for name in self.company_names:
                if self.total_tweets(name) > self.minimum_tweets:
                    companies_to_keep.append(name)
                else:
                    print("Dropping:", name)

            if companies_to_keep:
                pattern = '|'.join(companies_to_keep)
                self.data = self.data[self.data["Text"].str.contains(pattern, flags=re.IGNORECASE, regex=True)]
                self.company_names = companies_to_keep
            else:
                self.data = None
                self.company_names = None
        else:
            pass

    def total_tweets(self, text):
        if isinstance(self.data, pd.DataFrame):
            return self.data[self.data["Text"].str.contains(text, flags=re.IGNORECASE, regex=True)].shape[0]
        else:
            return None

    def format_tweet_text(self):
        if isinstance(self.data, pd.DataFrame):
            self.data.loc[:, "Text"] = self.data.loc[:, "Text"].apply(lambda x: x[2:-1])
        else:
            pass


if __name__ == '__main__':
    tweets = TweetPreprocessor("../twitterScraper-main/output")
    tweets.collate()
    tweets.save_pickle("data/all/all.pkl")
    tweets.save_csv("data/all/all.csv")
    tweets.clean()
    tweets.save_pickle("data/all/clean.pkl")
    tweets.save_csv("data/all/clean.csv")
    tweets.save_subset_csv("data/all/model_data.csv")
