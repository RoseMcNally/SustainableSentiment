import os
import pandas as pd


class TweetPreprocessor:

    def __init__(self, input_folder_path: str):
        self.path = input_folder_path
        self.data = None

    def collate(self):
        if not self.data:
            self.group()
            self.drop_duplicates()
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

    def group(self):
        data_list = list()
        for file in os.listdir(self.path):
            if file.endswith(".csv"):
                file_df = pd.read_csv(self.path + "/" + file)
                data_list.append(file_df)
        self.data = pd.concat(data_list)

    def drop_duplicates(self):
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data.drop_duplicates("Tweet_ID")
        else:
            pass


if __name__ == '__main__':
    tweets = TweetPreprocessor("../twitterScraper-main/output")
    tweets.collate()
    tweets.save_pickle("data/all.pkl")
    tweets.save_csv("data/all.csv")
    pass
