import unittest
import pandas as pd
import tweetPreprocessing


class TestTweetPreprocessor(unittest.TestCase):

    def test_group(self):
        tweets = tweetPreprocessing.TweetPreprocessor("test_tweets/inputs")
        tweets.group()
        correct_output = pd.read_pickle("test_tweets/outputs/group.pkl")
        self.assertTrue(tweets.data.equals(correct_output))

    def test_drop_duplicates(self):
        tweets = tweetPreprocessing.TweetPreprocessor("test_tweets/inputs")
        tweets.group()
        tweets.drop_duplicates()
        correct_output = pd.read_pickle("test_tweets/outputs/drop_duplicates.pkl")
        self.assertTrue(tweets.data.equals(correct_output))

    def test_drop_missing(self):
        tweets = tweetPreprocessing.TweetPreprocessor("test_tweets/inputs")
        tweets.group()
        tweets.drop_duplicates()
        tweets.drop_missing(['@UniofAdelaide', 'Switzerland', '@BBCGOODFOOD'])
        correct_output = pd.read_pickle("test_tweets/outputs/drop_missing.pkl")
        self.assertTrue(tweets.data.equals(correct_output))

    def test_drop_small_companies(self):
        tweets = tweetPreprocessing.TweetPreprocessor("test_tweets/inputs")
        tweets.data = pd.read_pickle("test_tweets/outputs/drop_missing.pkl")
        tweets.company_names = ["with", "@UniofAdelaide"]
        tweets.minimum_tweets = 2
        tweets.drop_small_companies()
        correct_output = pd.read_pickle("test_tweets/outputs/drop_small_companies.pkl")
        self.assertTrue(tweets.data.equals(correct_output))

    def test_fix_utf8_encodings(self):
        tweets = tweetPreprocessing.TweetPreprocessor("test_tweets/inputs")
        tweets.data = pd.read_pickle("test_tweets/outputs/drop_duplicates.pkl")
        tweets.fix_utf8_encodings()
        correct_output = pd.read_pickle("test_tweets/outputs/fix_utf8_encodings.pkl")
        self.assertTrue(tweets.data.equals(correct_output))

    def test_format_tweet_text(self):
        tweets = tweetPreprocessing.TweetPreprocessor("test_tweets/inputs")
        tweets.data = pd.read_pickle("test_tweets/outputs/drop_small_companies.pkl")
        tweets.format_tweet_text()
        correct_output = pd.read_pickle("test_tweets/outputs/format_tweet_text.pkl")
        self.assertTrue(tweets.data.equals(correct_output))

    def test_get_subset(self):
        tweets = tweetPreprocessing.TweetPreprocessor("test_tweets/inputs")
        tweets.data = pd.read_pickle("test_tweets/outputs/drop_missing.pkl")
        tweets.sample_size = 3
        sub = tweets.get_subset()
        correct_output = pd.read_pickle("test_tweets/outputs/get_subset.pkl")
        self.assertTrue(sub.equals(correct_output))


if __name__ == '__main__':
    unittest.main()
