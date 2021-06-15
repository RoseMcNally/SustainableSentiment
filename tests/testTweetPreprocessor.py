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


if __name__ == '__main__':
    unittest.main()
