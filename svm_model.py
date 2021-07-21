import os
import re
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec


class SvmModel:

    @staticmethod
    def preprocess(data: pd.DataFrame):
        data['Text'] = data['Text'].apply(lambda x: re.sub(r'http\S+', '', x))
        data['Text'] = data['Text'].apply(lambda x: re.sub('[^A-Za-z0-9 #]+', '', x))
        data['Text'] = data['Text'].apply(lambda x: x.lower())
        return data


class SvmTokenizer:

    def __init__(self):
        self.embedding = None
        self.vector_size = None

    def train_embedding(self, data_path, save_path, vector_size):
        if os.path.isfile(save_path):
            self.embedding = KeyedVectors.load(save_path)
            self.vector_size = self.embedding.vector_size
        else:
            data = pd.read_csv(data_path)
            data = SvmModel.preprocess(data)
            wordVec = [nltk.word_tokenize(tweet) for tweet in data["Text"]]
            self.embedding = Word2Vec(wordVec, min_count=1, vector_size=vector_size)
            self.embedding.save(save_path)
            self.vector_size = vector_size

    def apply(self, data: pd.DataFrame):
        word_train = np.zeros((1, self.vector_size + 1))
        for index, row in data.iterrows():
            total_word_vec = np.zeros((1, self.vector_size))
            text = row['Text']
            for token in text:
                token = str(token)
                try:
                    word_vec = self.embedding.wv[token]
                    word_vec.reshape(1, self.vector_size)
                    total_word_vec += word_vec
                except:
                    continue
            total_word_vec = np.append(total_word_vec, row["Relevance"])
            word_train = np.vstack((word_train, total_word_vec))
        word_train = np.delete(word_train, (0), axis=0)
        return word_train
