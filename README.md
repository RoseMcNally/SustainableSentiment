# Sustainable Sentiment 

This GitHub repository contains the files for the MSc individual project "Sustainable Sentiment: Estimating the sentiment for sustainability strategies through machine learning".

## Tweet Collection

Twitter data stored in `/data/all/` was collected using a script edited from Jake Lever's [twitterScraper](https://github.com/DL-WG/twitterScraper) project, and pre-processed using code in the file `tweetPreprocessing.py`. The raw data is not included here.

## Natural Language Processing Models

Two NLP models have been developed for this research: a support vector machine (source: `svm.py`, model: `models/svm`) and a BERT model (source: `bert.py`, model: `models/bert`). The evaluation script `nlp_eval.py` outputs the performance of each of the models. The hand-labelled data used to train and test the models can be found in `data/labelled` and the full output data from the models are stored in `data/svm_output` and `data/bert_output`.

## Sustainable Action vs Sentiment

## Random Forest Predictor
