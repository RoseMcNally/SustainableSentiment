# Sustainable Sentiment 

This GitHub repository contains the files for the MSc individual project "Noise Filtration in Twitter Data and its Application in a Study on Corporate Sustainability Sentiment". The tweet data is not included due to the large file sizes and Twitter's Developer Policy. To request any tweet data collected or processed as part of this project, please email me at rosemcnally@hotmail.co.uk. Note that it can only be used for academic research as stipulated by the [Twitter API policy](https://developer.twitter.com/en/developer-terms/policy). The MSCI data is not available.

## Tweet Collection

The raw tweets were collected using a script edited from Jake Lever's [twitterScraper](https://github.com/DL-WG/twitterScraper) project, and pre-processed using code in the file `tweetPreprocessing.py`. 

`methods.py` contains a helper function for reading the labelled data from csv.

## Natural Language Processing Models

- Support Vector Machine: Pre-processing and tokenizer code in `svm_model.py`, optimisation and results in `svm_optimisation.py`.
- General BERT: general architecture in `general_bert_architecture.py`, optimisation code in `general_bert_optimizer.py`, specific models:
  - BERT: pre-processor class in `bert_preprocessor.py`, model optimisation and results in `bert_optimisation.py`
  - BERTweet: model optimisation and results in `bertweet_optimisation.py`

The results from the optimisation procedures are documented in comments at the end of the optimisation files.

To train your own BERTweet noise filtration model, you will need the general BERT files and the `bert_optimisation.py` file, and you can follow the code in the `bertweet_optimisation.py` file. Your labelled data should be in a pandas dataframe with the tweets series named `Text` and the labels series named `Relevance`. 

The `filter_full_data.py` file runs the successful BERTweet model on the full tweet dataset.

## Sustainable Behaviour vs Sentiment

The `sentiment_analysis.py` file runs the BERTweet-based sentiment model from [pysentimiento](https://github.com/pysentimiento/pysentimiento) on the full dataset.

Comparision between sentiment and MSCI indexes:
- `sentiment_vs_msci_raw.py` plots the raw sentiment and msci indexes and calculates the correlation
- `sentiment_vs_msci_smooth.py` smoothes the sentiment and msci indexes, plots them, calculates the correlation and scans from a lag of -12 to +12 months to find the maximum correlation

Comparision between sentiment and change in MSCI indexes:
- `sentiment_vs_change_msci_raw.py` plots the raw sentiment and change in msci indexes and calculates the correlation
- `sentiment_vs_change_msci_smooth.py` smoothes the sentiment and change in msci indexes, plots them, calculates the correlation and scans from a lag of -12 to +12 months to find the maximum correlation

