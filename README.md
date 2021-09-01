# Sustainable Sentiment 

This GitHub repository contains the files for the MSc individual project "Noise Filtration in Twitter Data and its Application in a Study on Corporate Sustainability Sentiment". The tweet data is not included due to the large file sizes and Twitter's Developer Policy. To request any tweet data collected or processed as part of this project, please email me at rosemcnally@hotmail.co.uk. Note that it can only be used for academic research as stipulated by the [Twitter API policy](https://developer.twitter.com/en/developer-terms/policy).

## Tweet Collection

The raw tweets were collected using a script edited from Jake Lever's [twitterScraper](https://github.com/DL-WG/twitterScraper) project, and pre-processed using code in the file `tweetPreprocessing.py`. 

## Natural Language Processing Models

- Support Vector Machine: Pre-processing and tokenizer code in `svm_model.py`, optimisation and results in `svm_optimisation.py`.
- General BERT: general architecture in `general_bert_architecture.py`, optimisation code in `general_bert_optimizer.py`, specific models:
  - BERT: pre-processor class in `bert_preprocessor.py`, model optimisation and results in `bert_optimisation.py`
  - BERTweet: model optimisation and results in `bertweet_optimisation.py`

The `filter_full_data.py` file runs the successful BERTweet model on the full tweet dataset.

## Sustainable Behaviour vs Sentiment

The `sentiment_analysis.py` file runs the BERTweet-based sentiment model from [pysentimiento](https://github.com/pysentimiento/pysentimiento) on the full dataset.

The `correlation_raw.py` file plots the raw sustainability indices and sentiment data, and finds the correlation between the two. It also scans from a lag of -12 months to +12 months and finds the maximum correlation.

The `correlation_smooth.py` file smoothes the sustainability indices and sentiment data, plots them and finds the correlation and lag.

The `correlation_smooth_derivative.py` file smoothes the sustainability indices and sentiment data, differentiates the sustainability indices, plots them and finds the correlation and lag.

