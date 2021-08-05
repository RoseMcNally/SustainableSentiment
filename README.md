# Sustainable Sentiment 

This GitHub repository contains the files for the MSc individual project "Sustainable Sentiment: Estimating the Sentiment for the Sustainability of Technology Companies through Machine Learning". The data is not included due to the large file sizes. To request any data collected or processed as part of this project, please email me at rosemcnally@hotmail.co.uk.

## Tweet Collection

The raw tweets were collected using a script edited from Jake Lever's [twitterScraper](https://github.com/DL-WG/twitterScraper) project, and pre-processed using code in the file `tweetPreprocessing.py`. 

## Natural Language Processing Models

- Support Vector Machine: Pre-processing and tokenizer code in `svm_model.py`, optimisation and results in `svm_optimisation.py`.
- General BERT: general architecture in `general_bert_architecture.py`, optimisation code in `general_bert_optimizer.py`, specific models:
  - BERT: pre-processor class in `bert_preprocessor.py`, model optimisation and results in `bert_optimisation.py`
  - BERTweet: model optimisation and results in `bertweet_optimisation.py`

The `filter_full_data.py` file runs the successful BERTweet model on the full tweet dataset.

## Sustainable Action vs Sentiment

The `sentiment_analysis.py` file runs the BERTweet-based sentiment model from [pysentimiento](https://github.com/pysentimiento/pysentimiento) on the full dataset.

The `correlation_raw.py` file plots the raw sustainability indices and sentiment data, and finds the correlation between the two. It also scans from a lag of -24 months to +24 months and finds the maximum correlation.

The `correlation_smooth.py` file smoothes the sustainability indices and sentiment data, plots them and finds the correlation and lag.

The `correlation_smooth_derivative.py` file smoothes the sustainability indices and sentiment data, differentiates the sustainability indices, plots them and finds the correlation and lag.

