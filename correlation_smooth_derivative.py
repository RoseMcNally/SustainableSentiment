from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from correlation_raw import name_translation, get_indices, get_sentiment, print_corr
from correlation_smooth import loess


def smooth_data(sent_data, index_data, name):
    monthly = sent_data[sent_data["Text"].str.contains(name_translation.get(name), case=False)]
    monthly = monthly.groupby(pd.Grouper(key="Date", freq="1M")).sum().filter(regex="Total")
    ind = index_data[index_data.clean_name == name].groupby(pd.Grouper(key="AS_OF_DATE", freq="1M")).sum(min_count=1)
    ind = ind.interpolate(method="pad")

    min_date = pd.Timestamp(2010, 1, 1)
    max_date = pd.Timestamp(2020, 12, 1)

    # Plot sentiment
    eval_sent = loess("Total", data=monthly, alpha=0.15, poly_degree=2)
    eval_sent["v"] = eval_sent["v"].apply(lambda x: datetime.fromtimestamp(x))
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(111)
    ax1.plot(eval_sent['v'], eval_sent['g'], color='red', linewidth=3, label="Sentiment")
    plt.title('Sustainability Sentiment for ' + name_translation.get(
        name) + ' with LOWESS, alpha = 0.15, polynomial degree = 2')
    plt.xlabel("Date")
    plt.ylabel("Sentiment")
    plt.xlim((min_date, max_date))
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/lowess_derivative/" + name_translation.get(name) + "_sentiment.png", dpi=150)
    plt.close(fig)

    # Plot indices
    evalDF_ind = loess("INDUSTRY_ADJUSTED_SCORE", data=ind, alpha=0.18, poly_degree=2)
    evalDF_env = loess("ENVIRONMENTAL_PILLAR_SCORE", data=ind, alpha=0.18, poly_degree=2)
    evalDF_soc = loess("SOCIAL_PILLAR_SCORE", data=ind, alpha=0.18, poly_degree=2)
    evalDF_ind["v"] = evalDF_ind["v"].apply(lambda x: datetime.fromtimestamp(x))
    evalDF_env["v"] = evalDF_env["v"].apply(lambda x: datetime.fromtimestamp(x))
    evalDF_soc["v"] = evalDF_soc["v"].apply(lambda x: datetime.fromtimestamp(x))
    evalDF_ind["g"] = evalDF_ind["g"].diff()
    evalDF_env["g"] = evalDF_env["g"].diff()
    evalDF_soc["g"] = evalDF_soc["g"].diff()
    fig2 = plt.figure(figsize=(15, 10))
    ax2 = fig2.add_subplot(111)
    ax2.plot(evalDF_ind['v'], evalDF_ind['g'], color='blue', linewidth=3, label="Industry")
    ax2.plot(evalDF_env['v'], evalDF_env['g'], color='green', linewidth=3, label="Environmental")
    ax2.plot(evalDF_soc['v'], evalDF_soc['g'], color='orange', linewidth=3, label="Social")
    plt.title('Sustainbility Indices Derivatives for ' + name_translation.get(
        name) + ' with LOWESS, alpha = 0.18, polynomial degree = 2')
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.xlim((min_date, max_date))
    plt.legend()
    plt.tight_layout()
    ax2.get_figure().savefig("data/lowess_derivative/" + name_translation.get(name) + "_indices_derivatives.png",
                             dpi=150)
    plt.close(fig2)

    # Plot combined
    fig3 = plt.figure(figsize=(15, 10))
    ax3 = fig3.add_subplot(111)
    ax4 = ax3.twinx()
    ax3.plot(eval_sent['v'], eval_sent['g'], color='red', linewidth=3, label="Sentiment")
    ax4.plot(evalDF_ind['v'], evalDF_ind['g'], color='blue', linewidth=3, label="Industry")
    ax4.plot(evalDF_env['v'], evalDF_env['g'], color='green', linewidth=3, label="Environmental")
    ax4.plot(evalDF_soc['v'], evalDF_soc['g'], color='orange', linewidth=3, label="Social")
    plt.title('Sustainability Sentiment and Indices Derivatives for ' + name_translation.get(
        name) + ' with LOWESS, alpha = 0.15, polynomial degree = 2')
    plt.xlabel("Date")
    ax3.set_ylabel("Sentiment")
    ax4.set_ylabel("Index Derivative")
    plt.xlim((min_date, max_date))
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/lowess_derivative/" + name_translation.get(name) + "_combined.png", dpi=150)
    plt.close(fig3)

    # Find correlation (no lag)

    evalDF = eval_sent.groupby(pd.Grouper(key="v", freq="1M")).sum().rename(columns={"g": "sentiment"})
    evalDF_ind = evalDF_ind.groupby(pd.Grouper(key="v", freq="1M")).sum().rename(columns={"g": "industry"})
    evalDF_env = evalDF_env.groupby(pd.Grouper(key="v", freq="1M")).sum().rename(columns={"g": "environment"})
    evalDF_soc = evalDF_soc.groupby(pd.Grouper(key="v", freq="1M")).sum().rename(columns={"g": "social"})
    eval_tot = pd.concat([evalDF, evalDF_ind, evalDF_env, evalDF_soc], axis=1)
    no_lag_corr = eval_tot.corr()
    no_lag_corr.to_csv("data/lowess_derivative/no_lag_" + name_translation.get(name) + ".csv")
    print("Industry correlation:", no_lag_corr.sentiment.iloc[1])
    print("Environmental correlation:", no_lag_corr.sentiment.iloc[2])
    print("Social correlation:", no_lag_corr.sentiment.iloc[3])

    # Find correlation (with lag)

    corr_list = []
    for i in range(-24, 25):
        eval_temp = eval_tot.copy(deep=True)
        eval_temp.sentiment = eval_temp.sentiment.shift(i)
        corr = eval_temp.corr()
        corr_list.append(pd.DataFrame([[i, corr.sentiment.iloc[1], corr.sentiment.iloc[2], corr.sentiment.iloc[3]]],
                                      columns=["Shift", "Industry", "Environmental", "Social"]))

    results = pd.concat(corr_list, ignore_index=True)
    results.to_csv("data/lowess_derivative/lag_" + name_translation.get(name) + ".csv")
    print_corr(results, "Industry")
    print_corr(results, "Environmental")
    print_corr(results, "Social")


if __name__ == '__main__':
    indices = get_indices()
    sentiment = get_sentiment()

    print("Processing...")
    for name in name_translation.keys():
        print(name_translation.get(name))
        smooth_data(sentiment, indices, name)
