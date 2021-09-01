from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from sentiment_vs_msci_raw import name_translation, get_indices, get_sentiment, print_corr
from sentiment_vs_msci_smooth import loess
from sentiment_vs_change_msci_raw import align_yaxis

def smooth_data(sent_data, index_data, name):
    min_date = pd.Timestamp(2010, 1, 1)
    max_date = pd.Timestamp(2020, 12, 1)
    monthly = sent_data[sent_data["Text"].str.contains(name_translation.get(name), case=False)]
    monthly = monthly.groupby(pd.Grouper(key="Date", freq="1M")).sum().filter(regex="Total")
    ind = index_data[index_data.clean_name == name].groupby(pd.Grouper(key="AS_OF_DATE", freq="1M")).sum(min_count=1)
    ind = ind[min_date:]
    ind = ind.interpolate(method="pad")

    # Smooth

    eval_sent = loess("Total", data=monthly, alpha=0.15, poly_degree=2)
    eval_sent["v"] = eval_sent["v"].apply(lambda x: datetime.fromtimestamp(x))
    ind.ENVIRONMENTAL_PILLAR_SCORE = ind.ENVIRONMENTAL_PILLAR_SCORE.diff()
    ind.SOCIAL_PILLAR_SCORE = ind.SOCIAL_PILLAR_SCORE.diff()
    ind = ind.dropna()
    evalDF_env = loess("ENVIRONMENTAL_PILLAR_SCORE", data=ind, alpha=0.18, poly_degree=2)
    evalDF_soc = loess("SOCIAL_PILLAR_SCORE", data=ind, alpha=0.18, poly_degree=2)
    evalDF_env["v"] = evalDF_env["v"].apply(lambda x: datetime.fromtimestamp(x))
    evalDF_soc["v"] = evalDF_soc["v"].apply(lambda x: datetime.fromtimestamp(x))

    # Plot combined

    fig3 = plt.figure(figsize=(9, 3.5))
    ax3 = fig3.add_subplot(111)
    ax4 = ax3.twinx()
    ax3.plot(eval_sent['v'], eval_sent['g'], color='red')
    ax4.plot(evalDF_env['v'], evalDF_env['g'], color='green')
    ax4.plot(evalDF_soc['v'], evalDF_soc['g'], color='orange')
    plt.xlabel("Date")
    ax3.set_ylabel("Tweet Sentiment Count")
    ax4.set_ylabel("Index Score Difference")
    ax3.set_xlabel("Date")
    plt.xlim((min_date, max_date))
    align_yaxis(ax3,ax4)
    plt.tight_layout()
    plt.savefig("data/index_diff/v4_other_ind_diff_" + name_translation.get(name) + ".png", dpi=400)
    plt.close(fig3)

    # Find correlation (no lag)

    evalDF = eval_sent.groupby(pd.Grouper(key="v", freq="1M")).sum().rename(columns={"g": "sentiment"})
    evalDF_env = evalDF_env.groupby(pd.Grouper(key="v", freq="1M")).sum().rename(columns={"g": "environment"})
    evalDF_soc = evalDF_soc.groupby(pd.Grouper(key="v", freq="1M")).sum().rename(columns={"g": "social"})
    eval_tot = pd.concat([evalDF, evalDF_env, evalDF_soc], axis=1)
    no_lag_corr = eval_tot.corr()
    no_lag_corr.to_csv("data/lowess_derivative/no_lag_" + name_translation.get(name) + ".csv")
    print("Environmental correlation:", no_lag_corr.sentiment.iloc[1])
    print("Social correlation:", no_lag_corr.sentiment.iloc[2])

    # Find correlation (with lag)

    corr_list = []
    for i in range(-12, 13):
        eval_temp = eval_tot.copy(deep=True)
        eval_temp.sentiment = eval_temp.sentiment.shift(i)
        corr = eval_temp.corr()
        corr_list.append(pd.DataFrame([[i, corr.sentiment.iloc[1], corr.sentiment.iloc[2]]],
                                      columns=["Shift", "Environmental", "Social"]))

    results = pd.concat(corr_list, ignore_index=True)
    results.to_csv("data/lowess_derivative/lag_" + name_translation.get(name) + ".csv")
    print_corr(results, "Environmental")
    print_corr(results, "Social")


if __name__ == '__main__':
    indices = get_indices()
    sentiment = get_sentiment()

    print("Processing...")
    for name in name_translation.keys():
        print(name_translation.get(name))
        smooth_data(sentiment, indices, name)
