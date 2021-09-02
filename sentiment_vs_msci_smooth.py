from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sentiment_vs_msci_raw import name_translation, get_indices, get_sentiment, print_corr

# The following two functions are taken from https://github.com/MikLang/Lowess_simulation

def loc_eval(x, b):
    loc_est = 0
    for i in enumerate(b): loc_est += i[1] * (x ** i[0])
    return (loc_est)


def loess(yvals, data, alpha, poly_degree=1):
    x = [t.timestamp() for t in data.index.tolist()]
    all_data = sorted(zip(x, data[yvals].tolist()), key=lambda x: x[0])
    xvals, yvals = zip(*all_data)
    evalDF = pd.DataFrame(columns=['v', 'g'])
    n = len(xvals)
    m = n + 1
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    avg_interval = ((max(xvals) - min(xvals)) / len(xvals))
    v_lb = min(xvals) - (.5 * avg_interval)
    v_ub = (max(xvals) + (.5 * avg_interval))
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)
    xcols = [np.ones_like(xvals)]
    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    X = np.vstack(xcols).T
    for i in v:
        iterpos = i[0]
        iterval = i[1]
        iterdists = sorted([(j, np.abs(j - iterval)) for j in xvals], key=lambda x: x[1])
        _, raw_dists = zip(*iterdists)
        scale_fact = raw_dists[q - 1]
        scaled_dists = [(j[0], (j[1] / scale_fact)) for j in iterdists]
        weights = [(j[0], ((1 - np.abs(j[1] ** 3)) ** 3 if j[1] <= 1 else 0)) for j in scaled_dists]
        _, weights = zip(*sorted(weights, key=lambda x: x[0]))
        _, raw_dists = zip(*sorted(iterdists, key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists, key=lambda x: x[0]))
        W = np.diag(weights)
        b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ yvals)
        local_est = loc_eval(iterval, b)
        iterDF2 = pd.DataFrame({
            'v': [iterval],
            'g': [local_est]
        })
        evalDF = pd.concat([evalDF, iterDF2])
    evalDF = evalDF[['v', 'g']]
    return (evalDF)


def smooth_data(sent_data, index_data, name):
    min_date = pd.Timestamp(2010, 1, 1)
    max_date = pd.Timestamp(2020, 12, 1)
    monthly = sent_data[sent_data["Text"].str.contains(name_translation.get(name), case=False)]
    monthly = monthly.groupby(pd.Grouper(key="Date", freq="1M")).sum().filter(regex="Total")
    ind = index_data[index_data.clean_name == name].groupby(pd.Grouper(key="AS_OF_DATE", freq="1M")).sum(min_count=1)
    ind = ind[min_date:]
    ind = ind.interpolate(method="pad")
    ind = ind.dropna()

    # Smooth

    eval_sent = loess("Total", data=monthly, alpha=0.15, poly_degree=2)
    eval_sent["v"] = eval_sent["v"].apply(lambda x: datetime.fromtimestamp(x))
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
    ax4.set_ylabel("Index Score")
    ax3.set_xlabel("Date")
    plt.xlim((min_date, max_date))
    plt.tight_layout()
    plt.savefig("data/lowess/report/other_smoothed_" + name_translation.get(name) + ".png", dpi=400)
    plt.close(fig3)

    # Find correlation (no lag)

    evalDF = eval_sent.groupby(pd.Grouper(key="v", freq="1M")).sum().rename(columns={"g": "sentiment"})
    evalDF_env = evalDF_env.groupby(pd.Grouper(key="v", freq="1M")).sum().rename(columns={"g": "environment"})
    evalDF_soc = evalDF_soc.groupby(pd.Grouper(key="v", freq="1M")).sum().rename(columns={"g": "social"})
    eval_tot = pd.concat([evalDF, evalDF_env, evalDF_soc], axis=1)
    no_lag_corr = eval_tot.corr()
    no_lag_corr.to_csv("data/lowess/no_lag_" + name_translation.get(name) + ".csv")
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
    results.to_csv("data/lowess/lag_" + name_translation.get(name) + ".csv")
    print_corr(results, "Environmental")
    print_corr(results, "Social")


if __name__ == '__main__':
    indices = get_indices()
    sentiment = get_sentiment()

    print("Processing...")
    for name in name_translation.keys():
        print(name_translation.get(name))
        smooth_data(sentiment, indices, name)
