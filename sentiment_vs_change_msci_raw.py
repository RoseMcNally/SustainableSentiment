import pandas as pd
import matplotlib.pyplot as plt

from sentiment_vs_msci_raw import name_translation, get_indices, get_sentiment


def align_yaxis(ax1, ax2):
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]

    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])


def smooth_data(sent_data, index_data, name):
    min_date = pd.Timestamp(2010, 1, 1)
    max_date = pd.Timestamp(2020, 12, 1)
    monthly = sent_data[sent_data["Text"].str.contains(name_translation.get(name), case=False)]
    monthly = monthly.groupby(pd.Grouper(key="Date", freq="10D")).sum().filter(regex="Total")
    ind = index_data[index_data.clean_name == name].groupby(pd.Grouper(key="AS_OF_DATE", freq="1M")).sum(min_count=1)
    ind = ind[min_date:]
    ind_int = ind.interpolate(method="pad")

    # Smooth

    ind.ENVIRONMENTAL_PILLAR_SCORE = ind.ENVIRONMENTAL_PILLAR_SCORE.diff()
    ind.SOCIAL_PILLAR_SCORE = ind.SOCIAL_PILLAR_SCORE.diff()
    ind = ind.dropna()
    ind_int.ENVIRONMENTAL_PILLAR_SCORE = ind_int.ENVIRONMENTAL_PILLAR_SCORE.diff()
    ind_int.SOCIAL_PILLAR_SCORE = ind_int.SOCIAL_PILLAR_SCORE.diff()
    ind_int = ind_int.dropna()

    # Plot combined

    fig3 = plt.figure(figsize=(9, 3.5))
    ax3 = fig3.add_subplot(111)
    ax4 = ax3.twinx()
    ax3.plot(monthly.index, monthly.Total, color='red')
    ax4.plot(ind_int.index, ind_int.ENVIRONMENTAL_PILLAR_SCORE, '--', color='green')
    ax4.plot(ind_int.index, ind_int.SOCIAL_PILLAR_SCORE, '--', color='orange')
    ax4.plot(ind.index, ind.ENVIRONMENTAL_PILLAR_SCORE, color='green')
    ax4.plot(ind.index, ind.SOCIAL_PILLAR_SCORE, color='orange')
    plt.xlabel("Date")
    ax3.set_ylabel("Tweet Sentiment Count")
    ax4.set_ylabel("Index Score Difference")
    ax3.set_xlabel("Date")
    plt.xlim((min_date, max_date))
    align_yaxis(ax3,ax4)
    plt.tight_layout()
    plt.savefig("data/unsmoothed_diff/v2_" + name_translation.get(name) + ".png", dpi=400)
    plt.close(fig3)

    # Find correlation

    eval_tot = pd.concat([monthly, ind_int], axis=1)
    no_lag_corr = eval_tot.corr()
    no_lag_corr.to_csv("data/lowess_derivative/no_lag_" + name_translation.get(name) + ".csv")
    print("Environmental correlation:", no_lag_corr.Total.iloc[1])
    print("Social correlation:", no_lag_corr.Total.iloc[2])


if __name__ == '__main__':
    indices = get_indices()
    sentiment = get_sentiment()

    print("Processing...")
    for name in name_translation.keys():
        print(name_translation.get(name))
        smooth_data(sentiment, indices, name)
