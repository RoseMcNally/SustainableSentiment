import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

name_changes = {"ericssontelephoneab": "telefonaktiebolagetlmericsson",
                "samsungelectronicscoltd": "samsungelectronicsco", "qualcommincorporated": "qualcomm",
                "hitachiltd": "hitachi", "internationalbusinessmachines": "internationalbusinessmachine",
                "asustekcomputerincorporation": "asustekcomputer"}

final_names = ["telefonaktiebolagetlmericsson", "samsungelectronicsco", "apple", "microsoft", "ciscosystems",
               "qualcomm", "fujitsu", "sony", "hitachi", "toshiba", "lenovogroup", "foxconntechnologyco",
               "huaweiinvestmentandholdingco", "internationalbusinessmachine", "dell", "intel",
               "siemensaktiengesellschaft", "asustekcomputer", "panasonic", "nokiaoyj"]

name_translation = {"telefonaktiebolagetlmericsson": "Ericsson", "samsungelectronicsco": "Samsung", "apple": "Apple",
                    "microsoft": "Microsoft", "ciscosystems": "Cisco", "qualcomm": "Qualcomm", "fujitsu": "Fujitsu",
                    "sony": "Sony", "hitachi": "Hitachi", "toshiba": "Toshiba", "lenovogroup": "Lenovo",
                    "foxconntechnologyco": "Foxconn", "huaweiinvestmentandholdingco": "Huawei",
                    "internationalbusinessmachine": "IBM", "dell": "Dell", "intel": "Intel",
                    "siemensaktiengesellschaft": "Siemens", "asustekcomputer": "Asus", "panasonic": "Panasonic",
                    "nokiaoyj": "Nokia"}


def print_corr(results, column):
    max_corr_ind = results[column].max()
    print("Maximum " + column + " correlation of", max_corr_ind, "at shift of",
          list(results[results[column] == max_corr_ind]["Shift"]),
          "months")

def find_correlation(indices, sentiment, company_name):
    min_date = pd.Timestamp(2010, 1, 1)
    max_date = pd.Timestamp(2020, 12, 1)
    indices = indices[indices.clean_name == company_name].groupby(
        pd.Grouper(key="AS_OF_DATE", freq="1M")).sum(min_count=1)
    sentiment = sentiment[sentiment["Text"].str.contains(name_translation.get(company_name), case=False)].filter(
        regex="Total|Date")
    sentiment = sentiment.groupby(pd.Grouper(key="Date", freq="1M")).sum()
    indices_int = indices[min_date:]
    indices_int = indices_int.interpolate(method='pad')

    # Plot

    fig, ax1 = plt.subplots(figsize=(9,3))
    ax2 = ax1.twinx()
    ax1.plot(sentiment.index, sentiment.Total, color="Red")
    ax2.plot(indices_int.index, indices_int.ENVIRONMENTAL_PILLAR_SCORE,'--', color="Green")
    ax2.plot(indices.index, indices.ENVIRONMENTAL_PILLAR_SCORE, color="Green")
    ax2.plot(indices_int.index, indices_int.SOCIAL_PILLAR_SCORE, '--', color="Orange")
    ax2.plot(indices.index, indices.SOCIAL_PILLAR_SCORE, color="Orange")
    ax1.set_xlim((min_date, max_date))
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Tweet Sentiment Count")
    ax2.set_ylabel("Index Score")
    plt.tight_layout()
    fig.savefig("data/correlation_output_indices/report/a_raw_" + name_translation.get(company_name) + ".png", dpi=400)

    # Find correlation

    indices = indices_int
    no_lag_corr_ind = np.round(indices.INDUSTRY_ADJUSTED_SCORE.corr(np.squeeze(sentiment)), decimals=4)
    no_lag_corr_env = np.round(indices.ENVIRONMENTAL_PILLAR_SCORE.corr(np.squeeze(sentiment)), decimals=4)
    no_lag_corr_soc = np.round(indices.SOCIAL_PILLAR_SCORE.corr(np.squeeze(sentiment)), decimals=4)
    print("Industry correlation:", no_lag_corr_ind)
    print("Environmental correlation:", no_lag_corr_env)
    print("Social correlation:", no_lag_corr_soc)


def get_indices():
    indices = pd.read_csv("Monthly_msci_data.csv")
    indices = indices.filter(
        regex="AS_OF_DATE|SOCIAL_PILLAR_SCORE|ENVIRONMENTAL_PILLAR_SCORE|clean_name")
    indices.AS_OF_DATE = pd.to_datetime(indices.AS_OF_DATE)
    for name in name_changes.keys():
        indices.clean_name = indices.clean_name.str.replace(name, name_changes.get(name))
    pattern = "|".join(final_names)
    indices = indices[indices.clean_name.str.fullmatch(pattern)]
    return indices


def get_sentiment():
    sentiment = pd.read_csv("data/sentiment_output/sentiment.csv")
    sentiment['Date'] = pd.to_datetime(sentiment['Date'])
    sentiment = pd.concat([sentiment, pd.get_dummies(sentiment["Sentiment"], prefix="i")], axis=1)
    sentiment["Total"] = pd.to_numeric(
        pd.to_numeric(sentiment["i_POS"], downcast='signed') - pd.to_numeric(sentiment["i_NEG"], downcast='signed'),
        downcast='signed')
    return sentiment


if __name__ == '__main__':
    indices_data = get_indices()
    sentiment_data = get_sentiment()

    print("Processing...")
    for name in name_translation.keys():
        if name == "lenovogroup":
            print(name_translation.get(name))
            find_correlation(indices_data, sentiment_data, name)
