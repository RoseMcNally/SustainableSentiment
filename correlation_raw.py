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

def find_correlation_and_lag(indices, sentiment, company_name):
    # Filter company
    indices = indices[indices.clean_name == company_name].groupby(
        pd.Grouper(key="AS_OF_DATE", freq="1M")).sum(min_count=1)
    sentiment = sentiment[sentiment["Text"].str.contains(name_translation.get(company_name), case=False)].filter(
        regex="Total|Date")

    # Create plot
    min_date = pd.Timestamp(2010, 1, 1)
    max_date = pd.Timestamp(2020, 12, 1)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    sentiment.groupby(pd.Grouper(key="Date", freq="1M")).sum().plot(y=["Total"], figsize=(10, 7), ax=ax1,
                                                                    ylabel="Sentiment", legend=True, colormap='autumn')
    indices.plot(figsize=(10, 7), ax=ax2, legend=True, ylabel="Score", ylim=(0, 11))
    ax1.set_xlim((min_date, max_date))
    fig.suptitle("Sentiment and Sustainability Indices for " + name_translation.get(company_name) + ", 2010-2020")
    fig.savefig("data/correlation_output_indices/" + name_translation.get(company_name) + ".png")

    # Find correlation (no lag)
    indices = indices.interpolate(limit_direction='both')
    month_sentiment = sentiment.groupby(pd.Grouper(key="Date", freq="1M")).sum().filter(regex="Total")
    no_lag_corr_ind = np.round(indices.INDUSTRY_ADJUSTED_SCORE.corr(np.squeeze(month_sentiment)), decimals=4)
    no_lag_corr_env = np.round(indices.ENVIRONMENTAL_PILLAR_SCORE.corr(np.squeeze(month_sentiment)), decimals=4)
    no_lag_corr_soc = np.round(indices.SOCIAL_PILLAR_SCORE.corr(np.squeeze(month_sentiment)), decimals=4)
    print("Industry correlation:", no_lag_corr_ind)
    print("Environmental correlation:", no_lag_corr_env)
    print("Social correlation:", no_lag_corr_soc)

    # Find correlation (with lag)
    corr_list = []
    for i in range(-24, 25):
        shifted_sentiment = sentiment.copy(deep=True)
        shifted_sentiment["Date"] = shifted_sentiment["Date"] + pd.DateOffset(months=i)
        shifted_sentiment = shifted_sentiment.groupby(pd.Grouper(key="Date", freq="1M")).sum().filter(regex="Total")

        corr_ind = np.round(indices["INDUSTRY_ADJUSTED_SCORE"].corr(np.squeeze(shifted_sentiment)), decimals=4)
        corr_env = np.round(indices.ENVIRONMENTAL_PILLAR_SCORE.corr(np.squeeze(shifted_sentiment)), decimals=4)
        corr_soc = np.round(indices.SOCIAL_PILLAR_SCORE.corr(np.squeeze(shifted_sentiment)), decimals=4)
        corr_list.append(
            pd.DataFrame([[i, corr_ind, corr_env, corr_soc]], columns=["Shift", "Industry", "Environmental", "Social"]))

    results = pd.concat(corr_list, ignore_index=True)
    results.to_csv("data/correlation_output_indices/" + company_name + ".csv")
    print_corr(results, "Industry")
    print_corr(results, "Environmental")
    print_corr(results, "Social")


def get_indices():
    indices = pd.read_csv("Monthly_msci_data.csv")
    indices = indices.filter(
        regex="AS_OF_DATE|INDUSTRY_ADJUSTED_SCORE|SOCIAL_PILLAR_SCORE|ENVIRONMENTAL_PILLAR_SCORE|clean_name")
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
        print(name_translation.get(name))
        find_correlation_and_lag(indices_data, sentiment_data, name)
