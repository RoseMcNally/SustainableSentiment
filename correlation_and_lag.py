import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

company_names = {'Ericsson': 'Telefonaktiebolaget LM Ericsson', 'Samsung': 'SAMSUNG ELECTRONICS COMPANY LIMITED',
                 'Apple': 'APPLE INC.', 'Microsoft': 'MICROSOFT', 'Cisco': 'CISCO SYSTEMS INCORPORATED',
                 'Qualcomm': 'QUALCOMM INCORPORATED', 'Fujitsu': 'FUJITSU LIMITED', 'Sony': 'SONY CORPORATION',
                 'Hitachi': 'HITACHI LIMITED', 'Toshiba': 'TOSHIBA CORPORATION', 'Lenovo': 'LENOVO GROUP LIMITED',
                 'Foxconn': 'FOXCONN TECHNOLOGY COMPANY LIMITED', 'Huawei': 'Huawei', 'IBM': 'IBM',
                 'Dell': 'DELL TECHNOLOGIES INC.', 'Intel': 'INTEL CORPORATION', 'Siemens': 'Siemens AG',
                 'Asus': 'ASUSTEK COMPUTER INC.', 'Panasonic': 'PANASONIC CORPORATION', 'Nokia': 'NOKIA OY'}


def find_correlation_and_lag(golden, sentiment, company_name):
    golden = golden[golden["company_name"] == company_names.get(company_name)].groupby(
        pd.Grouper(key="pdfyear", freq="1Y")).sum(min_count=1)
    sentiment = sentiment[sentiment["Text"].str.contains(company_name, case=False)].filter(regex="Total|Date")
    tweet_count = sentiment["Total"].sum()
    sentiment["Total"] = sentiment["Total"] * 1000 / tweet_count

    # Create plot

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    sentiment.groupby(pd.Grouper(key="Date", freq="1Y")).sum().plot(y=["Total"], figsize=(10, 7), ax=ax1,
                                                                    ylabel="Sentiment", legend=False, colormap='autumn',
                                                                    ylim=(-500, 500))
    golden.plot(y=["MSCI"], figsize=(10, 7), ax=ax2, legend=False, ylabel="MSCI", ylim=(-12, 12))
    fig.legend(["Sentiment", "MSCI"])
    fig.suptitle("Sentiment score and MSCI for " + company_name + ", 2010-2020")
    fig.savefig("data/correlation_output/" + company_name + ".png")

    golden["MSCI"] = golden["MSCI"].interpolate(limit_direction='both')
    # year_sentiment = sentiment.groupby(pd.Grouper(key="Date", freq="1Y")).sum().filter(regex="Total")
    # no_lag_corr = np.round(golden["MSCI"].corr(year_sentiment["Total"]), decimals=4)
    # print("Correlation:", no_lag_corr)

    # Find correlation and lag

    corr_list = []
    for i in range(-24, 25):
        shifted_sentiment = sentiment.copy(deep=True)
        shifted_sentiment["Date"] = shifted_sentiment["Date"] + pd.DateOffset(months=i)
        shifted_sentiment = shifted_sentiment.groupby(pd.Grouper(key="Date", freq="1Y")).sum().filter(regex="Total")

        min_date = max(shifted_sentiment.index.min(), golden.index.min())
        if i > 0 and i % 12 != 0:
            min_date = min_date + pd.DateOffset(years=1)
        max_date = min(shifted_sentiment.index.max(), golden.index.max())
        if i < 0 and i % 12 != 0:
            max_date = max_date - pd.DateOffset(years=1)

        shifted_sentiment = shifted_sentiment[shifted_sentiment.index <= max_date]
        shifted_sentiment = shifted_sentiment[shifted_sentiment.index >= min_date]

        corr = np.round(golden["MSCI"].corr(shifted_sentiment["Total"]), decimals=4)
        corr_list.append(pd.DataFrame([[i, corr]], columns=["Shift", "Correlation"]))

    results = pd.concat(corr_list, ignore_index=True)
    results.to_csv("data/correlation_output/" + company_name + ".csv")
    max_corr = results["Correlation"].max()
    print("Maximum correlation of", max_corr, "at shift of", list(results[results["Correlation"] == max_corr]["Shift"]),
          "months")


if __name__ == '__main__':
    golden = pd.read_csv("../davinci_huawei_output_Integrated (1).csv")
    golden['pdfyear'] = pd.to_datetime(golden['pdfyear'], format="%Y")
    golden = golden.filter(regex="company_name|MSCI|pdfyear")

    sentiment = pd.read_csv("data/sentiment_output/sentiment.csv")
    sentiment['Date'] = pd.to_datetime(sentiment['Date'])
    sentiment = pd.concat([sentiment, pd.get_dummies(sentiment["Sentiment"], prefix="i")], axis=1)
    sentiment["Total"] = pd.to_numeric(
        pd.to_numeric(sentiment["i_POS"], downcast='signed') - pd.to_numeric(sentiment["i_NEG"], downcast='signed'),
        downcast='signed')

    print("Processing...")
    for name in company_names.keys():
        print(name)
        find_correlation_and_lag(golden, sentiment, name)
