# Stock Price Estimation Using Twitter Sentiment Analysis

# For data processing
import pandas as pd
import numpy as np

# For csv file handling
import csv
import pickle
import os
import re

# For date time utilities
import datetime
import dateutil.parser
import unicodedata

# For adding wait time between requests
import time

# For downloading data from Twitter
import snscrape.modules.twitter as sntwitter

# For downloading data from Yahoo! Finance
import yfinance as yf

# For data visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import seaborn as sns

# For building and training prediction model
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# For NLP & Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
nltk.download('vader_lexicon')
nltk.download('punkt')

# matplotlib.style.use('ggplot')


# The function below is created for downloading tweets for the given date range,
# along with the date and timestamp information.

def DownloadTweets(username='#', strt_dt='1000-01-01', end_dt='9999-12-31'):

    if username == '':
        raise ValueError("User name cannot be blank")

# Building query string with username, start date and end date
    query = "from:" + username + " " + "since:" + strt_dt + " " + " until:" + end_dt + "lang:en"
    
    tweets = []
    
# Scrapes tweets from the specified user's account, and adds them to a list
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    
    return tweets


# The function below is created for downloading historical stock prices data for the given stocks. 

def DownloadStockPrices(ticker, strt_dt, end_dt):
    
    per = '1mo'
# no output if username is not specified    
    if ticker == '':
        print("Ticker is not specified; returning without any data")
        return
# if start or end date is not specified, by default fetching last 1 month's stock prices    
    if strt_dt == '' or end_dt == '':
        print("\nStart data and/or end date is not specified; fetching last 1 month data")
        print()
        stock_data = yf.download(ticker,
                                 period=per)
    else:
# The yfinance API fetches data from the start date to the end date - 1; 
# therefore, adding 1 to specified end date
        end_dt_d = datetime.datetime.strptime(end_dt, "%Y-%m-%d").date()
        end_dt_d += datetime.timedelta(days=1)
        end_dt = str(end_dt_d)
        stock_data = yf.download(ticker,
                                 start=strt_dt,
                                 end=end_dt)
    
    print("Stock prices downloaded for", ticker)
    return stock_data


# The function below will scan through the dataframe and for missing values performs data imputation using a concave function.

def ConcaveImpute(df):
    
    for col in df.columns: # iterating over each column
        while df[col].isna().sum() > 0: # perform until there is no null value
            for i in range(df.shape[0] - 1): # iterate over all rows of the column
                if pd.isnull(df.loc[i,col]):
                    seq = [i] # list of consecutive null values
                    j = i
                    while pd.isnull(df.loc[j,col]):
                        j = j + 1
                        seq.append(j)
                    # data imputation using concave function
                    if len(seq) % 2 == 0:
                        df.loc[seq[int((len(seq) - 1)/2)],col] = (df.loc[i - 1,col] + df.loc[seq[len(seq) - 1],col])/2
                    else:
                        df.loc[seq[int((len(seq) - 1)/2)],col] = (df.loc[i - 1,col] + df.loc[seq[len(seq) - 1],col])/2
                # if value is not null, then no change
                else:
                    df.loc[i,col] = df.loc[i,col]
    
    return df


def AddColumns(df, ticker):
    
    # Adding a column with the ticker name in all rows
    df.insert(0,
              "Stock",
              ticker.split(".")[0])
    
    # Adding column for normalised percentage change in closing price
    df["Close_N"] = df["Close"].pct_change()
       
    # Adding column to represent month
    df["Month"] = df['Date'].dt.month
    
    return df


# The function below will perform 4 steps - 
# 1) Fetch historical stock prices for the specified ticker; for this purpose, it will call the DownloadStockPrices function
# 2) Add rows for missing days
# 3) Perform data imputation; for this purpose, it will call the ConcaveImpute function
# 4) Add a column with the ticker name in all rows

def PrepareStocksData(ticker, start, end):
    
    # Downloading stock prices data
    stock_df = DownloadStockPrices(ticker,
                                   start,
                                   end)
    
    # Inserting rows for missing days  
    idx = pd.date_range(start, end)
    stock_df = stock_df.reindex(idx)
    stock_df.reset_index(inplace=True)
    stock_df.rename(columns={'index':'Date'}, inplace=True)
    
    # Data imputation
    stock_df = ConcaveImpute(stock_df)
    
    # Adding columns
    stock_df = AddColumns(stock_df, ticker)
    
    print("Stock prices data ready for", ticker)
    return stock_df


#==========================================================================================================

# Stock Price Prediction Based on Historical Prices
# The prediction workflow will have the following steps - 
# 
# 1) Data preparation
# 2) Train - test split
# 3) Model building and training
# 4) Prediction
# 5) Evaluation of results
# 6) Visualization of results


def FilterData(df, cols):

    if not type(cols) is list:
        raise TypeError("A list expected  for argument 'cols'")
        return
    
    df_filt = df.filter(cols)
    return df_filt


def SplitFtrs(df, vals, cols):
 
    if not type(vals) is list:
        raise TypeError("Lists expected for argument 'vals'")
        return
    
    if not type(cols) is list:
        raise TypeError("Lists expected for argument 'cols'")
        return
    
    x_cl = df.loc[vals].values
    x_ftrs = df.loc[cols].values
    return x_cl, x_ftrs


def ScaleData(data):
    scaler = RobustScaler()
    data = scaler.fit_transform(data)
    return data, scaler


def CreateTrainData(x, trn_data_len, time_steps):
    
    if not type(trn_data_len) is int:
        raise TypeError("Integer value expected for argument 'trn_data_len'")
        return
    
    if not type(time_steps) is int:
        raise TypeError("Integer value expected for argument 'time_steps'")
        return
    
    trn_data = x[0:trn_data_len, :]
    x_train = []
    y_train = []

    for i in range(time_steps, trn_data_len):
        x_train.append(trn_data[i-time_steps:i,:])
        y_train.append(trn_data[i,-1])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    y_train = np.reshape(y_train, (x_train.shape[0], 1))
    
    return x_train, y_train


def CreateTestData(x_y, x, trn_data_len, time_steps):
    
    if not type(trn_data_len) is int:
        raise TypeError("Integer value expected for argument 'trn_data_len'")
        return
    
    if not type(time_steps) is int:
        raise TypeError("Integer value expected for argument 'time_steps'")
        return
    
    test_data = x[trn_data_len - time_steps:, :]
    x_test = []
    y_test = x_y[trn_data_len:, :]

    for i in range(time_steps, len(test_data)):
        x_test.append(test_data[i-time_steps:i, :])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    
    return x_test, y_test


def PrepareData(df, vals, cols, trn_data_pct=0.8, time_steps=10):
    
    x_val, x_ftrs = SplitFtrs(df, vals, cols)
    
    x_sc, scaler = ScaleData(x_val)
    
    x = np.concatenate((x_ftrs, x_sc), axis=1)
    print("Input data shape before train-test split:", x.shape)
    
    trn_data_len = int(np.ceil(len(x) * trn_data_pct))
    print("Training data length:", trn_data_len)
    tst_data_len = int(len(x) - trn_data_len)
    print("Test data length:", tst_data_len)
    
    x_train, y_train = CreateTrainData(x, trn_data_len, time_steps)
    print("Training input shape:", x_train.shape)
    print("Training output shape:", y_train.shape)
    
    x_test, y_test = CreateTestData(x_val, x, trn_data_len, time_steps)
    print("Test input shape:", x_test.shape)
    print("Test output shape:", y_test.shape)
    
    return x_train, y_train, x_test, y_test, scaler, trn_data_len, tst_data_len


#==========================================================================================================

def BuildModel(unit_i, unit_h, act_fn='relu'):
    
    model = Sequential()
    
    model.add(LSTM(units=unit_i,
                   activation=act_fn,
                   return_sequences=True,
                   input_shape=(x_train.shape[1], x_train.shape[2])))    
    
    model.add(LSTM(units=unit_h,
                   activation=act_fn,
                   return_sequences=True,
                   input_shape=(x_train.shape[1], x_train.shape[2])))    
    
    model.add(LSTM(units=unit_h,
                   activation=act_fn,
                   return_sequences=False))
      
    model.add(Dense(units=10))
        
    model.add(Dense(units=1))
    
    return model
    

def FitModel(model, x_train, y_train, opt='adam', loss='mean_squared_error', btc_size=64, epochs=25):
    
    model.compile(optimizer=opt,
                loss=loss)
    
    model.fit(x_train, 
              y_train, 
              batch_size=btc_size,
              epochs=epochs)
    
    return model


def ModelPrediction(model, x_test, scaler):
    pred = model.predict(x_test)
    pred = scaler.inverse_transform(pred)
    
    return pred


def EvaluateModel(pred, orig):
    rmse = np.sqrt(np.mean((pred - orig) ** 2))
    mape = np.mean((np.abs(pred - orig)/orig)) * 100

    return rmse, mape


def BuildModel2(unit_i, unit_h, act_fn='relu'):
    
    model = Sequential()
    
    model.add(LSTM(units=unit_i,
                   activation=act_fn,
                   return_sequences=True,
                   input_shape=(x_train.shape[1], x_train.shape[2])))
    
    model.add(LSTM(units=unit_i,
                   activation=act_fn,
                   return_sequences=True,
                   input_shape=(x_train.shape[1], x_train.shape[2])))    
    
    model.add(LSTM(units=unit_h,
                   activation=act_fn,
                   return_sequences=True,
                   input_shape=(x_train.shape[1], x_train.shape[2])))    
    
    model.add(LSTM(units=unit_h,
                   activation=act_fn,
                   return_sequences=False))
      
    model.add(Dense(units=10))
        
    model.add(Dense(units=1))
    
    return model


def SentimentScore(sent):
    score = sent_analyzer.polarity_scores(sent)
    return score


#==========================================================================================================
#==========================================================================================================

if __name__ == '__main__':

    n_out = ''
    # The code segment below will fetch tweets from the twitter handles listed in the

    # "news_hdls" python list. 
    # news_twts = []
    # start = '2020-01-01'
    # end = '2020-12-31'

    # Twitter Handles of News Accounts
    # news_hdls = ['CNBCTV18Live',
    #              'ZeeBusiness',
    #              'livemint',
    #              'EconomicTimes',
    #              'FinancialXpress',
    #              'bsindia',
    #              'FinancialTimes',
    #              'BT_India',
    #              'NDTVProfit',
    #              'ReutersIndia',
    #              'moneycontrolcom',
    #              '_groww',
    #              'zerodhaonline']

    # Calling the function to download tweets iteratively
    # for hdl in news_hdls:
    #     news_twts.extend(DownloadTweets(hdl, start, end))
    
    # Creating a dataframe from the tweets list  
    # news_twts_df = pd.DataFrame(news_twts,
    #                             columns=["Date", "ID", "TwtsTxt", "User"])


    twts1_df = pd.read_pickle('twts1.pkl')
    twts2_df = pd.read_pickle('twts2.pkl')
    twts3_df = pd.read_pickle('twts3.pkl')
    twts4_df = pd.read_pickle('twts4.pkl')
    news_twts_df = pd.concat([twts1_df, twts2_df, twts3_df, twts4_df], ignore_index=True)

    # print("No. of tweets in set-1:", len(twts1_df))
    # print("No. of tweets in set-2:", len(twts2_df))
    # print("No. of tweets in set-3:", len(twts3_df))
    # print("No. of tweets in set-4:", len(twts4_df))
    # print("No. of tweets in total dataset:", len(news_twts_df))


    # Below, PrepareStocksData function is being called to fetch and prepare the stock prices data for the
    # selected stocks. The stock prices data for all the stocks will be combined into a single dataframe.

    tickers = ["ADANIPORTS.NS", "APOLLOHOSP.NS", "JSWSTEEL.NS", "RELIANCE.NS", "TCS.NS"]
    start = '2018-01-01'
    end = '2022-12-30'
    stocks_data = []

    tickr = input("Enter stock name:")
    if tickr == '' or tickr is False:
        print("Stock name is mandatory, exiting")

    n_out = int(input("Enter number of days for prediction:"))
    if n_out == '' or n_out is False or n_out == 0:
        n_out = 1
        print("No input provided, predicting stock price for next 1 day")

    # Calling function to fetch and prepare stock prices data
    stocks = PrepareStocksData(tickr,
                               start,
                               end)
    
    stocks['Close_N'].fillna(0, inplace=True)
    

    # Data Preprocessing

    stocks_cl = stocks.filter(['Stock', 'Date', 'Close', 'Close_N', 'Month'])
    stocks_cl.set_index('Date', inplace=True)
    stocks_cl = stocks_cl.pivot(columns='Stock',
                                values=['Close'])

    stocks_cl.columns = [x for x in [col[1] for col in stocks_cl.columns.values]]
    labels = stocks_cl.columns.to_list()

    stocks_cln = stocks.filter(['Stock', 'Date', 'Close', 'Close_N', 'Month'])
    stocks_cln.set_index('Date', inplace=True)
    stocks_cln = stocks_cln.pivot(columns='Stock',
                                  values=['Close_N'])

    stocks_cln.columns = [x for x in [col[1] for col in stocks_cln.columns.values]]


    # Data Cleaning and Preprocessing on Tweets Data
    news_twts_df["ID"] = news_twts_df["ID"].values.astype('object')

    # Data preparation
    st_cl = FilterData(stocks, ['Date', 'Close', 'Close_N', 'Month'])

    # # One-hot encoding on the filtered dataset
    #
    # ohe = OneHotEncoder(drop='first', sparse=False)
    # st_enc = ohe.fit_transform(st_cl[['Stock']])
    # st_enc = pd.DataFrame(st_enc, columns=ohe.get_feature_names_out())

    # Merging the One-hot encoding features with the actual data

    # st_cl_enc = st_cl.merge(st_enc, left_index=True, right_index=True)
    
    x_trn_l = []
    y_trn_l = []
    x_tst_l = []
    y_tst_l = []
    scaler = []

    ohe_cols = ['Date', 'Stock']
    vals = ['Close']
    cols = [x for x in st_cl.columns if x not in vals and x not in ohe_cols]

    x_trn, y_trn, x_tst, y_tst, sc, train_len, test_len = PrepareData(st_cl, vals, cols)

    # Training samples
    trn_len = train_len - 10
    train = stocks_cl[:trn_len].copy()

    # Test samples
    valid = stocks_cl[train_len:].copy()

    # Sentiment Analysis on Twitter Data

    # Determining Sentiment Polarities
    sent_analyzer = SentimentIntensityAnalyzer()
    sent_scores = news_twts_df['TwtsTxt'].apply(SentimentScore)
 
    news_twts_df = news_twts_df.assign(Neg = [x['neg'] for x in sent_scores],
                                       Neu = [x['neu'] for x in sent_scores],
                                       Pos = [x['pos'] for x in sent_scores],
                                       Comp = [x['compound'] for x in sent_scores])

    news_twts_df['Sent'] = news_twts_df['Comp'].apply(lambda x: -1 if x < -0.05 else 1 if x > 0.05 else 0)

    # Sentiment Aggregation
    twt_sent_day = news_twts_df.groupby(news_twts_df['Date'].dt.date).mean()
    twt_sent_day.drop(columns=['Sent'], inplace=True)

    # Feature Engineering
    news_twt_sents = pd.DataFrame(news_twts_df.groupby(news_twts_df['Date'].dt.date)['Comp'].mean())
    news_twt_sents['Comp_Min'] = news_twts_df.groupby(news_twts_df['Date'].dt.date)['Comp'].min()
    news_twt_sents['Comp_Max'] = news_twts_df.groupby(news_twts_df['Date'].dt.date)['Comp'].max()

    news_twt_sents['Neg_Cnt_N'] = news_twts_df.loc[news_twts_df['Comp'] < -0.05].\
                                  groupby(news_twts_df['Date'].dt.date)['Comp'].count()/\
                                  news_twts_df.groupby(news_twts_df['Date'].dt.date)['Comp'].count()

    news_twt_sents['Neu_Cnt_N'] = news_twts_df.loc[(news_twts_df['Comp'] >= -0.05) & (news_twts_df['Comp'] <= 0.05)].\
                                  groupby(news_twts_df['Date'].dt.date)['Comp'].count()/\
                                  news_twts_df.groupby(news_twts_df['Date'].dt.date)['Comp'].count()

    news_twt_sents['Pos_Cnt_N'] = news_twts_df.loc[news_twts_df['Comp'] > 0.05].\
                                  groupby(news_twts_df['Date'].dt.date)['Comp'].count()/\
                                  news_twts_df.groupby(news_twts_df['Date'].dt.date)['Comp'].count()

    news_twt_sents['Comp_3'] = news_twt_sents['Comp'].ewm(span=3,
                                              adjust=True).mean()

    news_twt_sents['Comp_7'] = news_twt_sents['Comp'].ewm(span=7,
                                              adjust=True).mean()

    news_twt_sents['Comp_10'] = news_twt_sents['Comp'].ewm(span=10,
                                              adjust=True).mean()

    news_twt_sents.reset_index(inplace=True)
    news_twt_sents['Date'] = pd.to_datetime(news_twt_sents['Date'], format="%Y-%m-%d")

    stocks_sent = stocks.merge(news_twt_sents,
                               left_on='Date',
                               right_on='Date')

    # The combined dataset has all the features from the two individual datasets -
    # the historical stock prices dataset and the sentiment scores dataset.

    # Data Preparation

    all_cols = stocks_sent.columns.to_list()
    cols_exc = ['Stock', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
    cols_inc = [x for x in all_cols if x not in cols_exc]

    st_cl_sent = FilterData(stocks_sent, cols_inc)

    ohe_cols = ['Date', 'Stock']
    vals = ['Close']
    cols = [x for x in st_cl_sent.columns if x not in vals and x not in ohe_cols]

    x_trn, y_trn, x_tst, y_tst, sc, train_len, test_len = PrepareData(st_cl_sent, vals, cols)

    # Model Building and Price Predictions
    # Commented as model will be loaded from saved pickle file
    # model_f = BuildModel(10, 5, 'relu')
    # model_f = FitModel(model_f,
    #                  x_train,
    #                  y_train,
    #                  opt='adam',
    #                  loss='mean_squared_error',
    #                  btc_size=32,
    #                  epochs=25)

    # Saving the trained model to pickle file for future reuse
    # pickle.dump(model_f, open("final_model.pkl", 'wb'))

    # Loading the pre-trained model from saved pickle file
    model_f = pickle.load(open('final_model.pkl', 'rb'))

    # Predicting Stock Prices using Sentiment Scores
    y_pred = ModelPrediction(model_f,
                             x_tst,
                             sc)

    rmse, mape = EvaluateModel(y_pred, y_tst)
    print("\nStock:", tickr)
    print("RMSE =", rmse)
    print("MAPE =", mape)

    # (fig, ax) = plt.subplots(figsize=(12,8))
    # ax.set_title("Previous 10 days' stock price & Prediction for next day")
