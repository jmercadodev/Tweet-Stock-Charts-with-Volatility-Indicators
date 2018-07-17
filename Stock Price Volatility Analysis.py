import pandas_datareader.data as data
import matplotlib.pyplot as plt
from pandas import Timedelta
from matplotlib import style
from datetime import date
from keys import api
import pandas as pd
import numpy as np
import datetime
import os

today = datetime.date.today()
def tweet_chart_image(stock):
    photo = open('images/{}.png'.format(stock), 'rb')
    response = api.upload_media(media=photo)
    api.update_status(status='$' + str.upper(stock) + ' ' + cat + ' Chart ' + str(today), media_ids=[response['media_id']])

def get_quotes(y,m,d, slow, fast, stock):
    start = datetime.datetime(y, m, d)
    end = datetime.datetime.now()

    df = data.DataReader(stock, "morningstar", start, end)
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    df = df.drop("Symbol", axis=1)

    Close = df['Close']
    df['Slow_Ma'] = Close.rolling(window=slow).mean()
    df['Fast_Ma'] = Close.rolling(window=fast).mean()
    Slow_Ma = df['Slow_Ma']
    Fast_Ma = df['Fast_Ma']

    slow_diff = Close - Slow_Ma
    df['slow_vlty'] = round((slow_diff / Close) *100, 2)
    slow_vlty = df['slow_vlty']

    fast_diff = Close - Fast_Ma
    df['fast_vlty'] = round((fast_diff / Close) *100, 2)
    fast_vlty = df['fast_vlty']

    df['close_slow_std'] = Close.rolling(window=slow).std()
    df['close_fast_std'] = Close.rolling(window=fast).std()
    df['fast_ma_std'] = Fast_Ma.rolling(window=fast).std()

    df['slow_ma_std'] = Slow_Ma.rolling(window=slow).std()
    slow_ma_std = df['slow_ma_std']
    df['lower_std'] = Close - slow_ma_std
    df['upper_std'] = Close + slow_ma_std
    lower_std = df['lower_std']
    upper_std = df['upper_std']
  
    df['lower_vlty_std'] = -slow_vlty - slow_ma_std
    df['upper_vlty_std'] = slow_vlty + slow_ma_std
    lower_vlty_std = df['lower_vlty_std']
    upper_vlty_std = df['upper_vlty_std']

    slow_trades = slow_vlty.apply(np.sign)
    slow_run = round(float(len(Slow_Ma.index)), 2)
    slow_rise = round(int(slow_trades.sum()), 2)
    slow_trend = round((slow_rise / slow_run), 2)

    fast_trades = fast_vlty.apply(np.sign)
    fast_run = round(float(len(Fast_Ma.index)), 2)
    fast_rise = round(int(fast_trades.sum()), 2)
    fast_trend = round((fast_rise / fast_run), 2)

    global cat
    if slow_trend >= 0.2 and fast_trend >= 0.2:
        cat = 'Long Trending'
    elif slow_trend <= -0.2 and fast_trend <= -0.2:
        cat = 'Short Trending'
    else:
        cat = 'Mean Reverting'

    slow_trades = slow_trades.shift(1) 
    fast_trades = fast_trades.shift(1) 

    style.use('fivethirtyeight')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))

    ax1 = plt.subplot2grid((20,1),(0,0), rowspan=12, colspan=1)
    ax2 = plt.subplot2grid((20,1),(13,0), rowspan=7, colspan=1)

    ax1.plot(df[['Close', 'Fast_Ma', 'Slow_Ma']], lw = 3)

    ax2.plot(df[['slow_vlty']], lw = 0)
    ax2.plot(df[['fast_vlty']], lw = 0)

    df.dropna(inplace=True)

    box1 = dict(boxstyle = 'round4', fc = '#f2f2f2', ec = '#1992f8', lw = 2)
    ax1.annotate('{}'.format(Close[-1]),
        (df.index[-1], Close[-1]),
        xytext = (df.index[-1] + pd.Timedelta(days=8), Close[-1]) , bbox = box1)

    box2 = dict(boxstyle = 'round4', fc = '#f2f2f2', ec = 'deepskyblue', lw = 2)
    ax2.annotate('{}'.format(slow_vlty[-1]),
        (df.index[-1], slow_vlty[-1]),
        xytext = (df.index[-1] + pd.Timedelta(days=5) , slow_vlty[-1]+4), bbox = box2)

    box3 = dict(boxstyle = 'round4', fc = '#f2f2f2', ec = 'darkorchid', lw = 2)
    ax2.annotate('{}'.format(fast_vlty[-1]),
        (df.index[-1], fast_vlty[-1]),
        xytext = (df.index[-1] + pd.Timedelta(days=5), fast_vlty[-1] -4), bbox = box3)  

    ax2.fill_between(slow_vlty.index, 0, slow_vlty, where=(slow_vlty) > 0,  facecolor='deepskyblue', interpolate=True)
    ax2.fill_between(slow_vlty.index, 0, slow_vlty, where=(slow_vlty) < 0, facecolor='deeppink', interpolate=True)

    ax2.fill_between(fast_vlty.index, 0, fast_vlty, where=(fast_vlty) > 0,  facecolor='darkorchid', interpolate=True)
    ax2.fill_between(fast_vlty.index, 0, fast_vlty, where=(fast_vlty) < 0, facecolor='orange', interpolate=True)

    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax1.legend(('Close', '{}D MA'.format(fast), '{}D MA' .format(slow))) # dont know why but legends need 2 sets of parenthesis
    ax2.legend(('{} Fast Trend' .format(fast_trend), '{} Slow Trend'.format(slow_trend)), loc = 3) # dont know why but legends need 2 sets of parenthesis
    ax1.set_title('{} {} {}' .format(str.upper(stock), cat, today))

    ax1.set_ylabel('Price')

    ax2.set_ylabel('Percent Volatility')
    ax2.set_xlabel('Date')

    plt.subplots_adjust(left=.05, bottom=.1, right=.90, top=.90)
    plt.savefig('images/{}.png'.format(stock))

    plt.show()

    # plt.close()
    # tweet_chart_image(stock)
    # os.remove('images/{}.png'.format(stock)) 


stocks = ['GPRO', 'AMD', 'MU', 'TSLA', 'SNAP',
        'GE', 'OAS', 'NVDA', 'BOX', 'GPS', 'X',
        'TWTR', 'PBR', 'AAPL', 'BOX', 'CRON', 'JCP',
        'PYPL', 'JPM', 'QCOM', 'MA', 'XOM', 'V', 'NFLX', 
        'VRX', 'TWLO', 'SQ', 'ETSY', 'MSFT', 'AMZN', 'FB', 
        'CGC', 'USO']

for each_stock in stocks:
    get_quotes(2016, 1, 1, 90, 10, each_stock)
