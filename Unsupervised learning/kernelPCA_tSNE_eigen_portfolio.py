# Data visualization with t-SNE

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns


def check_for_nulls(df):
    """
    Test and report number of NAs in each column of the input data frame
    :param df: pandas.DataFrame
    :return: None
    """
    for col in df.columns.values:
        num_nans = np.sum(df[col].isnull())
        if num_nans > 0:
            print('%d Nans in col %s' % (num_nans, col))
    return "No null values. "


def plot_tsne_2D(df_tsne, label_column, plot_title):
    """
    plot_tsne_2D - plots t-SNE as two-dimensional graph
    Arguments:
    label_column - column name where labels data is stored
    df_tsne - pandas.DataFrame with columns x-tsne, y-tsne
    plot_title - string
    """
    unique_labels = df_tsne[label_column].unique()
    print('Data labels:', unique_labels)
    print(df_tsne.shape)

    colors = [ 'b', 'g','r']
    markers = ['s', 'x', 'o']
    y_train = df_tsne.regime.values

    plt.figure(figsize=(8, 8))
    ix = 0
    bars = [None] * len(unique_labels)
    for label, c, m in zip(unique_labels, colors, markers):
        plt.scatter(df_tsne.loc[df_tsne[label_column]==label, 'x-tsne'],
                    df_tsne.loc[df_tsne[label_column]==label, 'y-tsne'],
                    c=c, label=label, marker=m, s=15)
        bars[ix] = plt.bar([0, 1, 2], [0.2, 0.3, 0.1], width=0.4, align="center", color=c)
        ix += 1

    plt.legend(bars, unique_labels)
    plt.xlabel('first dimension')
    plt.ylabel('second dimension')
    plt.title(plot_title)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # load dataset
    asset_prices = pd.read_csv('spx_holdings_and_spx_closeprice_tSNE.csv',
                         date_parser=lambda dt: pd.to_datetime(dt, format='%Y-%m-%d'),
                         index_col = 0).dropna()
    n_stocks_show = 12
    print('Asset prices shape of the first 12 stocks: ', asset_prices.shape)
    print(asset_prices.iloc[:, :n_stocks_show].head())

    print('\nLast 10 columns contains SPX index prices:')
    print(asset_prices.iloc[:, -10:].head())

    print("\nCheck if nulls exist: ")
    print(check_for_nulls(asset_prices))

    # Calculate price log-returns
    asset_returns = np.log(asset_prices) - np.log(asset_prices.shift(1))
    asset_returns = asset_returns.iloc[1:, :]
    print("Asset prices shape of the first 12 stocks: ")
    print(asset_returns.iloc[:, :n_stocks_show].head())

    # Calculate 20 and 100-day moving average of SPX Index price based on spx_index
    # Get the SPX time series. This now returns a Pandas Series object indexed by date.
    spx_index = asset_prices.loc[:, 'SPX']

    short_rolling_spx = pd.core.series.Series(np.zeros(len(asset_prices.index)), index=asset_prices.index)
    long_rolling_spx = short_rolling_spx

    # 20 days moving averages of log-returns
    short_rolling_spx = spx_index.rolling(window=20).mean()
    # 100 days moving averages of log-returns
    long_rolling_spx = spx_index.rolling(window=100).mean()

    # Plot the index and rolling averages
    fig=plt.figure(figsize=(12, 5), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(spx_index.index, spx_index, label='SPX Index')
    ax.plot(short_rolling_spx.index, short_rolling_spx, label='20 days rolling')
    ax.plot(long_rolling_spx.index, long_rolling_spx, label='100 days rolling')
    ax.set_xlabel('Date')
    ax.set_ylabel('Log returns')
    ax.legend(loc=2)
    plt.show()

    # Apply scikit-learn StandardScaler to stocks log-returns

    # Standardize features by removing the mean and scaling to unit variance
    # Centering and scaling happen independently on each feature by computing the relevant statistics
    # on the samples in the training set. Mean and standard deviation are then stored to be used on later
    # data using the transform method.
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

    log_ret_mat_std = StandardScaler().fit_transform(asset_returns.values)
    log_ret_df_std = pd.DataFrame(data=log_ret_mat_std,
                                  index=asset_returns.index,
                                  columns=asset_returns.columns.values)
    print("\nStandardize asset returns to zero mean unit variance")
    print(log_ret_df_std.iloc[:, :10].head())

    # Calculate the 20 and 100 days moving averages of the log-returns of standarized return
    short_rolling_spx = log_ret_df_std[['SPX']].rolling(window=20).mean()
    long_rolling_spx = log_ret_df_std[['SPX']].rolling(window=100).mean()

    # Plot the index and rolling averages
    fig=plt.figure(figsize=(12, 5), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    ax.plot(log_ret_df_std.index, log_ret_df_std[['SPX']], label='SPX Index')
    ax.plot(short_rolling_spx.index, short_rolling_spx, label='20 days rolling')
    ax.plot(long_rolling_spx.index, long_rolling_spx, label='100 days rolling')
    ax.set_xlabel('Date')
    ax.set_ylabel('Log returns')
    ax.legend(loc=2)
    plt.show()

    # Assign a label 'regime' to each date:
    # 'regime' = 'benign' for all points except two intervals
    # 'regime' = 'crisis_2001_2002', or
    # 'regime = ', 'crisis_2007-2009'

    # first assign the default value for all rows
    log_ret_df_std['regime'] = 'benign'
    dt_start = np.datetime64('2000-03-24T00:00:00.000000000')
    dt_end = np.datetime64('2002-10-09T00:00:00.000000000')
    flag_crisis_2001_2002 = np.logical_and(log_ret_df_std.index > dt_start, log_ret_df_std.index < dt_end)

    dt_start = np.datetime64('2007-10-09T00:00:00.000000000')
    dt_end = np.datetime64('2009-03-09T00:00:00.000000000')
    flag_crisis_2007_2009 = np.logical_and(log_ret_df_std.index > dt_start, log_ret_df_std.index < dt_end)

    log_ret_df_std.loc[flag_crisis_2001_2002,'regime'] = 'crisis_2001_2002'
    log_ret_df_std.loc[flag_crisis_2007_2009, 'regime'] = 'crisis_2007_2009'

    print('crisis_2001_2002 days: ', log_ret_df_std[log_ret_df_std.regime == 'crisis_2001_2002'].shape[0])
    print('crisis_2007_2009 days: ', log_ret_df_std[log_ret_df_std.regime == 'crisis_2007_2009'].shape[0])

    # Train/test  data split
    # use data before 2012-03-26 for training, and data after it for testing
    train_end = datetime.datetime(2012, 3, 26)
    df_train = log_ret_df_std[log_ret_df_std.index <= train_end].copy()
    df_test = log_ret_df_std[log_ret_df_std.index > train_end].copy()
    print('Train dataset:', df_train.shape)
    print('Test dataset:', df_test.shape)

    # Returns regression on SPX Index
    # regress each individual stock on the market

    # create a Linear Regression object
    lm = LinearRegression()
    stock_tickers = asset_returns.columns.values[:-1]  # exclude SPX

    # compute betas for all stocks in the dataset
    R2_in_sample = [0.] * len(stock_tickers)
    R2_out_sample = [0.] * len(stock_tickers)
    betas = [0.] * len(stock_tickers)
    alphas = [0.] * len(stock_tickers)

    stocks = df_train.columns.values[:-2]
    for i, stock in enumerate(stocks):
        lm.fit(df_train['SPX'].values.reshape((3055, 1)), df_train[stock].values.reshape((3055, 1)))
        betas[i] = lm.coef_[0][0]
        alphas[i] = lm.intercept_[0]

        df_pred = lm.predict(df_train['SPX'].values.reshape((3055, 1)))
        R2_in_sample[i] = r2_score(df_train[stock], df_pred)

        df_pred = lm.predict(df_test['SPX'].values.reshape((437, 1)))
        R2_out_sample[i] = r2_score(df_test[stock].values.reshape((437, 1)), df_pred)

    df_lr = pd.DataFrame({'R2 in-sample': R2_in_sample, 'R2 out-sample': R2_out_sample, 'Alpha': alphas, 'Beta': betas},
                         index=stock_tickers)
    print("\nR2 in-sample/out-sample, Alpha and Beta for the first 10 stocks: ")
    print(df_lr.head(10))

    # Calculate unexplained log-returns as difference between the stock return and its value, "predicted" by the index
    # return (by alphas and betas).
    df_unexplained = df_train.loc[:, stock_tickers]

    df_ = pd.DataFrame(index=df_train.index, columns=stocks)
    df_ = df_.fillna(0)
    df_pred = (df_.T + df_train['SPX'].values).T
    df_unexplained = df_train[stocks] - df_lr['Alpha'] - df_lr['Beta'] * df_pred

    print('Unexplained log-returns of S&P 500 Index stocks', df_unexplained.shape)
    print('Unexplained log-returns of S&P 500 Index stocks of the first 10 stocks:')
    print(df_unexplained.iloc[:, :10].head())

    # Kernel PCA of Covariance Matrix of Returns
    sns.pairplot(df_train.loc[:, ['SPX', 'GE', 'AAPL', 'MSFT', 'regime']],
                 vars=['SPX', 'GE', 'AAPL', 'MSFT'], hue="regime", size=4.5)

    stock_tickers = asset_returns.columns.values[:-1]
    assert 'SPX' not in stock_tickers, "By accident included SPX index"
    data = df_test[stock_tickers].values
    # Perform Kernel PCA with 1 component using returns data df_test for all stocks in df_test
    df_index_test = pd.DataFrame(data=df_test['SPX'].values, index=df_test.index, columns=['SPX'])
    df_index_test['PCA_1'] = np.ones(len(df_test.index))

    # Transform original mapping in the coordinates of the first principal component
    pca = KernelPCA(n_components=1, random_state=42)
    df_index_test['PCA_1'] = pca.fit_transform(df_test[stock_tickers])

    # draw the two plots
    df_plot = df_index_test[['SPX', 'PCA_1']].apply(lambda x: (x - x.mean()) / x.std())
    df_plot.plot(figsize=(12, 6), title='Index replication via PCA')

    # Visualization with t-SNE
    np.random.seed(42)
    tsne_results = np.zeros((log_ret_df_std[stock_tickers].shape[0], 2))
    perplexity = 50
    n_iter = 300
    time_start = time.time()

    # Fit TSNE with 2 components, 300 iterations. Set perplexity to 50.
    tsne = TSNE(n_components=2, n_iter=300, perplexity=50, random_state=42)
    tsne.fit(log_ret_df_std[stock_tickers])
    tsne_results = tsne.fit_transform(log_ret_df_std[stock_tickers])

    df_tsne = pd.DataFrame({'regime': log_ret_df_std.regime.values,
                            'x-tsne': tsne_results[:,0],
                            'y-tsne': tsne_results[:,1]},
                           index=log_ret_df_std.index)
    print('t-SNE (perplexity=%.0f) data of first 10 timestamps:' % perplexity)
    print(df_tsne.head(10))

    plot_tsne_2D(df_tsne, 'regime', 'S&P 500 dimensionality reduction with t-SNE (perplexity=%d)' % perplexity)
