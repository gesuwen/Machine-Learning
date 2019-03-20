# Eigen-portfolio construction using Principal Component Analysis (PCA)

import os
import os.path
import numpy as np
import datetime

import matplotlib.pyplot as plt
import pandas as pd


def sharpe_ratio(ts_returns, periods_per_year=252):
    """
    sharpe_ratio - Calculates annualized return, annualized vol, and annualized sharpe ratio,
                    where sharpe ratio is defined as annualized return divided by annualized volatility

    Arguments:
    ts_returns - pd.Series of returns of a single eigen portfolio

    Return:
    a tuple of three doubles: annualized return, volatility, and sharpe ratio
    """

    annualized_return = 0.
    annualized_vol = 0.
    annualized_sharpe = 0.

    n_years = ts_returns.shape[0] / periods_per_year
    annualized_return = np.power(np.prod(1 + ts_returns), (1 / n_years)) - 1
    annualized_vol = ts_returns.std() * np.sqrt(periods_per_year)
    annualized_sharpe = annualized_return / annualized_vol

    return annualized_return, annualized_vol, annualized_sharpe


if __name__ == '__main__':

    # load dataset
    asset_prices = pd.read_csv('spx_holdings_and_spx_closeprice.csv',
                         date_parser=lambda dt: pd.to_datetime(dt, format='%Y-%m-%d'),
                         index_col = 0).dropna()
    n_stocks_show = 12
    print('Asset prices shape: ', asset_prices.shape)
    print("\n Top 5 rows of the first n_stocks_show stocks table: ")
    print(asset_prices.iloc[:, :n_stocks_show].head())

    print('\n Last 10 columns contains SPX index prices:')
    print(asset_prices.iloc[:, -10:].head())

    # Asset Returns Calculation
    asset_returns = pd.DataFrame(data=np.zeros(shape=(len(asset_prices.index), asset_prices.shape[1])),
                                 columns=asset_prices.columns.values,
                                 index=asset_prices.index)
    normed_returns = asset_returns
    # Calculate percent returns, also known as simple returns using asset_prices.
    asset_returns = asset_prices.pct_change().dropna()
    # Calculate de-meaned returns and scale them by standard deviation 𝜎
    normed_returns = (asset_returns - asset_returns.mean()) / asset_returns.std()
    print('\nDe-meaned returns of last 10 columns: ')
    print(normed_returns.iloc[-5:, -10:].head())

    # Train/test data split
    train_end = datetime.datetime(2012, 3, 26)

    df_train = None
    df_test = None
    df_raw_train = None
    df_raw_test = None

    df_train = normed_returns[normed_returns.index <= train_end].copy()
    df_test = normed_returns[normed_returns.index > train_end].copy()

    df_raw_train = asset_returns[asset_returns.index <= train_end].copy()
    df_raw_test = asset_returns[asset_returns.index > train_end].copy()

    print('\nTrain dataset size:', df_train.shape)
    print('Test dataset size:', df_test.shape)

    # PCA fitting
    from sklearn.decomposition import PCA
    import seaborn as sns

    stock_tickers = normed_returns.columns.values[:-1] # exclude SPX index
    assert 'SPX' not in stock_tickers, "By accident included SPX index"

    n_tickers = len(stock_tickers)
    pca = None
    cov_matrix = pd.DataFrame(data=np.ones(shape=(n_tickers, n_tickers)), columns=stock_tickers)
    cov_matrix_raw = cov_matrix

    if df_train is not None and df_raw_train is not None:
        stock_tickers = asset_returns.columns.values[:-1] # exclude SPX index
        assert 'SPX' not in stock_tickers, "By accident included SPX index"
        # Calculate covariance matrix using training data set
        cov_matrix = df_train.loc[:, df_train.columns != "SPX"].cov()
        # computing PCA on S&P 500 stocks
        pca = PCA().fit(cov_matrix)
        # not normed covariance matrix
        cov_matrix_raw = df_raw_train.loc[:, df_raw_train.columns != 'SPX'].cov()

        cov_raw_df = pd.DataFrame({'Variance': np.diag(cov_matrix_raw)}, index=stock_tickers)
        # cumulative variance explained
        var_threshold = 0.8
        var_explained = np.cumsum(pca.explained_variance_ratio_)
        num_comp = np.where(np.logical_not(var_explained < var_threshold))[0][0] + 1  # +1 due to zero based-arrays
        print('%d components explain %.2f%% of variance' % (num_comp, 100 * var_threshold))

    if pca is not None:
        bar_width = 0.9
        n_asset = int((1 / 10) * normed_returns.shape[1])
        x_indx = np.arange(n_asset)
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 4)
        # Eigenvalues are measured as percentage of explained variance.
        rects = ax.bar(x_indx, pca.explained_variance_ratio_[:n_asset], bar_width, color='deepskyblue')
        ax.set_xticks(x_indx + bar_width / 2)
        ax.set_xticklabels(list(range(n_asset)), rotation=45)
        ax.set_title('Percent variance explained')
        ax.legend((rects[0],), ('Percent variance explained by principal components',))

    if pca is not None:
        projected = pca.fit_transform(cov_matrix)


    # Eigen-portfolios construction https://math.nyu.edu/faculty/avellane/AvellanedaLeeStatArb20090616.pdf
    # the first two eigen-portfolio weights# the fi
    pc_w = np.zeros(len(stock_tickers))
    eigen_prtf1 = pd.DataFrame(data={'weights': pc_w.squeeze() * 100}, index=stock_tickers)
    if pca is not None:
        pcs = pca.components_

        # normalized portfilio sum to 1
        pc_w = pcs[:, 0] / sum(pcs[:, 0])

        # first eigen portfolio
        eigen_prtf1 = pd.DataFrame(data={'weights': pc_w.squeeze() * 100}, index=stock_tickers)
        eigen_prtf1.sort_values(by=['weights'], ascending=False, inplace=True)
        print('Sum of weights of first eigen-portfolio: %.2f' % np.sum(eigen_prtf1))
        eigen_prtf1.plot(title='First eigen-portfolio weights',
                         figsize=(12, 6),
                         xticks=range(0, len(stock_tickers), 10),
                         rot=45,
                         linewidth=3)

    pc_w = np.zeros(len(stock_tickers))
    eigen_prtf2 = pd.DataFrame(data={'weights': pc_w.squeeze() * 100}, index=stock_tickers)

    if pca is not None:
        pcs = pca.components_

        # normalized portfilio sum to 1
        pc_w = pcs[:, 1] / sum(pcs[:, 1])

        # second eigen portfolio
        eigen_prtf2 = pd.DataFrame(data={'weights': pc_w.squeeze() * 100}, index=stock_tickers)
        eigen_prtf2.sort_values(by=['weights'], ascending=False, inplace=True)
        print('Sum of weights of second eigen-portfolio: %.2f' % np.sum(eigen_prtf2))
        eigen_prtf2.plot(title='Second eigen-portfolio weights',
                         figsize=(12, 6),
                         xticks=range(0, len(stock_tickers), 10),
                         rot=45,
                         linewidth=3)

    # Compute performance of several eigen portfolios
    # Compute the annualized return, volatility, and Sharpe ratio of the first two eigen portfolios
    if df_raw_test is not None:
        eigen_prtf1_returns = np.dot(df_raw_test.loc[:, eigen_prtf1.index], eigen_prtf1 / 100)
        eigen_prtf1_returns = pd.Series(eigen_prtf1_returns.squeeze(), index=df_test.index)
        er, vol, sharpe = sharpe_ratio(eigen_prtf1_returns)
        print('First eigen-portfolio:\nReturn = %.2f%%\nVolatility = %.2f%%\nSharpe = %.2f' % (er*100, vol*100, sharpe))
        year_frac = (eigen_prtf1_returns.index[-1] - eigen_prtf1_returns.index[0]).days / 252

        df_plot = pd.DataFrame({'PC1': eigen_prtf1_returns, 'SPX': df_raw_test.loc[:, 'SPX']}, index=df_test.index)
        np.cumprod(df_plot + 1).plot(title='Returns of the market-cap weighted index vs. First eigen-portfolio',
                                 figsize=(12,6), linewidth=3)

    if df_raw_test is not None:
        eigen_prtf2_returns = np.dot(df_raw_test.loc[:, eigen_prtf2.index], eigen_prtf2 / 100)
        eigen_prtf2_returns = pd.Series(eigen_prtf2_returns.squeeze(), index=df_test.index)
        er, vol, sharpe = sharpe_ratio(eigen_prtf2_returns)
        print('Second eigen-portfolio:\nReturn = %.2f%%\nVolatility = %.2f%%\nSharpe = %.2f' % (er*100, vol*100, sharpe))

    # computing Sharpe ratio for the first 120 portfolios and select portfolio with the highest postive Sharpe ratio
    n_portfolios = 120
    annualized_ret = np.array([0.] * n_portfolios)
    sharpe_metric = np.array([0.] * n_portfolios)
    annualized_vol = np.array([0.] * n_portfolios)
    idx_highest_sharpe = 0  # index into sharpe_metric which identifies a portfolio with rhe highest Sharpe ratio

    if pca is not None:
        for ix in range(n_portfolios):
            pc_w = pcs[:, ix] / sum(pcs[:, ix])
            eigen_prtfix = pd.DataFrame(data={'weights': pc_w.squeeze() * 100}, index=stock_tickers)
            eigen_prtfix.sort_values(by=['weights'], ascending=False, inplace=True)

            eigen_prtix_returns = np.dot(df_raw_test.loc[:, eigen_prtfix.index], eigen_prtfix / 100)
            eigen_prtix_returns = pd.Series(eigen_prtix_returns.squeeze(), index=df_test.index)
            er, vol, sharpe = sharpe_ratio(eigen_prtix_returns)
            annualized_ret[ix] = er
            annualized_vol[ix] = vol
            sharpe_metric[ix] = sharpe

        # find portfolio with the highest Sharpe ratio
        idx_highest_sharpe = np.nanargmax(sharpe_metric)

        print('Eigen portfolio #%d with the highest Sharpe. Return %.2f%%, vol = %.2f%%, Sharpe = %.2f' %
              (idx_highest_sharpe,
               annualized_ret[idx_highest_sharpe] * 100,
               annualized_vol[idx_highest_sharpe] * 100,
               sharpe_metric[idx_highest_sharpe]))

        fig, ax = plt.subplots()
        fig.set_size_inches(12, 4)
        ax.plot(sharpe_metric, linewidth=3)
        ax.set_title('Sharpe ratio of eigen-portfolios')
        ax.set_ylabel('Sharpe ratio')
        ax.set_xlabel('Portfolios')

    results = pd.DataFrame(data={'Return': annualized_ret, 'Vol': annualized_vol, 'Sharpe': sharpe_metric})
    results.sort_values(by=['Sharpe'], ascending=False, inplace=True)
    print("\n Eigen portfolio with top 10 sharpe ratio: ")
    print(results.head(10))
