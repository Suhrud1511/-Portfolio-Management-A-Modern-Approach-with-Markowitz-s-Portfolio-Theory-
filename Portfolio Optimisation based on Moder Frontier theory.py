import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def fetch_stock_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date)
    return data['Adj Close']

def calculate_portfolio_performance(weights, returns):
    portfolio_return = np.sum(weights * returns.mean()) * 252
    portfolio_volatility = np.sqrt(weights @ (returns.cov() * 252) @ weights)
    return portfolio_return, portfolio_volatility

def objective_function(weights, returns):
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, returns)
    return -portfolio_return / portfolio_volatility  # Negative return for maximization

def optimize_portfolio(returns, weights=None):
    """
    Optimize portfolio allocation for given returns and optional weights.

    Parameters:
        returns (pd.DataFrame): Returns of assets.
        weights (list, optional): Initial guess for portfolio weights. Defaults to None.

    Returns:
        float: Optimal portfolio return.
        float: Optimal portfolio volatility (standard deviation).
        float: Optimal Sharpe ratio.
        list: Optimal portfolio weights.
    """
    num_assets = len(returns.columns)
    
    if weights is None:
        weights = np.ones(num_assets) / num_assets  # Initialize with equal weights
    
    def objective(weights):
        portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized return
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized volatility
        sharpe_ratio = portfolio_return / portfolio_volatility
        return -sharpe_ratio  # Minimize the negative Sharpe ratio for maximum Sharpe ratio
    
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Constraint: weights sum to 1
    bounds = tuple((0, 1) for asset in range(num_assets))  # Bounds for asset weights (between 0 and 1)

    result = minimize(objective, weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_return = -result.fun  # Maximum Sharpe ratio
    optimal_volatility = np.sqrt(np.dot(result.x.T, np.dot(returns.cov() * 252, result.x)))  # Annualized volatility
    optimal_weights = result.x
    
    return optimal_return, optimal_volatility, result.fun, optimal_weights

def plot_efficient_frontier(returns, num_portfolios):
    results = simulate_random_portfolios(returns, num_portfolios)
    plt.figure(figsize=(12, 6))
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.colorbar(label='Sharpe Ratio')
    optimal_return, optimal_volatility, _, _ = optimize_portfolio(returns)
    plt.scatter(optimal_volatility, optimal_return, c='red', marker='*', s=100)
    plt.show()

def simulate_random_portfolios(returns, num_portfolios):
    results = np.zeros((3, num_portfolios))
    risk_free_rate = 0.03
    for i in range(num_portfolios):
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, returns)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = sharpe_ratio
    return results


def plot_std(STD):
    # Your code for plotting STD goes here
    # For example:
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_title(r"Standard Deviation ($) of all instruments for all years")
    ax.set_facecolor((0.95, 0.95, 0.99))
    ax.grid(c=(0.75, 0.75, 0.99))
    ax.set_ylabel(r"Standard Deviation ($)")
    ax.set_xlabel(r"Years")
    STD.plot(ax=plt.gca(), grid=True)

    for instr in STD:
        stds = STD[instr]
        years = list(STD.index)
        for year, std in zip(years, stds):
            label = "%.3f" % std
            plt.annotate(
                label,
                xy=(year, std),
                xytext=((-1) * 50, 40),
                textcoords='offset points',
                ha='right',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

    plt.show()



def plot_return_vs_risk(STD_avg, APY_avg):
    # Your code for plotting Return vs. Risk goes here
    # For example:
    c = [y + x for y, x in zip(APY_avg, STD_avg)]
    c = list(map(lambda x: x / max(c), c))
    s = list(map(lambda x: x * 600, c))

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_title(r"Risk ($) vs Return (%) of all instruments")
    ax.set_facecolor((0.95, 0.95, 0.99))
    ax.grid(c=(0.75, 0.75, 0.99))
    ax.set_xlabel(r"Standard Deviation ($)")
    ax.set_ylabel(r"Annualized Percentage Yield (%) or Return ($)")
    ax.scatter(STD_avg, APY_avg, s=s, c=c, cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)
    ax.axhline(y=0.0, xmin=0, xmax=5, c="blue", linewidth=1.5, zorder=0, linestyle='dashed')
    ax.axvline(x=0.0, ymin=0, ymax=40, c="blue", linewidth=1.5, zorder=0, linestyle='dashed')

    plt.show()

def calculate_apy_avg(returns):
    # Calculate APY for each year
    apy_avg = returns.groupby([returns.index.year]).apply(lambda x: (1 + x).prod() - 1)

    # Calculate the average APY
    apy_avg = apy_avg.mean()

    return apy_avg

def print_optimal_portfolio(optimal_return, optimal_volatility, symbols, optimal_weights):
    print("Optimal Portfolio Return: {:.2%}".format(optimal_return))
    print("Optimal Portfolio Volatility: {:.2%}".format(optimal_volatility))
    print("Optimal Portfolio Weights:")
    for symbol, weight in zip(symbols, optimal_weights):
        print("{}: {:.2%}".format(symbol, weight))


def generate_random_weights(num_assets):
    """
    Generate random portfolio weights.
    
    Parameters:
        num_assets (int): Number of assets in the portfolio.
    
    Returns:
        numpy.ndarray: Randomly generated portfolio weights.
    """
    weights = np.random.rand(num_assets)
    weights /= weights.sum()  # Normalize to ensure they sum to 1
    return weights

def calculate_apy_avg(returns):
    # Calculate APY for each year
    apy_avg = returns.groupby([returns.index.year]).apply(lambda x: (1 + x).prod() - 1)

    # Calculate the average APY
    apy_avg = apy_avg.mean()

    return apy_avg

def compare_portfolios(returns, num_portfolios):
    optimal_return, optimal_volatility, optimal_sharpe_ratio, optimal_weights = optimize_portfolio(returns)

    random_portfolios = []
    random_returns = []

    for _ in range(num_portfolios):
        random_weights = generate_random_weights(len(returns.columns))
        random_portfolio_return, _, _, _ = optimize_portfolio(returns, random_weights)
        random_portfolios.append(random_weights)
        random_returns.append(random_portfolio_return)

    # Compare optimal portfolio to random portfolios
    print("Optimal Portfolio:")
    print("Return: {:.2%}".format(optimal_return))
    print("Volatility: {:.2%}".format(optimal_volatility))
    print("Sharpe Ratio: {:.4f}".format(optimal_sharpe_ratio))
    print("Weights:")
    for symbol, weight in zip(returns.columns, optimal_weights):
        print("{}: {:.2%}".format(symbol, weight))

    # Plot return percentages
    plt.figure(figsize=(10, 6))
    plt.hist(random_returns, bins=30, alpha=0.5, label='Random Portfolios')
    plt.axvline(x=optimal_return, color='r', linestyle='dashed', linewidth=2, label='Optimal Portfolio')
    plt.xlabel('Return Percentage')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Return Percentage Distribution')
    plt.show()

def main():
    # Fetch stock data and calculate returns
    symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    start_date = '2012-01-01'
    end_date = '2022-01-01'

    stock_data = fetch_stock_data(symbols, start_date, end_date)
    returns = stock_data.pct_change().dropna()

    # Calculate logarithmic returns
    log_returns = np.log(1 + returns)

    # Calculate Standard Deviation
    N=252
    STD = log_returns.groupby([log_returns.index.year]).agg('std') * np.sqrt(N)
    STD_avg = STD.mean()

    # Calculate Variance
    VAR = STD ** 2
    VAR_avg = VAR.mean()

    # Calculate APY average
    apy_avg = calculate_apy_avg(returns)

    # Visualize Standard Deviation
    plot_std(STD)

    # Visualize Return vs. Risk
    plot_return_vs_risk(STD_avg, apy_avg)

    num_portfolios = 10000
    plot_efficient_frontier(returns, num_portfolios)
    optimal_return, optimal_volatility, _, optimal_weights = optimize_portfolio(returns)

    print_optimal_portfolio(optimal_return, optimal_volatility, symbols, optimal_weights)

    # Compare portfolios
    num_random_portfolios = 1000  # Adjust the number of random portfolios as needed
    compare_portfolios(returns, num_random_portfolios)

    plt.show()

if __name__ == '__main__':
    main()


