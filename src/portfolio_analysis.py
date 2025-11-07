"""
Investment Portfolio Analysis
Analyzes a stock portfolio and benchmarks against S&P 500
Calculates risk-adjusted returns and generates performance visualizations
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    """
    Portfolio Analysis Tool
    Fetches stock data, calculates metrics, and generates visualizations
    """
    
    def __init__(self, tickers, weights=None, start_date=None, end_date=None):
        """
        Initialize Portfolio Analyzer
        
        Parameters:
        -----------
        tickers : list
            List of stock ticker symbols
        weights : list, optional
            Portfolio weights (must sum to 1). Defaults to equal weights
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.benchmark = '^GSPC'  # S&P 500 index
        
        # Set equal weights if not specified
        if weights is None:
            self.weights = np.array([1/len(tickers)] * len(tickers))
        else:
            self.weights = np.array(weights)
            if not np.isclose(self.weights.sum(), 1.0):
                raise ValueError("Weights must sum to 1.0")
            
        # Set date range (default: 3 years of historical data)
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=3*365)
        else:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    def fetch_data(self):
        """Fetch historical price data from Yahoo Finance"""
        print("="*60)
        print("FETCHING MARKET DATA")
        print("="*60)
        print(f"Tickers: {', '.join(self.tickers)}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print("Downloading data from Yahoo Finance...")
        
        # Download portfolio stocks
        portfolio_raw = yf.download(
            self.tickers, 
            start=self.start_date, 
            end=self.end_date,
            progress=False
        )
        
        # Extract 'Adj Close' column - handle both MultiIndex and single-level columns
        if isinstance(portfolio_raw.columns, pd.MultiIndex):
            # Multiple tickers: try level 0 first (price types), then level 1
            try:
                self.data = portfolio_raw.xs('Adj Close', level=0, axis=1)
            except KeyError:
                try:
                    self.data = portfolio_raw.xs('Close', level=0, axis=1)
                except KeyError:
                    try:
                        self.data = portfolio_raw.xs('Adj Close', level=1, axis=1)
                    except KeyError:
                        self.data = portfolio_raw.xs('Close', level=1, axis=1)
        else:
            # Single ticker: try different column names
            adj_close_name = None
            for name in ['Adj Close', 'AdjClose', 'Close']:
                if name in portfolio_raw.columns:
                    adj_close_name = name
                    break
            
            if adj_close_name is None:
                raise ValueError(f"Could not find adjusted close price column. Available columns: {portfolio_raw.columns.tolist()}")
            
            self.data = portfolio_raw[adj_close_name]
            if isinstance(self.data, pd.Series):
                self.data = self.data.to_frame(name=self.tickers[0])
        
        # Download benchmark (S&P 500)
        benchmark_raw = yf.download(
            self.benchmark, 
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        
        # Extract 'Adj Close' for benchmark
        if isinstance(benchmark_raw.columns, pd.MultiIndex):
            # Try level 0 first (price types), then level 1
            try:
                benchmark_adj_close = benchmark_raw.xs('Adj Close', level=0, axis=1)
            except KeyError:
                try:
                    benchmark_adj_close = benchmark_raw.xs('Close', level=0, axis=1)
                except KeyError:
                    try:
                        benchmark_adj_close = benchmark_raw.xs('Adj Close', level=1, axis=1)
                    except KeyError:
                        benchmark_adj_close = benchmark_raw.xs('Close', level=1, axis=1)
            
            self.benchmark_data = benchmark_adj_close.iloc[:, 0] if benchmark_adj_close.shape[1] > 0 else benchmark_adj_close
        else:
            # Single ticker: try different column names
            adj_close_name = None
            for name in ['Adj Close', 'AdjClose', 'Close']:
                if name in benchmark_raw.columns:
                    adj_close_name = name
                    break
            
            if adj_close_name is None:
                raise ValueError(f"Could not find adjusted close price column for benchmark. Available columns: {benchmark_raw.columns.tolist()}")
            
            self.benchmark_data = benchmark_raw[adj_close_name]
        
        print(f"✓ Successfully fetched {len(self.data)} trading days of data")
        
    def calculate_returns(self):
        """Calculate daily and cumulative returns"""
        print("\nCalculating returns...")
        
        # Calculate daily returns
        self.returns = self.data.pct_change().dropna()
        self.benchmark_returns = self.benchmark_data.pct_change().dropna()
        
        # Calculate weighted portfolio returns
        self.portfolio_returns = (self.returns * self.weights).sum(axis=1)
        
        # Calculate cumulative returns
        self.portfolio_cumulative = (1 + self.portfolio_returns).cumprod()
        self.benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
        
        print("✓ Returns calculated")
        
    def calculate_metrics(self):
        """Calculate comprehensive portfolio performance metrics"""
        print("Calculating performance metrics...")
        
        # Constants
        TRADING_DAYS = 252  # Annual trading days
        RISK_FREE_RATE = 0.04  # 4% annual risk-free rate (approximate)
        
        # Portfolio metrics
        portfolio_daily_return = self.portfolio_returns.mean()
        portfolio_annual_return = portfolio_daily_return * TRADING_DAYS
        portfolio_volatility = self.portfolio_returns.std() * np.sqrt(TRADING_DAYS)
        portfolio_sharpe = (portfolio_annual_return - RISK_FREE_RATE) / portfolio_volatility
        
        # Benchmark metrics
        benchmark_daily_return = self.benchmark_returns.mean()
        benchmark_annual_return = benchmark_daily_return * TRADING_DAYS
        benchmark_volatility = self.benchmark_returns.std() * np.sqrt(TRADING_DAYS)
        benchmark_sharpe = (benchmark_annual_return - RISK_FREE_RATE) / benchmark_volatility
        
        # Total returns
        portfolio_total_return = self.portfolio_cumulative.iloc[-1] - 1
        benchmark_total_return = self.benchmark_cumulative.iloc[-1] - 1
        
        # Maximum drawdown calculation
        portfolio_running_max = self.portfolio_cumulative.cummax()
        portfolio_drawdown = (self.portfolio_cumulative - portfolio_running_max) / portfolio_running_max
        portfolio_max_drawdown = portfolio_drawdown.min()
        
        benchmark_running_max = self.benchmark_cumulative.cummax()
        benchmark_drawdown = (self.benchmark_cumulative - benchmark_running_max) / benchmark_running_max
        benchmark_max_drawdown = benchmark_drawdown.min()
        
        # Alpha and Beta calculations
        covariance = np.cov(self.portfolio_returns, self.benchmark_returns)[0][1]
        benchmark_variance = np.var(self.benchmark_returns)
        beta = covariance / benchmark_variance
        alpha = portfolio_annual_return - (RISK_FREE_RATE + beta * (benchmark_annual_return - RISK_FREE_RATE))
        
        # Store metrics
        self.metrics = {
            'Portfolio': {
                'Total Return': portfolio_total_return,
                'Annual Return': portfolio_annual_return,
                'Annual Volatility': portfolio_volatility,
                'Sharpe Ratio': portfolio_sharpe,
                'Max Drawdown': portfolio_max_drawdown,
                'Beta': beta,
                'Alpha': alpha
            },
            'S&P 500': {
                'Total Return': benchmark_total_return,
                'Annual Return': benchmark_annual_return,
                'Annual Volatility': benchmark_volatility,
                'Sharpe Ratio': benchmark_sharpe,
                'Max Drawdown': benchmark_max_drawdown,
                'Beta': 1.0,
                'Alpha': 0.0
            }
        }
        
        print("✓ Metrics calculated")
        
    def print_results(self):
        """Display analysis results in formatted table"""
        print("\n" + "="*70)
        print("PORTFOLIO ANALYSIS RESULTS")
        print("="*70)
        
        # Portfolio composition
        print("\n📊 PORTFOLIO COMPOSITION")
        print("-" * 70)
        for ticker, weight in zip(self.tickers, self.weights):
            print(f"  {ticker:6s} : {weight:6.1%}")
        print(f"  Total  : {self.weights.sum():6.1%}")
        
        # Performance comparison
        print("\n📈 PERFORMANCE METRICS")
        print("-" * 70)
        print(f"{'Metric':<25} {'Portfolio':>20} {'S&P 500':>20}")
        print("-" * 70)
        
        metrics_order = [
            'Total Return', 
            'Annual Return', 
            'Annual Volatility', 
            'Sharpe Ratio',
            'Max Drawdown',
            'Beta',
            'Alpha'
        ]
        
        for metric in metrics_order:
            port_val = self.metrics['Portfolio'][metric]
            bench_val = self.metrics['S&P 500'][metric]
            
            # Format based on metric type
            if metric in ['Total Return', 'Annual Return', 'Annual Volatility', 'Max Drawdown', 'Alpha']:
                port_str = f"{port_val:>19.2%}"
                bench_str = f"{bench_val:>19.2%}"
            else:
                port_str = f"{port_val:>20.3f}"
                bench_str = f"{bench_val:>20.3f}"
            
            print(f"{metric:<25} {port_str} {bench_str}")
        
        # Key insights
        print("\n💡 KEY INSIGHTS")
        print("-" * 70)
        
        outperformance = (self.metrics['Portfolio']['Total Return'] - 
                         self.metrics['S&P 500']['Total Return']) * 100
        
        if outperformance > 0:
            print(f"  ✓ Portfolio OUTPERFORMED S&P 500 by {outperformance:.2f}%")
        else:
            print(f"  ✗ Portfolio UNDERPERFORMED S&P 500 by {abs(outperformance):.2f}%")
        
        if self.metrics['Portfolio']['Sharpe Ratio'] > self.metrics['S&P 500']['Sharpe Ratio']:
            print(f"  ✓ Better risk-adjusted returns (Higher Sharpe Ratio)")
        else:
            print(f"  ✗ Lower risk-adjusted returns (Lower Sharpe Ratio)")
            
        print("\n" + "="*70)
        
    def plot_performance(self, save_path='portfolio_analysis.png'):
        """Generate comprehensive visualization dashboard"""
        print(f"\nGenerating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        
        # Create subplot layout
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Investment Portfolio Analysis Dashboard', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Cumulative Returns Comparison (Large, top)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.portfolio_cumulative.index, 
                (self.portfolio_cumulative - 1) * 100, 
                label='Portfolio', linewidth=2.5, color='#2E86AB')
        ax1.plot(self.benchmark_cumulative.index, 
                (self.benchmark_cumulative - 1) * 100, 
                label='S&P 500', linewidth=2.5, color='#A23B72', alpha=0.7)
        ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # 2. Individual Stock Performance
        ax2 = fig.add_subplot(gs[1, 0])
        individual_cumulative = (1 + self.returns).cumprod()
        colors = sns.color_palette('husl', len(self.tickers))
        
        for i, ticker in enumerate(self.tickers):
            ax2.plot(individual_cumulative.index, 
                    (individual_cumulative[ticker] - 1) * 100, 
                    label=ticker, alpha=0.8, linewidth=2, color=colors[i])
        
        ax2.set_title('Individual Stock Performance', fontsize=12, fontweight='bold', pad=10)
        ax2.set_ylabel('Return (%)', fontsize=10)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Portfolio Allocation Pie Chart
        ax3 = fig.add_subplot(gs[1, 1])
        wedges, texts, autotexts = ax3.pie(
            self.weights, 
            labels=self.tickers, 
            autopct='%1.1f%%',
            colors=colors, 
            startangle=90,
            textprops={'fontsize': 10}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax3.set_title('Portfolio Allocation', fontsize=12, fontweight='bold', pad=10)
        
        # 4. Risk-Return Scatter Plot
        ax4 = fig.add_subplot(gs[2, 0])
        annual_returns = self.returns.mean() * 252 * 100
        annual_vol = self.returns.std() * np.sqrt(252) * 100
        
        # Plot individual stocks
        for i, ticker in enumerate(self.tickers):
            ax4.scatter(annual_vol[i], annual_returns[i], 
                       s=150, alpha=0.7, c=[colors[i]], edgecolors='black', linewidth=1.5)
            ax4.annotate(ticker, (annual_vol[i], annual_returns[i]),
                        xytext=(7, 7), textcoords='offset points', fontsize=9)
        
        # Plot portfolio
        port_return = self.metrics['Portfolio']['Annual Return'] * 100
        port_vol = self.metrics['Portfolio']['Annual Volatility'] * 100
        ax4.scatter(port_vol, port_return, s=300, marker='*', 
                   c='#FF6B35', label='Portfolio', edgecolors='black', linewidth=2, zorder=5)
        
        # Plot benchmark
        bench_return = self.metrics['S&P 500']['Annual Return'] * 100
        bench_vol = self.metrics['S&P 500']['Annual Volatility'] * 100
        ax4.scatter(bench_vol, bench_return, s=300, marker='s', 
                   c='#004E89', label='S&P 500', edgecolors='black', linewidth=2, zorder=5)
        
        ax4.set_xlabel('Annual Volatility (%) - Risk', fontsize=10)
        ax4.set_ylabel('Annual Return (%)', fontsize=10)
        ax4.set_title('Risk-Return Profile', fontsize=12, fontweight='bold', pad=10)
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Metrics Table
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        # Create table data
        table_data = []
        metrics_display = [
            ('Total Return', 'Total Return'),
            ('Annual Return', 'Annual Return'),
            ('Volatility', 'Annual Volatility'),
            ('Sharpe Ratio', 'Sharpe Ratio'),
            ('Max Drawdown', 'Max Drawdown')
        ]
        
        for display_name, metric_key in metrics_display:
            port_val = self.metrics['Portfolio'][metric_key]
            bench_val = self.metrics['S&P 500'][metric_key]
            
            if metric_key == 'Sharpe Ratio':
                table_data.append([display_name, f"{port_val:.3f}", f"{bench_val:.3f}"])
            else:
                table_data.append([display_name, f"{port_val:.1%}", f"{bench_val:.1%}"])
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Metric', 'Portfolio', 'S&P 500'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style table
        for i in range(len(table_data) + 1):
            if i == 0:
                for j in range(3):
                    table[(i, j)].set_facecolor('#2E86AB')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                for j in range(3):
                    table[(i, j)].set_facecolor('#E8F4F8' if i % 2 == 0 else 'white')
        
        ax5.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Visualization saved as '{save_path}'")
        plt.show()
        
    def run_analysis(self):
        """Execute complete portfolio analysis pipeline"""
        self.fetch_data()
        self.calculate_returns()
        self.calculate_metrics()
        self.print_results()
        self.plot_performance()
        print("\n✓ Analysis complete!")


def main():
    """Main execution function"""
    # Define portfolio
    tickers = ['AXP', 'AAPL', 'NVDA', 'GOOG', 'TSLA']
    
    # Create analyzer with equal weights
    analyzer = PortfolioAnalyzer(tickers)
    
    # Run complete analysis
    analyzer.run_analysis()


if __name__ == "__main__":
    main()