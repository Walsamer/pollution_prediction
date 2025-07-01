import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)


def plot_pm25_trend(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot daily average PM2.5 trend and save to output_dir/eda_pm25_trend.pdf

    Returns the path to the saved figure.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Calculate daily average (datetime index was set in preprocessing)
    daily_avg = df['pm2.5'].resample('D').mean()

    fig, ax = plt.subplots()
    ax.plot(daily_avg.index, daily_avg.values)
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Average PM2.5')
    ax.set_title('Daily Average PM2.5 Trend')

    out_path = os.path.join(output_dir, 'eda_pm25_trend.pdf')
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved PM2.5 trend plot to %s", out_path)
    return out_path


def plot_correlation(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot heatmap of correlations between numeric features and save to output_dir/eda_correlation_heatmap.pdf

    Returns the path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute correlation matrix
    corr = df.corr()

    # Plot heatmap
    fig, ax = plt.subplots()
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    fig.colorbar(cax)

    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title('Feature Correlation Heatmap')

    out_path = os.path.join(output_dir, 'eda_correlation_heatmap.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved correlation heatmap to %s", out_path)
    return out_path


def plot_histogram_pm25(df: pd.DataFrame, output_dir: str, n_bins: int = 20) -> str:
    """
    Plot histogram of daily average PM2.5 distribution and save to output_dir/eda_pm25_histogram.pdf

    Returns the path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute daily average if datetime column exists
    if 'datetime' in df.columns:
        df = df.set_index(pd.to_datetime(df['datetime']))
    daily = df['pm2.5'].resample('D').mean()
    data = daily.dropna().values

    fig, ax = plt.subplots()
    ax.hist(data, bins=n_bins)
    ax.set_xlabel('PM2.5')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Daily Average PM2.5')

    out_path = os.path.join(output_dir, 'eda_pm25_histogram.pdf')
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved PM2.5 histogram to %s", out_path)
    return out_path
