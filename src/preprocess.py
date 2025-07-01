import pandas as pd
import zipfile
import io
import os
import logging

logger = logging.getLogger(__name__)

def preprocess_data(zip_path: str, station: str, output_csv: str) -> pd.DataFrame:
    """
    Since the original zip file had a d zipfiles by mistake. This script is 
    additionally handling the nested zip file structure. Other than that, extracting
    the CSV for `station` at `zip_path`, cleaning the data bby dropping non-numeric and NaN
    rows, constructing datetime index and savin the cleaned data to 'output_csv'.

    Parameters:
      zip_path : str
          Path to the raw ZIP file containing air quality data.
      station : str
          Station name to filter the CSV filename inside the archive.
      output_csv : str
          Path where the cleaned CSV will be saved.

    Returns:
      pd.DataFrame
        The cleaned DataFrame with a datetime index.
    """
  
    assert zip_path.endswith('.zip'), f"Expected a .zip file, got {zip_path!r}"

    all_zfs = []
    df_loaded = None
    #unpacking nested zipfiles
    try:
        outer_zf = zipfile.ZipFile(zip_path, 'r')
        all_zfs.append(outer_zf)
        queue = [outer_zf]
        # searching through nested zipfiles
        while queue:
            zf = queue.pop(0)
            for name in zf.namelist():
                if station in name and name.lower().endswith('.csv'):
                    with zf.open(name) as f:
                        df_loaded = pd.read_csv(f)
                    break
                if name.lower().endswith('.zip'):
                    inner_bytes = zf.read(name)
                    inner_buf = io.BytesIO(inner_bytes)
                    inner_zf = zipfile.ZipFile(inner_buf, 'r')
                    all_zfs.append(inner_zf)
                    queue.append(inner_zf)
            if df_loaded is not None:
                break
        if df_loaded is None:
            raise FileNotFoundError(f"No CSV for station {station!r} in {zip_path}")
    finally:
        # Ensure all open ZipFile objects are closed
        for zf in all_zfs:
            zf.close()

    # Drop wind direction if present (non-numerical data)
    if 'wd' in df_loaded.columns:
        df_loaded = df_loaded.drop('wd', axis=1)

    # Drop rows(samples) with any missing values
    df_clean = df_loaded.dropna()

    #uncomment for getting information about the number of dropped samples.
    #n_orig = len(df_loaded)
    #n_clean = len(df_clean)
    #print(f"Dropped {n_orig - n_clean} rows with 'NA' as a value.
    #\nsamples before cleaning: {n_orig}
    #\nsamples after cleaning: {n_clean}")

    # Construct a datetime column from year, month, day, hour
    date_cols = {'year', 'month', 'day', 'hour'}
    if date_cols.issubset(df_clean.columns):
        df_clean['datetime'] = pd.to_datetime(df_clean[list(date_cols)])
        df_clean = df_clean.drop(columns=list(date_cols))
    else:
        logger.warning("Date columns %s not all present; skipping datetime index", date_cols)

    # Set the datetime column as index and sort
    df_clean = df_clean.set_index('datetime').sort_index()

    # Save cleaned data to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_clean.to_csv(output_csv, index=True)
    logger.info("Saved cleaned data to %s", output_csv)

    return df_clean
