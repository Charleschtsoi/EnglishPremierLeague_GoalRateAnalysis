import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(path):
    """Load raw CSV file."""
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} matches from {path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise

def preprocess_data(df):
    """Filter, create features, handle missing values."""
    logger.info("Starting preprocessing...")
    
    # Drop unnecessary columns
    columns_to_drop = ['Date', 'Referee', 'Display_Order', 'League']
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], axis=1)
    
    # Filter: Keep only seasons from 2000/01 onwards
    seasons_to_exclude = ['1993/94', '1994/95', '1995/96', '1996/97', '1997/98', '1998/99', '1999/00']
    df = df[~df['Season'].isin(seasons_to_exclude)].copy()
    
    # Create target features
    df['FT_total_goals'] = df['FTH Goals'] + df['FTA Goals']
    df['HT_total_goals'] = df['HTH Goals'] + df['HTA Goals']
    df['SHT_total_goals'] = df['FT_total_goals'] - df['HT_total_goals']
    
    logger.info(f"Preprocessed data: {len(df)} matches, {df['Season'].nunique()} seasons")
    return df
