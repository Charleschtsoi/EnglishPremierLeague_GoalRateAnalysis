import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from config import DATA_PATH, FEATURES_BASE, RESULTS_PATH
from src.data_loader import load_data, preprocess_data
from src.clustering import cluster_pure, cluster_engineered
from src.models import train_and_evaluate

def main():
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    logger.info("="*60)
    logger.info("PREMIER LEAGUE CLUSTERING ANALYSIS")
    logger.info("="*60)
    
    # Load and preprocess data
    logger.info("\n[Step 1] Loading data...")
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    
    # Run clustering
    logger.info("\n[Step 2] Running clustering analysis...")
    df, k_pure, score_pure = cluster_pure(df, FEATURES_BASE)
    df, k_eng, score_eng = cluster_engineered(df, FEATURES_BASE)
    
    # Train and evaluate models
    logger.info("\n[Step 3] Training and evaluating models...")
    results = train_and_evaluate(df, FEATURES_BASE)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"R² (Pure Clusters):      {results['r2_pure']:.4f}")
    logger.info(f"R² (Engineered Clusters):{results['r2_eng']:.4f}")
    logger.info(f"Leakage Effect:          {results['leakage']:+.4f}")
    logger.info("="*60)
    
    # Save results to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"{RESULTS_PATH}results_{timestamp}.json"
    
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'r2_pure': float(results['r2_pure']),
        'r2_eng': float(results['r2_eng']),
        'leakage': float(results['leakage']),
        'samples': len(df),
        'features': FEATURES_BASE
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Plots saved to: {RESULTS_PATH}clusters_*.png")

if __name__ == "__main__":
    main()
