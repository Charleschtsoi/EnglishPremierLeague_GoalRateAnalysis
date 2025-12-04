import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import RANDOM_STATE, RF_N_ESTIMATORS, TRAIN_TEST_SPLIT

def train_and_evaluate(df, base_features):
    """Train and compare Pure vs Engineered clustering models."""
    logger.info("Training models...")
    
    y = df['FT_total_goals'].copy()
    
    # ============ MODEL 1: PURE CLUSTERS ============
    rf_feats_pure = base_features + ['cluster_km_pure']
    X_pure = df[rf_feats_pure].dropna()
    y_pure = df.loc[X_pure.index, 'FT_total_goals']
    
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_pure, y_pure, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE
    )
    
    rf_pure = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    rf_pure.fit(X_train_p, y_train_p)
    r2_pure = rf_pure.score(X_test_p, y_test_p)
    
    logger.info(f"Model 1 - R² with PURE clusters: {r2_pure:.4f}")
    
    # ============ MODEL 2: ENGINEERED CLUSTERS ============
    rf_feats_eng = base_features + ['cluster_km_engineered']
    X_eng = df[rf_feats_eng].dropna()
    y_eng = df.loc[X_eng.index, 'FT_total_goals']
    
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        X_eng, y_eng, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE
    )
    
    rf_eng = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    rf_eng.fit(X_train_e, y_train_e)
    r2_eng = rf_eng.score(X_test_e, y_test_e)
    
    logger.info(f"Model 2 - R² with ENGINEERED clusters: {r2_eng:.4f}")
    
    leakage = r2_eng - r2_pure
    logger.info(f"Leakage effect (difference): {leakage:+.4f}")
    
    if abs(leakage) < 0.05:
        logger.info("✓ Small difference = genuine clustering signal, not leakage")
    else:
        logger.info("✗ Large difference = significant leakage detected")
        logger.info("   → Use PURE clusters for honest model evaluation")
    
    # ============ FEATURE IMPORTANCES ============
    logger.info("\n[Pure Model] Feature importances:")
    imp_pure = pd.Series(rf_pure.feature_importances_, index=rf_feats_pure).sort_values(ascending=False)
    logger.info(imp_pure.round(4))
    
    logger.info("\n[Engineered Model] Feature importances:")
    imp_eng = pd.Series(rf_eng.feature_importances_, index=rf_feats_eng).sort_values(ascending=False)
    logger.info(imp_eng.round(4))
    
    return {
        'r2_pure': r2_pure,
        'r2_eng': r2_eng,
        'leakage': leakage,
        'imp_pure': imp_pure,
        'imp_eng': imp_eng,
        'model_pure': rf_pure,
        'model_eng': rf_eng
    }
