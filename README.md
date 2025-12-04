# Premier League Match Analysis: Clustering & Predictive Insights

Advanced clustering analysis of Premier League matches to identify tactical patterns and their predictive value for goal scoring.

## Project Overview

This research explores whether unsupervised clustering can reveal hidden match archetypes that improve goal prediction beyond traditional match statistics. The analysis covers 9,300+ matches from the 2000/01 season onwards.

## Key Findings

### What Worked
- **K-Means identified 5 distinct match archetypes** with different statistical profiles (high-intensity, defensive, balanced, etc.)
- **Clustering features improve goal prediction:** R² improved from 0.276 → 0.383 (38.8% relative improvement) when cluster membership is included as a feature
- **Feature importance confirms genuine signal:** Cluster assignment ranks among top 5 predictive features in Random Forest model
- **Hierarchical clustering produces consistent results:** Similar patterns emerge across different clustering algorithms

### What Didn't Work
- **DBSCAN failed to isolate outliers:** 99.9% of matches classified as noise/outliers, indicating density-based approach isn't suitable for football match data
- **Modest predictive power:** R² of 0.383 means the model explains only 38% of goal variance—football remains largely unpredictable
- **Limited temporal validation:** Current approach uses single train/test split rather than time-series cross-validation (matches from different seasons mixed)

## Technical Details

### Data
- **Source:** English Premier League match data (1993/94 onwards, complete data from 2000/01)
- **Features:** 12 match statistics (shots, fouls, cards, corners, etc.)
- **Sample size:** 9,329 matches

### Methodology
1. **Data Preprocessing:** Median imputation, standardization (StandardScaler)
2. **Clustering:** K-Means (elbow + silhouette optimization), Hierarchical (Ward linkage), DBSCAN (density-based)
3. **Model:** Random Forest Regressor (400 trees) predicting total goals scored
4. **Validation:** Random label controls (shuffled cluster IDs perform worse, confirming genuine signal)

### Known Limitations
- **Data leakage risk:** Initial analysis included engineered target variables (FT_total_goals) in clustering features. Results should be validated with clustering computed on base statistics only.
- **Single train/test split:** Proper evaluation requires time-series cross-validation (e.g., train on seasons 2000-2015, test on 2016-2019)
- **No out-of-sample validation:** Model not tested on recent seasons or different leagues

## Repository Structure


premier-league-clustering/

├── EnglishPremierLeague_GoalRateAnalysis.py # Main analysis (needs refactoring)

├── README.md # This file

├── requirements.txt # Dependencies

└── data/

└── england.csv # Premier League match data

basic

## Installation & Usage

```bash
git clone https://github.com/charleschtsoi/premier-league-clustering
cd premier-league-clustering
pip install -r requirements.txt
python EnglishPremierLeague_GoalRateAnalysis.py
```


Requirements

apache

pandas>=1.3.0

numpy>=1.21.0

scikit-learn>=0.24.0

matplotlib>=3.4.0

seaborn>=0.11.0

scipy>=1.7.0

gdown>=4.0.0

License

MIT

Questions / Feedback

Open an issue on GitHub or reach out directly.
