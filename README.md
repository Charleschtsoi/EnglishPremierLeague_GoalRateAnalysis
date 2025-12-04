## Installation

### Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

### Setup
1. Clone the repository
2. Create virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
bash
pip install -r requirements.txt

### 2. **Project Structure**
         ```markdown
         ## Project Structure
         
         
         ├── data/
         │ └── England.csv # Premier League match data
         ├── src/
         │ ├── data_loader.py # Data loading & preprocessing
         │ ├── clustering.py # K-Means clustering logic
         │ └── models.py # Random Forest models
         ├── config.py # Configuration parameters
         ├── main.py # Main execution script
         └── requirements.txt # Dependencies

## Results

The analysis generates:
- Cluster visualizations (scatter plots, heatmaps)
- Model performance metrics (accuracy, precision, recall)
- JSON output with cluster statistics

## Key Features

- **9,300+ matches** analyzed (2000/01 season onwards)
- **K-Means Clustering** to identify match archetypes
- **Random Forest** models for goal prediction
- **Comparative analysis** vs baseline models

## Contributing

Pull requests welcome!

## License

MIT


