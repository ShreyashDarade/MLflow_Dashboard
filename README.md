## Repository Overview

### Files

- **mlflow_utils.py**  
  Contains utility functions for interacting with MLflow, including:
  - Fetching experiments and runs
  - Logging and comparing metrics
  - Managing registered models (listing, promoting, and transitioning versions)

- **streamlit_app.py**  
  An interactive [Streamlit](https://streamlit.io/) application that:
  - Allows you to upload a CSV file and preview raw and preprocessed data
  - Lets you choose between regression and classification tasks
  - Provides options for training a model (e.g., RandomForest, XGBoost, LogisticRegression) with live parameter tuning
  - Logs experiments to MLflow and visualizes metrics with plots and parallel coordinates

- **train_multiple_models.py**  
  A command-line script that:
  - Trains multiple models on a given dataset (supports both regression and classification)
  - Logs parameters, metrics, and model artifacts to MLflow
  - Registers new model versions and promotes the best-performing model based on a specified metric

---

## Getting Started

### Prerequisites

- **Python 3.7+**
- [MLflow](https://mlflow.org/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Plotly](https://plotly.com/)

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/ShreyashDarade/MLflow_Dashboard.git
cd ShreyashDarade
pip install -r requirements.txt
```

*Tip: If you don't have a `requirements.txt` yet, create one listing the packages mentioned above.*

---

## Usage

### Running the Streamlit App

The interactive app lets you run experiments, visualize metrics, and manage models:

```bash
streamlit run streamlit_app.py
```

### Training Multiple Models via Command Line

To train multiple models and log experiments with MLflow, use the command-line script. For example:

```bash
python train_multiple_models.py --csv_path path/to/your_dataset.csv --target target_column_name --task_type Classification --experiment_name YourExperimentName --registered_model_name YourRegisteredModelName
```

For regression tasks, set `--task_type Regression` and adjust any hyperparameters as needed.

---

## MLflow Configuration

Both the Streamlit app and training script use a local MLflow tracking server by default:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

Make sure your MLflow server is running, or update the URI if you are using a different MLflow server.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please contact **Shreyash Darade** or open an issue in the repository.

Happy experimenting!
```

You can add or modify sections as needed. Simply create this file at the root of your repository under the name **README.md** (or **ShreyashDarade.md** if you prefer that naming convention).
