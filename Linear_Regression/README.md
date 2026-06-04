# Medical Insurance Charge Predictor

A machine learning project that predicts annual medical insurance charges from a
person's demographic and health details, deployed as an interactive Streamlit web app.

> **Note:** This is a learning / portfolio project trained on a small public
> teaching dataset (1,338 rows). It demonstrates an end-to-end ML workflow and is
> **not** a real insurance-pricing tool or medical advice.

## Demo

Enter age, sex, BMI, number of children, smoking status, and region — the app
returns an estimated annual charge.

*(Add your Streamlit Cloud link here once deployed, e.g. `https://your-app.streamlit.app`)*

## Project Overview

The goal is to predict the `charges` column (annual medical cost) using linear
regression, with a focus on a clean, leakage-safe, reproducible workflow.

**Dataset:** the public insurance dataset — 1,338 records with the features
`age`, `sex`, `bmi`, `children`, `smoker`, `region`, and the target `charges`.

## Approach

The workflow follows a target-first EDA and a pipeline-based modeling process:

1. **EDA (target-first)** — examined the distribution of `charges` (right-skewed,
   skew ≈ 1.5), checked correlations, and studied how each feature moves the target.
   Smoking status emerged as by far the strongest driver.
2. **Feature engineering** — added `is_smoker_and_obese`, an interaction flag for
   patients who both smoke and have a BMI ≥ 30. Linear regression cannot capture
   interactions on its own, and this group has dramatically higher costs.
3. **Target transform** — modeled `log(charges)` instead of raw charges to correct
   the skew, then inverted predictions back to dollars with `expm1`.
4. **Leakage-safe pipeline** — split the data *before* any fitting, then used a
   `ColumnTransformer` inside a `Pipeline` (one-hot encoding for nominal
   categoricals, scaling for numerics) so all preprocessing is learned from the
   training set only.
5. **Baseline check** — compared against a `DummyRegressor` (predict the mean) to
   confirm the model adds real value.
6. **Evaluation & interpretation** — checked R², RMSE, MAE, and inspected
   coefficients to confirm the model's logic matched the EDA findings.

## Results

| Metric | Value |
| ------ | ----- |
| R² (log scale) | ~0.84 |
| MAE (dollars) | ~$3,800 |
| RMSE (dollars) | ~$7,700 |

The coefficients confirmed the story from EDA: smoking status was the dominant
predictor, followed by age, with the smoker-and-obese interaction adding further
signal — while BMI on its own contributed little.

## Tech Stack

- **Python** — pandas, NumPy
- **scikit-learn** — `Pipeline`, `ColumnTransformer`, `LinearRegression`
- **Visualization** — matplotlib, seaborn
- **App & deployment** — Streamlit, joblib

## Project Structure

```
Linear Regression/
├── data/
│   └── raw/
│       └── insurance.csv
├── notebooks/
│   └── medical_price.ipynb     # EDA + model training
├── app.py                       # Streamlit app
├── insurance_model.pkl          # saved pipeline (preprocessing + model)
├── requirements.txt
└── README.md
```

## Run Locally

```bash
# clone and enter the repo
git clone https://github.com/Pratik-sharma-py/datascience_project.git
cd datascience_project

# install dependencies
pip install -r requirements.txt

# launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How It Works

The saved `insurance_model.pkl` is a full scikit-learn `Pipeline`, so it carries
its own encoding and scaling. The app collects the six raw user inputs, recreates
the `is_smoker_and_obese` feature exactly as in training, feeds the row to the
pipeline, and inverts the log prediction back to dollars.

## License

Released for educational and portfolio use.
