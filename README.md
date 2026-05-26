# Bike Sharing Demand Prediction

This project is a small machine learning baseline for predicting daily bike
sharing demand. It uses 2011 daily rental data for training and predicts 2012
daily rental counts.

## Project Files

- `bike_sharing_local.py`: local Python script for loading data, training the model, evaluating validation performance, and exporting predictions.
- `day_train.csv`: 2011 training data with daily rental count.
- `day_test.csv`: 2012 test data used for prediction.
- `23610252kn_pred.csv`: saved prediction output for submission/reference.
- `requirements.txt`: Python package requirements.

Resume PDFs and temporary base64 files are intentionally excluded from this
repository because this repo is intended to be public portfolio material.

## Method

- Model: `LinearRegression` from scikit-learn.
- Feature used: `atemp`, the normalized feeling temperature.
- Validation: the final 20% of the 2011 training data is used as a simple
  chronological validation split.
- Output: a CSV with `instant` and predicted `count`.

This is a baseline model, so the goal is clarity and reproducibility rather
than maximum score.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
python bike_sharing_local.py
```

For a quick run without saving charts:

```bash
python bike_sharing_local.py --skip-plots
```

The script writes prediction files to `submissions/` and charts to `figures/`.
Those generated folders are ignored by Git.

## Example Output

The included prediction file starts like this:

```csv
dteday,cnt
2012/1/1,3367.74
2012/1/2,2883.61
2012/1/3,3596.46
```

When running `bike_sharing_local.py`, the generated submission format is:

```csv
instant,count
366,2828
367,2048
368,1250
```
