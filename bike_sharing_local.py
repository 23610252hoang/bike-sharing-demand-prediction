"""Bike sharing demand baseline model.

This script trains a simple Linear Regression model on the 2011 daily bike
sharing data and predicts daily demand for 2012.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR
FEATURE_COLUMNS = ["atemp"]
VALID_RATIO = 0.2


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSV files and add readable helper columns."""
    train_df = pd.read_csv(data_dir / "day_train.csv").rename(columns={"cnt": "count"})
    test_df = pd.read_csv(data_dir / "day_test.csv")

    for df in (train_df, test_df):
        if "yr" in df.columns:
            df["year"] = np.where(df["yr"] == 0, 2011, 2012)
        if "weathersit" in df.columns:
            df["weather"] = df["weathersit"].map(
                {
                    1: "clear",
                    2: "misty",
                    3: "rain",
                    4: "heavy_rain",
                }
            )
        df["dteday"] = pd.to_datetime(df["dteday"])

    train_df = train_df.sort_values("dteday").reset_index(drop=True)
    test_df = test_df.sort_values("dteday").reset_index(drop=True)
    return train_df, test_df


def save_exploratory_plots(train_df: pd.DataFrame, output_dir: Path) -> None:
    """Save basic exploratory charts for portfolio review."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.title("Daily bike rentals in 2011")
    plt.plot(train_df["dteday"], train_df["count"])
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.tight_layout()
    plt.savefig(output_dir / "rental_trend_2011.png", dpi=160)
    plt.close()

    try:
        import seaborn as sns
    except ImportError:
        print("seaborn is not installed; skipped pairplot and categorical charts.")
        return

    drop_cols = [
        "instant",
        "dteday",
        "season",
        "yr",
        "year",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "weather",
        "casual",
        "registered",
    ]
    pairplot_df = train_df.drop([c for c in drop_cols if c in train_df.columns], axis=1)
    pair_grid = sns.pairplot(pairplot_df, height=2)
    pair_grid.fig.suptitle("Numerical feature relationships", y=1.02)
    pair_grid.fig.tight_layout()
    pair_grid.fig.savefig(output_dir / "feature_pairplot.png", dpi=160)
    plt.close(pair_grid.fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Categorical variables and rental count")
    sns.barplot(data=train_df, x="season", y="count", ax=axes[0, 0])
    sns.barplot(data=train_df, x="holiday", y="count", ax=axes[0, 1])
    sns.barplot(data=train_df, x="weekday", y="count", ax=axes[0, 2])
    sns.barplot(data=train_df, x="workingday", y="count", ax=axes[1, 0])
    sns.barplot(data=train_df, x="weather", y="count", ax=axes[1, 1])
    sns.barplot(data=train_df, x="mnth", y="count", ax=axes[1, 2])
    plt.tight_layout()
    fig.savefig(output_dir / "categorical_count_comparison.png", dpi=160)
    plt.close(fig)


def train_validate(train_df: pd.DataFrame) -> tuple[LinearRegression, float, float, pd.DataFrame]:
    """Train on the first part of 2011 data and validate on the final period."""
    split_idx = int(len(train_df) * (1 - VALID_RATIO))
    train_part = train_df.iloc[:split_idx].copy()
    valid_part = train_df.iloc[split_idx:].copy()

    model = LinearRegression()
    model.fit(train_part[FEATURE_COLUMNS].values, train_part["count"].values)

    valid_pred = model.predict(valid_part[FEATURE_COLUMNS].values)
    rmse = float(np.sqrt(mean_squared_error(valid_part["count"].values, valid_pred)))
    mae = float(mean_absolute_error(valid_part["count"].values, valid_pred))

    validation_df = valid_part[["dteday", "count"]].copy()
    validation_df["predicted_count"] = valid_pred
    return model, rmse, mae, validation_df


def predict_test(train_df: pd.DataFrame, test_df: pd.DataFrame, round_to_int: bool) -> pd.DataFrame:
    """Train on all 2011 data and predict 2012 count values."""
    model = LinearRegression()
    model.fit(train_df[FEATURE_COLUMNS].values, train_df["count"].values)

    predictions = model.predict(test_df[FEATURE_COLUMNS].values)
    output_df = test_df[["instant"]].copy()
    output_df["count"] = predictions
    if round_to_int:
        output_df["count"] = np.rint(output_df["count"]).astype(int)
    return output_df


def save_prediction_plot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save a chart comparing 2011 actuals and 2012 predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.title("2011 actual rentals and 2012 predicted rentals")
    plt.plot(train_df["dteday"], train_df["count"], label="2011 actual")
    plt.plot(test_df["dteday"], prediction_df["count"], label="2012 predicted")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_2012.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a bike sharing baseline model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "submissions")
    parser.add_argument("--plot-dir", type=Path, default=PROJECT_DIR / "figures")
    parser.add_argument("--student-id", default="23610252kn")
    parser.add_argument("--no-round", action="store_true", help="Keep prediction values as floats.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip saving exploratory charts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")

    train_df, test_df = load_data(args.data_dir)
    print("Loaded data")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape : {test_df.shape}")
    print()
    print("Training columns")
    print(FEATURE_COLUMNS)
    print()
    print("Missing values in train data")
    print(train_df.isnull().sum())

    _, rmse, mae, validation_df = train_validate(train_df)
    print()
    print("Validation metrics")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE : {mae:.3f}")

    prediction_df = predict_test(train_df, test_df, round_to_int=not args.no_round)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = args.output_dir / f"submission_{args.student_id}_{timestamp}.csv"
    prediction_df.to_csv(output_path, index=False, encoding="utf-8")

    if not args.skip_plots:
        save_exploratory_plots(train_df, args.plot_dir)
        save_prediction_plot(train_df, test_df, prediction_df, args.plot_dir)

    print()
    print(f"Saved predictions to: {output_path}")
    print()
    print("Prediction preview")
    print(prediction_df.head(10))
    print()
    print("Validation preview")
    print(validation_df.head(5))


if __name__ == "__main__":
    main()
