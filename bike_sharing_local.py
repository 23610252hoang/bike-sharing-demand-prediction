"""レンタサイクル需要予測のベースラインモデル。

2011年の日別データで線形回帰モデルを学習し、2012年の日別需要を予測します。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR
FEATURE_COLUMNS = ["atemp"]
VALID_RATIO = 0.2


def configure_japanese_font() -> None:
    """実行環境で利用可能な日本語フォントをMatplotlibへ設定します。"""
    preferred_fonts = [
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
        "IPAexGothic",
    ]
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in installed_fonts:
            plt.rcParams["font.family"] = font_name
            return

    print("日本語フォントが見つからないため、グラフ内の日本語が正しく表示されない場合があります。")


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """学習・予測用CSVを読み込み、確認しやすい補助列を追加します。"""
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
    """ポートフォリオ確認用の基本的な可視化グラフを保存します。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.title("2011年の日別レンタサイクル利用数")
    plt.plot(train_df["dteday"], train_df["count"])
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.tight_layout()
    plt.savefig(output_dir / "rental_trend_2011.png", dpi=160)
    plt.close()

    try:
        import seaborn as sns
    except ImportError:
        print("seabornが未インストールのため、pairplotとカテゴリ別グラフをスキップしました。")
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
    pair_grid.fig.suptitle("数値特徴量の関係", y=1.02)
    pair_grid.fig.tight_layout()
    pair_grid.fig.savefig(output_dir / "feature_pairplot.png", dpi=160)
    plt.close(pair_grid.fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("カテゴリ変数と利用数の関係")
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
    """2011年データの前半で学習し、後半期間で検証します。"""
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
    """2011年データ全体で再学習し、2012年の利用数を予測します。"""
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
    """2011年の実績値と2012年の予測値を比較するグラフを保存します。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.title("2011年実績値と2012年予測値")
    plt.plot(train_df["dteday"], train_df["count"], label="2011年実績")
    plt.plot(test_df["dteday"], prediction_df["count"], label="2012年予測")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_2012.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="レンタサイクル需要予測のベースラインモデルを学習します。")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "submissions")
    parser.add_argument("--plot-dir", type=Path, default=PROJECT_DIR / "figures")
    parser.add_argument("--student-id", default="23610252kn")
    parser.add_argument("--no-round", action="store_true", help="予測値を小数のまま保存します。")
    parser.add_argument("--skip-plots", action="store_true", help="探索的グラフの保存をスキップします。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    configure_japanese_font()

    train_df, test_df = load_data(args.data_dir)
    print("データを読み込みました")
    print(f"  学習データ形状: {train_df.shape}")
    print(f"  予測データ形状: {test_df.shape}")
    print()
    print("学習に使用する特徴量")
    print(FEATURE_COLUMNS)
    print()
    print("学習データの欠損値")
    print(train_df.isnull().sum())

    _, rmse, mae, validation_df = train_validate(train_df)
    print()
    print("検証指標")
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
    print(f"予測結果を保存しました: {output_path}")
    print()
    print("予測結果プレビュー")
    print(prediction_df.head(10))
    print()
    print("検証結果プレビュー")
    print(validation_df.head(5))


if __name__ == "__main__":
    main()
