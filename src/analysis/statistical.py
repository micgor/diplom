"""Statistical analysis module: trends, seasonality, correlations."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("figures")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "font.family": "DejaVu Sans"})
sns.set_theme(style="whitegrid")


def load_data() -> pd.DataFrame:
    return pd.read_parquet("data/processed/flights_enriched.parquet")


def airline_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute key performance metrics per airline."""
    agg = df.groupby("Airline").agg(
        total_flights=("Arr_Delay", "size"),
        avg_dep_delay=("Dep_Delay", "mean"),
        avg_arr_delay=("Arr_Delay", "mean"),
        median_arr_delay=("Arr_Delay", "median"),
        pct_delayed_15=("is_delayed_dep", "mean"),
        avg_carrier_delay=("Delay_Carrier", "mean"),
        avg_weather_delay=("Delay_Weather", "mean"),
        avg_nas_delay=("Delay_NAS", "mean"),
        avg_late_aircraft=("Delay_LastAircraft", "mean"),
    ).round(2)
    agg["pct_delayed_15"] = (agg["pct_delayed_15"] * 100).round(1)
    agg = agg.sort_values("avg_arr_delay", ascending=False)
    logger.info("Airline performance table:\n%s", agg.to_string())
    return agg


def monthly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregated stats for seasonality analysis."""
    monthly = df.groupby("Month").agg(
        total_flights=("Arr_Delay", "size"),
        avg_dep_delay=("Dep_Delay", "mean"),
        avg_arr_delay=("Arr_Delay", "mean"),
        pct_delayed=("is_delayed_dep", "mean"),
        avg_carrier=("Delay_Carrier", "mean"),
        avg_weather=("Delay_Weather", "mean"),
        avg_nas=("Delay_NAS", "mean"),
        avg_late_aircraft=("Delay_LastAircraft", "mean"),
    ).round(2)
    monthly["pct_delayed"] = (monthly["pct_delayed"] * 100).round(1)
    logger.info("Monthly seasonality:\n%s", monthly.to_string())
    return monthly


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix for numeric features."""
    cols = [
        "Dep_Delay", "Arr_Delay", "Flight_Duration", "Aicraft_age",
        "Delay_Carrier", "Delay_Weather", "Delay_NAS",
        "Delay_Security", "Delay_LastAircraft",
        "weather_temp_avg", "weather_precip", "weather_snow",
        "weather_wind_speed", "weather_pressure",
    ]
    corr = df[cols].corr().round(3)

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Корреляционная матрица: задержки рейсов и метеопараметры")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png")
    plt.close()
    logger.info("Saved correlation_matrix.png")
    return corr


def plot_seasonality_decomposition(df: pd.DataFrame) -> None:
    """Weekly average delay time series."""
    weekly = df.groupby("WeekOfYear").agg(
        avg_dep_delay=("Dep_Delay", "mean"),
        avg_arr_delay=("Arr_Delay", "mean"),
        total_flights=("Arr_Delay", "size"),
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(weekly["WeekOfYear"], weekly["avg_dep_delay"], "o-", color="tab:red", markersize=4)
    axes[0].plot(weekly["WeekOfYear"], weekly["avg_arr_delay"], "s-", color="tab:blue", markersize=4)
    axes[0].set_ylabel("Средняя задержка (мин)")
    axes[0].set_title("Средние задержки рейсов по неделям (2023)")
    axes[0].legend(["Задержка вылета", "Задержка прибытия"])

    axes[1].bar(weekly["WeekOfYear"], weekly["total_flights"], color="steelblue", alpha=0.7)
    axes[1].set_xlabel("Неделя года")
    axes[1].set_ylabel("Количество рейсов")
    axes[1].set_title("Еженедельный объём рейсов")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "weekly_delay_trend.png")
    plt.close()
    logger.info("Saved weekly_delay_trend.png")


def plot_delay_causes_by_airline(df: pd.DataFrame) -> None:
    """Stacked bar: delay cause breakdown per airline."""
    causes = df.groupby("Airline")[
        ["Delay_Carrier", "Delay_Weather", "Delay_NAS", "Delay_Security", "Delay_LastAircraft"]
    ].mean()
    causes = causes.sort_values("Delay_Carrier", ascending=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    causes.plot(kind="barh", stacked=True, ax=ax,
                color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"])
    ax.set_xlabel("Средняя задержка (мин)")
    ax.set_title("Структура причин задержек по авиакомпаниям (2023)")
    ax.legend(["Авиакомпания", "Погода", "Система ОрВД", "Безопасность", "Позднее ВС"],
              bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "delay_causes_by_airline.png")
    plt.close()
    logger.info("Saved delay_causes_by_airline.png")


def hypothesis_test_weather(df: pd.DataFrame) -> dict:
    """Test: do rainy days have significantly higher delays?"""
    rainy = df[df["weather_precip"] > 5]["Dep_Delay"]
    dry = df[df["weather_precip"] == 0]["Dep_Delay"]

    t_stat, p_value = stats.ttest_ind(rainy, dry, equal_var=False)
    result = {
        "rainy_mean": round(rainy.mean(), 2),
        "dry_mean": round(dry.mean(), 2),
        "t_statistic": round(t_stat, 4),
        "p_value": p_value,
        "significant": p_value < 0.05,
    }
    logger.info("Weather hypothesis test: %s", result)
    return result


def run_statistical_analysis() -> dict:
    """Run all statistical analyses."""
    df = load_data()

    perf = airline_performance_table(df)
    perf.to_csv("data/processed/airline_performance.csv")

    monthly = monthly_seasonality(df)
    monthly.to_csv("data/processed/monthly_seasonality.csv")

    corr = correlation_analysis(df)
    plot_seasonality_decomposition(df)
    plot_delay_causes_by_airline(df)
    weather_test = hypothesis_test_weather(df)

    return {
        "airline_count": len(perf),
        "worst_airline": perf.index[0],
        "best_airline": perf.index[-1],
        "peak_month": monthly["avg_dep_delay"].idxmax(),
        "weather_test": weather_test,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_statistical_analysis()
    print("\n=== Statistical Analysis Summary ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
