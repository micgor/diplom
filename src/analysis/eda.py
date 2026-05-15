"""Exploratory Data Analysis: generate key visualizations for VKR."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.family": "DejaVu Sans",
})
sns.set_theme(style="whitegrid")


def load_data() -> pd.DataFrame:
    """Load enriched dataset."""
    df = pd.read_parquet("data/processed/flights_enriched.parquet")
    logger.info("Loaded %d rows for EDA", len(df))
    return df


def plot_flights_by_airline(df: pd.DataFrame) -> None:
    """Bar chart: number of flights per airline."""
    fig, ax = plt.subplots(figsize=(14, 7))
    counts = df["Airline"].value_counts()
    colors = sns.color_palette("viridis", len(counts))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.set_xlabel("Количество рейсов")
    ax.set_title("Количество рейсов по авиакомпаниям (2023)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(val + 10000, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "flights_by_airline.png")
    plt.close()
    logger.info("Saved flights_by_airline.png")


def plot_monthly_delays(df: pd.DataFrame) -> None:
    """Line chart: average delay by month."""
    monthly = df.groupby("Month").agg(
        avg_dep_delay=("Dep_Delay", "mean"),
        avg_arr_delay=("Arr_Delay", "mean"),
        pct_delayed=("is_delayed_dep", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(monthly["Month"], monthly["avg_dep_delay"], "o-", color="tab:red", label="Ср. задержка вылета (мин)")
    ax1.plot(monthly["Month"], monthly["avg_arr_delay"], "s-", color="tab:blue", label="Ср. задержка прибытия (мин)")
    ax1.set_xlabel("Месяц")
    ax1.set_ylabel("Средняя задержка (мин)")
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
                          "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"])
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.bar(monthly["Month"], monthly["pct_delayed"] * 100, alpha=0.2, color="gray", label="% задержанных >15 мин")
    ax2.set_ylabel("% рейсов с задержкой >15 мин")
    ax2.legend(loc="upper right")

    ax1.set_title("Сезонность задержек рейсов по месяцам (2023)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "monthly_delays.png")
    plt.close()
    logger.info("Saved monthly_delays.png")


def plot_delay_causes(df: pd.DataFrame) -> None:
    """Pie chart: breakdown of delay causes."""
    delayed = df[df["has_delay_cause"] == 1]
    causes = {
        "Авиакомпания": delayed["Delay_Carrier"].sum(),
        "Погода": delayed["Delay_Weather"].sum(),
        "Система ОрВД": delayed["Delay_NAS"].sum(),
        "Безопасность": delayed["Delay_Security"].sum(),
        "Позднее ВС": delayed["Delay_LastAircraft"].sum(),
    }
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    wedges, texts, autotexts = ax.pie(
        causes.values(), labels=causes.keys(), autopct="%1.1f%%",
        colors=colors, startangle=90, textprops={"fontsize": 12}
    )
    ax.set_title("Распределение причин задержек (суммарные минуты, 2023)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "delay_causes_pie.png")
    plt.close()
    logger.info("Saved delay_causes_pie.png")


def plot_delay_by_airline(df: pd.DataFrame) -> None:
    """Horizontal bar chart: average arrival delay by airline."""
    avg_delay = df.groupby("Airline")["Arr_Delay"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in avg_delay.values]
    ax.barh(avg_delay.index, avg_delay.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Средняя задержка прибытия (мин)")
    ax.set_title("Средняя задержка прибытия по авиакомпаниям (2023)")
    for i, (val, name) in enumerate(zip(avg_delay.values, avg_delay.index)):
        ax.text(val + 0.3 if val >= 0 else val - 0.3, i,
                f"{val:.1f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "delay_by_airline.png")
    plt.close()
    logger.info("Saved delay_by_airline.png")


def plot_delay_heatmap_dow_time(df: pd.DataFrame) -> None:
    """Heatmap: average delay by day of week and time of day."""
    pivot = df.pivot_table(
        values="Dep_Delay", index="DepTime_label", columns="Day_Of_Week", aggfunc="mean"
    )
    order = ["Morning", "Afternoon", "Evening", "Night"]
    pivot = pivot.reindex(order)
    pivot.index = ["Утро", "День", "Вечер", "Ночь"]
    pivot.columns = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Средняя задержка вылета (мин) по дням недели и времени суток")
    ax.set_ylabel("Время суток")
    ax.set_xlabel("День недели")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "heatmap_delay_dow_time.png")
    plt.close()
    logger.info("Saved heatmap_delay_dow_time.png")


def plot_weather_vs_delay(df: pd.DataFrame) -> None:
    """Scatter: weather features vs average delay."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    weather_features = [
        ("weather_wind_speed", "Скорость ветра (км/ч)"),
        ("weather_precip", "Осадки (мм)"),
        ("weather_snow", "Снежный покров (мм)"),
    ]
    for ax, (col, label) in zip(axes, weather_features):
        # Bin the weather feature and compute mean delay
        df_valid = df[df[col].notna() & (df[col] >= 0)].copy()
        df_valid["bin"] = pd.qcut(df_valid[col], q=20, duplicates="drop")
        agg = df_valid.groupby("bin", observed=True)["Dep_Delay"].mean().reset_index()
        agg["bin_mid"] = agg["bin"].apply(lambda x: x.mid)
        ax.scatter(agg["bin_mid"], agg["Dep_Delay"], alpha=0.7, s=50)
        z = np.polyfit(agg["bin_mid"].astype(float), agg["Dep_Delay"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(agg["bin_mid"].min(), agg["bin_mid"].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel("Ср. задержка вылета (мин)")
        ax.set_title(f"Задержка vs {label}")

    plt.suptitle("Влияние погоды на задержки рейсов (2023)", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "weather_vs_delay.png")
    plt.close()
    logger.info("Saved weather_vs_delay.png")


def plot_delay_distribution(df: pd.DataFrame) -> None:
    """Histogram: distribution of departure delays."""
    fig, ax = plt.subplots(figsize=(12, 6))
    data = df["Dep_Delay"].clip(-60, 180)
    ax.hist(data, bins=100, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="green", linestyle="--", linewidth=1.5, label="Вовремя")
    ax.axvline(15, color="red", linestyle="--", linewidth=1.5, label="Задержка (>15 мин)")
    ax.set_xlabel("Задержка вылета (мин)")
    ax.set_ylabel("Количество рейсов")
    ax.set_title("Распределение задержек вылета (диапазон [-60, 180] мин)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "delay_distribution.png")
    plt.close()
    logger.info("Saved delay_distribution.png")


def plot_top_routes_delayed(df: pd.DataFrame) -> None:
    """Top 15 routes with highest average delay."""
    df["route"] = df["Dep_Airport"] + " → " + df["Arr_Airport"]
    route_stats = df.groupby("route").agg(
        avg_delay=("Arr_Delay", "mean"),
        count=("Arr_Delay", "size"),
    )
    # Only routes with 1000+ flights
    route_stats = route_stats[route_stats["count"] >= 1000].sort_values("avg_delay", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("Reds_r", len(route_stats))
    ax.barh(route_stats.index[::-1], route_stats["avg_delay"].values[::-1], color=colors[::-1])
    ax.set_xlabel("Средняя задержка прибытия (мин)")
    ax.set_title("Топ-15 маршрутов с наибольшими задержками (мин. 1000 рейсов, 2023)")
    for i, (val, name) in enumerate(zip(route_stats["avg_delay"].values[::-1], route_stats.index[::-1])):
        ax.text(val + 0.2, i, f"{val:.1f} мин", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "top_delayed_routes.png")
    plt.close()
    logger.info("Saved top_delayed_routes.png")


def run_eda() -> None:
    """Run all EDA visualizations."""
    df = load_data()

    plot_flights_by_airline(df)
    plot_monthly_delays(df)
    plot_delay_causes(df)
    plot_delay_by_airline(df)
    plot_delay_heatmap_dow_time(df)
    plot_weather_vs_delay(df)
    plot_delay_distribution(df)
    plot_top_routes_delayed(df)

    logger.info("EDA complete: %d figures saved to %s", len(list(FIGURES_DIR.glob("*.png"))), FIGURES_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_eda()