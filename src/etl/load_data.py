"""ETL module: loading, cleaning, and merging aviation datasets."""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_flights(path: Path = RAW_DIR / "US_flights_2023.csv") -> pd.DataFrame:
    """Load main flights dataset with proper dtypes."""
    logger.info("Loading flights from %s", path)
    df = pd.read_csv(
        path,
        parse_dates=["FlightDate"],
        dtype={
            "Day_Of_Week": "int8",
            "Dep_Delay": "int32",
            "Arr_Delay": "int32",
            "Dep_Delay_Tag": "int8",
            "Flight_Duration": "int32",
            "Delay_Carrier": "int32",
            "Delay_Weather": "int32",
            "Delay_NAS": "int32",
            "Delay_Security": "int32",
            "Delay_LastAircraft": "int32",
            "Aicraft_age": "int16",
        },
    )
    logger.info("Loaded %d flights", len(df))
    return df


def load_cancelled(path: Path = RAW_DIR / "Cancelled_Diverted_2023.csv") -> pd.DataFrame:
    """Load cancelled/diverted flights dataset."""
    logger.info("Loading cancelled/diverted from %s", path)
    df = pd.read_csv(path, parse_dates=["FlightDate"])
    logger.info("Loaded %d cancelled/diverted records", len(df))
    return df


def load_weather(path: Path = RAW_DIR / "weather_meteo_by_airport.csv") -> pd.DataFrame:
    """Load weather dataset."""
    logger.info("Loading weather from %s", path)
    df = pd.read_csv(path, parse_dates=["time"])
    df.rename(columns={"time": "date", "airport_id": "airport"}, inplace=True)
    logger.info("Loaded %d weather records for %d airports", len(df), df["airport"].nunique())
    return df


def merge_flights_weather(
    flights: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """Merge flights with departure airport weather."""
    logger.info("Merging flights with weather data")
    weather_renamed = weather.rename(
        columns={
            "airport": "Dep_Airport",
            "date": "FlightDate",
            "tavg": "weather_temp_avg",
            "tmin": "weather_temp_min",
            "tmax": "weather_temp_max",
            "prcp": "weather_precip",
            "snow": "weather_snow",
            "wdir": "weather_wind_dir",
            "wspd": "weather_wind_speed",
            "pres": "weather_pressure",
        }
    )
    merged = flights.merge(
        weather_renamed,
        on=["Dep_Airport", "FlightDate"],
        how="left",
    )
    matched = merged["weather_temp_avg"].notna().sum()
    logger.info("Weather matched for %d / %d flights (%.1f%%)",
                matched, len(merged), 100 * matched / len(merged))
    return merged


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for analysis."""
    df["Month"] = df["FlightDate"].dt.month
    df["DayOfMonth"] = df["FlightDate"].dt.day
    df["WeekOfYear"] = df["FlightDate"].dt.isocalendar().week.astype(int)
    df["is_delayed_dep"] = (df["Dep_Delay"] > 15).astype("int8")
    df["is_delayed_arr"] = (df["Arr_Delay"] > 15).astype("int8")
    df["total_delay_causes"] = (
        df["Delay_Carrier"]
        + df["Delay_Weather"]
        + df["Delay_NAS"]
        + df["Delay_Security"]
        + df["Delay_LastAircraft"]
    )
    df["has_delay_cause"] = (df["total_delay_causes"] > 0).astype("int8")
    logger.info("Added derived features")
    return df


def run_etl() -> pd.DataFrame:
    """Run full ETL pipeline: load, merge, enrich, save."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    flights = load_flights()
    weather = load_weather()
    cancelled = load_cancelled()

    merged = merge_flights_weather(flights, weather)
    enriched = add_derived_features(merged)

    output_path = PROCESSED_DIR / "flights_enriched.parquet"
    enriched.to_parquet(output_path, index=False)
    logger.info("Saved enriched dataset to %s (%d rows)", output_path, len(enriched))

    cancelled_path = PROCESSED_DIR / "cancelled.parquet"
    cancelled.to_parquet(cancelled_path, index=False)
    logger.info("Saved cancelled dataset to %s", cancelled_path)

    return enriched


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_etl()
