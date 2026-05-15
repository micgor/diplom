"""Streamlit dashboard: Аналитика авиакомпаний.

Интерактивный дашборд с четырьмя страницами:
- Обзор: KPI, рейтинг авиакомпаний, тепловая карта времени, сезонность
- Авиакомпании: структура причин задержек, таблица показателей
- Маршруты: поиск маршрута и анализ причин задержек
- Гео-карта: интерактивная карта аэропортов с индикацией задержек
"""

import airportsdata
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

DATA_DIR = Path("data/processed")

MONTH_NAMES = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
}
MONTH_SHORT = {
    1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр", 5: "Май", 6: "Июн",
    7: "Июл", 8: "Авг", 9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек",
}
DAY_NAMES = {1: "Пн", 2: "Вт", 3: "Ср", 4: "Чт", 5: "Пт", 6: "Сб", 7: "Вс"}
TIME_LABELS = {"Morning": "Утро", "Afternoon": "День", "Evening": "Вечер", "Night": "Ночь"}

CAUSE_LABELS = {
    "Delay_Carrier": "Авиакомпания",
    "Delay_Weather": "Погода",
    "Delay_NAS": "ОрВД (диспетчерская)",
    "Delay_Security": "Безопасность",
    "Delay_LastAircraft": "Опоздание самолёта",
}

CAUSE_DESCRIPTIONS = {
    "Авиакомпания": "Задержки по вине перевозчика: техническое обслуживание, "
                    "заправка, погрузка багажа, проблемы с экипажем, "
                    "оформление пассажиров.",
    "Погода": "Задержки из-за неблагоприятных метеоусловий в аэропорту "
              "вылета или прибытия (сильный ветер, гроза, снегопад, низкая "
              "видимость), делающих полёт небезопасным.",
    "ОрВД (диспетчерская)": "Задержки от системы организации воздушного "
                            "движения (National Airspace System): очереди "
                            "на взлёт/посадку в загруженных хабах, "
                            "ограничения по слотам, перегрузка воздушного "
                            "пространства, неблагоприятная погода на "
                            "маршруте.",
    "Безопасность": "Задержки из-за инцидентов на досмотре, эвакуации "
                    "терминалов, проверок служб безопасности, обнаружения "
                    "оставленного багажа.",
    "Опоздание самолёта": "Каскадная задержка: воздушное судно, "
                          "выполняющее данный рейс, прибыло из предыдущего "
                          "рейса с опозданием. На опоздание самолёта "
                          "приходится наибольшая доля совокупных задержек "
                          "в отрасли.",
}


def render_cause_legend():
    """Раскрывающаяся легенда с описанием категорий причин."""
    with st.expander("О категориях причин задержек"):
        for name, desc in CAUSE_DESCRIPTIONS.items():
            st.markdown(f"**{name}** — {desc}")


@st.cache_data
def load_data():
    flights = pd.read_parquet(DATA_DIR / "flights_enriched.parquet")
    cancelled = pd.read_parquet(DATA_DIR / "cancelled.parquet")
    return flights, cancelled


@st.cache_data
def load_performance():
    return pd.read_csv(DATA_DIR / "airline_performance.csv", index_col=0)


@st.cache_data
def get_route_map(flights: pd.DataFrame) -> dict:
    return flights.groupby("Dep_Airport")["Arr_Airport"].apply(
        lambda x: sorted(x.unique())
    ).to_dict()


@st.cache_data
def get_airport_labels(flights: pd.DataFrame) -> dict:
    dep = flights[["Dep_Airport", "Dep_CityName"]].drop_duplicates()
    dep.columns = ["code", "city"]
    arr = flights[["Arr_Airport", "Arr_CityName"]].drop_duplicates()
    arr.columns = ["code", "city"]
    combined = pd.concat([dep, arr]).drop_duplicates(subset="code")
    return dict(zip(combined["code"], combined["code"] + " — " + combined["city"]))


@st.cache_data
def get_airport_coords() -> pd.DataFrame:
    """Load airport coordinates from airportsdata."""
    data = airportsdata.load("IATA")
    rows = [
        {"code": code, "name": info["name"], "lat": info["lat"], "lon": info["lon"]}
        for code, info in data.items()
    ]
    return pd.DataFrame(rows)


def render_kpi_row(df: pd.DataFrame) -> None:
    cols = st.columns(5)
    cols[0].metric("Всего рейсов", f"{len(df):,.0f}")
    cols[1].metric("Ср. задержка прибытия", f"{df['Arr_Delay'].mean():.1f} мин")
    cols[2].metric("Пунктуальность", f"{(df['is_delayed_arr'] == 0).mean() * 100:.1f}%")
    cols[3].metric("Авиакомпании", df["Airline"].nunique())
    cols[4].metric("Аэропорты", df["Dep_Airport"].nunique())


# ============================================================
#  СТРАНИЦА 1: ОБЗОР
# ============================================================
def page_overview(df: pd.DataFrame) -> None:
    st.header("Обзор")
    render_kpi_row(df)
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Рейтинг пунктуальности авиакомпаний")
        ranking = df.groupby("Airline").agg(
            on_time_pct=("is_delayed_arr", lambda x: (1 - x.mean()) * 100),
            avg_delay=("Arr_Delay", "mean"),
        ).sort_values("on_time_pct", ascending=False).round(1)

        fig = px.bar(
            ranking.reset_index(), x="on_time_pct", y="Airline",
            orientation="h", color="avg_delay",
            color_continuous_scale="RdYlGn_r",
            labels={"on_time_pct": "Пунктуальность, %",
                    "avg_delay": "Ср. задержка (мин)",
                    "Airline": "Авиакомпания"},
            title="Пунктуальность авиакомпаний",
        )
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Лучшее время для вылета")
        time_stats = df.groupby(["DepTime_label", "Day_Of_Week"]).agg(
            avg_delay=("Dep_Delay", "mean"),
        ).reset_index()
        time_stats["Day_Of_Week"] = time_stats["Day_Of_Week"].map(DAY_NAMES)
        time_stats["DepTime_label"] = time_stats["DepTime_label"].map(TIME_LABELS)
        pivot = time_stats.pivot(index="DepTime_label",
                                  columns="Day_Of_Week", values="avg_delay")
        time_order = ["Утро", "День", "Вечер", "Ночь"]
        day_order = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        pivot = pivot.reindex(time_order)[day_order]

        fig = px.imshow(
            pivot, color_continuous_scale="RdYlGn_r",
            labels=dict(x="День недели", y="Время суток",
                        color="Ср. задержка (мин)"),
            title="Средняя задержка по дням и времени суток",
            aspect="auto",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Сезонность задержек по месяцам")
    monthly = df.groupby("Month").agg(
        avg_dep=("Dep_Delay", "mean"),
        avg_arr=("Arr_Delay", "mean"),
        flights=("Arr_Delay", "size"),
    ).reset_index()
    monthly["Месяц"] = monthly["Month"].map(MONTH_SHORT)
    fig = px.line(
        monthly, x="Месяц", y=["avg_dep", "avg_arr"],
        labels={"value": "Задержка (мин)", "Месяц": "Месяц",
                "variable": "Тип задержки"},
        title="Средняя задержка вылета и прибытия по месяцам",
        markers=True,
    )
    fig.for_each_trace(lambda t: t.update(
        name="Вылет" if t.name == "avg_dep" else "Прибытие"
    ))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
#  СТРАНИЦА 2: АВИАКОМПАНИИ
# ============================================================
def page_airlines(df: pd.DataFrame) -> None:
    st.header("Авиакомпании")
    render_kpi_row(df)
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Распределение причин задержек")
        delayed = df[df["has_delay_cause"] == 1]
        causes = pd.DataFrame({
            "Причина": list(CAUSE_LABELS.values()),
            "Минуты": [
                delayed["Delay_Carrier"].sum(),
                delayed["Delay_Weather"].sum(),
                delayed["Delay_NAS"].sum(),
                delayed["Delay_Security"].sum(),
                delayed["Delay_LastAircraft"].sum(),
            ]
        })
        fig = px.pie(
            causes, values="Минуты", names="Причина",
            title="Доля причин в суммарных задержках",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Структура задержек по авиакомпаниям")
        cause_by_airline = df.groupby("Airline")[
            ["Delay_Carrier", "Delay_Weather", "Delay_NAS",
             "Delay_Security", "Delay_LastAircraft"]
        ].mean().round(2)
        cause_by_airline = cause_by_airline.sort_values(
            "Delay_Carrier", ascending=True)

        fig = px.bar(
            cause_by_airline.reset_index(), x="Airline",
            y=list(CAUSE_LABELS.keys()),
            title="Средние задержки по категориям причин",
            barmode="stack",
            labels={"Airline": "Авиакомпания", "value": "Задержка (мин)",
                    "variable": "Причина"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.for_each_trace(lambda t: t.update(
            name=CAUSE_LABELS.get(t.name, t.name)))
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

    render_cause_legend()

    st.divider()
    st.subheader("Таблица показателей авиакомпаний")
    perf = load_performance()
    perf_display = perf.copy()
    perf_display.columns = [
        "Всего рейсов", "Ср. задержка вылета", "Ср. задержка прибытия",
        "Медиана задержки", "% задержанных >15 мин",
        "Задержка (вина авиакомп.)", "Задержка (погода)",
        "Задержка (ОрВД)", "Задержка (опоздание ВС)",
    ]
    st.dataframe(
        perf_display.style.format({
            "Ср. задержка вылета": "{:.1f}",
            "Ср. задержка прибытия": "{:.1f}",
            "% задержанных >15 мин": "{:.1f}%",
        }),
        use_container_width=True,
    )


# ============================================================
#  СТРАНИЦА 3: МАРШРУТЫ
# ============================================================
def page_routes(df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    st.header("Анализ маршрутов")

    route_map = get_route_map(full_df)
    airport_labels = get_airport_labels(full_df)
    dep_airports = sorted(route_map.keys())

    col_a, col_b = st.columns(2)
    with col_a:
        default_origin = dep_airports.index("JFK") if "JFK" in dep_airports else 0
        origin = st.selectbox(
            "Аэропорт вылета", dep_airports, index=default_origin,
            format_func=lambda c: airport_labels.get(c, c),
        )
    available_dests = route_map.get(origin, [])
    with col_b:
        default_dest_idx = available_dests.index("LAX") if "LAX" in available_dests else 0
        dest = st.selectbox(
            "Аэропорт прибытия", available_dests, index=default_dest_idx,
            format_func=lambda c: airport_labels.get(c, c),
        )

    route_df = df[(df["Dep_Airport"] == origin) & (df["Arr_Airport"] == dest)]

    if len(route_df) == 0:
        st.info(f"Нет рейсов {origin} → {dest} за выбранный период")
        return

    st.divider()
    cols = st.columns(4)
    cols[0].metric("Рейсов", f"{len(route_df):,}")
    cols[1].metric("Ср. задержка прибытия", f"{route_df['Arr_Delay'].mean():.1f} мин")
    cols[2].metric("Пунктуальность",
                   f"{(1 - route_df['is_delayed_arr'].mean()) * 100:.1f}%")
    cols[3].metric("Авиакомпаний", route_df["Airline"].nunique())

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Авиакомпании на маршруте {origin} → {dest}")
        airline_route = route_df.groupby("Airline").agg(
            avg_delay=("Arr_Delay", "mean"),
            flights=("Arr_Delay", "size"),
            on_time_pct=("is_delayed_arr", lambda x: (1 - x.mean()) * 100),
        ).sort_values("avg_delay").round(2)

        fig = px.bar(
            airline_route.reset_index(), x="Airline", y="avg_delay",
            color="avg_delay", color_continuous_scale="RdYlGn_r",
            labels={"Airline": "Авиакомпания",
                    "avg_delay": "Ср. задержка (мин)"},
            title="Средняя задержка по авиакомпаниям",
            hover_data={"flights": True, "on_time_pct": ":.1f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Причины задержек на маршруте")
        delayed_route = route_df[route_df["has_delay_cause"] == 1]
        if len(delayed_route) > 0:
            causes = pd.DataFrame({
                "Причина": list(CAUSE_LABELS.values()),
                "Минуты": [
                    delayed_route["Delay_Carrier"].sum(),
                    delayed_route["Delay_Weather"].sum(),
                    delayed_route["Delay_NAS"].sum(),
                    delayed_route["Delay_Security"].sum(),
                    delayed_route["Delay_LastAircraft"].sum(),
                ],
            })
            fig = px.pie(
                causes, values="Минуты", names="Причина",
                title="Доля причин в суммарных задержках",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("На маршруте нет задержек с зарегистрированной причиной")

    render_cause_legend()

    st.divider()
    st.subheader("Сравнение авиакомпаний на маршруте по месяцам")
    airline_monthly = route_df.groupby(["Month", "Airline"]).agg(
        avg_delay=("Arr_Delay", "mean"),
        flights=("Arr_Delay", "size"),
    ).reset_index()
    airline_monthly = airline_monthly[airline_monthly["flights"] >= 5]
    if len(airline_monthly) > 0:
        airline_monthly["Месяц"] = airline_monthly["Month"].map(MONTH_SHORT)
        month_order = [MONTH_SHORT[m] for m in sorted(airline_monthly["Month"])]
        fig = px.line(
            airline_monthly, x="Месяц", y="avg_delay", color="Airline",
            labels={"avg_delay": "Ср. задержка прибытия (мин)",
                    "Месяц": "Месяц", "Airline": "Авиакомпания"},
            title=f"Помесячная задержка по авиакомпаниям: {origin} → {dest}",
            markers=True,
            category_orders={"Месяц": month_order},
            hover_data={"flights": True},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Недостаточно данных для сравнения авиакомпаний по месяцам")



# ============================================================
#  СТРАНИЦА 4: ГЕО-КАРТА
# ============================================================
def page_geomap(df: pd.DataFrame) -> None:
    st.header("Гео-карта проблемных зон")
    st.caption("Карта аэропортов США с индикацией уровня задержек. "
               "Размер точки — количество рейсов, цвет — средняя задержка прибытия.")

    metric = st.radio(
        "Метрика для цветовой индикации:",
        ["Средняя задержка прибытия", "% задержанных рейсов",
         "Средняя задержка вылета"],
        horizontal=True,
    )
    min_flights = st.slider(
        "Минимум рейсов в аэропорту для отображения", 100, 50000, 1000, 100)

    airport_stats = df.groupby("Dep_Airport").agg(
        avg_arr_delay=("Arr_Delay", "mean"),
        avg_dep_delay=("Dep_Delay", "mean"),
        pct_delayed=("is_delayed_arr", lambda x: x.mean() * 100),
        flights=("Arr_Delay", "size"),
        city=("Dep_CityName", "first"),
    ).round(2).reset_index()
    airport_stats = airport_stats[airport_stats["flights"] >= min_flights]

    coords = get_airport_coords()
    airport_stats = airport_stats.merge(
        coords, left_on="Dep_Airport", right_on="code", how="inner")

    if len(airport_stats) == 0:
        st.warning("Нет аэропортов, удовлетворяющих условию фильтра")
        return

    metric_map = {
        "Средняя задержка прибытия": ("avg_arr_delay", "Ср. задержка прибытия (мин)"),
        "% задержанных рейсов": ("pct_delayed", "% задержанных рейсов"),
        "Средняя задержка вылета": ("avg_dep_delay", "Ср. задержка вылета (мин)"),
    }
    color_col, color_label = metric_map[metric]

    fig = px.scatter_geo(
        airport_stats,
        lat="lat", lon="lon",
        size="flights",
        color=color_col,
        color_continuous_scale="RdYlGn_r",
        hover_name="Dep_Airport",
        hover_data={
            "city": True, "flights": ":,",
            "avg_arr_delay": ":.1f", "pct_delayed": ":.1f",
            "lat": False, "lon": False,
        },
        scope="usa",
        size_max=40,
        labels={color_col: color_label, "flights": "Рейсов",
                "avg_arr_delay": "Ср. задержка (мин)",
                "pct_delayed": "% задержанных",
                "city": "Город"},
        title=f"Аэропорты США: {color_label}",
    )
    fig.update_geos(
        showcoastlines=True,
        showland=True,
        showlakes=True,
        showsubunits=True,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "doubleClick": "reset",
            "displaylogo": False,
            "modeBarButtonsToRemove": ["pan2d", "select2d", "lasso2d"],
        },
    )

    st.divider()
    st.subheader("Топ-10 самых проблемных аэропортов")
    top = airport_stats.sort_values(color_col, ascending=False).head(10)[
        ["Dep_Airport", "city", "flights", "avg_arr_delay", "pct_delayed"]
    ]
    top.columns = ["Код", "Город", "Рейсов", "Ср. задержка (мин)",
                   "% задержанных"]
    st.dataframe(
        top.style.format({
            "Рейсов": "{:,.0f}",
            "Ср. задержка (мин)": "{:.1f}",
            "% задержанных": "{:.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )


# ============================================================
#  MAIN
# ============================================================
def main():
    st.set_page_config(
        page_title="Аналитика авиакомпаний",
        page_icon="✈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Дашборд аналитики авиакомпаний")
    st.caption("Модель анализа больших данных для построения дашбордов — ВКР")

    df, cancelled = load_data()

    with st.sidebar:
        st.header("Навигация")
        page = st.radio(
            "Раздел",
            ["Обзор", "Авиакомпании", "Маршруты", "Гео-карта"],
            index=0,
        )

        st.divider()
        st.subheader("Период")
        all_months = list(range(1, 13))
        select_all = st.checkbox("Весь год", value=True)
        if select_all:
            selected_months = all_months
        else:
            selected_months = []
            cols_m = st.columns(2)
            for i, m in enumerate(all_months):
                with cols_m[i % 2]:
                    if st.checkbox(MONTH_NAMES[m], value=True, key=f"m_{m}"):
                        selected_months.append(m)
            if not selected_months:
                selected_months = all_months
                st.warning("Выбран весь год")

        st.divider()
        st.caption(f"Датасет: {len(df):,} рейсов | "
                   f"{cancelled['Cancelled'].sum():,.0f} отменённых")
        st.caption("Источник: US DOT BTS, 2023")

    filtered = df[df["Month"].isin(selected_months)]
    st.sidebar.metric("Отфильтровано рейсов", f"{len(filtered):,}")

    if page == "Обзор":
        page_overview(filtered)
    elif page == "Авиакомпании":
        page_airlines(filtered)
    elif page == "Маршруты":
        page_routes(filtered, df)
    elif page == "Гео-карта":
        page_geomap(filtered)


if __name__ == "__main__":
    main()
