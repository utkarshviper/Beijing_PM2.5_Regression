"""
app.py - PEKING AIR · 北京空气
An IQAir-inspired, voxel-style ("Minecraft") air-quality analytics dashboard for Beijing,
rendered in the Imperial Peking palette.

Data  : Beijing Multi-Site Air Quality (PRSA 2013-2017), UCI ML Repository (DOI 10.24432/C5RK5G),
        condensed to station-day means by src/build_app_data.py.
Course: Introduction to Machine Learning (BS Analytics, Sem V), TISS School of Analytics.
Layout inspired by IQAir's city pages; this project is not affiliated with IQAir.
"""

from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="Peking Air · 北京空气", page_icon="🏯", layout="wide")

BASE = Path(__file__).parent
DATA_DIR = BASE / "data" / "processed"
FIG_DIR = BASE / "reports" / "figures"
W = {"width": "stretch"}  # full-width charts (Streamlit >= 1.49)

# ---------------------------------------------------------------------------
# Imperial Peking palette
# ---------------------------------------------------------------------------
P = {
    "bg": "#FAF8F2", "paper": "#F1EBDD", "ink": "#2B231D", "red": "#9B1D20",
    "gold": "#DDA12A", "blue": "#1F4E79", "jade": "#2D6A4F", "grey": "#6C757D",
}

# Station coordinates (approx.) and spatial zone
STATIONS = {
    "Aotizhongxin": (39.982, 116.397, "Urban"),
    "Changping": (40.217, 116.230, "Rural"),
    "Dingling": (40.292, 116.220, "Rural"),
    "Dongsi": (39.929, 116.417, "Urban"),
    "Guanyuan": (39.929, 116.339, "Urban"),
    "Gucheng": (39.914, 116.184, "Urban"),
    "Huairou": (40.328, 116.628, "Rural"),
    "Nongzhanguan": (39.937, 116.461, "Urban"),
    "Shunyi": (40.127, 116.655, "Rural"),
    "Tiantan": (39.886, 116.407, "Urban"),
    "Wanliu": (39.987, 116.287, "Urban"),
    "Wanshouxigong": (39.878, 116.352, "Urban"),
}
FORBIDDEN_CITY = (39.9163, 116.3972)

WD_MAP = {
    "N": "NW", "NNW": "NW", "NW": "NW", "WNW": "NW",
    "ENE": "NE", "NE": "NE", "NNE": "NE",
    "W": "SW", "WSW": "SW", "SW": "SW", "SSW": "SW",
    "S": "SE", "SSE": "SE", "SE": "SE", "ESE": "SE", "E": "SE",
}

# US EPA PM2.5 AQI breakpoints (2024 revision): (C_lo, C_hi, I_lo, I_hi)
PM_BREAKS = [
    (0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500),
]

# AQI bands recoloured in Peking tones (jade -> imperial gold -> persimmon -> vermilion -> Forbidden purple -> lacquer)
BANDS = [
    {"lo": 0, "hi": 50, "label": "Good", "color": "#3A8D5C", "text": "#FFFFFF",
     "advice": [("🏃", "Enjoy outdoor activities"), ("🪟", "Open windows to bring in fresh air")]},
    {"lo": 51, "hi": 100, "label": "Moderate", "color": "#DDA12A", "text": "#2B231D",
     "advice": [("🚴", "Sensitive groups should limit long outdoor exertion"), ("🪟", "Ventilate in short bursts")]},
    {"lo": 101, "hi": 150, "label": "Unhealthy for Sensitive Groups", "color": "#D9772B", "text": "#FFFFFF",
     "advice": [("😷", "Sensitive groups should wear a mask outdoors"), ("🏠", "Reduce outdoor exercise"),
                ("💨", "Run an air purifier")]},
    {"lo": 151, "hi": 200, "label": "Unhealthy", "color": "#9B1D20", "text": "#FFFFFF",
     "advice": [("😷", "Wear an N95 mask outdoors"), ("🚫", "Avoid outdoor exercise"),
                ("🪟", "Close windows"), ("💨", "Run an air purifier")]},
    {"lo": 201, "hi": 300, "label": "Very Unhealthy", "color": "#5B2A6E", "text": "#FFFFFF",
     "advice": [("😷", "Everyone should wear an N95 mask"), ("🏠", "Stay indoors"),
                ("🪟", "Seal windows"), ("💨", "Purifier on high")]},
    {"lo": 301, "hi": 500, "label": "Hazardous", "color": "#3B1414", "text": "#FFFFFF",
     "advice": [("🚨", "Health emergency: avoid all outdoor activity"), ("🏠", "Stay indoors with purified air"),
                ("😷", "N95 mask if going out is unavoidable")]},
]


def pm_to_aqi(pm):
    if pm is None or pd.isna(pm):
        return np.nan
    c = np.floor(float(pm) * 10) / 10
    for c_lo, c_hi, i_lo, i_hi in PM_BREAKS:
        if c <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (max(c, c_lo) - c_lo) + i_lo)
    return 500  # beyond the index


def band_of(aqi):
    for b in BANDS:
        if aqi <= b["hi"]:
            return b
    return BANDS[-1]


def hex_rgb(h, alpha=235):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]


def rel_humidity(t, td):
    """Magnus approximation of relative humidity (%) from temperature and dew point (°C)."""
    a, b = 17.625, 243.04
    return float(np.clip(100 * np.exp(a * td / (b + td)) / np.exp(a * t / (b + t)), 0, 100))


def season_of(month):
    return {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer"}.get(month, "Autumn")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    daily = pd.read_csv(DATA_DIR / "beijing_daily.csv", parse_dates=["date"])
    profile = pd.read_csv(DATA_DIR / "beijing_diurnal_profile.csv")
    daily["zone"] = daily["station"].map(lambda s: STATIONS[s][2])
    daily["lat"] = daily["station"].map(lambda s: STATIONS[s][0])
    daily["lon"] = daily["station"].map(lambda s: STATIONS[s][1])
    daily["aqi"] = daily["PM2.5"].apply(pm_to_aqi)
    daily["season"] = daily["date"].dt.month.map(season_of)
    daily["wd_sector"] = daily["wd"].map(WD_MAP).fillna("CV")
    return daily, profile


# ---------------------------------------------------------------------------
# Models (syllabus-aligned: Ridge + Decision Tree), trained on station-day means
# ---------------------------------------------------------------------------
NUM_FEATS = ["TEMP", "PRES", "WSPM", "RAIN", "spread"]


def make_X(df):
    X = pd.DataFrame({c: df[c].astype(float) for c in NUM_FEATS})
    X["is_urban"] = (df["zone"] == "Urban").astype(int)
    for s in ["Spring", "Summer", "Autumn"]:
        X[f"season_{s}"] = (df["season"] == s).astype(int)
    for w in ["NE", "NW", "SE", "SW"]:
        X[f"wd_{w}"] = (df["wd_sector"] == w).astype(int)
    return X


@st.cache_resource
def train_models(daily):
    d = daily.dropna(subset=["PM2.5", "TEMP", "DEWP", "PRES", "WSPM", "RAIN"]).copy()
    d["spread"] = d["TEMP"] - d["DEWP"]
    X, y = make_X(d), d["PM2.5"].values
    tr = (d["date"] < "2016-03-01").values  # same chronological split as the notebook

    scaler = StandardScaler().fit(X.loc[tr, NUM_FEATS])

    def scaled(frame):
        out = frame.copy()
        out[NUM_FEATS] = scaler.transform(frame[NUM_FEATS])
        return out

    ridge = Ridge(alpha=1.0).fit(scaled(X[tr]), y[tr])
    tree = DecisionTreeRegressor(max_depth=7, min_samples_leaf=25, random_state=42).fit(X[tr], y[tr])

    metrics = {}
    for name, model, Xte in [("Ridge Regression", ridge, scaled(X[~tr])), ("Decision Tree", tree, X[~tr])]:
        p = model.predict(Xte)
        metrics[name] = {"R²": r2_score(y[~tr], p), "MAE": mean_absolute_error(y[~tr], p)}
    return {"ridge": ridge, "tree": tree, "scaled": scaled, "metrics": metrics,
            "n_train": int(tr.sum()), "n_test": int((~tr).sum())}


def predict(models, which, frame):
    X = make_X(frame)
    if which == "Ridge Regression":
        return np.clip(models["ridge"].predict(models["scaled"](X)), 1, None)
    return np.clip(models["tree"].predict(X), 1, None)


# ---------------------------------------------------------------------------
# Styling: pixel fonts, block shadows, Forbidden City roof + wall strips
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');
.stApp {{
  background-color: {P['bg']};
  background-image:
    linear-gradient(rgba(43,35,29,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(43,35,29,.035) 1px, transparent 1px);
  background-size: 16px 16px;
}}
h1, h2, h3, h4 {{ font-family: 'Press Start 2P', monospace !important; color: {P['red']} !important;
  letter-spacing: .5px; line-height: 1.5 !important; }}
h3 {{ font-size: 15px !important; }}
.vt, .stMarkdown p, .stMarkdown li {{ font-family: 'VT323', monospace; font-size: 21px; }}
section[data-testid="stSidebar"] {{ background: {P['paper']}; border-right: 4px solid {P['ink']}; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'Press Start 2P', monospace; font-size: 11px;
  border: 3px solid {P['ink']}; background: {P['paper']}; margin-right: 6px; padding: 10px 14px; }}
.stTabs [aria-selected="true"] {{ background: {P['red']} !important; color: #fff !important; }}
.roof {{ height: 14px; background: repeating-linear-gradient(90deg, {P['gold']} 0 22px, #B9831C 22px 26px);
  border: 3px solid {P['ink']}; border-bottom: none; }}
.wall {{ height: 18px; border: 3px solid {P['ink']}; border-top: none; margin-bottom: 18px;
  background:
    repeating-linear-gradient(90deg, transparent 0 30px, rgba(0,0,0,.25) 30px 32px),
    linear-gradient({P['red']}, #7E1618); }}
.topbar {{ display: flex; align-items: center; gap: 18px; padding: 14px 18px; background: #fff;
  border-left: 3px solid {P['ink']}; border-right: 3px solid {P['ink']}; }}
.seal {{ background: {P['red']}; color: #fff; font-weight: 700; font-size: 20px; line-height: 1.15;
  padding: 8px 10px; border: 3px solid {P['ink']}; box-shadow: 4px 4px 0 {P['ink']}; text-align: center; }}
.brand {{ font-family: 'Press Start 2P', monospace; font-size: 24px; color: {P['red']}; }}
.tag {{ font-family: 'VT323', monospace; font-size: 22px; color: {P['grey']}; }}
.block {{ background: #fff; border: 4px solid {P['ink']}; box-shadow: 6px 6px 0 {P['ink']};
  padding: 16px 18px; margin-bottom: 20px; }}
.px {{ font-family: 'Press Start 2P', monospace; }}
.hero-num {{ font-family: 'Press Start 2P', monospace; font-size: 64px; line-height: 1.1; margin: 10px 0; }}
.scale {{ display: flex; gap: 4px; margin-top: 12px; }}
.scale div {{ flex: 1; height: 16px; border: 2px solid {P['ink']}; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
.cell {{ border: 3px solid {P['ink']}; background: {P['paper']}; padding: 8px 10px; }}
.cell .k {{ font-family: 'Press Start 2P', monospace; font-size: 9px; color: {P['grey']}; }}
.cell .v {{ font-family: 'VT323', monospace; font-size: 28px; color: {P['ink']}; }}
.rank {{ width: 100%; border-collapse: collapse; font-family: 'VT323', monospace; font-size: 22px; }}
.rank th {{ font-family: 'Press Start 2P', monospace; font-size: 10px; text-align: left;
  background: {P['blue']}; color: #fff; padding: 8px; }}
.rank td {{ padding: 6px 8px; border-bottom: 2px dashed {P['paper']}; }}
.badge {{ display: inline-block; min-width: 54px; text-align: center; padding: 2px 6px;
  border: 2px solid {P['ink']}; font-family: 'Press Start 2P', monospace; font-size: 11px; }}
.advice {{ display: flex; align-items: center; gap: 10px; border: 3px solid {P['ink']};
  background: {P['paper']}; padding: 6px 10px; margin-bottom: 8px; font-family: 'VT323', monospace; font-size: 21px; }}
.advice span {{ font-size: 24px; }}
div[data-testid="stMetric"] {{ background: #fff; border: 3px solid {P['ink']}; box-shadow: 4px 4px 0 {P['ink']}; padding: 10px; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="roof"></div>
<div class="topbar">
  <div class="seal">北京<br>空气</div>
  <div>
    <div class="brand">PEKING AIR</div>
    <div class="tag">Beijing air-quality explorer · 12 monitoring stations · Mar 2013 – Feb 2017 · PRSA / UCI #501</div>
  </div>
</div>
<div class="wall"></div>
""", unsafe_allow_html=True)

try:
    daily, profile = load_data()
except FileNotFoundError:
    st.error("Processed data not found. Run `python src/build_app_data.py` once to create data/processed/.")
    st.stop()

city = daily.groupby("date").agg(pm=("PM2.5", "mean")).dropna()
worst_day, best_day = city["pm"].idxmax().date(), city["pm"].idxmin().date()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🧭 Controls")
episodes = {
    "Pick a date": None,
    f"☠️ Worst day on record ({worst_day})": worst_day,
    f"🌿 Cleanest day on record ({best_day})": best_day,
    "🧧 Spring Festival eve 2016 (2016-02-07)": date(2016, 2, 7),
    "🎄 Winter heating peak (2015-12-25)": date(2015, 12, 25),
}
episode = st.sidebar.selectbox("Jump to an episode", list(episodes))
picked = st.sidebar.date_input("Date", value=date(2015, 12, 25),
                               min_value=date(2013, 3, 1), max_value=date(2017, 2, 28))
sel_date = pd.Timestamp(episodes[episode] or picked)
place = st.sidebar.selectbox("Location", ["Beijing (city average)"] + list(STATIONS))
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='vt'>AQI uses the US EPA PM2.5 scale (2024 breakpoints) on <b>daily mean</b> PM2.5. "
    "Colours are Peking tones: jade → imperial gold → persimmon → vermilion → Forbidden purple → lacquer.</div>",
    unsafe_allow_html=True)

day = daily[daily["date"] == sel_date].dropna(subset=["PM2.5"]).copy()
if day.empty:
    st.warning("No valid readings for this date. Pick another day.")
    st.stop()

if place.startswith("Beijing"):
    row = day[["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "DEWP", "PRES", "WSPM", "RAIN"]].mean()
    wd_txt = day["wd"].mode().iat[0] if day["wd"].notna().any() else "—"
else:
    sub = day[day["station"] == place]
    if sub.empty:
        st.warning(f"{place} has no PM2.5 reading on this date; showing city average instead.")
        sub = day
    row = sub.iloc[0] if len(sub) == 1 else sub.mean(numeric_only=True)
    wd_txt = sub["wd"].iat[0] if len(sub) == 1 else "—"

aqi = pm_to_aqi(row["PM2.5"])
band = band_of(aqi)

tab_map, tab_hist, tab_sim, tab_about = st.tabs(["🗺️ LIVE MAP", "📅 HISTORY", "🧪 SIMULATOR", "📜 METHOD"])

# ---------------------------------------------------------------------------
# TAB 1 - IQAir-style city page + voxel map
# ---------------------------------------------------------------------------
with tab_map:
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        scale_html = "".join(
            f"<div style='background:{b['color']};{'outline:4px solid #000;outline-offset:1px;' if b is band else ''}'></div>"
            for b in BANDS)
        st.markdown(f"""
        <div class="block" style="background:{band['color']};color:{band['text']};">
          <div class="px" style="font-size:10px;">US AQI · {place.upper()} · {sel_date:%d %b %Y}</div>
          <div class="hero-num">{aqi}{'+' if aqi >= 500 else ''}</div>
          <div class="px" style="font-size:13px;">{band['label'].upper()}</div>
          <div class="vt" style="margin-top:8px;">Main pollutant: PM2.5 · <b>{row['PM2.5']:.1f} µg/m³</b>
            · {row['PM2.5'] / 5:.0f}× the WHO annual guideline (5 µg/m³)</div>
          <div class="scale">{scale_html}</div>
        </div>""", unsafe_allow_html=True)

        rh = rel_humidity(row["TEMP"], row["DEWP"])
        st.markdown(f"""
        <div class="block"><div class="px" style="font-size:11px;color:{P['blue']};margin-bottom:10px;">⛅ WEATHER</div>
        <div class="grid">
          <div class="cell"><div class="k">TEMP</div><div class="v">{row['TEMP']:.1f} °C</div></div>
          <div class="cell"><div class="k">HUMIDITY</div><div class="v">{rh:.0f} %</div></div>
          <div class="cell"><div class="k">WIND</div><div class="v">{row['WSPM']:.1f} m/s {wd_txt}</div></div>
          <div class="cell"><div class="k">PRESSURE</div><div class="v">{row['PRES']:.0f} hPa</div></div>
          <div class="cell"><div class="k">RAIN</div><div class="v">{row['RAIN']:.1f} mm</div></div>
          <div class="cell"><div class="k">DEW SPREAD</div><div class="v">{row['TEMP'] - row['DEWP']:.1f} °C</div></div>
        </div></div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="block"><div class="px" style="font-size:11px;color:{P['blue']};margin-bottom:10px;">🧪 POLLUTANTS (µg/m³)</div>
        <div class="grid">
          <div class="cell" style="border-color:{P['red']};"><div class="k">PM2.5 ★</div><div class="v">{row['PM2.5']:.0f}</div></div>
          <div class="cell"><div class="k">PM10</div><div class="v">{row['PM10']:.0f}</div></div>
          <div class="cell"><div class="k">NO2</div><div class="v">{row['NO2']:.0f}</div></div>
          <div class="cell"><div class="k">SO2</div><div class="v">{row['SO2']:.0f}</div></div>
          <div class="cell"><div class="k">CO</div><div class="v">{row['CO']:.0f}</div></div>
          <div class="cell"><div class="k">O3</div><div class="v">{row['O3']:.0f}</div></div>
        </div>
        <div class="vt" style="font-size:17px;color:{P['grey']};margin-top:8px;">Co-pollutants are shown for context only;
        they are deliberately excluded from the ML models to avoid leakage.</div></div>""", unsafe_allow_html=True)

        advice_html = "".join(f"<div class='advice'><span>{i}</span>{t}</div>" for i, t in band["advice"])
        st.markdown(f"<div class='block'><div class='px' style='font-size:11px;color:{P['blue']};margin-bottom:10px;'>"
                    f"❤️ HEALTH RECOMMENDATIONS</div>{advice_html}</div>", unsafe_allow_html=True)

    with right:
        st.markdown("### 🧱 Voxel map of Beijing")
        st.caption("Towers = monitoring stations (height ∝ AQI). Ground blocks = inverse-distance-weighted "
                   "surface between stations (illustrative). Gold block = Forbidden City. Drag to rotate, scroll to zoom.")

        # Interpolated voxel "terrain" (IDW, power 2) over the station extent
        lats = np.arange(39.80, 40.42, 0.03)
        lons = np.arange(116.10, 116.76, 0.035)
        glat, glon = np.meshgrid(lats, lons)
        glat, glon = glat.ravel(), glon.ravel()
        d2 = (glat[:, None] - day["lat"].values[None, :]) ** 2 + (glon[:, None] - day["lon"].values[None, :]) ** 2
        w = 1 / np.maximum(d2, 1e-6)
        gpm = (w * day["PM2.5"].values[None, :]).sum(1) / w.sum(1)
        grid = pd.DataFrame({"lat": glat, "lon": glon, "pm": gpm.round(1)})
        grid["aqi"] = grid["pm"].apply(pm_to_aqi)
        grid["label"] = grid["aqi"].apply(lambda a: band_of(a)["label"])
        grid["rgb"] = grid["aqi"].apply(lambda a: hex_rgb(band_of(a)["color"], 150))
        grid["name"] = "Interpolated block"

        towers = day[["station", "lat", "lon", "PM2.5", "aqi", "zone"]].copy()
        towers["pm"] = towers["PM2.5"].round(1)
        towers["label"] = towers["aqi"].apply(lambda a: band_of(a)["label"])
        towers["rgb"] = towers["aqi"].apply(lambda a: hex_rgb(band_of(a)["color"]))
        towers["name"] = towers["station"] + " (" + towers["zone"] + ")"
        towers["z"] = towers["aqi"] * 45 + 900

        landmark = pd.DataFrame([{"lat": FORBIDDEN_CITY[0], "lon": FORBIDDEN_CITY[1], "name": "Forbidden City 故宫",
                                  "label": "Landmark", "aqi": "—", "pm": "—", "rgb": hex_rgb(P["gold"], 255)}])

        layers = [
            pdk.Layer("GridCellLayer", grid, get_position=["lon", "lat"], cell_size=2700, extruded=True,
                      get_elevation="aqi", elevation_scale=6, get_fill_color="rgb", pickable=True),
            pdk.Layer("ColumnLayer", towers, get_position=["lon", "lat"], get_elevation="aqi", elevation_scale=45,
                      radius=1300, disk_resolution=4, angle=45, extruded=True, get_fill_color="rgb",
                      pickable=True, auto_highlight=True),
            pdk.Layer("ColumnLayer", landmark, get_position=["lon", "lat"], get_elevation=2500, radius=900,
                      disk_resolution=4, angle=45, extruded=True, get_fill_color="rgb", pickable=True),
            pdk.Layer("TextLayer", towers, get_position=["lon", "lat", "z"], get_text="station", get_size=13,
                      get_color=hex_rgb(P["ink"], 255), billboard=True, font_weight=700),
        ]
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=40.06, longitude=116.42, zoom=9.2, pitch=52, bearing=-18),
            map_provider="carto", map_style="light",
            tooltip={"html": "<b>{name}</b><br/>AQI {aqi} · {label}<br/>PM2.5 {pm} µg/m³",
                     "style": {"backgroundColor": P["ink"], "color": "#fff", "fontFamily": "monospace"}},
        )
        st.pydeck_chart(deck, height=620, **W)

    # Station ranking (IQAir "live ranking" style)
    st.markdown("### 🏆 Station ranking")
    rk = day.sort_values("aqi", ascending=False).reset_index(drop=True)
    rows_html = "".join(
        f"<tr><td>{i + 1}</td><td>{'🏙️' if r.zone == 'Urban' else '🌲'} {r.station}</td><td>{r.zone}</td>"
        f"<td><span class='badge' style='background:{band_of(r.aqi)['color']};color:{band_of(r.aqi)['text']};'>"
        f"{r.aqi}</span></td><td>{r._asdict()['_4'] if False else r[4]:.1f}</td><td>{band_of(r.aqi)['label']}</td></tr>"
        for i, r in enumerate(rk[["station", "zone", "aqi", "wd", "PM2.5"]].itertuples(index=False)))
    urban_m = day.loc[day.zone == "Urban", "PM2.5"].mean()
    rural_m = day.loc[day.zone == "Rural", "PM2.5"].mean()
    st.markdown(f"""<div class="block"><table class="rank">
      <tr><th>#</th><th>STATION</th><th>ZONE</th><th>US AQI</th><th>PM2.5 µg/m³</th><th>STATUS</th></tr>{rows_html}
      </table><div class="vt" style="margin-top:10px;">Urban mean <b>{urban_m:.1f}</b> vs rural mean <b>{rural_m:.1f}</b> µg/m³
      → raw gap <b>{urban_m - rural_m:+.1f}</b> µg/m³ on this day.</div></div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 2 - History: block calendar, monthly trend, diurnal heatmap
# ---------------------------------------------------------------------------
band_domain = [b["label"] for b in BANDS]
band_range = [b["color"] for b in BANDS]

with tab_hist:
    hist_place = place
    if hist_place.startswith("Beijing"):
        series = daily.groupby("date", as_index=False)["PM2.5"].mean()
    else:
        series = daily[daily["station"] == hist_place][["date", "PM2.5"]]
    series = series.dropna().copy()
    series["aqi"] = series["PM2.5"].apply(pm_to_aqi)
    series["band"] = series["aqi"].apply(lambda a: band_of(a)["label"])

    year = st.select_slider("Year for the block calendar", options=[2013, 2014, 2015, 2016, 2017],
                            value=int(sel_date.year))
    cal = series[series["date"].dt.year == year].copy()
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["dow"] = cal["date"].dt.day_name().str[:3]
    cal.loc[(cal["date"].dt.month == 1) & (cal["week"] > 50), "week"] = 0
    cal.loc[(cal["date"].dt.month == 12) & (cal["week"] == 1), "week"] = 53

    st.markdown(f"### 📅 {hist_place} · {year} block calendar")
    cal_chart = alt.Chart(cal).mark_rect(stroke=P["ink"], strokeWidth=1.2).encode(
        x=alt.X("week:O", title="ISO week", axis=alt.Axis(labelAngle=0, values=list(range(0, 54, 4)))),
        y=alt.Y("dow:O", sort=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], title=None),
        color=alt.Color("band:N", scale=alt.Scale(domain=band_domain, range=band_range), title="AQI band",
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), alt.Tooltip("PM2.5:Q", format=".1f"), "aqi:Q", "band:N"],
    ).properties(height=210)
    st.altair_chart(cal_chart, **W)

    shares = cal["band"].value_counts(normalize=True).reindex(band_domain).fillna(0)
    cols = st.columns(len(BANDS))
    for c, b in zip(cols, BANDS):
        c.metric(b["label"], f"{shares[b['label']] * 100:.0f}% of days")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### 📈 Monthly trend")
        monthly = daily.dropna(subset=["PM2.5"]).copy()
        monthly["month"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
        monthly = monthly.groupby(["month", "zone"], as_index=False)["PM2.5"].mean()
        line = alt.Chart(monthly).mark_line(point=alt.OverlayMarkDef(shape="square", size=40), strokeWidth=3).encode(
            x=alt.X("month:T", title=None), y=alt.Y("PM2.5:Q", title="Mean PM2.5 (µg/m³)"),
            color=alt.Color("zone:N", scale=alt.Scale(domain=["Urban", "Rural"], range=[P["red"], P["jade"]])),
            tooltip=[alt.Tooltip("month:T", format="%b %Y"), "zone:N", alt.Tooltip("PM2.5:Q", format=".1f")])
        rule = alt.Chart(pd.DataFrame({"x": [pd.Timestamp("2015-01-01")]})).mark_rule(
            color=P["gold"], strokeDash=[6, 4], strokeWidth=3).encode(x="x:T")
        who = alt.Chart(pd.DataFrame({"y": [35]})).mark_rule(color=P["grey"], strokeDash=[2, 2]).encode(y="y:Q")
        st.altair_chart((line + rule + who).properties(height=330), **W)
        st.caption("Gold line: 2015 clean-air turning point. Grey line: 35 µg/m³ (WHO interim target 1 / China Grade II).")

    with c2:
        st.markdown("### 🕒 Hour × month heatmap")
        prof = profile if hist_place.startswith("Beijing") else profile[profile["station"] == hist_place]
        prof = prof.groupby(["month", "hour"], as_index=False)["PM2.5"].mean()
        heat = alt.Chart(prof).mark_rect(stroke="#fff", strokeWidth=1).encode(
            x=alt.X("hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0, values=list(range(0, 24, 3)))),
            y=alt.Y("month:O", title="Month"),
            color=alt.Color("PM2.5:Q", scale=alt.Scale(range=[P["jade"], P["gold"], P["red"], "#5B2A6E"]),
                            title="µg/m³"),
            tooltip=["month:O", "hour:O", alt.Tooltip("PM2.5:Q", format=".1f")]).properties(height=330)
        st.altair_chart(heat, **W)
        st.caption("Night-time winter peaks reveal boundary-layer inversions trapping heating emissions.")

# ---------------------------------------------------------------------------
# TAB 3 - What-if simulator (Ridge vs Decision Tree)
# ---------------------------------------------------------------------------
with tab_sim:
    models = train_models(daily)
    st.markdown("### 🧪 Weather → PM2.5 simulator")
    st.markdown("<div class='vt'>Models are retrained here on <b>station-day means</b> with the notebook's chronological "
                "split (train &lt; 1 Mar 2016, test after) and the same leakage rules (no co-pollutants, DEWP replaced "
                "by the dew-point spread). Use it to explore how weather moves PM2.5 - not as a forecast.</div>",
                unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        t_in = st.slider("Temperature (°C)", -15.0, 35.0, 0.0, 0.5)
        dew_in = st.slider("Dew point (°C)", -30.0, 28.0, -4.0, 0.5)
        dew_in = min(dew_in, t_in)
    with s2:
        ws_in = st.slider("Wind speed (m/s)", 0.0, 8.0, 1.2, 0.1)
        pres_in = st.slider("Pressure (hPa)", 990.0, 1040.0, 1022.0, 1.0)
        rain_in = st.slider("Daily rain (mm)", 0.0, 40.0, 0.0, 0.5)
    with s3:
        season_in = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
        wd_in = st.selectbox("Dominant wind sector", ["SW (polluted corridor)", "SE", "NE", "NW (clean Siberian air)", "CV (calm)"])
        model_in = st.radio("Model", ["Decision Tree", "Ridge Regression"], horizontal=True)

    base = {"TEMP": t_in, "DEWP": dew_in, "spread": t_in - dew_in, "PRES": pres_in, "WSPM": ws_in, "RAIN": rain_in,
            "season": season_in, "wd_sector": wd_in.split()[0]}
    scen = pd.DataFrame([{**base, "zone": "Urban"}, {**base, "zone": "Rural"}])
    p_urban, p_rural = predict(models, model_in, scen)

    a1, a2, a3 = st.columns(3)
    for col, title, pm in [(a1, "🏙️ URBAN CORE", p_urban), (a2, "🌲 RURAL BACKGROUND", p_rural)]:
        a = pm_to_aqi(pm)
        b = band_of(a)
        col.markdown(f"""<div class="block" style="background:{b['color']};color:{b['text']};">
          <div class="px" style="font-size:10px;">{title}</div><div class="hero-num" style="font-size:44px;">{a}</div>
          <div class="px" style="font-size:10px;">{b['label'].upper()}</div>
          <div class="vt">PM2.5 ≈ {pm:.0f} µg/m³ (daily mean)</div></div>""", unsafe_allow_html=True)
    m = models["metrics"][model_in]
    a3.markdown(f"""<div class="block"><div class="px" style="font-size:10px;color:{P['blue']};">COUNTERFACTUAL GAP</div>
      <div class="hero-num" style="font-size:40px;color:{P['red']};">{p_urban - p_rural:+.1f}</div>
      <div class="vt">µg/m³ urban − rural, same weather</div>
      <div class="vt" style="color:{P['grey']};">Test R² {m['R²']:.2f} · MAE {m['MAE']:.1f} µg/m³ ·
      {models['n_train']:,} train / {models['n_test']:,} test station-days</div></div>""", unsafe_allow_html=True)

    st.markdown("### 💨 Wind sweep: how fast does wind clean the air?")
    sweep = pd.DataFrame([{**base, "WSPM": v, "zone": z} for v in np.arange(0, 8.01, 0.25) for z in ["Urban", "Rural"]])
    sweep["PM2.5"] = predict(models, model_in, sweep)
    sweep_chart = alt.Chart(sweep).mark_line(strokeWidth=4, interpolate="step-after").encode(
        x=alt.X("WSPM:Q", title="Wind speed (m/s)"), y=alt.Y("PM2.5:Q", title="Predicted PM2.5 (µg/m³)"),
        color=alt.Color("zone:N", scale=alt.Scale(domain=["Urban", "Rural"], range=[P["red"], P["jade"]])),
        tooltip=["zone:N", "WSPM:Q", alt.Tooltip("PM2.5:Q", format=".1f")])
    now = alt.Chart(pd.DataFrame({"x": [ws_in]})).mark_rule(color=P["gold"], strokeWidth=3).encode(x="x:Q")
    st.altair_chart((sweep_chart + now).properties(height=300), **W)
    st.caption("Step-shaped lines come from the tree's split thresholds; Ridge gives straight lines. "
               "Associations, not causal effects.")

# ---------------------------------------------------------------------------
# TAB 4 - Method & notebook results
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("### 📜 Notebook benchmark (hourly data, 15,000-row shared test sample)")
    bench = pd.DataFrame([
        ["Naive baseline (Dummy)", 80.48, 81.28, 59.55, -0.001],
        ["Linear Regression (OLS)", 68.00, 67.50, 48.41, 0.310],
        ["Ridge Regression", 68.00, 67.50, 48.41, 0.310],
        ["Decision Tree (depth 8)", 61.03, 63.74, 44.26, 0.384],
        ["K-Nearest Neighbors (K=15)", 0.00, 63.62, 44.52, 0.387],
        ["SVR (RBF)", 63.18, 62.07, 40.03, 0.416],
    ], columns=["Model", "Train RMSE", "Test RMSE", "Test MAE", "Test R²"])
    st.dataframe(bench, hide_index=True, **W)
    st.markdown("""
- **Task:** supervised regression of hourly PM2.5 from weather, time and location (420,768 rows, 12 stations).
- **Leakage guards:** chronological split at 1 Mar 2016, co-pollutants excluded, scaling fitted on train only, DEWP dropped (collinear with the dew-point spread).
- **Urban penalty:** a decoupled spatial model (no station dummies) gives β_urban ≈ **+8.75 µg/m³**, holding weather fixed.
- **Three-tier verification (literature):** US Embassy StateAir monitor, Berkeley Earth audit, NASA AERONET/MODIS AOD.
""")
    figs = [("fig2_model_performance_diagnostics.png", "Bias–variance diagnostic"),
            ("fig3_feature_importance_weights.png", "Feature importance vs Ridge weights"),
            ("fig4_counterfactual_matrix.png", "Counterfactual P&C matrix")]
    with st.expander("Notebook figures"):
        for f, cap in figs:
            if (FIG_DIR / f).exists():
                st.image(str(FIG_DIR / f), caption=cap, **W)

st.markdown(f"""<div class="wall" style="margin-top:24px;"></div>
<div class="vt" style="text-align:center;color:{P['grey']};">Data: Chen (2019), Beijing Multi-Site Air Quality, UCI ML Repository ·
Layout inspired by IQAir (not affiliated) · Built for Introduction to Machine Learning, TISS</div>""", unsafe_allow_html=True)
