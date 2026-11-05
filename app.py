#!/usr/bin/env python3
"""
Front-end app Morpheus

Applicazione web per la pianificazione di attivita extraveicolari (EVA).
Integra analisi GIS multi-livello (Mappa Obiettivo e Mappa Rischio),
micro-routing su griglia tramite algoritmo A* con funzione di Tobler marziana
e ottimizzazione del percorso tramite Orienteering Problem (OP / TOP).

Avvio:
    streamlit run app-v2.py
"""

# Rivedere commenti e codice

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk

# Librerie opzionali con fallback automatico
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage.graph import route_through_array
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# ==============================================================================
# 0. BLOCCO TEMA STREAMLIT (PREVIENE BUG LIGHT/DARK MODE)
# ==============================================================================
def ensure_locked_streamlit_theme() -> None:
    """
    Crea automaticamente .streamlit/config.toml per bloccare il tema nativo
    ed evitare conflitti cromatici tra Light Mode e Dark Mode del browser/OS.
    """
    config_dir = Path(".streamlit")
    config_file = config_dir / "config.toml"
    desired_config = (
        "[theme]\n"
        'base = "dark"\n'
        'primaryColor = "#3b82f6"\n'
        'backgroundColor = "#0d1117"\n'
        'secondaryBackgroundColor = "#161b22"\n'
        'textColor = "#e6edf3"\n'
        'font = "sans serif"\n\n'
        "[client]\n"
        'toolbarMode = "minimal"\n'
    )
    try:
        config_dir.mkdir(exist_ok=True)
        if not config_file.exists() or config_file.read_text(encoding="utf-8") != desired_config:
            config_file.write_text(desired_config, encoding="utf-8")
    except Exception:
        pass


ensure_locked_streamlit_theme()


# ==============================================================================
# 1. CONFIGURAZIONE PAGINA E STILE CSS ORDINATO (LOCKED PALETTE)
# ==============================================================================
st.set_page_config(
    page_title="Morpheus - NASA SAC",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOCKED_UI_CSS = """
<style>
    /* Forza schema colori fisso indipendentemente da impostazioni OS o browser */
    :root,
    html,
    body,
    [data-theme="light"],
    [data-theme="dark"],
    .stApp {
        color-scheme: dark !important;
        --primary-color: #3b82f6 !important;
        --background-color: #0d1117 !important;
        --secondary-background-color: #161b22 !important;
        --text-color: #e6edf3 !important;
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Contenitore principale */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }

    /* Barra laterale */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #161b22 !important;
        border-right: 1px solid #262c36 !important;
        color: #e6edf3 !important;
    }

    /* Etichette, testi e titoli ovunque */
    h1, h2, h3, h4, h5, h6, p, span, label,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] p {
        color: #e6edf3 !important;
    }

    .stCaption, [data-testid="stCaptionContainer"] p {
        color: #8b949e !important;
        font-size: 0.82rem !important;
    }

    /* Input, Selectbox, Dropdown e Popover (evita testo bianco su sfondo bianco in Light Mode) */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        background-color: #0d1117 !important;
        border-color: #30363d !important;
        color: #e6edf3 !important;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[data-baseweb="menu"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
    }

    li[role="option"] {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
    }

    li[role="option"]:hover,
    li[aria-selected="true"] {
        background-color: #1f2937 !important;
        color: #ffffff !important;
    }

    /* Bottoni e Download Button */
    .stButton > button,
    .stDownloadButton > button {
        background-color: #1f2937 !important;
        color: #e6edf3 !important;
        border: 1px solid #374151 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #2563eb !important;
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #8b949e !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #e6edf3 !important;
        border-bottom-color: #3b82f6 !important;
    }

    /* Metriche KPI compatte e pulite */
    div[data-testid="stMetric"] {
        background-color: #161b22 !important;
        border: 1px solid #262c36 !important;
        padding: 12px 14px !important;
        border-radius: 6px !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #8b949e !important;
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #f0f6fc !important;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
        font-size: 1.35rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricDelta"] div {
        font-size: 0.75rem !important;
        color: #8b949e !important;
    }
    div[data-testid="stMetricDelta"] svg {
        display: none !important;
    }

    /* Intestazione applicazione minimale */
    .app-topbar {
        background-color: #161b22;
        border: 1px solid #262c36;
        border-radius: 6px;
        padding: 14px 20px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .app-topbar-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f0f6fc;
        letter-spacing: 0.02em;
        margin: 0;
    }
    .app-topbar-sub {
        font-size: 0.82rem;
        color: #8b949e;
        margin-top: 2px;
    }
    .app-topbar-meta {
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.76rem;
        color: #9ca3af;
        background-color: #0d1117;
        border: 1px solid #30363d;
        padding: 5px 10px;
        border-radius: 4px;
    }

    /* Box scrollabile per il Piano di Volo (evita allungamento pagina) */
    .flight-plan-box {
        background-color: #161b22;
        border: 1px solid #262c36;
        border-radius: 6px;
        height: 410px;
        overflow-y: auto;
        padding: 12px;
    }
    .flight-plan-box::-webkit-scrollbar {
        width: 6px;
    }
    .flight-plan-box::-webkit-scrollbar-track {
        background: #0d1117;
    }
    .flight-plan-box::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 3px;
    }
    .fp-section-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8b949e;
        margin: 8px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #262c36;
    }
    .fp-item {
        background-color: #0d1117;
        border: 1px solid #262c36;
        border-left: 3px solid #3b82f6;
        border-radius: 4px;
        padding: 9px 12px;
        margin-bottom: 8px;
    }
    .fp-item.rover {
        border-left-color: #d97706;
    }
    .fp-item.base {
        border-left-color: #6b7280;
    }
    .fp-row-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.84rem;
        font-weight: 600;
        color: #f0f6fc;
    }
    .fp-score {
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.76rem;
        color: #60a5fa;
        background-color: rgba(59, 130, 246, 0.12);
        padding: 2px 6px;
        border-radius: 3px;
    }
    .fp-row-meta {
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.75rem;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Legenda compatta sotto mappa */
    .map-legend {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        background-color: #161b22;
        border: 1px solid #262c36;
        border-radius: 6px;
        padding: 8px 14px;
        margin-top: 8px;
        font-size: 0.78rem;
        color: #8b949e;
    }
    .legend-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 2px;
        margin-right: 6px;
        vertical-align: middle;
    }
</style>
"""
st.markdown(LOCKED_UI_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. STRUTTURE DATI E MOTORE GIS + RICERCA OPERATIVA
# ==============================================================================
@dataclass
class Waypoint:
    id: int
    code: str
    name: str
    row: int
    col: int
    lat: float
    lon: float
    elevation_m: float
    slope_deg: float
    science_score: float
    spectral_val: float
    thermal_inertia: float
    radiation_svf: float
    feature_type: str
    recommended_tool: str
    rover_accessible: bool
    pov_image_url: str
    geology_notes: str


MARS_SITES = {
    "Mount Sharp (Gale Crater) - Formazioni Boxwork": {
        "base_lat": -5.0850,
        "base_lon": 137.8500,
        "base_elev": -4120.0,
        "seed": 42,
        "description": (
            "Reticolato di creste mineralizzate (boxwork) cementate da antiche acque "
            "sotterranee lungo le pendici del Monte Sharp."
        ),
    },
    "Jezero Crater - Fronte Occidentale del Delta": {
        "base_lat": 18.4447,
        "base_lon": 77.4508,
        "base_elev": -2540.0,
        "seed": 108,
        "description": (
            "Depositi fluvio-lacustri stratificati ricchi di carbonati e fillosilicati "
            "con elevato potenziale di conservazione di biofirme."
        ),
    },
}

NASA_POV_IMAGES = [
    "https://mars.nasa.gov/msl-raw-images/msss/04100/mcam/4100ML1062040000108827E01_DXXX.jpg",
    "https://mars.nasa.gov/msl-raw-images/proj/msl/redops/ods/surface/sol/03800/opgs/edr/ncam/NLB_734974685EDR_F1001108NCAM00259M_.JPG",
    "https://mars.nasa.gov/msl-raw-images/msss/03950/mcam/3950MR1051450001904970C00_DXXX.jpg",
]


def smooth_2d(arr: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """Filtro di smoothing 2D con fallback se scipy non e presente."""
    if SCIPY_AVAILABLE:
        return gaussian_filter(arr, sigma=sigma)
    k = max(3, int(sigma * 2) | 1)
    pad = k // 2
    padded = np.pad(arr, pad, mode="edge")
    out = np.zeros_like(arr, dtype=float)
    for i in range(k):
        for j in range(k):
            out += padded[i : i + arr.shape[0], j : j + arr.shape[1]]
    return out / (k * k)


def read_uploaded_geotiff(uploaded_file, target_size: int = 75) -> Optional[np.ndarray]:
    """
    Legge un file GeoTIFF (.tif) di test, pulisce i valori NoData
    e ricampiona la matrice alla dimensione della griglia di calcolo.
    """
    if uploaded_file is None:
        return None
    if RASTERIO_AVAILABLE:
        try:
            with rasterio.open(uploaded_file) as src:
                data = src.read(
                    1,
                    out_shape=(target_size, target_size),
                    resampling=Resampling.bilinear,
                ).astype(float)
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan
                data[data < -1e5] = np.nan
                if np.isnan(data).any():
                    median_val = float(np.nanmedian(data)) if not np.isnan(data).all() else 0.0
                    data = np.nan_to_num(data, nan=median_val)
                return data
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def generate_mars_layers(
    site_name: str,
    grid_size: int = 75,
    res_m: float = 15.0,
    w_spectral: float = 0.45,
    w_boxwork: float = 0.35,
    w_thermal: float = 0.20,
    spe_alert_level: float = 1.0,
    custom_dem: Optional[np.ndarray] = None,
    custom_spectral: Optional[np.ndarray] = None,
    custom_thermal: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Genera o elabora i 5 layer co-registrati della superficie marziana:
    1. DEM (Topografia in metri)
    2. Pendenza (Gradi)
    3. Indice Spettrale / Boxwork (CRISM / Mastcam-Z)
    4. Inerzia Termica (THEMIS J m^-2 K^-1 s^-1/2)
    5. Rateo di Dose Radiazioni (RAD + Sky-View Factor)
    """
    site_cfg = MARS_SITES[site_name]
    rng = np.random.default_rng(site_cfg["seed"])

    y, x = np.mgrid[0:grid_size, 0:grid_size]
    yn, xn = y / grid_size, x / grid_size

    ridges = 6.5 * (
        np.abs(np.sin(10 * xn * np.pi + 1.5 * yn))
        + np.abs(np.cos(10 * yn * np.pi - 1.2 * xn))
    )

    if custom_dem is not None and custom_dem.shape == (grid_size, grid_size):
        dem = custom_dem.copy()
    else:
        dem = (
            site_cfg["base_elev"]
            + 65.0 * yn
            + 25.0 * np.sin(3.2 * xn * np.pi) * np.cos(2.5 * yn * np.pi)
            + ridges
        )
        craters = [(0.30, 0.38, 0.11, 22.0), (0.68, 0.62, 0.14, 28.0), (0.25, 0.78, 0.09, 18.0)]
        for cy, cx, rad, depth in craters:
            dist = np.sqrt((yn - cy) ** 2 + (xn - cx) ** 2)
            bowl = np.where(dist < rad, -depth * (1 - (dist / rad) ** 2), 0.0)
            rim = np.where(
                (dist >= rad * 0.85) & (dist <= rad * 1.25),
                depth * 0.35 * np.exp(-(((dist - rad) / (rad * 0.18)) ** 2)),
                0.0,
            )
            dem += bowl + rim
        dem = smooth_2d(dem, sigma=1.2)

    dy, dx = np.gradient(dem, res_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    boxwork_index = np.clip((ridges - ridges.min()) / (ridges.max() - ridges.min() + 1e-6), 0, 1)

    if custom_spectral is not None and custom_spectral.shape == (grid_size, grid_size):
        s_min, s_max = float(custom_spectral.min()), float(custom_spectral.max())
        spectral_norm = np.clip((custom_spectral - s_min) / (s_max - s_min + 1e-6), 0, 1)
    else:
        spectral_raw = smooth_2d(rng.random((grid_size, grid_size)), sigma=3.0)
        spectral_norm = (spectral_raw - spectral_raw.min()) / (spectral_raw.max() - spectral_raw.min() + 1e-6)
        craters = [(0.30, 0.38, 0.11, 22.0), (0.68, 0.62, 0.14, 28.0), (0.25, 0.78, 0.09, 18.0)]
        for cy, cx, rad, _ in craters:
            dist = np.sqrt((yn - cy) ** 2 + (xn - cx) ** 2)
            spectral_norm += 0.45 * np.exp(-(((dist - rad * 1.1) / 0.06) ** 2))
        spectral_norm = np.clip(spectral_norm, 0, 1)

    curv = smooth_2d(dem - smooth_2d(dem, sigma=4.0), sigma=1.0)
    if custom_thermal is not None and custom_thermal.shape == (grid_size, grid_size):
        thermal_inertia = np.clip(custom_thermal, 70.0, 680.0)
    else:
        thermal_inertia = np.clip(
            340.0 + 140.0 * (curv / (np.std(curv) + 1e-6)) + rng.normal(0, 25, (grid_size, grid_size)),
            70.0,
            680.0,
        )
    thermal_interest = np.clip((thermal_inertia - 200.0) / 450.0, 0, 1)

    svf = np.clip(1.0 - (slope_deg / 55.0) - 0.12 * np.maximum(0, -curv), 0.35, 1.0)
    rad_dose_usv_hr = 26.0 * svf * spe_alert_level

    total_w = max(w_spectral + w_boxwork + w_thermal, 1e-6)
    objective_map = 100.0 * (
        (w_spectral * spectral_norm + w_boxwork * boxwork_index + w_thermal * thermal_interest)
        / total_w
    )

    slope_risk = np.clip(slope_deg / 25.0, 0, 1.5) ** 1.6
    sand_trap_risk = np.where(thermal_inertia < 180.0, (180.0 - thermal_inertia) / 110.0, 0.0)
    rad_risk = np.clip((rad_dose_usv_hr - 15.0) / 60.0, 0, 1.0)
    risk_map = np.clip(0.50 * slope_risk + 0.30 * sand_trap_risk + 0.20 * rad_risk, 0.01, 1.0)

    return {
        "dem": dem,
        "slope_deg": slope_deg,
        "spectral": spectral_norm,
        "boxwork": boxwork_index,
        "thermal_inertia": thermal_inertia,
        "svf": svf,
        "rad_dose_usv_hr": rad_dose_usv_hr,
        "objective_map": objective_map,
        "risk_map": risk_map,
    }


def extract_candidate_waypoints(
    layers: Dict[str, np.ndarray],
    site_name: str,
    max_safe_slope: float = 18.0,
    min_thermal_inertia: float = 155.0,
    num_candidates: int = 16,
    res_m: float = 15.0,
) -> List[Waypoint]:
    """
    Filtra i pixel pericolosi (pendenza eccessiva o sabbia profonda) ed estrae
    i massimi locali dalla Mappa Obiettivo come nodi candidati del grafo.
    """
    site_cfg = MARS_SITES[site_name]
    dem = layers["dem"]
    slope = layers["slope_deg"]
    obj = layers["objective_map"]
    ti = layers["thermal_inertia"]
    spec = layers["spectral"]
    boxw = layers["boxwork"]
    svf = layers["svf"]

    n = dem.shape[0]
    base_r, base_c = n // 2, n // 2

    def rc_to_latlon(r: int, c: int) -> Tuple[float, float]:
        dlat = ((n // 2) - r) * res_m / 59200.0
        dlon = (c - (n // 2)) * res_m / 59200.0
        return site_cfg["base_lat"] + dlat, site_cfg["base_lon"] + dlon

    base_lat, base_lon = rc_to_latlon(base_r, base_c)
    waypoints: List[Waypoint] = [
        Waypoint(
            id=0,
            code="BASE",
            name="Modulo Habitat / Airlock EVA",
            row=base_r,
            col=base_c,
            lat=base_lat,
            lon=base_lon,
            elevation_m=float(dem[base_r, base_c]),
            slope_deg=float(slope[base_r, base_c]),
            science_score=0.0,
            spectral_val=float(spec[base_r, base_c]),
            thermal_inertia=float(ti[base_r, base_c]),
            radiation_svf=float(svf[base_r, base_c]),
            feature_type="Punto di Partenza e Rientro",
            recommended_tool="Ricarica PLSS e Stoccaggio Campioni",
            rover_accessible=True,
            pov_image_url=NASA_POV_IMAGES[0],
            geology_notes="Area pianeggiante su regolite consolidata idonea alle operazioni di sbarco.",
        )
    ]

    safe_mask = (slope <= max_safe_slope) & (ti >= min_thermal_inertia)
    masked_obj = np.where(safe_mask, obj, -1.0)

    candidates = []
    min_dist_px = max(6, n // 11)
    flat_indices = np.argsort(masked_obj, axis=None)[::-1]

    feature_catalog = [
        ("Cresta Fratturata Boxwork", "APXS + Micro-Imager", "Vene di solfato depositate da circolazione idrotermale."),
        ("Affioramento Solfati Idrati", "Carotatore Rotativo", "Strati sedimentari ricchi di minerali idrati rilevati da CRISM."),
        ("Blocco Ejecta Bordo Cratere", "Spettrometro Raman UV", "Litologia profonda esposta da impatto; elevata inerzia termica."),
        ("Vena Poligonale da Essiccazione", "Fluorescenza Raggi X (PIXL)", "Strutture poligonali legate a cicli di evaporazione lacustre."),
        ("Nicchia Rocciosa Schermata", "Georadar Sottosuolo", "Parete sub-verticale con schermatura parziale dalle radiazioni."),
    ]

    for idx in flat_indices:
        if len(candidates) >= num_candidates:
            break
        r, c = int(idx // n), int(idx % n)
        if masked_obj[r, c] <= 10.0:
            break
        if math.hypot(r - base_r, c - base_c) < min_dist_px:
            continue
        if any(math.hypot(r - pr, c - pc) < min_dist_px for pr, pc in candidates):
            continue
        candidates.append((r, c))

    for i, (r, c) in enumerate(candidates, start=1):
        lat, lon = rc_to_latlon(r, c)
        ftype, tool, notes = feature_catalog[(i - 1) % len(feature_catalog)]
        if boxw[r, c] > 0.68:
            ftype, tool = "Cresta Mineralizzata Boxwork", "APXS + Carotatore"
        rover_ok = bool(slope[r, c] <= 12.5 and ti[r, c] >= 200.0)
        code_str = f"WP-{i:02d}"
        waypoints.append(
            Waypoint(
                id=i,
                code=code_str,
                name=f"{code_str} | {ftype}",
                row=r,
                col=c,
                lat=lat,
                lon=lon,
                elevation_m=float(dem[r, c]),
                slope_deg=float(slope[r, c]),
                science_score=round(float(obj[r, c]), 1),
                spectral_val=round(float(spec[r, c]), 2),
                thermal_inertia=round(float(ti[r, c]), 0),
                radiation_svf=round(float(svf[r, c]), 2),
                feature_type=ftype,
                recommended_tool=tool,
                rover_accessible=rover_ok,
                pov_image_url=NASA_POV_IMAGES[i % len(NASA_POV_IMAGES)],
                geology_notes=notes,
            )
        )

    return waypoints


def compute_astar_pairwise(
    waypoints: List[Waypoint],
    layers: Dict[str, np.ndarray],
    res_m: float = 15.0,
    max_safe_slope: float = 18.0,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[Tuple[int, int]]]]:
    """
    Calcola i percorsi minimi A* sulla griglia di Rischio e Pendenza per ogni coppia
    di waypoint, usando la funzione di Tobler adattata alla gravita marziana.
    """
    slope_deg = layers["slope_deg"]
    risk_map = layers["risk_map"]
    ti = layers["thermal_inertia"]

    slope_rad = np.radians(slope_deg)
    base_speed_ms = 1.15 * np.exp(-3.5 * np.abs(np.tan(slope_rad) + 0.05))
    sand_factor = np.where(ti < 180.0, 0.45, 1.0)
    slope_barrier = np.where(slope_deg > max_safe_slope, 0.08, 1.0)
    eff_speed_ms = np.clip(base_speed_ms * sand_factor * slope_barrier, 0.05, 1.35)

    cell_cost_min = (res_m / eff_speed_ms) / 60.0 * (1.0 + 1.4 * risk_map)

    n_wp = len(waypoints)
    cost_matrix = np.zeros((n_wp, n_wp), dtype=float)
    paths_dict: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for i in range(n_wp):
        for j in range(i + 1, n_wp):
            start = (waypoints[i].row, waypoints[i].col)
            end = (waypoints[j].row, waypoints[j].col)
            if SKIMAGE_AVAILABLE:
                indices, weight = route_through_array(
                    cell_cost_min, start, end, fully_connected=True, geometric=True
                )
                path = [(int(r), int(c)) for r, c in indices]
                travel_min = float(weight)
            else:
                steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1)
                rs = np.linspace(start[0], end[0], steps + 1).astype(int)
                cs = np.linspace(start[1], end[1], steps + 1).astype(int)
                path = list(zip(rs.tolist(), cs.tolist()))
                diag = math.hypot(end[0] - start[0], end[1] - start[1]) / steps
                travel_min = float(sum(cell_cost_min[r, c] * diag for r, c in path))

            cost_matrix[i, j] = travel_min
            cost_matrix[j, i] = travel_min
            paths_dict[(i, j)] = path
            paths_dict[(j, i)] = path[::-1]

    return cost_matrix, paths_dict


def solve_orienteering_problem(
    waypoints: List[Waypoint],
    cost_matrix: np.ndarray,
    budget_minutes: float = 180.0,
    science_time_per_wp: float = 15.0,
    enable_rover_team: bool = True,
) -> Dict[str, List[int]]:
    """
    Risolve l'Orienteering Problem (o Team Orienteering Problem con Rover).
    Massimizza il punteggio scientifico totale rispettando il budget di tempo EVA.
    """
    n = len(waypoints)
    if ORTOOLS_AVAILABLE and not enable_rover_team:
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            u = manager.IndexToNode(from_index)
            v = manager.IndexToNode(to_index)
            service = science_time_per_wp if u != 0 else 0.0
            return int(round((cost_matrix[u, v] + service) * 10))

        transit_idx = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        routing.AddDimension(
            transit_idx,
            0,
            int(round(budget_minutes * 10)),
            True,
            "Time",
        )

        for node in range(1, n):
            penalty = int(round(waypoints[node].science_score * 1000))
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = 1

        solution = routing.SolveWithParameters(search_params)
        if solution:
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(0)
            return {"astronaut": route, "rover": [0, 0]}

    def build_route(allowed_nodes: set, speed_multiplier: float, budget: float, service_t: float) -> List[int]:
        route = [0]
        visited = {0}
        used_time = 0.0
        curr = 0
        while True:
            best_next = None
            best_ratio = -1.0
            for cand in allowed_nodes - visited:
                step_cost = cost_matrix[curr, cand] * speed_multiplier + service_t
                return_cost = cost_matrix[cand, 0] * speed_multiplier
                if used_time + step_cost + return_cost <= budget:
                    ratio = (waypoints[cand].science_score ** 1.3) / max(step_cost, 1.0)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_next = cand
            if best_next is None:
                break
            used_time += cost_matrix[curr, best_next] * speed_multiplier + service_t
            route.append(best_next)
            visited.add(best_next)
            curr = best_next
        route.append(0)

        improved = True
        while improved and len(route) > 4:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    old_d = (
                        cost_matrix[route[i - 1], route[i]]
                        + cost_matrix[route[j], route[j + 1]]
                    )
                    new_d = (
                        cost_matrix[route[i - 1], route[j]]
                        + cost_matrix[route[i], route[j + 1]]
                    )
                    if new_d + 1e-4 < old_d:
                        route[i : j + 1] = reversed(route[i : j + 1])
                        improved = True
        return route

    all_nodes = set(range(1, n))
    astro_route = build_route(all_nodes, speed_multiplier=1.0, budget=budget_minutes, service_t=science_time_per_wp)

    rover_route = [0, 0]
    if enable_rover_team:
        remaining = {
            idx for idx in (all_nodes - set(astro_route)) if waypoints[idx].rover_accessible
        }
        rover_route = build_route(remaining, speed_multiplier=0.62, budget=budget_minutes * 1.1, service_t=8.0)

    return {"astronaut": astro_route, "rover": rover_route}


def evaluate_route_telemetry(
    route: List[int],
    waypoints: List[Waypoint],
    paths_dict: Dict[Tuple[int, int], List[Tuple[int, int]]],
    layers: Dict[str, np.ndarray],
    res_m: float = 15.0,
    science_time_per_wp: float = 15.0,
) -> Dict:
    """Calcola la telemetria passo-passo lungo il tracciato A* selezionato."""
    if len(route) <= 2:
        return {
            "full_grid_path": [(waypoints[0].row, waypoints[0].col)],
            "latlon_path": [[waypoints[0].lon, waypoints[0].lat, waypoints[0].elevation_m]],
            "total_dist_m": 0.0,
            "total_time_min": 0.0,
            "total_science": 0.0,
            "total_rad_usv": 0.0,
            "o2_used_pct": 0.0,
            "profile_df": pd.DataFrame(),
        }

    site_lat = waypoints[0].lat
    site_lon = waypoints[0].lon
    n = layers["dem"].shape[0]
    base_r, base_c = n // 2, n // 2

    full_cells: List[Tuple[int, int]] = []
    for k in range(len(route) - 1):
        u, v = route[k], route[k + 1]
        seg = paths_dict.get((u, v), [(waypoints[u].row, waypoints[u].col), (waypoints[v].row, waypoints[v].col)])
        if k > 0 and full_cells:
            seg = seg[1:]
        full_cells.extend(seg)

    records = []
    cum_dist_m = 0.0
    cum_time_min = 0.0
    cum_rad_usv = 0.0
    latlon_path = []

    for idx, (r, c) in enumerate(full_cells):
        if idx > 0:
            pr, pc = full_cells[idx - 1]
            step_m = math.hypot(r - pr, c - pc) * res_m
        else:
            step_m = 0.0
        cum_dist_m += step_m
        elev = float(layers["dem"][r, c])
        slp = float(layers["slope_deg"][r, c])
        risk = float(layers["risk_map"][r, c])
        rad_rate = float(layers["rad_dose_usv_hr"][r, c])

        speed_ms = max(0.15, 1.15 * math.exp(-3.5 * abs(math.tan(math.radians(slp)) + 0.05)))
        dt_min = (step_m / speed_ms) / 60.0
        cum_time_min += dt_min
        cum_rad_usv += rad_rate * (dt_min / 60.0)

        lat = site_lat + ((base_r - r) * res_m / 59200.0)
        lon = site_lon + ((c - base_c) * res_m / 59200.0)
        latlon_path.append([lon, lat, elev - waypoints[0].elevation_m + 5.0])

        records.append(
            {
                "step": idx,
                "dist_m": round(cum_dist_m, 1),
                "elevation_m": round(elev, 1),
                "slope_deg": round(slp, 1),
                "risk": round(risk, 2),
                "rad_usv_hr": round(rad_rate, 1),
                "cum_time_min": round(cum_time_min, 1),
                "cum_rad_usv": round(cum_rad_usv, 2),
            }
        )

    visited_wps = [w for w in waypoints if w.id in route and w.id != 0]
    science_time = len(visited_wps) * science_time_per_wp
    total_time = cum_time_min + science_time
    for w in visited_wps:
        cum_rad_usv += float(layers["rad_dose_usv_hr"][w.row, w.col]) * (science_time_per_wp / 60.0)

    total_science = sum(w.science_score for w in visited_wps)
    o2_used_pct = min(100.0, (total_time / 300.0) * 100.0 * (1.0 + 0.25 * np.mean([r["risk"] for r in records])))

    return {
        "full_grid_path": full_cells,
        "latlon_path": latlon_path,
        "total_dist_m": cum_dist_m,
        "total_time_min": total_time,
        "walk_time_min": cum_time_min,
        "science_time_min": science_time,
        "total_science": total_science,
        "total_rad_usv": cum_rad_usv,
        "o2_used_pct": o2_used_pct,
        "profile_df": pd.DataFrame(records),
    }


# ==============================================================================
# 3. BARRA LATERALE DI CONTROLLO
# ==============================================================================
with st.sidebar:
    st.markdown("### Morpheus")
    st.caption("Pianificazione Percorsi EVA e Analisi Rischio")

    ui_mode = st.radio(
        "Modalita Visualizzazione",
        [
            "Pianificatore Missione (Mappa 3D)",
            "Visore Casco Astronauta (HUD)",
        ],
        index=0,
    )

    st.markdown("---")
    selected_site = st.selectbox("Area Operativa", list(MARS_SITES.keys()))

    with st.expander("Sorgente Dati (Test / GeoTIFF)", expanded=False):
        st.caption("Carica file .tif reali per sovrascrivere i layer sintetici di test.")
        dem_file = st.file_uploader("DEM Topografico (.tif)", type=["tif", "tiff"], key="u_dem")
        spec_file = st.file_uploader("Mappa Spettrale (.tif)", type=["tif", "tiff"], key="u_spec")
        therm_file = st.file_uploader("Inerzia Termica (.tif)", type=["tif", "tiff"], key="u_therm")

    st.markdown("#### Vincoli Operativi EVA")
    eva_budget_min = st.slider("Autonomia Massima EVA (min)", 60, 360, 180, step=15)
    science_duration_min = st.slider("Tempo Analisi per Waypoint (min)", 5, 30, 15, step=5)
    max_safe_slope = st.slider("Pendenza Massima Consentita (gradi)", 10.0, 25.0, 18.0, step=0.5)
    enable_rover = st.toggle("Includi Rover di Supporto (TOP)", value=True)

    st.markdown("#### Pesi Mappa Obiettivo")
    w_spectral = st.slider("Peso Spettrale (CRISM)", 0.0, 1.0, 0.45, 0.05)
    w_boxwork = st.slider("Peso Morfologia Boxwork", 0.0, 1.0, 0.35, 0.05)
    w_thermal = st.slider("Peso Inerzia Termica (THEMIS)", 0.0, 1.0, 0.20, 0.05)

    st.markdown("#### Condizioni Ambientali e Layer")
    spe_level = st.select_slider(
        "Livello Attivita Solare (RAD)",
        options=[1.0, 1.5, 2.5, 4.0],
        value=1.0,
        format_func=lambda x: {
            1.0: "Nominale (25 uSv/h)",
            1.5: "Moderato (Classe M)",
            2.5: "Elevato (Classe X)",
            4.0: "Critico (Allerta Rifugio)",
        }[x],
    )
    active_heatmap_layer = st.selectbox(
        "Livello Superficie 3D",
        [
            "Sintesi Missione (Obiettivo / Rischio)",
            "Mappa Obiettivo (Valore Scientifico)",
            "Mappa Rischio (Pendenza, Sabbia, Rad)",
            "Inerzia Termica (THEMIS)",
            "Dose Radiazioni (RAD + SVF)",
        ],
    )


# ==============================================================================
# 4. ESECUZIONE PIPELINE
# ==============================================================================
custom_dem_arr = read_uploaded_geotiff(dem_file, target_size=75)
custom_spec_arr = read_uploaded_geotiff(spec_file, target_size=75)
custom_therm_arr = read_uploaded_geotiff(therm_file, target_size=75)

layers = generate_mars_layers(
    site_name=selected_site,
    grid_size=75,
    res_m=15.0,
    w_spectral=w_spectral,
    w_boxwork=w_boxwork,
    w_thermal=w_thermal,
    spe_alert_level=spe_level,
    custom_dem=custom_dem_arr,
    custom_spectral=custom_spec_arr,
    custom_thermal=custom_therm_arr,
)

waypoints = extract_candidate_waypoints(
    layers=layers,
    site_name=selected_site,
    max_safe_slope=max_safe_slope,
    num_candidates=16,
    res_m=15.0,
)

cost_matrix, paths_dict = compute_astar_pairwise(
    waypoints=waypoints,
    layers=layers,
    res_m=15.0,
    max_safe_slope=max_safe_slope,
)

routes = solve_orienteering_problem(
    waypoints=waypoints,
    cost_matrix=cost_matrix,
    budget_minutes=float(eva_budget_min),
    science_time_per_wp=float(science_duration_min),
    enable_rover_team=enable_rover,
)

astro_route = routes["astronaut"]
rover_route = routes["rover"]

astro_telemetry = evaluate_route_telemetry(
    route=astro_route,
    waypoints=waypoints,
    paths_dict=paths_dict,
    layers=layers,
    res_m=15.0,
    science_time_per_wp=float(science_duration_min),
)

rover_telemetry = evaluate_route_telemetry(
    route=rover_route,
    waypoints=waypoints,
    paths_dict=paths_dict,
    layers=layers,
    res_m=15.0,
    science_time_per_wp=8.0,
)


# ==============================================================================
# 5. INTESTAZIONE E METRICHE DI SINTESI
# ==============================================================================
solver_label = "OR-TOOLS + A*" if (ORTOOLS_AVAILABLE and not enable_rover) else "TOP HEURISTIC + 2-OPT + A*"
data_source_label = "GEOTIFF UTENTE" if custom_dem_arr is not None else "MODELLO SINTETICO"

st.markdown(
    f"""
    <div class="app-topbar">
        <div>
            <div class="app-topbar-title">Morpheus - Sistema di Pianificazione EVA</div>
            <div class="app-topbar-sub">{selected_site} &mdash; {MARS_SITES[selected_site]['description']}</div>
        </div>
        <div class="app-topbar-meta">MOTORE: {solver_label} &nbsp;|&nbsp; DATI: {data_source_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

visited_astro_ids = [i for i in astro_route if i != 0]
visited_rover_ids = [i for i in rover_route if i != 0]
combined_science = astro_telemetry["total_science"] + rover_telemetry["total_science"]

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric(
    "Punteggio Scientifico",
    f"{combined_science:.0f} pt",
    f"EVA: {astro_telemetry['total_science']:.0f} | Rover: {rover_telemetry['total_science']:.0f}",
)
m2.metric(
    "Waypoint Selezionati",
    f"{len(visited_astro_ids) + len(visited_rover_ids)} / {len(waypoints) - 1}",
    f"{len(visited_astro_ids)} astronauta, {len(visited_rover_ids)} rover",
)
m3.metric(
    "Durata Stimata EVA",
    f"{astro_telemetry['total_time_min']:.0f} / {eva_budget_min} min",
    f"Margine: {eva_budget_min - astro_telemetry['total_time_min']:.0f} min",
)
m4.metric(
    "Distanza Percorsa (A*)",
    f"{astro_telemetry['total_dist_m'] / 1000.0:.2f} km",
    f"Rover: {rover_telemetry['total_dist_m'] / 1000.0:.2f} km",
)
m5.metric(
    "Consumo Ossigeno PLSS",
    f"{astro_telemetry['o2_used_pct']:.1f}%",
    "Nominale" if astro_telemetry["o2_used_pct"] < 85 else "Riserva ridotta",
)
m6.metric(
    "Dose Radiazioni Assorbita",
    f"{astro_telemetry['total_rad_usv']:.1f} uSv",
    f"Fattore solare: {spe_level}x",
)


# ==============================================================================
# 6. MODALITA 1: PIANIFICATORE MISSIONE (MAPPA 3D + BOX PIANO DI VOLO)
# ==============================================================================
if ui_mode.startswith("Pianificatore"):
    col_map, col_details = st.columns([1.6, 1.0], gap="medium")

    with col_map:
        st.markdown("#### Mappa Tattica 3D (Nodi Obiettivo e Percorsi A*)")

        n = layers["dem"].shape[0]
        base_r, base_c = n // 2, n // 2
        site_lat = waypoints[0].lat
        site_lon = waypoints[0].lon
        base_elev = waypoints[0].elevation_m

        grid_records = []
        for r in range(0, n, 2):
            for c in range(0, n, 2):
                lat = site_lat + ((base_r - r) * 15.0 / 59200.0)
                lon = site_lon + ((c - base_c) * 15.0 / 59200.0)
                obj_v = float(layers["objective_map"][r, c])
                risk_v = float(layers["risk_map"][r, c])
                ti_v = float(layers["thermal_inertia"][r, c])
                rad_v = float(layers["rad_dose_usv_hr"][r, c])
                elev_rel = float(layers["dem"][r, c] - base_elev)

                if active_heatmap_layer == "Mappa Obiettivo (Valore Scientifico)":
                    color = [int(25 + 1.8 * obj_v), int(160 * (obj_v / 100.0)), int(220 - 1.2 * obj_v), 145]
                elif active_heatmap_layer == "Mappa Rischio (Pendenza, Sabbia, Rad)":
                    color = [int(min(240, 235 * risk_v * 1.25)), int(max(20, 170 * (1 - risk_v))), 45, 150]
                elif active_heatmap_layer == "Inerzia Termica (THEMIS)":
                    tn = (ti_v - 70.0) / 600.0
                    color = [int(220 * (1 - tn)), int(130 * tn + 55), int(230 * tn), 145]
                elif active_heatmap_layer == "Dose Radiazioni (RAD + SVF)":
                    rn = min(1.0, rad_v / 80.0)
                    color = [int(230 * rn), int(65 * (1 - rn)), int(200 * rn), 145]
                else:
                    color = [
                        int(min(230, 220 * risk_v)),
                        int(min(220, 2.1 * obj_v * (1 - 0.5 * risk_v))),
                        int(min(210, 175 * (obj_v / 100.0))),
                        140,
                    ]

                grid_records.append(
                    {
                        "lon": lon,
                        "lat": lat,
                        "elev_rel": max(1.0, elev_rel + 25.0),
                        "color": color,
                        "obj": round(obj_v, 1),
                        "risk": round(risk_v, 2),
                        "slope": round(float(layers["slope_deg"][r, c]), 1),
                    }
                )

        grid_df = pd.DataFrame(grid_records)

        wp_records = []
        for w in waypoints:
            if w.id == 0:
                status = "Base Habitat"
                col = [240, 246, 252, 255]
            elif w.id in visited_astro_ids:
                seq = visited_astro_ids.index(w.id) + 1
                status = f"Selezionato EVA (#{seq})"
                col = [59, 130, 246, 245]
            elif w.id in visited_rover_ids:
                seq = visited_rover_ids.index(w.id) + 1
                status = f"Selezionato Rover (#{seq})"
                col = [217, 119, 6, 245]
            else:
                status = "Escluso (Vincoli Tempo/Rischio)"
                col = [100, 116, 139, 150]

            wp_records.append(
                {
                    "id": w.id,
                    "name": w.name,
                    "lon": w.lon,
                    "lat": w.lat,
                    "science_score": w.science_score,
                    "height": 28.0 if w.id == 0 else max(12.0, w.science_score * 0.9),
                    "slope_deg": round(w.slope_deg, 1),
                    "thermal": round(w.thermal_inertia, 0),
                    "tool": w.recommended_tool,
                    "status": status,
                    "color": col,
                }
            )
        wp_df = pd.DataFrame(wp_records)

        paths_data = [
            {
                "name": "Percorso Ottimo Astronauta (A* + OP)",
                "path": [[pt[0], pt[1]] for pt in astro_telemetry["latlon_path"]],
                "color": [56, 189, 248, 255],
            }
        ]
        if enable_rover and len(rover_route) > 2:
            paths_data.append(
                {
                    "name": "Percorso Rover di Supporto (TOP A*)",
                    "path": [[pt[0], pt[1]] for pt in rover_telemetry["latlon_path"]],
                    "color": [217, 119, 6, 240],
                }
            )
        paths_df = pd.DataFrame(paths_data)

        deck_layers = [
            pdk.Layer(
                "ColumnLayer",
                data=grid_df,
                get_position=["lon", "lat"],
                get_elevation="elev_rel",
                elevation_scale=1.2,
                radius=13,
                get_fill_color="color",
                pickable=False,
                extruded=True,
            ),
            pdk.Layer(
                "PathLayer",
                data=paths_df,
                get_path="path",
                get_color="color",
                width_scale=4,
                width_min_pixels=4,
                pickable=True,
            ),
            pdk.Layer(
                "ColumnLayer",
                data=wp_df,
                get_position=["lon", "lat"],
                get_elevation="height",
                elevation_scale=2.2,
                radius=16,
                get_fill_color="color",
                pickable=True,
                extruded=True,
            ),
        ]

        view_state = pdk.ViewState(
            latitude=site_lat,
            longitude=site_lon,
            zoom=14.6,
            pitch=50,
            bearing=-18,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=deck_layers,
                initial_view_state=view_state,
                map_style=None,
                tooltip={
                    "html": (
                        "<b>{name}</b><br/>"
                        "Stato: {status}<br/>"
                        "Valore Scientifico: <b>{science_score} pt</b><br/>"
                        "Pendenza: {slope_deg}° | Inerzia Termica: {thermal}<br/>"
                        "Strumento: {tool}"
                    ),
                    "style": {
                        "backgroundColor": "#161b22",
                        "color": "#e6edf3",
                        "fontSize": "12px",
                        "border": "1px solid #30363d",
                    },
                },
            ),
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="map-legend">
                <span><span class="legend-dot" style="background:#f0f6fc;"></span>Base Habitat</span>
                <span><span class="legend-dot" style="background:#3b82f6;"></span>Waypoint Astronauta (EVA)</span>
                <span><span class="legend-dot" style="background:#d97706;"></span>Waypoint Rover (TOP)</span>
                <span><span class="legend-dot" style="background:#64748b;"></span>Waypoint Escluso</span>
                <span><span class="legend-dot" style="background:#38bdf8;"></span>Tracciato Sicuro A*</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_details:
        st.markdown("#### Dettaglio Operativo e Livelli")
        tab_seq, tab_layers, tab_export = st.tabs(
            ["Piano di Volo", "Livelli Input GIS", "Esportazione"]
        )

        with tab_seq:
            # Costruzione HTML compatto dentro un unico box a scorrimento verticale fisso
            items_html = ['<div class="flight-plan-box">']
            items_html.append('<div class="fp-section-label">Sequenza EVA Astronauta</div>')

            for step_idx, wp_id in enumerate(astro_route):
                w = waypoints[wp_id]
                if step_idx == 0:
                    items_html.append(
                        f'<div class="fp-item base">'
                        f'<div class="fp-row-top"><span>0. Partenza: {w.code}</span><span>{w.elevation_m:.0f} m</span></div>'
                        f'<div class="fp-row-meta">{w.name}</div>'
                        f'</div>'
                    )
                elif step_idx == len(astro_route) - 1:
                    prev_id = astro_route[step_idx - 1]
                    leg_min = cost_matrix[prev_id, wp_id]
                    items_html.append(
                        f'<div class="fp-item base">'
                        f'<div class="fp-row-top"><span>{step_idx}. Rientro Airlock: {w.code}</span><span>{leg_min:.1f} min</span></div>'
                        f'<div class="fp-row-meta">Fine escursione e ricompilazione campioni</div>'
                        f'</div>'
                    )
                else:
                    prev_id = astro_route[step_idx - 1]
                    leg_min = cost_matrix[prev_id, wp_id]
                    items_html.append(
                        f'<div class="fp-item">'
                        f'<div class="fp-row-top">'
                        f'<span>{step_idx}. {w.name}</span>'
                        f'<span class="fp-score">{w.science_score:.1f} pt</span>'
                        f'</div>'
                        f'<div class="fp-row-meta">'
                        f'Cammino A*: {leg_min:.1f} min &nbsp;|&nbsp; Pendenza: {w.slope_deg:.1f}&deg; &nbsp;|&nbsp; {w.recommended_tool}'
                        f'</div>'
                        f'</div>'
                    )

            if enable_rover and len(rover_route) > 2:
                items_html.append('<div class="fp-section-label" style="margin-top:14px;">Unità Rover Autonoma</div>')
                for step_idx, wp_id in enumerate(rover_route):
                    if wp_id == 0:
                        continue
                    w = waypoints[wp_id]
                    prev_id = rover_route[step_idx - 1]
                    leg_min = cost_matrix[prev_id, wp_id] * 0.62
                    items_html.append(
                        f'<div class="fp-item rover">'
                        f'<div class="fp-row-top">'
                        f'<span>R-{step_idx}. {w.name}</span>'
                        f'<span class="fp-score" style="color:#fbbf24;background:rgba(217,119,6,0.15);">{w.science_score:.1f} pt</span>'
                        f'</div>'
                        f'<div class="fp-row-meta">'
                        f'Transito: {leg_min:.1f} min &nbsp;|&nbsp; Pendenza: {w.slope_deg:.1f}&deg; &nbsp;|&nbsp; Terreno accessibile'
                        f'</div>'
                        f'</div>'
                    )

            items_html.append("</div>")
            st.markdown("".join(items_html), unsafe_allow_html=True)

        with tab_layers:
            fig_layers = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Pendenza (HiRISE)",
                    "Valore Obiettivo (CRISM)",
                    "Inerzia Termica (THEMIS)",
                    "Mappa Rischio + Rotta A*",
                ),
                vertical_spacing=0.14,
                horizontal_spacing=0.10,
            )
            fig_layers.add_trace(
                go.Heatmap(z=layers["slope_deg"], colorscale="YlOrRd", showscale=False),
                row=1,
                col=1,
            )
            fig_layers.add_trace(
                go.Heatmap(z=layers["objective_map"], colorscale="Viridis", showscale=False),
                row=1,
                col=2,
            )
            fig_layers.add_trace(
                go.Heatmap(z=layers["thermal_inertia"], colorscale="Cividis", showscale=False),
                row=2,
                col=1,
            )
            fig_layers.add_trace(
                go.Heatmap(z=layers["risk_map"], colorscale="Hot", reversescale=True, showscale=False),
                row=2,
                col=2,
            )
            path_r = [pt[0] for pt in astro_telemetry["full_grid_path"]]
            path_c = [pt[1] for pt in astro_telemetry["full_grid_path"]]
            fig_layers.add_trace(
                go.Scatter(
                    x=path_c,
                    y=path_r,
                    mode="lines",
                    line=dict(color="#38bdf8", width=2.5),
                    name="Tracciato A*",
                ),
                row=2,
                col=2,
            )
            fig_layers.update_layout(
                height=410,
                margin=dict(l=8, r=8, t=30, b=8),
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=dict(color="#e6edf3", size=11),
                showlegend=False,
            )
            st.plotly_chart(fig_layers, use_container_width=True)

        with tab_export:
            st.caption("Esporta la rotta calcolata e i metadati dei waypoint in formato standard GeoJSON.")
            geojson_payload = {
                "type": "FeatureCollection",
                "metadata": {
                    "site": selected_site,
                    "eva_budget_min": eva_budget_min,
                    "total_science_score": combined_science,
                    "astronaut_time_min": round(astro_telemetry["total_time_min"], 1),
                    "radiation_dose_usv": round(astro_telemetry["total_rad_usv"], 2),
                },
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"role": "Astronaut_EVA_Route"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[p[0], p[1]] for p in astro_telemetry["latlon_path"]],
                        },
                    }
                ]
                + [
                    {
                        "type": "Feature",
                        "properties": asdict(w),
                        "geometry": {"type": "Point", "coordinates": [w.lon, w.lat]},
                    }
                    for w in waypoints
                ],
            }
            st.download_button(
                "Scarica Piano di Volo (.geojson)",
                data=json.dumps(geojson_payload, indent=2),
                file_name="mars_eva_mission_plan.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )

    st.markdown("---")
    c_prof, c_bar = st.columns([1.5, 1.0], gap="medium")

    with c_prof:
        st.markdown("#### Profilo Altimetrico e Pendenza lungo il Percorso")
        pdf = astro_telemetry["profile_df"]
        if not pdf.empty:
            fig_prof = make_subplots(specs=[[{"secondary_y": True}]])
            fig_prof.add_trace(
                go.Scatter(
                    x=pdf["dist_m"],
                    y=pdf["elevation_m"],
                    fill="tozeroy",
                    name="Altitudine (m)",
                    line=dict(color="#3b82f6", width=2),
                    fillcolor="rgba(59, 130, 246, 0.14)",
                ),
                secondary_y=False,
            )
            fig_prof.add_trace(
                go.Scatter(
                    x=pdf["dist_m"],
                    y=pdf["slope_deg"],
                    name="Pendenza (gradi)",
                    line=dict(color="#d97706", width=1.8, dash="dot"),
                ),
                secondary_y=True,
            )
            fig_prof.add_hline(
                y=max_safe_slope,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Limite ({max_safe_slope}°)",
                secondary_y=True,
            )
            fig_prof.update_layout(
                height=290,
                margin=dict(l=12, r=12, t=18, b=12),
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=dict(color="#e6edf3", size=11),
                legend=dict(orientation="h", y=1.14),
                xaxis_title="Distanza Progressiva (m)",
            )
            fig_prof.update_yaxes(title_text="Quota (m)", gridcolor="#262c36", secondary_y=False)
            fig_prof.update_yaxes(title_text="Pendenza (°)", gridcolor="#262c36", secondary_y=True)
            st.plotly_chart(fig_prof, use_container_width=True)

    with c_bar:
        st.markdown("#### Riparto Waypoint (Orienteering Problem)")
        wp_plot_df = pd.DataFrame(
            [
                {
                    "Waypoint": w.code,
                    "Scienza": w.science_score,
                    "Assegnazione": (
                        "Astronauta EVA"
                        if w.id in visited_astro_ids
                        else ("Rover Supporto" if w.id in visited_rover_ids else "Escluso")
                    ),
                }
                for w in waypoints
                if w.id != 0
            ]
        )
        color_map = {
            "Astronauta EVA": "#3b82f6",
            "Rover Supporto": "#d97706",
            "Escluso": "#4b5563",
        }
        fig_bar = go.Figure()
        for stato, grp in wp_plot_df.groupby("Assegnazione"):
            fig_bar.add_trace(
                go.Bar(
                    x=grp["Waypoint"],
                    y=grp["Scienza"],
                    name=stato,
                    marker_color=color_map.get(stato, "#64748b"),
                )
            )
        fig_bar.update_layout(
            barmode="group",
            height=290,
            margin=dict(l=12, r=12, t=18, b=12),
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3", size=11),
            legend=dict(orientation="h", y=1.14),
            yaxis_title="Valore Scientifico (0-100)",
            yaxis=dict(gridcolor="#262c36"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# 7. MODALITA 2: VISORE CASCO ASTRONAUTA (HUD PRIMA PERSONA)
# ==============================================================================
else:
    st.markdown("#### Visore Casco (HUD) - Navigazione e Telemetria Locale")
    st.caption(
        "Interfaccia a sovrimpressione per la navigazione di superficie e l'analisi "
        "dei waypoint selezionati."
    )

    if len(visited_astro_ids) == 0:
        st.warning("Aumentare il budget di tempo EVA nella barra laterale per selezionare almeno un waypoint.")
    else:
        hud_col_ctrl, hud_col_info = st.columns([1.0, 2.0])
        with hud_col_ctrl:
            selected_wp_id = st.selectbox(
                "Tappa Attiva",
                options=visited_astro_ids,
                format_func=lambda idx: f"Tappa {visited_astro_ids.index(idx)+1}: {waypoints[idx].name} ({waypoints[idx].science_score} pt)",
            )
        curr_wp = waypoints[selected_wp_id]
        curr_pos_in_route = astro_route.index(selected_wp_id)
        next_wp = waypoints[astro_route[curr_pos_in_route + 1]]

        dy_m = (curr_wp.row - next_wp.row) * 15.0
        dx_m = (next_wp.col - curr_wp.col) * 15.0
        dist_to_next_m = math.hypot(dx_m, dy_m)
        bearing_deg = (math.degrees(math.atan2(dx_m, dy_m)) + 360.0) % 360.0
        leg_time_min = cost_matrix[curr_wp.id, next_wp.id]

        with hud_col_info:
            st.markdown(
                f"""
                <div style="background:#161b22; border:1px solid #262c36; border-radius:6px; padding:12px 16px; margin-top:4px; font-size:0.86rem; color:#e6edf3;">
                    <b>Prossimo Waypoint:</b> {next_wp.name} &nbsp;|&nbsp;
                    <b>Rilevamento:</b> {bearing_deg:.0f}&deg; &nbsp;|&nbsp;
                    <b>Distanza A*:</b> {dist_to_next_m:.0f} m (~{leg_time_min:.1f} min)
                </div>
                """,
                unsafe_allow_html=True,
            )

        ground_state = "Sabbia soffice (Cautela)" if curr_wp.thermal_inertia < 210 else "Substrato consolidato"
        rad_rate_local = float(layers["rad_dose_usv_hr"][curr_wp.row, curr_wp.col])

        hud_html = f"""
        <div style="
            position: relative;
            width: 100%;
            height: 520px;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #30363d;
            background: linear-gradient(180deg, #1c1917 0%, #292524 55%, #0d1117 100%);
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            color: #e6edf3;
        ">
            <img src="{curr_wp.pov_image_url}"
                 onerror="this.style.display='none'"
                 style="width:100%; height:100%; object-fit:cover; opacity:0.62; filter: contrast(1.08);" />

            <svg viewBox="0 0 1000 520" preserveAspectRatio="none"
                 style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;">
                <polygon points="0,330 180,275 370,305 580,250 790,290 1000,260 1000,520 0,520"
                         fill="rgba(28, 25, 23, 0.35)" stroke="rgba(168, 162, 158, 0.35)" stroke-width="1"/>
                <polygon points="420,520 580,520 530,345 485,345"
                         fill="rgba(59, 130, 246, 0.16)" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="6,4"/>
                <rect x="600" y="290" width="80" height="80" fill="none" stroke="#60a5fa" stroke-width="1.5"/>
                <line x1="680" y1="330" x2="730" y2="295" stroke="#60a5fa" stroke-width="1.5"/>
            </svg>

            <div style="
                position:absolute; top:16px; left:50%; transform:translateX(-50%);
                background:rgba(13, 17, 23, 0.85); border:1px solid #30363d;
                padding:6px 24px; border-radius:4px; font-size:12px; letter-spacing:1.5px; color:#9ca3af;
            ">
                N 000&deg; &nbsp;|&nbsp; NE 045&deg; &nbsp;|&nbsp; <b style="color:#60a5fa;">HDG {bearing_deg:.0f}&deg; ({dist_to_next_m:.0f} m)</b> &nbsp;|&nbsp; SE 135&deg; &nbsp;|&nbsp; S 180&deg;
            </div>

            <div style="
                position:absolute; top:68px; left:20px; width:265px;
                background:rgba(13, 17, 23, 0.86); border:1px solid #30363d;
                border-left:3px solid #3b82f6; padding:12px 14px; border-radius:4px;
                font-size:12px; line-height:1.65;
            ">
                <div style="font-weight:600; color:#f0f6fc; border-bottom:1px solid #262c36; padding-bottom:4px; margin-bottom:6px;">
                    TELEMETRIA TUTA (PLSS)
                </div>
                RISERVA O2: <b>{100.0 - astro_telemetry['o2_used_pct']:.1f}%</b><br/>
                AUTONOMIA RESIDUA: <b>{eva_budget_min - astro_telemetry['total_time_min']:.0f} min</b><br/>
                PRESSIONE INTERNA: <b>4.3 psi</b><br/>
                PENDENZA LOCALE: <b>{curr_wp.slope_deg:.1f}&deg; (Max {max_safe_slope:.0f}&deg;)</b>
            </div>

            <div style="
                position:absolute; top:68px; right:20px; width:285px;
                background:rgba(13, 17, 23, 0.86); border:1px solid #30363d;
                border-left:3px solid #d97706; padding:12px 14px; border-radius:4px;
                font-size:12px; line-height:1.65;
            ">
                <div style="font-weight:600; color:#f0f6fc; border-bottom:1px solid #262c36; padding-bottom:4px; margin-bottom:6px;">
                    PARAMETRI AMBIENTALI
                </div>
                RATEO DOSE RAD: <b>{rad_rate_local:.1f} uSv/h</b><br/>
                SKY-VIEW FACTOR: <b>{curr_wp.radiation_svf:.2f}</b><br/>
                INERZIA TERMICA: <b>{curr_wp.thermal_inertia:.0f} TI ({ground_state})</b><br/>
                INDICE SPETTRALE: <b>{curr_wp.spectral_val:.2f}</b><br/>
                QUOTA LOCALE: <b>{curr_wp.elevation_m:.1f} m</b>
            </div>

            <div style="
                position:absolute; top:240px; left:68%; width:260px;
                background:rgba(13, 17, 23, 0.90); border:1px solid #3b82f6;
                padding:10px 12px; border-radius:4px; color:#e6edf3; font-size:11.5px; line-height:1.5;
            ">
                <div style="color:#60a5fa; font-weight:600;">OBIETTIVO [{curr_wp.science_score} PT]</div>
                <b>{curr_wp.feature_type}</b><br/>
                Strumento: {curr_wp.recommended_tool}<br/>
                <span style="color:#9ca3af;">{curr_wp.geology_notes}</span>
            </div>

            <div style="
                position:absolute; bottom:16px; left:20px; right:20px;
                background:rgba(13, 17, 23, 0.88); border:1px solid #30363d;
                padding:9px 16px; border-radius:4px;
                display:flex; justify-content:space-between; align-items:center; font-size:12px; color:#9ca3af;
            ">
                <div>POSIZIONE: <b style="color:#e6edf3;">{curr_wp.name}</b> ({curr_wp.lat:.4f} N, {curr_wp.lon:.4f} E)</div>
                <div>TRACCIATO A* ATTIVO</div>
                <div>SUCCESSIVO: <b style="color:#60a5fa;">{next_wp.code} ({dist_to_next_m:.0f} m)</b></div>
            </div>
        </div>
        """
        st.components.v1.html(hud_html, height=535)
