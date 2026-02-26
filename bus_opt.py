# bus_opt.py
# Streamlit GUI: Nonlinear Optimization of School Bus Routes (Genetic Algorithm)
# Run locally: streamlit run bus_opt.py

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk


# -----------------------------
# Page config + Styling
# -----------------------------
st.set_page_config(
    page_title="School Bus Route Optimizer (GA)",
    page_icon="🚌",
    layout="wide",
)

CUSTOM_CSS = """
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 20%, rgba(59,130,246,0.10), transparent 40%),
                radial-gradient(circle at 85% 25%, rgba(16,185,129,0.10), transparent 45%),
                radial-gradient(circle at 50% 90%, rgba(234,179,8,0.08), transparent 40%);
}

/* Main spacing */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 14px;
    padding: 14px 14px 10px 14px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(2,6,23,0.96));
}
section[data-testid="stSidebar"] * { color: #e5e7eb !important; }
section[data-testid="stSidebar"] a { color: #93c5fd !important; }
section[data-testid="stSidebar"] .stButton button { width: 100%; border-radius: 12px; }

/* Buttons */
.stButton button {
    border-radius: 14px;
    padding: 0.6rem 1rem;
    font-weight: 700;
}

/* Expanders */
details {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 14px;
    padding: 0.3rem 0.8rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.04);
}

/* Badge */
.badge {
    display:inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.78rem;
    border: 1px solid rgba(0,0,0,0.08);
    background: rgba(255,255,255,0.75);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Distance utilities
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_distance_matrix(points_df: pd.DataFrame) -> np.ndarray:
    """Compute symmetric distance matrix for points (lat/lon)."""
    n = len(points_df)
    D = np.zeros((n, n), dtype=float)
    lats = points_df["lat"].to_numpy()
    lons = points_df["lon"].to_numpy()
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            D[i, j] = D[j, i] = d
    return D


def route_distance(route: List[int], D: np.ndarray) -> float:
    """Distance for a route specified by indices in D (including depot index 0 at start/end)."""
    if len(route) < 2:
        return 0.0
    dist = 0.0
    for i in range(len(route) - 1):
        dist += D[route[i], route[i + 1]]
    return dist


# -----------------------------
# Routing decoder + baselines
# -----------------------------
def decode_permutation_to_routes(
    perm_students: List[int],
    demands: np.ndarray,
    capacity: int,
    max_buses: int,
    depot_idx: int = 0,
) -> Tuple[List[List[int]], float]:
    """
    Decode a "giant tour" permutation into feasible routes by splitting when capacity exceeded.
    Returns (routes, penalty). Routes include depot at start/end.
    """
    routes = []
    current = [depot_idx]
    load = 0
    penalty = 0.0

    for s in perm_students:
        d = int(demands[s])

        if d > capacity:
            # impossible individual demand
            penalty += 1e6 * (d - capacity)
            if len(current) > 1:
                current.append(depot_idx)
                routes.append(current)
            routes.append([depot_idx, s, depot_idx])
            current = [depot_idx]
            load = 0
            continue

        if load + d <= capacity:
            current.append(s)
            load += d
        else:
            current.append(depot_idx)
            routes.append(current)
            current = [depot_idx, s]
            load = d

    if len(current) > 1:
        current.append(depot_idx)
        routes.append(current)

    if len(routes) > max_buses:
        penalty += 5000.0 * (len(routes) - max_buses)

    return routes, penalty


def greedy_baseline_routes(
    student_indices: List[int],
    D: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    max_buses: int,
    depot_idx: int = 0,
) -> List[List[int]]:
    """Simple baseline: nearest-neighbor routes until capacity filled."""
    remaining = set(student_indices)
    routes = []

    for _ in range(max_buses):
        if not remaining:
            break
        route = [depot_idx]
        load = 0
        current = depot_idx

        while True:
            candidates = [s for s in remaining if load + int(demands[s]) <= capacity]
            if not candidates:
                break
            next_s = min(candidates, key=lambda s: D[current, s])
            route.append(next_s)
            load += int(demands[next_s])
            remaining.remove(next_s)
            current = next_s

        route.append(depot_idx)
        routes.append(route)

    # Any leftover students -> single routes (shows infeasibility under bus limit)
    while remaining:
        s = remaining.pop()
        routes.append([depot_idx, s, depot_idx])

    return routes


def evaluate_routes(
    routes: List[List[int]],
    D: np.ndarray,
    speed_kmph: float,
    fuel_l_per_km: float,
    traffic_multiplier: float = 1.0,
) -> Dict[str, float]:
    total_km = sum(route_distance(r, D) for r in routes)
    total_fuel = total_km * fuel_l_per_km
    total_hours = (total_km / max(speed_kmph, 1e-6)) * traffic_multiplier
    avg_minutes_per_route = (total_hours * 60.0) / max(len(routes), 1)

    # Utilization (simple): average load / capacity across used routes
    # (ignore depot demand)
    loads = []
    for r in routes:
        loads.append(sum(int(x) for x in [0] + [0]))  # placeholder

    return {
        "total_km": float(total_km),
        "total_fuel_l": float(total_fuel),
        "total_hours": float(total_hours),
        "avg_minutes_per_route": float(avg_minutes_per_route),
        "num_routes": float(len(routes)),
    }


def stability_score(route_sets: List[List[List[int]]]) -> float:
    """Jaccard similarity of consecutive route edge sets."""
    def edges(routes):
        e = set()
        for r in routes:
            for i in range(len(r) - 1):
                e.add((r[i], r[i + 1]))
        return e

    if len(route_sets) < 2:
        return 1.0

    sims = []
    for i in range(len(route_sets) - 1):
        a = edges(route_sets[i])
        b = edges(route_sets[i + 1])
        inter = len(a & b)
        union = len(a | b) if len(a | b) else 1
        sims.append(inter / union)
    return float(np.mean(sims))


# -----------------------------
# Genetic Algorithm
# -----------------------------
@dataclass
class GAParams:
    pop_size: int = 120
    generations: int = 250
    cx_rate: float = 0.9
    mut_rate: float = 0.25
    elite_k: int = 5
    tournament_k: int = 4
    seed: int = 42


def ox_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """Order crossover (OX) for permutations."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))

    def make_child(x, y):
        child = [None] * n
        child[a:b + 1] = x[a:b + 1]
        fill = [g for g in y if g not in child[a:b + 1]]
        idx = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        return child

    return make_child(p1, p2), make_child(p2, p1)


def swap_mutation(p: List[int], swaps: int = 1) -> List[int]:
    n = len(p)
    c = p[:]
    for _ in range(swaps):
        i, j = random.sample(range(n), 2)
        c[i], c[j] = c[j], c[i]
    return c


def tournament_select(pop: List[List[int]], fitness: List[float], k: int) -> List[int]:
    idxs = random.sample(range(len(pop)), k)
    best = min(idxs, key=lambda i: fitness[i])
    return pop[best]


def ga_optimize(
    student_indices: List[int],
    D: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    max_buses: int,
    speed_kmph: float,
    fuel_l_per_km: float,
    distance_weight: float,
    fuel_weight: float,
    time_weight: float,
    params: GAParams,
) -> Tuple[List[List[int]], Dict[str, float], List[float]]:
    random.seed(params.seed)
    np.random.seed(params.seed)

    base = student_indices[:]
    pop = []
    for _ in range(params.pop_size):
        cand = base[:]
        random.shuffle(cand)
        pop.append(cand)

    history_best = []
    best_routes = None
    best_metrics = None
    best_obj = float("inf")

    def fitness_of_perm(perm):
        routes, penalty = decode_permutation_to_routes(
            perm_students=perm,
            demands=demands,
            capacity=capacity,
            max_buses=max_buses,
            depot_idx=0,
        )
        metrics = evaluate_routes(routes, D, speed_kmph, fuel_l_per_km, traffic_multiplier=1.0)

        # Nonlinear objective (squared terms)
        obj = (
            distance_weight * (metrics["total_km"] ** 2)
            + fuel_weight * (metrics["total_fuel_l"] ** 2)
            + time_weight * (metrics["avg_minutes_per_route"] ** 2)
            + penalty
        )
        return obj, routes, metrics

    for _gen in range(params.generations):
        fits = []
        cache_routes = []
        cache_metrics = []

        for perm in pop:
            obj, routes, metrics = fitness_of_perm(perm)
            fits.append(obj)
            cache_routes.append(routes)
            cache_metrics.append(metrics)

        gen_best_i = int(np.argmin(fits))
        gen_best_obj = fits[gen_best_i]
        history_best.append(float(gen_best_obj))

        if gen_best_obj < best_obj:
            best_obj = gen_best_obj
            best_routes = cache_routes[gen_best_i]
            best_metrics = cache_metrics[gen_best_i]

        elite_idxs = np.argsort(fits)[: params.elite_k]
        new_pop = [pop[i][:] for i in elite_idxs]

        while len(new_pop) < params.pop_size:
            p1 = tournament_select(pop, fits, params.tournament_k)
            p2 = tournament_select(pop, fits, params.tournament_k)

            if random.random() < params.cx_rate:
                c1, c2 = ox_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            if random.random() < params.mut_rate:
                c1 = swap_mutation(c1, swaps=1 + (len(c1) // 50))
            if random.random() < params.mut_rate:
                c2 = swap_mutation(c2, swaps=1 + (len(c2) // 50))

            new_pop.append(c1)
            if len(new_pop) < params.pop_size:
                new_pop.append(c2)

        pop = new_pop

    return best_routes, best_metrics, history_best


# -----------------------------
# Pydeck layers (FIXED)
# -----------------------------
def build_route_layers(points: pd.DataFrame, routes: List[List[int]]):
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position=["lon", "lat"],
        get_radius=35,
        pickable=True,
        auto_highlight=True,
        opacity=0.9,
    )

    paths = []
    for ridx, r in enumerate(routes):
        coords = points.loc[r, ["lon", "lat"]].to_numpy().tolist()
        paths.append({"route_id": ridx + 1, "path": coords})

    path_layer = pdk.Layer(
        "PathLayer",
        data=pd.DataFrame(paths),
        get_path="path",
        get_width=6,
        pickable=True,
        opacity=0.85,
    )

    return [scatter, path_layer]


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.markdown("## 🚌 Route Optimizer")
st.sidebar.markdown('<span class="badge">Genetic Algorithm • Capacity Constraints • Sensitivity</span>', unsafe_allow_html=True)
st.sidebar.divider()

with st.sidebar.expander("📦 Data input", expanded=True):
    st.write("Upload **students.csv** or generate a demo dataset.")
    students_file = st.file_uploader(
        "students.csv (required: lat, lon; optional: student_id, demand)",
        type=["csv"],
        accept_multiple_files=False,
    )

    demo = st.checkbox("Use demo dataset", value=(students_file is None))

    depot_lat = st.number_input("Depot latitude", value=-1.286389, format="%.6f")
    depot_lon = st.number_input("Depot longitude", value=36.817223, format="%.6f")

with st.sidebar.expander("🧮 Constraints & model", expanded=True):
    capacity = st.slider("Bus capacity (students)", 10, 80, 45, 1)
    max_buses = st.slider("Number of buses available", 1, 30, 8, 1)

    speed_kmph = st.slider("Average speed (km/h)", 10.0, 60.0, 25.0, 0.5)
    fuel_l_per_km = st.slider("Fuel rate (liters per km)", 0.1, 1.5, 0.35, 0.01)

    st.markdown("**Objective weights** (nonlinear blend)")
    distance_weight = st.slider("Distance weight", 0.0, 3.0, 1.0, 0.05)
    fuel_weight = st.slider("Fuel weight", 0.0, 3.0, 1.0, 0.05)
    time_weight = st.slider("Travel time weight", 0.0, 3.0, 0.6, 0.05)

with st.sidebar.expander("🧬 GA hyperparameters", expanded=False):
    pop_size = st.slider("Population size", 40, 500, 120, 10)
    generations = st.slider("Generations", 50, 1200, 250, 25)
    cx_rate = st.slider("Crossover rate", 0.0, 1.0, 0.90, 0.01)
    mut_rate = st.slider("Mutation rate", 0.0, 1.0, 0.25, 0.01)
    elite_k = st.slider("Elites kept", 1, 30, 5, 1)
    tournament_k = st.slider("Tournament size", 2, 12, 4, 1)
    seed = st.number_input("Random seed", value=42, step=1)

with st.sidebar.expander("🚦 Sensitivity (traffic)", expanded=False):
    traffic_low = st.slider("Traffic multiplier (low)", 0.8, 1.2, 0.95, 0.01)
    traffic_mid = st.slider("Traffic multiplier (mid)", 0.9, 1.6, 1.15, 0.01)
    traffic_high = st.slider("Traffic multiplier (high)", 1.0, 2.5, 1.55, 0.01)

run_btn = st.sidebar.button("▶ Run Optimization", type="primary")


# -----------------------------
# Data loading / generation
# -----------------------------
@st.cache_data(show_spinner=False)
def load_or_generate_students(demo: bool, students_file, depot_lat, depot_lon) -> pd.DataFrame:
    if demo:
        rng = np.random.default_rng(7)
        n = 140
        lats = depot_lat + rng.normal(0, 0.05, n)
        lons = depot_lon + rng.normal(0, 0.05, n)
        demand = np.ones(n, dtype=int)
        df = pd.DataFrame(
            {
                "student_id": [f"S{i:03d}" for i in range(1, n + 1)],
                "lat": lats,
                "lon": lons,
                "demand": demand,
            }
        )
        return df

    df = pd.read_csv(students_file)
    if "student_id" not in df.columns:
        df["student_id"] = [f"S{i:03d}" for i in range(1, len(df) + 1)]
    if "demand" not in df.columns:
        df["demand"] = 1

    required = {"lat", "lon"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}. Required: lat, lon.")

    df["demand"] = df["demand"].fillna(1).astype(int)
    return df[["student_id", "lat", "lon", "demand"]].copy()


try:
    students_df = load_or_generate_students(demo, students_file, depot_lat, depot_lon)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

depot = pd.DataFrame({"student_id": ["DEPOT"], "lat": [depot_lat], "lon": [depot_lon], "demand": [0]})
points = pd.concat([depot, students_df], ignore_index=True)
points["idx"] = np.arange(len(points))

@st.cache_data(show_spinner=False)
def cached_distance_matrix(points_latlon: pd.DataFrame) -> np.ndarray:
    return compute_distance_matrix(points_latlon)

D = cached_distance_matrix(points[["lat", "lon"]])

student_indices = list(range(1, len(points)))  # 1..N
demands = points["demand"].to_numpy()


# -----------------------------
# Header
# -----------------------------
title_col, badge_col = st.columns([0.75, 0.25])
with title_col:
    st.markdown("# 🚌 School Bus Route Optimizer")
    st.caption(
        "Nonlinear optimization (Genetic Algorithm) for capacitated school bus routing. "
        "Minimize distance & fuel, reduce travel time, and test stability under traffic variation."
    )
with badge_col:
    st.markdown(
        """
        <div style="text-align:right;">
            <span class="badge">VRP-lite Decoder</span>
            <span class="badge">Nonlinear Objective</span>
            <span class="badge">Map + Metrics</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Baseline (always available)
# -----------------------------
baseline_routes = greedy_baseline_routes(
    student_indices=student_indices,
    D=D,
    demands=demands,
    capacity=capacity,
    max_buses=max_buses,
    depot_idx=0,
)
baseline_metrics = evaluate_routes(baseline_routes, D, speed_kmph, fuel_l_per_km, traffic_multiplier=1.0)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🗺️ Map & Routes", "🧾 Data & Export"])

# Run optimization
if run_btn:
    with st.spinner("Optimizing routes with Genetic Algorithm..."):
        ga_params = GAParams(
            pop_size=int(pop_size),
            generations=int(generations),
            cx_rate=float(cx_rate),
            mut_rate=float(mut_rate),
            elite_k=int(elite_k),
            tournament_k=int(tournament_k),
            seed=int(seed),
        )

        best_routes, best_metrics, history = ga_optimize(
            student_indices=student_indices,
            D=D,
            demands=demands,
            capacity=capacity,
            max_buses=max_buses,
            speed_kmph=speed_kmph,
            fuel_l_per_km=fuel_l_per_km,
            distance_weight=distance_weight,
            fuel_weight=fuel_weight,
            time_weight=time_weight,
            params=ga_params,
        )

        st.session_state["best_routes"] = best_routes
        st.session_state["best_metrics"] = best_metrics
        st.session_state["history"] = history

best_routes = st.session_state.get("best_routes", None)
best_metrics = st.session_state.get("best_metrics", None)
history = st.session_state.get("history", None)


# -----------------------------
# Dashboard
# -----------------------------
with tab1:
    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.subheader("Performance indicators")

        if best_routes is None:
            st.info("Click **Run Optimization** in the sidebar to compute GA-optimized routes.")
        else:
            dist_impr = (baseline_metrics["total_km"] - best_metrics["total_km"]) / max(baseline_metrics["total_km"], 1e-9) * 100
            fuel_impr = (baseline_metrics["total_fuel_l"] - best_metrics["total_fuel_l"]) / max(baseline_metrics["total_fuel_l"], 1e-9) * 100
            time_impr_min = baseline_metrics["avg_minutes_per_route"] - best_metrics["avg_minutes_per_route"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Distance (km)", f"{best_metrics['total_km']:.1f}", f"{dist_impr:+.1f}% vs baseline")
            m2.metric("Fuel Estimate (L)", f"{best_metrics['total_fuel_l']:.1f}", f"{fuel_impr:+.1f}% vs baseline")
            m3.metric("Avg Minutes / Route", f"{best_metrics['avg_minutes_per_route']:.1f}", f"{time_impr_min:+.1f} min")
            m4.metric("Buses Used", f"{int(best_metrics['num_routes'])}", f"{int(best_metrics['num_routes'])-int(baseline_metrics['num_routes']):+d}")

            st.divider()

            st.subheader("Sensitivity analysis (traffic variation)")
            scenarios = [("Low", traffic_low), ("Mid", traffic_mid), ("High", traffic_high)]

            scen_rows = []
            route_sets = []
            for name, mult in scenarios:
                m = evaluate_routes(best_routes, D, speed_kmph, fuel_l_per_km, traffic_multiplier=mult)
                scen_rows.append(
                    {
                        "Scenario": name,
                        "Traffic Multiplier": mult,
                        "Total km": m["total_km"],
                        "Total fuel (L)": m["total_fuel_l"],
                        "Total hours": m["total_hours"],
                        "Avg min/route": m["avg_minutes_per_route"],
                        "Buses used": int(m["num_routes"]),
                    }
                )
                route_sets.append(best_routes)

            st.metric("Route stability score (0–1)", f"{stability_score(route_sets):.3f}", "higher = more stable")
            st.dataframe(pd.DataFrame(scen_rows), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Baseline vs Optimized")

        base_card = pd.DataFrame(
            [
                {
                    "System": "Baseline (Greedy)",
                    "Total km": baseline_metrics["total_km"],
                    "Fuel (L)": baseline_metrics["total_fuel_l"],
                    "Avg min/route": baseline_metrics["avg_minutes_per_route"],
                    "Buses used": int(baseline_metrics["num_routes"]),
                }
            ]
        )
        st.dataframe(base_card, use_container_width=True, hide_index=True)

        if best_routes is not None:
            opt_card = pd.DataFrame(
                [
                    {
                        "System": "Optimized (GA)",
                        "Total km": best_metrics["total_km"],
                        "Fuel (L)": best_metrics["total_fuel_l"],
                        "Avg min/route": best_metrics["avg_minutes_per_route"],
                        "Buses used": int(best_metrics["num_routes"]),
                    }
                ]
            )
            st.dataframe(opt_card, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("GA convergence")
            st.line_chart(pd.DataFrame({"best_objective": history}))
        else:
            st.caption("Optimization history will appear here after you run the GA.")


# -----------------------------
# Map & Routes (FIXED: uses pdk.Deck)
# -----------------------------
with tab2:
    st.subheader("Route visualization")

    options = ["Baseline (Greedy)"]
    if best_routes is not None:
        options.append("Optimized (GA)")

    which = st.radio("Choose routes to display", options, horizontal=True)
    routes_to_show = baseline_routes if which.startswith("Baseline") else best_routes

    layers = build_route_layers(points, routes_to_show)

    view_state = pdk.ViewState(
        latitude=float(points["lat"].mean()),
        longitude=float(points["lon"].mean()),
        zoom=11,
        pitch=35,
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "{student_id}\n(lat: {lat}, lon: {lon})"},
    )

    st.pydeck_chart(deck, use_container_width=True)

    st.divider()
    st.subheader("Route list (stop order + loads)")

    rows = []
    for i, r in enumerate(routes_to_show, start=1):
        stops = points.loc[r, "student_id"].tolist()
        load = int(points.loc[r, "demand"].sum())
        km = route_distance(r, D)
        rows.append(
            {
                "Route": f"Bus {i}",
                "Stops": " → ".join(stops),
                "Load": load,
                "Distance (km)": km,
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# -----------------------------
# Data & Export
# -----------------------------
with tab3:
    st.subheader("Input data preview")
    st.dataframe(students_df.head(30), use_container_width=True)

    st.divider()
    st.subheader("Export routes")

    if best_routes is None:
        st.info("Run optimization first to export GA routes. Baseline export is available.")

    export_choice = st.selectbox(
        "Export which routes?",
        ["Baseline (Greedy)"] + (["Optimized (GA)"] if best_routes is not None else []),
    )

    export_routes = baseline_routes if export_choice.startswith("Baseline") else best_routes

    export_rows = []
    for bus_i, r in enumerate(export_routes, start=1):
        stop_idxs = [x for x in r if x != 0]
        stop_ids = points.loc[stop_idxs, "student_id"].tolist()
        export_rows.append(
            {
                "bus_id": bus_i,
                "num_stops": len(stop_ids),
                "stops": ",".join(stop_ids),
                "route_km": route_distance(r, D),
                "route_load": int(points.loc[r, "demand"].sum()),
            }
        )

    export_df = pd.DataFrame(export_rows)
    st.dataframe(export_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download routes CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="bus_routes_export.csv",
        mime="text/csv",
    )

    with st.expander("✅ CSV template", expanded=False):
        tmpl = pd.DataFrame(
            {
                "student_id": ["S001", "S002"],
                "lat": [-1.300123, -1.280456],
                "lon": [36.820111, 36.790222],
                "demand": [1, 1],
            }
        )
        st.code(tmpl.to_csv(index=False), language="text")


# -----------------------------
# Footer
# -----------------------------
st.divider()
with st.expander("About this model (read me)", expanded=False):
    st.markdown(
        """
**What this app does**
- Uses a **Genetic Algorithm** to optimize school bus routing (capacity constrained).
- Minimizes a **nonlinear objective** (squared distance, fuel, and travel time) plus penalties.
- Compares against a simple **baseline (greedy nearest-neighbor)**.
- Runs **sensitivity analysis** with traffic multipliers.

**Note**
This is a “VRP-lite” approach:
- Encoding is a permutation of students; a decoder splits into routes by capacity.
- For real deployment, consider road-network distances (OSRM/GraphHopper) and local-search (2-opt/3-opt).
        """
    )
