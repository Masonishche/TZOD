import os
import glob
import time
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
ELECTRICITY_DIR = os.path.join(BASE_DIR, "Electricity")
GAS_DIR         = os.path.join(BASE_DIR, "Gas")

N_CLUSTERS   = 5
N_PARTITIONS = 8

FEATURES = [
    "num_connections",
    "annual_consume",
    "annual_consume_lowtarif_perc",
    "smartmeter_perc",
    "delivery_perc",
    "perc_of_active_connections",
]

def load_csv_folder(folder: str, label: str) -> pd.DataFrame:
    """Зчитує всі CSV з папки та об'єднує в один DataFrame."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"CSV-файли не знайдено у '{folder}'.\n"
            f"Переконайтесь, що Electricity/ та Gas/ знаходяться поряд з Lr3.py"
        )

    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False)
            if df.shape[1] <= 2:                          # деякі файли через ";"
                df = pd.read_csv(path, low_memory=False, sep=";")
            df["source_file"] = os.path.basename(path)
            df["energy_type"] = label
            frames.append(df)
        except Exception as e:
            print(f"    [!] Не вдалось прочитати {os.path.basename(path)}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"    {label:12s}: {len(files):2d} файлів  →  {len(combined):>8,} рядків")
    return combined


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in FEATURES if c in df.columns]
    if not available:
        available = list(df.select_dtypes(include=[np.number]).columns[:6])
        print(f"    [!] Стандартні ознаки не знайдені. Використовуємо: {available}")

    feat = df[available].copy()
    for col in available:
        feat[col] = pd.to_numeric(feat[col], errors="coerce")
    feat.dropna(how="all", inplace=True)
    feat.fillna(feat.median(numeric_only=True), inplace=True)
    return feat.reset_index(drop=True)

def kmeans_pandas(feat: pd.DataFrame) -> tuple:
    X      = feat.values.astype(np.float32)
    means  = X.mean(axis=0)
    stds   = X.std(axis=0) + 1e-9
    X_norm = (X - means) / stds

    rng     = np.random.default_rng(42)
    centers = X_norm[rng.choice(len(X_norm), N_CLUSTERS, replace=False)].copy()

    for _ in range(30):
        dists  = np.linalg.norm(X_norm[:, None] - centers[None], axis=2)
        labels = dists.argmin(axis=1)
        new_c  = np.array([
            X_norm[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
            for k in range(N_CLUSTERS)
        ])
        if np.allclose(centers, new_c, atol=1e-4):
            break
        centers = new_c

    return labels, centers

def kmeans_dask(feat: pd.DataFrame) -> tuple:
    import dask.dataframe as dd

    ddf   = dd.from_pandas(feat, npartitions=N_PARTITIONS)
    means = ddf.mean().compute().values.astype(np.float32)
    stds  = ddf.std().compute().values.astype(np.float32) + 1e-9

    def normalize_partition(part):
        return (part - means) / stds

    norm_ddf = ddf.map_partitions(normalize_partition)

    sample_n = min(100_000, max(len(feat) // 10, N_CLUSTERS * 200))
    X_sample = feat.sample(sample_n, random_state=42).values.astype(np.float32)
    X_sample = (X_sample - means) / stds

    km      = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42, max_iter=300)
    km.fit(X_sample)
    centers = km.cluster_centers_

    def assign_clusters(part):
        Xp   = part.values.astype(np.float32)
        dist = np.linalg.norm(Xp[:, None] - centers[None], axis=2)
        return pd.Series(dist.argmin(axis=1), index=part.index, name="cluster")

    labels = norm_ddf.map_partitions(assign_clusters).compute().values
    return labels, centers

def kmeans_pyarrow(feat: pd.DataFrame) -> tuple:
    import pyarrow as pa
    import pyarrow.compute as pc

    table = pa.Table.from_pandas(feat, preserve_index=False)

    cols      = table.schema.names
    n_cols    = len(cols)
    n_rows    = table.num_rows

    arr_means = np.array([pc.mean(table[c]).as_py() for c in cols], dtype=np.float32)
    arr_stds  = np.array([pc.stddev(table[c]).as_py() for c in cols], dtype=np.float32) + 1e-9

    X = np.empty((n_rows, n_cols), dtype=np.float32)
    for i, c in enumerate(cols):
        raw = table[c].to_pydict() if hasattr(table[c], "to_pydict") else None
        col_np = table[c].to_pylist()
        X[:, i] = (np.asarray(col_np, dtype=np.float32) - arr_means[i]) / arr_stds[i]

    sample_n = min(100_000, max(len(feat) // 10, N_CLUSTERS * 200))
    idx      = np.random.default_rng(42).choice(n_rows, sample_n, replace=False)

    km      = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42, max_iter=300)
    km.fit(X[idx])
    centers = km.cluster_centers_

    dist   = np.linalg.norm(X[:, None] - centers[None], axis=2)
    labels = dist.argmin(axis=1)
    return labels, centers


def run_timed(func, feat: pd.DataFrame, label: str):
    print(f"    {label:<34}", end="", flush=True)
    t0      = time.perf_counter()
    result  = func(feat)
    elapsed = time.perf_counter() - t0
    print(f"{elapsed:8.3f} с")
    return result, elapsed


def print_cluster_stats(feat: pd.DataFrame, labels: np.ndarray, method: str):
    tmp = feat.copy()
    tmp["cluster"] = labels

    stat_cols = [c for c in ["annual_consume", "num_connections", "smartmeter_perc"]
                 if c in tmp.columns]
    if not stat_cols:
        stat_cols = list(feat.columns[:2])

    stats = tmp.groupby("cluster")[stat_cols].mean().round(1)
    count = tmp.groupby("cluster").size().rename("count")
    stats.insert(0, "count", count)
    print(f"\n  Статистика кластерів ({method}, K={N_CLUSTERS}):")
    print(stats.to_string())


def print_summary(results: dict):
    t_pd = results["pandas_sec"]
    print("\n" + "=" * 62)
    print("  ПІДСУМОК ПОРІВНЯННЯ")
    print("=" * 62)
    print(f"  {'Метод':<30} {'Час (с)':>8}   {'Прискорення':>10}")
    print("  " + "-" * 52)
    print(f"  {'Pandas K-Means (baseline)':<30} {t_pd:>8.3f}   {'1.00×':>10}")

    if results.get("dask_sec"):
        t = results["dask_sec"]
        x = results["dask_speedup"]
        print(f"  {'Dask K-Means':<30} {t:>8.3f}   {f'{x:.2f}×':>10}")

    if results.get("pyarrow_sec"):
        t = results["pyarrow_sec"]
        x = results["pyarrow_speedup"]
        print(f"  {'PyArrow K-Means (замість Vaex)':<30} {t:>8.3f}   {f'{x:.2f}×':>10}")

    print("=" * 62)


def main():
    print("=" * 62)
    print("  ПР №3, Варіант 21 — Dutch Energy Clustering")
    print("  Порівняння: Pandas | Dask | PyArrow")
    print("=" * 62)

    print(f"\n▶  Завантаження даних")
    elec_df = load_csv_folder(ELECTRICITY_DIR, "Electricity")
    gas_df  = load_csv_folder(GAS_DIR,         "Gas")
    raw_df  = pd.concat([elec_df, gas_df], ignore_index=True)
    print(f"    Разом рядків: {len(raw_df):,}")

    print(f"\n▶  Підготовка ознак")
    feat_df = prepare_features(raw_df)
    n  = len(feat_df)
    mb = feat_df.memory_usage(deep=True).sum() / 1e6
    print(f"    Рядків: {n:,}  |  RAM: {mb:.1f} MB")
    print(f"    Ознаки: {list(feat_df.columns)}")

    print(f"\n▶  Кластеризація (K={N_CLUSTERS}, Dask partitions={N_PARTITIONS})")
    print(f"    {'Метод':<34} {'Час (с)':>8}")
    print("    " + "-" * 44)

    (l_pd, _), t_pd = run_timed(kmeans_pandas,   feat_df, "Pandas K-Means (baseline)")
    t_dk = t_pa = None
    l_best = l_pd

    try:
        (l_dk, _), t_dk = run_timed(kmeans_dask,    feat_df, "Dask K-Means")
        l_best = l_dk
    except ImportError:
        print(f"    {'Dask K-Means':<34} не встановлено")
        print("    → pip install dask[dataframe]")

    try:
        (l_pa, _), t_pa = run_timed(kmeans_pyarrow, feat_df, "PyArrow K-Means")
    except ImportError:
        print(f"    {'PyArrow K-Means':<34} не встановлено")
        print("    → pip install pyarrow")

    print_cluster_stats(feat_df, l_best, "Dask" if t_dk else "Pandas")

    results = {
        "n_rows":          n,
        "n_clusters":      N_CLUSTERS,
        "features":        list(feat_df.columns),
        "pandas_sec":      round(t_pd, 3),
        "dask_sec":        round(t_dk, 3)        if t_dk else None,
        "pyarrow_sec":     round(t_pa, 3)        if t_pa else None,
        "dask_speedup":    round(t_pd / t_dk, 2) if t_dk else None,
        "pyarrow_speedup": round(t_pd / t_pa, 2) if t_pa else None,
    }

    print_summary(results)

    out = os.path.join(BASE_DIR, "benchmark_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Результати збережено → benchmark_results.json")


if __name__ == "__main__":
    main()
