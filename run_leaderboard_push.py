#!/usr/bin/env python3
"""Challenge 22 leaderboard push runner.

Creates reproducible OOF training artifacts and a pack of submission files.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)
    return float(np.mean(np.abs(y_true - y_pred) / y_true))


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_columns(columns: Iterable[str]) -> List[str]:
    used: Dict[str, int] = {}
    out: List[str] = []
    for c in columns:
        c2 = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(c))
        if not c2:
            c2 = "f"
        if c2 in used:
            used[c2] += 1
            c2 = f"{c2}_{used[c2]}"
        else:
            used[c2] = 0
        out.append(c2)
    return out


@dataclass
class DataBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    block_a: pd.DataFrame
    block_b: pd.DataFrame
    block_c: pd.DataFrame
    block_a_sanitized: pd.DataFrame
    block_b_sanitized: pd.DataFrame
    block_c_sanitized: pd.DataFrame
    cat_block: pd.DataFrame
    cat_test_block: pd.DataFrame
    feature_map_c: Dict[str, str]


@dataclass
class ModelResult:
    name: str
    trained: bool
    mean_mape: float
    fold_mapes: List[float]
    oof_pred: np.ndarray
    best_iterations: List[int]
    params: Dict[str, object]


def load_inputs(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = base_dir / "data" / "training_input.csv"
    test_path = base_dir / "data" / "testing_input.csv"
    target_path = base_dir / "data" / "challenge_34_cfm_trainingoutputfile (2).csv"

    train_df = pd.read_csv(train_path, sep=";")
    test_df = pd.read_csv(test_path, sep=";")

    target_df = pd.read_csv(target_path, sep=";")
    if len(target_df.columns) == 1:
        target_df = pd.read_csv(target_path, sep=",")

    if not {"ID", "TARGET"}.issubset(target_df.columns):
        raise ValueError("Target file must contain ID and TARGET columns")

    if not train_df["ID"].is_unique:
        raise ValueError("Duplicate IDs found in training_input.csv")
    if not test_df["ID"].is_unique:
        raise ValueError("Duplicate IDs found in testing_input.csv")
    if not target_df["ID"].is_unique:
        raise ValueError("Duplicate IDs found in target file")

    train_df = train_df.merge(target_df[["ID", "TARGET"]], on="ID", how="inner")
    if (train_df["TARGET"] <= 0).any():
        raise ValueError("TARGET contains non-positive values; MAPE would be unstable")

    return train_df, test_df, target_df


def _fill_vol(df: pd.DataFrame, vol_cols: List[str]) -> pd.DataFrame:
    vol = df[vol_cols].copy()
    vol = vol.interpolate(axis=1, limit_direction="both")
    vol = vol.fillna(0.0)
    return vol


def _fill_ret(df: pd.DataFrame, ret_cols: List[str]) -> pd.DataFrame:
    return df[ret_cols].fillna(0.0).copy()


def _build_base_blocks(df: pd.DataFrame, vol_cols: List[str], ret_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vol = _fill_vol(df, vol_cols)
    ret = _fill_ret(df, ret_cols)

    block_a = pd.concat(
        [
            df[["date", "product_id"]].reset_index(drop=True),
            vol.reset_index(drop=True),
        ],
        axis=1,
    )
    block_a["vol_mean"] = vol.mean(axis=1)
    block_a["vol_std"] = vol.std(axis=1)
    block_a["vol_min"] = vol.min(axis=1)
    block_a["vol_max"] = vol.max(axis=1)
    block_a["vol_first"] = vol.iloc[:, 0]
    block_a["vol_last"] = vol.iloc[:, -1]
    block_a["vol_range"] = block_a["vol_max"] - block_a["vol_min"]

    block_b = pd.concat([block_a.copy(), ret.reset_index(drop=True)], axis=1)
    block_b["ret_abs_mean"] = ret.abs().mean(axis=1)
    block_b["ret_pos_ratio"] = (ret > 0).mean(axis=1)
    block_b["ret_neg_ratio"] = (ret < 0).mean(axis=1)
    block_b["ret_zero_ratio"] = (ret == 0).mean(axis=1)
    block_b["ret_momentum_signed"] = ret.sum(axis=1)
    block_b["ret_momentum_late"] = ret.iloc[:, len(ret_cols) // 2 :].sum(axis=1)

    return block_a, block_b


def _build_transductive_features(
    train_block_b: pd.DataFrame,
    test_block_b: pd.DataFrame,
    use_transductive: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not use_transductive:
        empty_train = pd.DataFrame(index=train_block_b.index)
        empty_test = pd.DataFrame(index=test_block_b.index)
        return empty_train, empty_test

    use_cols = [
        "product_id",
        "date",
        "vol_mean",
        "vol_std",
        "ret_abs_mean",
        "ret_momentum_signed",
    ]

    t1 = train_block_b[use_cols].copy()
    t1["_is_train"] = 1
    t1["_row_id"] = np.arange(len(t1))

    t2 = test_block_b[use_cols].copy()
    t2["_is_train"] = 0
    t2["_row_id"] = np.arange(len(t2))

    merged = pd.concat([t1, t2], axis=0, ignore_index=True)
    merged = merged.sort_values(["product_id", "date", "_is_train", "_row_id"]).reset_index(drop=True)

    for col in ["vol_mean", "vol_std", "ret_abs_mean"]:
        merged[f"prod_{col}_roll5"] = (
            merged.groupby("product_id")[col].transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))
        )
        merged[f"prod_{col}_roll20"] = (
            merged.groupby("product_id")[col].transform(lambda s: s.rolling(20, min_periods=1).mean().shift(1))
        )
        merged[f"prod_{col}_ewm"] = (
            merged.groupby("product_id")[col].transform(lambda s: s.shift(1).ewm(alpha=0.1, adjust=False).mean())
        )

    trans_cols = [c for c in merged.columns if c.startswith("prod_")]
    merged[trans_cols] = merged[trans_cols].fillna(0.0)

    merged_train = merged[merged["_is_train"] == 1].copy()
    merged_test = merged[merged["_is_train"] == 0].copy()

    merged_train = merged_train.sort_values("_row_id")
    merged_test = merged_test.sort_values("_row_id")

    trans_train = merged_train[trans_cols].reset_index(drop=True)
    trans_test = merged_test[trans_cols].reset_index(drop=True)

    return trans_train, trans_test


def build_data_bundle(base_dir: Path, use_transductive: bool) -> DataBundle:
    train_df, test_df, _ = load_inputs(base_dir)

    vol_cols = [c for c in train_df.columns if c.startswith("volatility ")]
    ret_cols = [c for c in train_df.columns if c.startswith("return ")]

    if len(vol_cols) == 0 or len(ret_cols) == 0:
        raise ValueError("Could not infer volatility/return columns")

    block_a_train, block_b_train = _build_base_blocks(train_df, vol_cols, ret_cols)
    block_a_test, block_b_test = _build_base_blocks(test_df, vol_cols, ret_cols)

    trans_train, trans_test = _build_transductive_features(block_b_train, block_b_test, use_transductive)

    block_c_train = pd.concat([block_b_train.reset_index(drop=True), trans_train], axis=1)
    block_c_test = pd.concat([block_b_test.reset_index(drop=True), trans_test], axis=1)

    # CatBoost path: keep original names and NaN tolerance.
    cat_block = block_c_train.copy()
    cat_test_block = block_c_test.copy()

    # Fill numeric NaNs for linear/lgb/xgb models.
    for df in [block_a_train, block_b_train, block_c_train, block_a_test, block_b_test, block_c_test]:
        num_cols = [c for c in df.columns if c not in ["date", "product_id"]]
        df[num_cols] = df[num_cols].astype(float).fillna(0.0)

    # Sanitize names (persisted for block C, then mapped subsets).
    c_names = list(block_c_train.columns)
    c_sanitized = sanitize_columns(c_names)
    c_map = dict(zip(c_names, c_sanitized))

    block_c_train_s = block_c_train.rename(columns=c_map)
    block_c_test_s = block_c_test.rename(columns=c_map)

    a_map = {k: v for k, v in c_map.items() if k in block_a_train.columns}
    b_map = {k: v for k, v in c_map.items() if k in block_b_train.columns}

    block_a_train_s = block_a_train.rename(columns=a_map)
    block_b_train_s = block_b_train.rename(columns=b_map)
    block_a_test_s = block_a_test.rename(columns=a_map)
    block_b_test_s = block_b_test.rename(columns=b_map)

    # Keep order and consistent dtypes.
    for df in [
        block_a_train_s,
        block_b_train_s,
        block_c_train_s,
        block_a_test_s,
        block_b_test_s,
        block_c_test_s,
    ]:
        if "date" in df.columns:
            df["date"] = df["date"].astype(float)
        if "product_id" in df.columns:
            df["product_id"] = df["product_id"].astype(float)

    y = train_df["TARGET"].values.astype(float)
    groups = train_df["date"].values

    return DataBundle(
        train_df=train_df,
        test_df=test_df,
        y=y,
        groups=groups,
        train_ids=train_df["ID"].values,
        test_ids=test_df["ID"].values,
        block_a=block_a_train,
        block_b=block_b_train,
        block_c=block_c_train,
        block_a_sanitized=block_a_train_s,
        block_b_sanitized=block_b_train_s,
        block_c_sanitized=block_c_train_s,
        cat_block=cat_block,
        cat_test_block=cat_test_block,
        feature_map_c=c_map,
    )


def get_test_blocks(bundle: DataBundle, use_transductive: bool) -> Dict[str, pd.DataFrame]:
    # Rebuild from same data path to avoid accidental mutation and keep consistent transforms.
    # This is deterministic and cheap relative to model training.
    base_dir = Path(__file__).resolve().parent
    _, test_df, _ = load_inputs(base_dir)

    vol_cols = [c for c in bundle.train_df.columns if c.startswith("volatility ")]
    ret_cols = [c for c in bundle.train_df.columns if c.startswith("return ")]

    block_a_test, block_b_test = _build_base_blocks(test_df, vol_cols, ret_cols)
    _, test_trans = _build_transductive_features(bundle.block_b, block_b_test, use_transductive)
    block_c_test = pd.concat([block_b_test.reset_index(drop=True), test_trans], axis=1)

    cat_test = block_c_test.copy()

    for df in [block_a_test, block_b_test, block_c_test]:
        num_cols = [c for c in df.columns if c not in ["date", "product_id"]]
        df[num_cols] = df[num_cols].astype(float).fillna(0.0)

    a_map = {k: v for k, v in bundle.feature_map_c.items() if k in block_a_test.columns}
    b_map = {k: v for k, v in bundle.feature_map_c.items() if k in block_b_test.columns}

    block_a_test_s = block_a_test.rename(columns=a_map)
    block_b_test_s = block_b_test.rename(columns=b_map)
    block_c_test_s = block_c_test.rename(columns=bundle.feature_map_c)

    for df in [block_a_test_s, block_b_test_s, block_c_test_s]:
        if "date" in df.columns:
            df["date"] = df["date"].astype(float)
        if "product_id" in df.columns:
            df["product_id"] = df["product_id"].astype(float)

    return {
        "A": block_a_test_s,
        "B": block_b_test_s,
        "C": block_c_test_s,
        "CAT": cat_test,
    }


def _folds(groups: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(groups))
    return list(gkf.split(idx, groups=groups))


def train_ridge_oof(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    alpha: float,
) -> ModelResult:
    folds = _folds(groups, n_splits)
    oof = np.zeros(len(y), dtype=float)
    mapes: List[float] = []

    for tr_idx, va_idx in folds:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, random_state=seed)),
            ]
        )
        sw = 1.0 / np.maximum(y[tr_idx], 1e-6)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model.fit(X.iloc[tr_idx], np.log1p(y[tr_idx]), ridge__sample_weight=sw)
            pred = np.expm1(model.predict(X.iloc[va_idx]))
        pred = np.maximum(pred, 1e-6)
        oof[va_idx] = pred
        mapes.append(safe_mape(y[va_idx], pred))

    return ModelResult(
        name=name,
        trained=True,
        mean_mape=float(np.mean(mapes)),
        fold_mapes=[float(x) for x in mapes],
        oof_pred=oof,
        best_iterations=[],
        params={"alpha": alpha, "target_transform": "log1p", "sample_weight": "1/target"},
    )


def train_lgbm_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> ModelResult:
    import lightgbm as lgb

    folds = _folds(groups, n_splits)
    oof = np.zeros(len(y), dtype=float)
    mapes: List[float] = []
    best_iters: List[int] = []

    params = {
        "objective": "regression_l1",
        "learning_rate": 0.03,
        "num_leaves": 255,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": 6000,
        "metric": "None",
        "verbosity": -1,
        "seed": seed,
    }

    for tr_idx, va_idx in folds:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X.iloc[tr_idx],
            np.log1p(y[tr_idx]),
            eval_set=[(X.iloc[va_idx], np.log1p(y[va_idx]))],
            eval_metric=lambda yt, yp: (
                "mape",
                safe_mape(np.expm1(yt), np.expm1(yp)),
                False,
            ),
            callbacks=[lgb.early_stopping(300, verbose=False)],
        )
        best_iter = int(model.best_iteration_ or params["n_estimators"])
        pred = np.expm1(model.predict(X.iloc[va_idx], num_iteration=best_iter))
        pred = np.maximum(pred, 1e-6)
        oof[va_idx] = pred
        mapes.append(safe_mape(y[va_idx], pred))
        best_iters.append(best_iter)

    return ModelResult(
        name="M3_lgbm_l1",
        trained=True,
        mean_mape=float(np.mean(mapes)),
        fold_mapes=[float(x) for x in mapes],
        oof_pred=oof,
        best_iterations=best_iters,
        params=params,
    )


def train_xgb_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> ModelResult:
    import xgboost as xgb

    folds = _folds(groups, n_splits)
    oof = np.zeros(len(y), dtype=float)
    mapes: List[float] = []
    best_iters: List[int] = []

    params = {
        "objective": "reg:squarederror",
        "eta": 0.03,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "eval_metric": "mae",
        "seed": seed,
    }

    for tr_idx, va_idx in folds:
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=np.log1p(y[tr_idx]))
        dva = xgb.DMatrix(X.iloc[va_idx], label=np.log1p(y[va_idx]))
        model = xgb.train(
            params,
            dtr,
            num_boost_round=5000,
            evals=[(dva, "valid")],
            early_stopping_rounds=300,
            verbose_eval=False,
        )
        best_iter = int(model.best_iteration + 1)
        pred = np.expm1(model.predict(dva, iteration_range=(0, best_iter)))
        pred = np.maximum(pred, 1e-6)
        oof[va_idx] = pred
        mapes.append(safe_mape(y[va_idx], pred))
        best_iters.append(best_iter)

    return ModelResult(
        name="M4_xgb",
        trained=True,
        mean_mape=float(np.mean(mapes)),
        fold_mapes=[float(x) for x in mapes],
        oof_pred=oof,
        best_iterations=best_iters,
        params=params,
    )


def train_cat_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> ModelResult:
    from catboost import CatBoostRegressor

    folds = _folds(groups, n_splits)
    oof = np.zeros(len(y), dtype=float)
    mapes: List[float] = []
    best_iters: List[int] = []

    params = {
        "loss_function": "MAPE",
        "eval_metric": "MAPE",
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3,
        "iterations": 4000,
        "random_seed": seed,
        "od_type": "Iter",
        "od_wait": 300,
        "verbose": False,
    }

    cat_cols: List[str] = []
    if "date" in X.columns:
        cat_cols.append("date")
    if "product_id" in X.columns:
        cat_cols.append("product_id")

    for tr_idx, va_idx in folds:
        model = CatBoostRegressor(**params)
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            cat_features=cat_cols,
            eval_set=(X.iloc[va_idx], y[va_idx]),
            use_best_model=True,
        )
        pred = model.predict(X.iloc[va_idx])
        pred = np.maximum(pred, 1e-6)
        oof[va_idx] = pred
        mapes.append(safe_mape(y[va_idx], pred))
        best_iters.append(int(model.get_best_iteration() or params["iterations"]))

    return ModelResult(
        name="M5_cat",
        trained=True,
        mean_mape=float(np.mean(mapes)),
        fold_mapes=[float(x) for x in mapes],
        oof_pred=oof,
        best_iterations=best_iters,
        params=params,
    )


def min_corr_with_admitted(candidate: np.ndarray, admitted: Dict[str, np.ndarray]) -> float:
    if not admitted:
        return 1.0
    corrs = []
    for arr in admitted.values():
        c = np.corrcoef(candidate, arr)[0, 1]
        if np.isnan(c):
            c = 1.0
        corrs.append(float(c))
    return float(min(corrs))


def generate_simplex_weights(n: int, step: float) -> Iterable[np.ndarray]:
    total = int(round(1.0 / step))

    def rec(prefix: List[int], remaining: int, slots: int):
        if slots == 1:
            yield prefix + [remaining]
            return
        for val in range(remaining + 1):
            yield from rec(prefix + [val], remaining - val, slots - 1)

    for ints in rec([], total, n):
        arr = np.array(ints, dtype=float) * step
        # Numerical guard.
        arr[-1] = 1.0 - arr[:-1].sum()
        if (arr >= -1e-12).all():
            arr = np.maximum(arr, 0.0)
            arr /= arr.sum()
            yield arr


def blend_score(
    preds_matrix: np.ndarray,
    weights: np.ndarray,
    y: np.ndarray,
    sample_idx: np.ndarray | None = None,
) -> float:
    pred = preds_matrix @ weights
    if sample_idx is not None:
        return safe_mape(y[sample_idx], pred[sample_idx])
    return safe_mape(y, pred)


def fine_tune_weights(
    preds_matrix: np.ndarray,
    y: np.ndarray,
    start_w: np.ndarray,
    step: float = 0.01,
    max_rounds: int = 20,
) -> Tuple[np.ndarray, float]:
    best_w = start_w.copy()
    best_s = blend_score(preds_matrix, best_w, y)
    n = len(best_w)

    for _ in range(max_rounds):
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j or best_w[j] < step:
                    continue
                cand = best_w.copy()
                cand[i] += step
                cand[j] -= step
                cand = np.maximum(cand, 0.0)
                cand /= cand.sum()
                s = blend_score(preds_matrix, cand, y)
                if s + 1e-12 < best_s:
                    best_w, best_s = cand, s
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best_w, best_s


def fit_blend(
    blend_name: str,
    model_names: List[str],
    oof_dict: Dict[str, np.ndarray],
    y: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, float, pd.DataFrame]:
    if len(model_names) == 1:
        weights = np.array([1.0])
        score = safe_mape(y, oof_dict[model_names[0]])
        row = {
            "blend_name": blend_name,
            "phase": "single",
            "score": score,
            "weights_json": json.dumps({model_names[0]: 1.0}),
        }
        return weights, score, pd.DataFrame([row])

    preds = np.column_stack([oof_dict[m] for m in model_names])

    rng = np.random.default_rng(seed)
    if len(y) > 180_000:
        sample_idx = np.sort(rng.choice(len(y), size=180_000, replace=False))
    else:
        sample_idx = None

    records = []
    coarse: List[Tuple[np.ndarray, float]] = []

    for w in generate_simplex_weights(len(model_names), step=0.05):
        s = blend_score(preds, w, y, sample_idx=sample_idx)
        coarse.append((w, s))
        records.append(
            {
                "blend_name": blend_name,
                "phase": "coarse_0.05",
                "score": s,
                "weights_json": json.dumps({m: float(x) for m, x in zip(model_names, w)}),
            }
        )

    coarse.sort(key=lambda x: x[1])
    top = coarse[:20]

    best_w = top[0][0]
    best_s = blend_score(preds, best_w, y)

    for w, _ in top:
        tuned_w, tuned_s = fine_tune_weights(preds, y, w, step=0.01, max_rounds=30)
        records.append(
            {
                "blend_name": blend_name,
                "phase": "fine_0.01",
                "score": tuned_s,
                "weights_json": json.dumps({m: float(x) for m, x in zip(model_names, tuned_w)}),
            }
        )
        if tuned_s < best_s:
            best_w, best_s = tuned_w, tuned_s

    records.append(
        {
            "blend_name": blend_name,
            "phase": "selected",
            "score": best_s,
            "weights_json": json.dumps({m: float(x) for m, x in zip(model_names, best_w)}),
        }
    )

    return best_w, best_s, pd.DataFrame(records)


def calibrate_predictions(y: np.ndarray, pred: np.ndarray) -> Tuple[Dict[str, float], float, np.ndarray]:
    base = safe_mape(y, pred)
    best_score = base
    best_cfg = {"scale": 1.0, "low_q": 0.0, "high_q": 1.0}
    best_pred = pred.copy()

    scales = np.arange(0.95, 1.051, 0.005)
    lows = [0.0, 0.0001, 0.0005]
    highs = [1.0, 0.9999, 0.9995, 0.999]

    for s in scales:
        scaled = np.maximum(pred * s, 1e-6)
        for lq in lows:
            lo = np.quantile(scaled, lq)
            for hq in highs:
                if hq <= lq:
                    continue
                hi = np.quantile(scaled, hq)
                cand = np.clip(scaled, lo, hi)
                score = safe_mape(y, cand)
                if score + 1e-12 < best_score:
                    best_score = score
                    best_cfg = {"scale": float(s), "low_q": float(lq), "high_q": float(hq)}
                    best_pred = cand

    return best_cfg, best_score, best_pred


def apply_calibration(pred: np.ndarray, cfg: Dict[str, float]) -> np.ndarray:
    out = np.maximum(pred * float(cfg.get("scale", 1.0)), 1e-6)
    lq = float(cfg.get("low_q", 0.0))
    hq = float(cfg.get("high_q", 1.0))
    lo = np.quantile(out, lq)
    hi = np.quantile(out, hq)
    out = np.clip(out, lo, hi)
    return np.maximum(out, 1e-6)


def train_full_models(
    admitted_models: List[str],
    bundle: DataBundle,
    test_blocks: Dict[str, pd.DataFrame],
    results: Dict[str, ModelResult],
    seed: int,
) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}

    if "M1_ridge_a" in admitted_models:
        m1 = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=10.0, random_state=seed)),
            ]
        )
        sw = 1.0 / np.maximum(bundle.y, 1e-6)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            m1.fit(bundle.block_a_sanitized, np.log1p(bundle.y), ridge__sample_weight=sw)
            preds["M1_ridge_a"] = np.maximum(np.expm1(m1.predict(test_blocks["A"])), 1e-6)

    if "M2_ridge_b" in admitted_models:
        m2 = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=10.0, random_state=seed)),
            ]
        )
        sw = 1.0 / np.maximum(bundle.y, 1e-6)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            m2.fit(bundle.block_b_sanitized, np.log1p(bundle.y), ridge__sample_weight=sw)
            preds["M2_ridge_b"] = np.maximum(np.expm1(m2.predict(test_blocks["B"])), 1e-6)

    if "M3_lgbm_l1" in admitted_models:
        import lightgbm as lgb

        base = results["M3_lgbm_l1"].params.copy()
        n_est = int(np.median(results["M3_lgbm_l1"].best_iterations)) if results["M3_lgbm_l1"].best_iterations else 800
        base["n_estimators"] = max(100, n_est)
        model = lgb.LGBMRegressor(**base)
        model.fit(bundle.block_c_sanitized, np.log1p(bundle.y))
        preds["M3_lgbm_l1"] = np.maximum(np.expm1(model.predict(test_blocks["C"])), 1e-6)

    if "M4_xgb" in admitted_models:
        import xgboost as xgb

        params = results["M4_xgb"].params.copy()
        n_boost = int(np.median(results["M4_xgb"].best_iterations)) if results["M4_xgb"].best_iterations else 600
        dtr = xgb.DMatrix(bundle.block_c_sanitized, label=np.log1p(bundle.y))
        dte = xgb.DMatrix(test_blocks["C"])
        model = xgb.train(params, dtr, num_boost_round=max(100, n_boost), verbose_eval=False)
        preds["M4_xgb"] = np.maximum(np.expm1(model.predict(dte)), 1e-6)

    if "M5_cat" in admitted_models:
        from catboost import CatBoostRegressor

        params = results["M5_cat"].params.copy()
        n_est = int(np.median(results["M5_cat"].best_iterations)) if results["M5_cat"].best_iterations else 400
        params["iterations"] = max(100, n_est)
        params["verbose"] = False
        cat_cols = [c for c in ["date", "product_id"] if c in bundle.cat_block.columns]
        model = CatBoostRegressor(**params)
        model.fit(bundle.cat_block, bundle.y, cat_features=cat_cols)
        preds["M5_cat"] = np.maximum(model.predict(test_blocks["CAT"]), 1e-6)

    return preds


def save_submission_pair(
    ids: np.ndarray,
    pred: np.ndarray,
    semicolon_path: Path,
    comma_path: Path,
) -> None:
    df = pd.DataFrame({"ID": ids, "TARGET": np.maximum(pred, 1e-6)})
    if not df["TARGET"].notna().all():
        raise ValueError("Submission contains NaN")
    if not (df["TARGET"] > 0).all():
        raise ValueError("Submission contains non-positive values")

    semicolon_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(semicolon_path, sep=";", index=False)
    df.to_csv(comma_path, index=False)


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    base_dir = Path(__file__).resolve().parent
    run_dir = base_dir / "runs_push" / timestamp_tag()
    run_dir.mkdir(parents=True, exist_ok=True)

    start_ts = time.time()

    if args.stage == "baseline":
        args.max_submit_pack = min(int(args.max_submit_pack), 1)

    bundle = build_data_bundle(base_dir=base_dir, use_transductive=args.use_transductive)
    test_blocks = get_test_blocks(bundle, use_transductive=args.use_transductive)

    if len(bundle.test_ids) != len(test_blocks["A"]):
        raise ValueError("Test feature row count mismatch")

    folds = _folds(bundle.groups, args.n_splits)
    cv_records: List[Dict[str, object]] = []

    results: Dict[str, ModelResult] = {}

    # Stage baseline anchor.
    m1 = train_ridge_oof(
        name="M1_ridge_a",
        X=bundle.block_a_sanitized,
        y=bundle.y,
        groups=bundle.groups,
        n_splits=args.n_splits,
        seed=args.seed,
        alpha=10.0,
    )
    results[m1.name] = m1
    for fold_idx, score in enumerate(m1.fold_mapes):
        cv_records.append({"model": m1.name, "fold": fold_idx, "mape": score, "rows": int(len(folds[fold_idx][1]))})

    baseline_mape = m1.mean_mape

    if args.stage in {"baseline"}:
        admitted = ["M1_ridge_a"]
    else:
        m2 = train_ridge_oof(
            name="M2_ridge_b",
            X=bundle.block_b_sanitized,
            y=bundle.y,
            groups=bundle.groups,
            n_splits=args.n_splits,
            seed=args.seed,
            alpha=10.0,
        )
        results[m2.name] = m2
        for fold_idx, score in enumerate(m2.fold_mapes):
            cv_records.append({"model": m2.name, "fold": fold_idx, "mape": score, "rows": int(len(folds[fold_idx][1]))})

        elapsed_min = (time.time() - start_ts) / 60.0
        remaining_min = args.time_budget_min - elapsed_min

        if remaining_min > 20:
            m3 = train_lgbm_oof(
                X=bundle.block_c_sanitized,
                y=bundle.y,
                groups=bundle.groups,
                n_splits=args.n_splits,
                seed=args.seed,
            )
            results[m3.name] = m3
            for fold_idx, score in enumerate(m3.fold_mapes):
                cv_records.append({"model": m3.name, "fold": fold_idx, "mape": score, "rows": int(len(folds[fold_idx][1]))})

        elapsed_min = (time.time() - start_ts) / 60.0
        remaining_min = args.time_budget_min - elapsed_min

        if remaining_min > 30:
            try:
                m4 = train_xgb_oof(
                    X=bundle.block_c_sanitized,
                    y=bundle.y,
                    groups=bundle.groups,
                    n_splits=args.n_splits,
                    seed=args.seed,
                )
                results[m4.name] = m4
                for fold_idx, score in enumerate(m4.fold_mapes):
                    cv_records.append({"model": m4.name, "fold": fold_idx, "mape": score, "rows": int(len(folds[fold_idx][1]))})
            except Exception as exc:  # pragma: no cover
                print(f"[warn] Skipping XGBoost due to error: {exc}")

        elapsed_min = (time.time() - start_ts) / 60.0
        remaining_min = args.time_budget_min - elapsed_min

        if remaining_min > 30:
            try:
                m5 = train_cat_oof(
                    X=bundle.cat_block,
                    y=bundle.y,
                    groups=bundle.groups,
                    n_splits=args.n_splits,
                    seed=args.seed,
                )
                results[m5.name] = m5
                for fold_idx, score in enumerate(m5.fold_mapes):
                    cv_records.append({"model": m5.name, "fold": fold_idx, "mape": score, "rows": int(len(folds[fold_idx][1]))})
            except Exception as exc:  # pragma: no cover
                print(f"[warn] Skipping CatBoost due to error: {exc}")

        admitted = ["M1_ridge_a"]
        admitted_oof = {"M1_ridge_a": results["M1_ridge_a"].oof_pred}
        gate_rows = []

        for model_name in ["M2_ridge_b", "M3_lgbm_l1", "M4_xgb", "M5_cat"]:
            if model_name not in results:
                gate_rows.append({"model": model_name, "available": False, "accepted": False, "reason": "not_trained"})
                continue

            res = results[model_name]
            improvement = baseline_mape - res.mean_mape
            corr = min_corr_with_admitted(res.oof_pred, admitted_oof)
            accept = (improvement >= 0.0007) or (corr < 0.985)
            reason = f"improvement={improvement:.6f}, min_corr={corr:.6f}"

            if accept:
                admitted.append(model_name)
                admitted_oof[model_name] = res.oof_pred

            gate_rows.append(
                {
                    "model": model_name,
                    "available": True,
                    "accepted": accept,
                    "reason": reason,
                }
            )

        pd.DataFrame(gate_rows).to_csv(run_dir / "gating.csv", index=False)

    # Persist core artifacts from training stage.
    pd.DataFrame(cv_records).to_csv(run_dir / "cv_scores.csv", index=False)

    registry = {
        "seed": args.seed,
        "n_splits": args.n_splits,
        "time_budget_min": args.time_budget_min,
        "use_transductive": args.use_transductive,
        "admitted_models": admitted,
        "baseline_mape": baseline_mape,
        "models": {},
    }

    oof_frame = pd.DataFrame({"ID": bundle.train_ids, "TARGET": bundle.y})
    for name, res in results.items():
        registry["models"][name] = {
            "trained": res.trained,
            "mean_mape": res.mean_mape,
            "fold_mapes": res.fold_mapes,
            "best_iterations": res.best_iterations,
            "params": res.params,
            "accepted": name in admitted,
        }
        oof_frame[f"oof_{name}"] = res.oof_pred

    with open(run_dir / "model_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    oof_frame.to_parquet(run_dir / "oof_predictions.parquet", index=False)

    # Early return for train stage.
    if args.stage == "train":
        return {
            "run_dir": run_dir,
            "admitted_models": admitted,
            "baseline_mape": baseline_mape,
            "oof_models": list(results.keys()),
        }

    blend_rows = []
    oof_dict = {name: results[name].oof_pred for name in admitted}

    # Blend variants from OOF.
    linear_models = [m for m in ["M1_ridge_a", "M2_ridge_b"] if m in admitted]
    lgb_models = [m for m in ["M1_ridge_a", "M2_ridge_b", "M3_lgbm_l1"] if m in admitted]
    full_models = admitted.copy()

    w_linear, s_linear, df_linear = fit_blend("linear", linear_models, oof_dict, bundle.y, args.seed)
    blend_rows.append(df_linear)

    w_lgb, s_lgb, df_lgb = fit_blend("linear_lgb", lgb_models, oof_dict, bundle.y, args.seed)
    blend_rows.append(df_lgb)

    w_full, s_full, df_full = fit_blend("full", full_models, oof_dict, bundle.y, args.seed)
    blend_rows.append(df_full)

    full_blend_oof = np.column_stack([oof_dict[m] for m in full_models]) @ w_full
    cal_cfg, cal_score, cal_oof = calibrate_predictions(bundle.y, full_blend_oof)
    use_calibration = cal_score + 1e-12 < s_full

    blend_rows.append(
        pd.DataFrame(
            [
                {
                    "blend_name": "full",
                    "phase": "calibration",
                    "score": cal_score,
                    "weights_json": json.dumps(cal_cfg),
                }
            ]
        )
    )

    blend_df = pd.concat(blend_rows, ignore_index=True)
    blend_df.to_csv(run_dir / "blend_search.csv", index=False)

    if args.stage == "blend":
        return {
            "run_dir": run_dir,
            "admitted_models": admitted,
            "blend_scores": {
                "linear": s_linear,
                "linear_lgb": s_lgb,
                "full": s_full,
                "full_calibrated": cal_score,
            },
        }

    # Submit pack / full.
    test_preds = train_full_models(admitted, bundle, test_blocks, results, args.seed)

    # Build submission variants.
    variants: List[Tuple[str, np.ndarray, str]] = []

    v1 = test_preds.get("M1_ridge_a")
    if v1 is None:
        raise ValueError("Anchor model prediction missing")
    variants.append(("v1_anchor_linear", v1, "Anchor weighted Ridge on Block A"))

    v2 = np.column_stack([test_preds[m] for m in linear_models]) @ w_linear
    variants.append(("v2_linear_blend", v2, "Blend of admitted linear models"))

    v3 = np.column_stack([test_preds[m] for m in lgb_models]) @ w_lgb
    variants.append(("v3_linear_lgbm", v3, "Blend of linear + LGBM subset"))

    v4 = np.column_stack([test_preds[m] for m in full_models]) @ w_full
    variants.append(("v4_full_ensemble", v4, "Full admitted-model blend"))

    if use_calibration:
        v5 = apply_calibration(v4, cal_cfg)
        rationale = "Full blend with calibrated scale/quantile clipping"
    else:
        v5 = v4.copy()
        rationale = "Full blend (calibration kept off: no OOF gain)"
    variants.append(("v5_full_ensemble_calibrated", v5, rationale))

    variants = variants[: max(1, args.max_submit_pack)]

    submissions_dir = base_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    stamp = timestamp_tag()
    manifest_rows = []

    for i, (name, pred, rationale) in enumerate(variants, start=1):
        semicolon_path = submissions_dir / f"{stamp}_{name}.csv"
        comma_path = submissions_dir / f"{stamp}_{name}_comma.csv"
        save_submission_pair(bundle.test_ids, pred, semicolon_path, comma_path)

        manifest_rows.append(
            {
                "order": i,
                "variant": name,
                "semicolon_path": str(semicolon_path),
                "comma_path": str(comma_path),
                "rationale": rationale,
                "local_reference": {
                    "v1_anchor_linear": registry["models"]["M1_ridge_a"]["mean_mape"],
                    "v2_linear_blend": s_linear,
                    "v3_linear_lgbm": s_lgb,
                    "v4_full_ensemble": s_full,
                    "v5_full_ensemble_calibrated": cal_score if use_calibration else s_full,
                }.get(name, np.nan),
                "public_score": "",
                "rank_after_submit": "",
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(run_dir / "submission_manifest.csv", index=False)

    # Add recommended probing order for stage=full.
    if args.stage == "full":
        probe_rows = []
        probe_order = ["v1_anchor_linear", "v3_linear_lgbm", "v4_full_ensemble", "v5_full_ensemble_calibrated"]
        for rank, vname in enumerate(probe_order, start=1):
            match = manifest[manifest["variant"] == vname]
            if len(match) == 0:
                continue
            rec = match.iloc[0].to_dict()
            rec["probe_order"] = rank
            probe_rows.append(rec)
        pd.DataFrame(probe_rows).to_csv(run_dir / "probing_plan.csv", index=False)

    return {
        "run_dir": run_dir,
        "admitted_models": admitted,
        "variant_count": len(variants),
        "best_full_blend_oof": s_full,
        "best_calibrated_oof": cal_score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Challenge 22 leaderboard push runner")
    parser.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=["baseline", "train", "blend", "submit_pack", "full"],
        help="Execution stage",
    )
    parser.add_argument("--time-budget-min", type=int, default=180)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-submit-pack", type=int, default=5)
    parser.add_argument("--use-transductive", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
