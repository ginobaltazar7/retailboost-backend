"""
Temporal Demand Forecasting Pipeline for RetailBoost
Date: 2025-12
Author: Gino Baltazar
====================================

An automated feature engineering and ensemble learning system designed for 
time-series volume prediction. This module integrates historical activity logs 
with forward-looking reservation data to predict daily traffic.

Key Components:
---------------
1. EpanechnikovKernel: 
   Generates mathematically robust probability transition matrices using 
   Laplace-smoothed kernels for temporal weighting.

2. TimeSeriesFeatureEngine: 
   Object-oriented engine that generates sliding-window statistical aggregates 
   (lag features, rolling trends) via configuration injection.

3. Ensemble Modeling: 
   Implements LightGBM with K-Fold cross-validation and stacking, utilizing 
   temporal proximity weighting to prioritize recent signals.

Usage:
------
Prepare input dictionary with train/test/relation dataframes and pass to 
`train_forecasts(dict_dfs)`.
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import preprocessing
import lightgbm as lgb
from lightgbm.callback import early_stopping
import pandas as pd
import numpy as np
import gc
import os
from datetime import date, timedelta

# Import your existing utilities
from scripts.forecast_utils import (
    encode_onehot, encode_count, feat_count, feat_nunique, feat_mean,
    feat_kernelMedian, feat_std, feat_median, feat_max, feat_min,
    feat_sum, feat_var, feat_quantile, merge_count, merge_nunique,
    merge_median, merge_mean, merge_sum, merge_max, merge_min,
    merge_std, merge_var
)

# --- GLOBAL CONSTANTS ---
# Defined here so they are accessible to Class definitions below
# Maximum Lookback Horizon (in days) that the kernel smoothing logic can handle.
MAX_LOOKBACK_HORIZON = 2000 

# ==============================================================================
# 1. Transition Matrix / Probability Lookup
# ==============================================================================
class EpanechnikovKernel:
    def __init__(self, size=MAX_LOOKBACK_HORIZON, smoothing_method='laplace_strict'):
        self.size = size
        self.matrix = self._generate_matrix(smoothing_method)

    def _generate_matrix(self, method):
        Pr = np.zeros((self.size, self.size))
        Pr[1, 0] = 1
        Pr[2, :2] = [0.5, 0.5]
        for i in range(3, self.size):
            scale = (i - 1) / 2.
            x = np.arange(-(i + 1) / 2. + 1, (i + 1) / 2., step=1) / scale
            y = 3. / 4. * (1 - x**2)
            y = y / np.sum(y)
            Pr[i, :i] = y

        Pr_Smoothed = Pr.copy()
        if method == 'laplace_strict':
            for i in range(3, self.size):
                Pr_Smoothed[i, :i] = (Pr_Smoothed[i, :i] * i + 1) / (2 * i)
        return Pr_Smoothed

    def get_matrix(self):
        return self.matrix

# ==============================================================================
# 2. Feature Engineering Engine
# ==============================================================================
class TimeSeriesFeatureEngine:
    def __init__(self, kernel_matrix, config, store_info_df, date_info_df):
        self.kernel_matrix = kernel_matrix
        self.cfg = config
        self.store_info = store_info_df 
        self.date_info = date_info_df

    def _enrich_dates(self, df):
        """Standardizes date features and merges external info."""
        date_col = self.cfg['cols']['date']
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Merge Date Info (Holidays)
        if self.date_info is not None:
            df = df.merge(self.date_info, on=date_col, how="left").fillna(-1)

        # Merge Store Info (Genre, Area)
        if self.store_info is not None:
            df = df.merge(self.store_info, on=self.cfg['cols']['id'], how="left").fillna(-1)

        df["weekday"] = pd.to_datetime(df[date_col]).dt.weekday
        df["day"] = pd.to_datetime(df[date_col]).dt.day
        
        days_of_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        df["days_to_side"] = df[date_col].apply(
            lambda x: min(x.day, days_of_months[x.month-1]-x.day)
        )
        df["day"] = df["day"].apply(lambda x: 0 if x <= 7 else 2 if x >= 24 else 1)
        
        # Holiday Logic
        if 'holiday_flg' in df.columns:
             df["holiday"] = ((df["weekday"] >= 5) | (df["holiday_flg"] == 1)).astype(int)
        
        return df

    def _apply_agg(self, df_main, df_source, keys, target, agg_type, suffix):
        full_name = suffix
        if agg_type == 'sum': return feat_sum(df_main, df_source, keys, target, full_name)
        if agg_type == 'mean': return feat_mean(df_main, df_source, keys, target, full_name)
        if agg_type == 'median': return feat_median(df_main, df_source, keys, target, full_name)
        if agg_type == 'max': return feat_max(df_main, df_source, keys, target, full_name)
        if agg_type == 'min': return feat_min(df_main, df_source, keys, target, full_name)
        if agg_type == 'std': return feat_std(df_main, df_source, keys, target, full_name)
        if agg_type == 'count': return feat_count(df_main, df_source, keys, target, full_name)
        if agg_type == 'kernel': 
            return feat_kernelMedian(df_main, df_source, keys, target, self.kernel_matrix, full_name)
        return df_main

    def execute(self, df_target, df_history, bookings_primary, bookings_secondary):
        # Enforce Copies to prevent SettingWithCopyWarning on slices
        df_target = df_target.copy()
        
        # Standardize dates & Merge Info
        df_target = self._enrich_dates(df_target)
        # Note: We enrich history too, but we don't need to merge store_info 
        # again if it's just for aggregation. Keeping it simple:
        df_history = self._enrich_dates(df_history.copy())

        id_col = self.cfg['cols']['id']
        date_col = self.cfg['cols']['date']
        
        # --- A. Booking / Reservation Features ---
        booking_sources = [('primary', bookings_primary), ('secondary', bookings_secondary)]

        for source_name, df_book in booking_sources:
            # 1. Basic Aggregates
            df_target = self._apply_agg(df_target, df_book, [id_col, date_col], "lead_time", "sum", f"{source_name}_lead_time_sum")
            df_target = self._apply_agg(df_target, df_book, [id_col, date_col], "lead_time", "kernel", f"{source_name}_lead_time_mean")
            df_target = self._apply_agg(df_target, df_book, [id_col, date_col], "booking_size", "sum", f"{source_name}_vol_sum")
            df_target = self._apply_agg(df_target, df_book, [id_col, date_col], "booking_size", "kernel", f"{source_name}_vol_mean")
            df_target = self._apply_agg(df_target, df_book, [date_col], "booking_size", "sum", f"{source_name}_global_vol_sum")
            df_target = self._apply_agg(df_target, df_book, [date_col], "booking_size", "kernel", f"{source_name}_global_vol_mean")

            # 2. Windowed Booking Trends
            for w in self.cfg['windows']['short']:
                df_slice = df_book[df_book['day_gap'] >= -w].copy()
                grouped_book = pd.DataFrame(df_slice.groupby([id_col, date_col])['booking_size'].sum()).reset_index()
                grouped_book.columns = [id_col, date_col, "booking_sum"]
                grouped_book = self._enrich_dates(grouped_book) # Get weekday/holiday info
                
                grouped_book = feat_count(grouped_book, df_slice, [id_col, date_col], "booking_size", "booking_count")
                grouped_book = feat_kernelMedian(grouped_book, df_slice, [id_col, date_col], "booking_size", self.kernel_matrix, "booking_mean")

                for group_key in ['weekday', 'holiday']:
                    suffix_base = f"{source_name}_vol_{group_key}_mean_{w}"
                    df_target = self._apply_agg(df_target, grouped_book, [id_col, group_key], "booking_sum", "mean", f"{suffix_base}_sum")
                    df_target = self._apply_agg(df_target, grouped_book, [id_col, group_key], "booking_mean", "mean", f"{suffix_base}_mean")
                    df_target = self._apply_agg(df_target, grouped_book, [id_col, group_key], "booking_count", "mean", f"{suffix_base}_count")

        # --- B. Historical Activity Features ---
        target_val = self.cfg['cols']['target']
        df_target = self._apply_agg(df_target, df_history, [id_col, "day", "weekday"], target_val, "kernel", "hist_day_mean")
        df_target = self._apply_agg(df_target, df_history, [id_col, "day", "holiday"], target_val, "kernel", "hist_holiday_mean")

        for w in self.cfg['windows']['long']:
            df_slice = df_history[df_history['day_gap'] >= -w].copy()
            std_aggs = ['median', 'mean', 'kernel', 'max', 'min', 'std', 'count']
            
            groupings = [
                ([id_col], "hist"),
                ([id_col, "weekday"], "hist_week"),
                ([id_col, "holiday"], "hist_holiday"),
                ([self.cfg['cols']['genre'], "holiday"], "genre_holiday"),
                ([self.cfg['cols']['genre'], "weekday"], "genre_weekday"),
                ([self.cfg['cols']['area'], "holiday"], "area_holiday"),
                ([self.cfg['cols']['area'], self.cfg['cols']['genre'], "holiday"], "area_genre_holiday"),
            ]
            
            for cols, prefix in groupings:
                for agg in std_aggs:
                    col_name = f"{prefix}_{agg}_{w}"
                    if prefix == "hist" and agg == "median": col_name = f"hist_median_{w}"
                    df_target = self._apply_agg(df_target, df_slice, cols, target_val, agg, col_name)

        return df_target

# ==============================================================================
# 3. Main Workflow
# ==============================================================================

def train_forecasts(dict_dfs: dict):
    
    CONFIG = {
        'cols': {
            'id': 'air_store_id', 'date': 'visit_date', 'target': 'visitors',
            'genre': 'air_genre_name', 'area': 'air_area_name'
        },
        'windows': {
            'short': [35, 63, 140],
            'long': [21, 35, 63, 140, 280, 350, 420]
        }
    }
    FORECAST_HORIZONS = [14, 28, 42] 
    MAX_HISTORY_DAYS = 420

    df_train = dict_dfs["df_train"]
    df_test = dict_dfs["df_test"]
    store_id_relation = dict_dfs["store_id_relation"]
    air_reserve = dict_dfs["air_reserve"]
    hpg_reserve = dict_dfs["hpg_reserve"]
    df_date_info = dict_dfs["df_date_info"]
    air_info = dict_dfs["air_info"]
    hpg_info = dict_dfs["hpg_info"]

    # --- Preprocessing ---
    df_train["visit_date"] = pd.to_datetime(df_train["visit_date"])
    df_train["visitors"] = df_train["visitors"].apply(lambda x: np.log1p(float(x)) if float(x) > 0 else 0)

    df_test["air_store_id"] = df_test["id"].apply(lambda x: "_".join(x.split("_")[:2]))
    df_test["visit_date"] = pd.to_datetime(df_test["id"].apply(lambda x: x.split("_")[2]))
    if "visitors" in df_test.columns: del df_test["visitors"]

    for df_r in [air_reserve, hpg_reserve]:
        df_r["visit_date"] = pd.to_datetime(df_r["visit_datetime"].dt.date)
        df_r["reserve_date"] = pd.to_datetime(df_r["reserve_datetime"].dt.date)
        df_r["lead_time"] = (df_r["visit_date"] - df_r["reserve_date"]).dt.days
        df_r["booking_size"] = df_r["reserve_visitors"]

    hpg_reserve = pd.merge(hpg_reserve, store_id_relation, how='inner', on=['hpg_store_id'])
    df_train = df_train.merge(store_id_relation, on=["air_store_id"], how="left")

    # Clean Date Info
    df_date_info.columns = ["visit_date", "day_of_week", "holiday_flg"]
    df_date_info["visit_date"] = pd.to_datetime(df_date_info["visit_date"])
    df_date_info["day_of_week"] = df_date_info["day_of_week"].replace(
        {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}
    )
    del df_date_info["day_of_week"]

    air_info = encode_count(air_info, "air_genre_name")
    air_info = encode_count(air_info, "air_area_name")
    
    # --- Initialization ---
    # Instantiate with Global Max or explicit 500 override
    kernel_gen = EpanechnikovKernel(size=MAX_LOOKBACK_HORIZON, smoothing_method='laplace_strict')
    PrEp = kernel_gen.get_matrix()

    # Helper for date gaps
    def get_date_gap(x, y):
        # FIX: Check for missing dates (NaT) before math
        if pd.isnull(x):
            return np.nan
            
        if isinstance(x, pd.Timestamp): 
            # pd.Timestamp.date() returns NaT if the timestamp is NaT, 
            # but the pd.isnull check above handles that.
            return (x.date() - y).days
        elif isinstance(x, date): 
            return (x - y).days
        
        # Fallback for strings
        try:
            return (date(*map(int, x.split("-"))) - y).days
        except:
            return np.nan

    output_dir = os.path.join(os.getcwd(), 'output')
    stacking_dir = os.path.join(output_dir, 'stacking')
    os.makedirs(stacking_dir, exist_ok=True)

    # Initialize Engine (Passing Date Info here avoids the recursion/patching mess)
    feature_engine = TimeSeriesFeatureEngine(PrEp, CONFIG, air_info, df_date_info)

    # --- Training Loop ---
    for horizon_days in FORECAST_HORIZONS:
        print(f"\n   ->  Starting Validation Scenario: {horizon_days}-Day Forecast Horizon")
        t2017 = date(2017, 4, 23)
        window_step_size = horizon_days 

        all_data = []
        max_windows = MAX_HISTORY_DAYS // horizon_days
        
        for i in range(horizon_days, horizon_days * (max_windows + 1), horizon_days):
            delta = timedelta(days=i)
            t_begin = t2017 - delta
            print(f"\n   ->  Window: {i} days, t_begin: {t_begin}")
            
            df_train_temp = df_train.copy()
            air_res_temp = air_reserve.copy()
            hpg_res_temp = hpg_reserve.copy()

            df_train_temp["day_gap"] = df_train_temp["visit_date"].apply(lambda x: get_date_gap(x, t_begin))
            air_res_temp["day_gap"] = air_res_temp["reserve_date"].apply(lambda x: get_date_gap(x, t_begin))
            hpg_res_temp["day_gap"] = hpg_res_temp["reserve_date"].apply(lambda x: get_date_gap(x, t_begin))

            df_history = df_train_temp[df_train_temp.day_gap < 0].copy()
            df_air_res_hist = air_res_temp[air_res_temp.day_gap < 0].copy()
            df_hpg_res_hist = hpg_res_temp[hpg_res_temp.day_gap < 0].copy()

            df_label = df_train_temp[
                (df_train_temp.day_gap >= 0) & (df_train_temp.day_gap < window_step_size)
            ][["air_store_id", "hpg_store_id", "visit_date", "day_gap", "visitors"]].copy()

            print(f"   -> Generating features...")
            train_data_tmp = feature_engine.execute(df_label, df_history, df_air_res_hist, df_hpg_res_hist)
            all_data.append(train_data_tmp)

        train = pd.concat(all_data)

        # --- Online Test Suite ---
        print(f"\n   ->  Constructing online test suite...")
        t_begin = date(2017, 4, 23)
        df_label_test = df_test.merge(store_id_relation, on="air_store_id", how="left")
        df_label_test["day_gap"] = df_label_test["visit_date"].apply(lambda x: get_date_gap(x, t_begin))
        
        df_train_temp = df_train.copy()
        air_res_temp = air_reserve.copy()
        hpg_res_temp = hpg_reserve.copy()
        df_train_temp["day_gap"] = df_train_temp["visit_date"].apply(lambda x: get_date_gap(x, t_begin))
        air_res_temp["day_gap"] = air_res_temp["reserve_date"].apply(lambda x: get_date_gap(x, t_begin))
        hpg_res_temp["day_gap"] = hpg_res_temp["reserve_date"].apply(lambda x: get_date_gap(x, t_begin))
        
        df_label_test = df_label_test[["air_store_id", "hpg_store_id", "visit_date", "day_gap"]].copy()
        test = feature_engine.execute(df_label_test, df_train_temp, air_res_temp, hpg_res_temp)

        #train.to_csv(f"./stacking/train_{horizon_days}.csv", index=None)
        #test.to_csv(f"./stacking/test_{horizon_days}.csv", index=None)

        train.to_csv(os.path.join(stacking_dir, f"train_{horizon_days}.csv"), index=None)
        test.to_csv(os.path.join(stacking_dir, f"test_{horizon_days}.csv"), index=None)

        # --- LightGBM Stacking ---
        def run_stacking(clf, train_df, test_df, clf_name, n_folds=5, seed=2025):
            train_preds = np.zeros((train_df.shape[0], 1))
            test_preds = np.zeros((test_df.shape[0], 1))
            test_preds_folds = np.empty((n_folds, test_df.shape[0], 1))
            cv_scores = []
            
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            for i, (train_idx, val_idx) in enumerate(kf.split(train_df)):
                print(f"   -> Fold {i+1}/{n_folds}")
                fold_tr = train_df.iloc[train_idx]
                fold_val = train_df.iloc[val_idx]
                
                tr_x = fold_tr.drop(["visitors"], axis=1)
                tr_y = fold_tr['visitors'].values
                val_x = fold_val.drop(["visitors"], axis=1)
                val_y = fold_val['visitors'].values
                
                val_recent = fold_val[fold_val.day_gap <= 6]
                val_distant = fold_val[fold_val.day_gap > 6]
                
                # WEIGHING BIAS: 0.2 weight for older than 6 days
                weight_tr = fold_tr["day_gap"].apply(lambda x: 1 if x <= 6 else 0.2)
                weight_val = fold_val["day_gap"].apply(lambda x: 1 if x <= 6 else 0.2)

                d_train = clf.Dataset(tr_x, label=tr_y, weight=weight_tr)
                d_val = clf.Dataset(val_x, label=val_y, weight=weight_val)

                params = {
                    'num_leaves': 255, 'objective': 'regression_l2', 'max_depth': 9,
                    'min_data_in_leaf': 50, 'learning_rate': 0.007, 'feature_fraction': 0.6,
                    'bagging_fraction': 0.8, 'bagging_freq': 1, 'metric': 'rmse',
                    'num_threads': 4, 'seed': seed, 'verbosity': -1
                }

                model = clf.train(
                    params, d_train, num_boost_round=6000,
                    valid_sets=[d_val],
                    callbacks=[early_stopping(500, verbose=False)]
                )
                
                val_p = model.predict(val_x, num_iteration=model.best_iteration).reshape(-1, 1)
                train_preds[val_idx] = val_p
                test_preds_folds[i, :] = model.predict(test_df, num_iteration=model.best_iteration).reshape(-1, 1)
                
                score_total = mean_squared_error(val_y, val_p) ** 0.5
                score_recent = mean_squared_error(val_recent['visitors'], model.predict(val_recent.drop(['visitors'], axis=1))) ** 0.5 if len(val_recent) > 0 else 0
                score_distant = mean_squared_error(val_distant['visitors'], model.predict(val_distant.drop(['visitors'], axis=1))) ** 0.5 if len(val_distant) > 0 else 0
                cv_scores.append((score_total, score_recent, score_distant))

            test_preds[:] = test_preds_folds.mean(axis=0)
            mean_cv = np.mean([x[0] for x in cv_scores])
            print(f"   -> {clf_name} Mean RMSE: {mean_cv}")
            
            with open(os.path.join(output_dir, "score_cv.txt"), "a") as f:
                f.write(f"{clf_name} horizon {horizon_days} scores: {cv_scores}\nMean: {mean_cv}\n")

            return train_preds, test_preds, cv_scores

        train_cols_only = train.drop(["air_store_id", "hpg_store_id", "visit_date"], axis=1)
        test_cols_only = test.drop(["air_store_id", "hpg_store_id", "visit_date"], axis=1)

        lgb_train_pred, lgb_test_pred, metrics = run_stacking(lgb, train_cols_only, test_cols_only, "lgb")

        train["visitors_pre"] = lgb_train_pred
        train["visitors"] = np.clip(np.expm1(train["visitors"]), 0, 1000)
        train["visitors_pre"] = np.clip(np.expm1(train["visitors_pre"]), 0, 1000)
        
        fname = f"offline_cv_{metrics[0][0]}_horizon{horizon_days}.csv"
        train[["air_store_id", "visit_date", "visitors", "visitors_pre"]].to_csv(os.path.join(output_dir, fname), index=None)

        df_test["visitors"] = np.clip(np.expm1(lgb_test_pred), 0, 1000)
        sub_name = f"sub_forecast_horizon{horizon_days}_rmse{metrics[0][0]}.csv"
        df_test[["id", "visitors"]].to_csv(os.path.join(output_dir, sub_name), index=None)

    return
