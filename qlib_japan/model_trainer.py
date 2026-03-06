"""
model_trainer.py
Qlib のモデル学習・予測スコア取得モジュール

対応モデル:
  - LightGBM  (qlib.contrib.model.gbdt)
  - LSTM       (qlib.contrib.model.pytorch_lstm)
  - GRU        (qlib.contrib.model.pytorch_gru)
  - Transformer(qlib.contrib.model.pytorch_transformer)
  - XGBoost    (qlib.contrib.model.xgboost)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ─── モデル設定テンプレート ──────────────────────────────────────────────────

MODEL_CONFIGS = {
    "lightgbm": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    },
    "xgboost": {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "kwargs": {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
    "lstm": {
        "class": "LSTM",
        "module_path": "qlib.contrib.model.pytorch_lstm",
        "kwargs": {
            "d_feat": 6,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 200,
            "lr": 1e-3,
            "early_stop": 20,
            "batch_size": 800,
            "metric": "loss",
            "loss": "mse",
            "GPU": 0,
        },
    },
    "gru": {
        "class": "GRU",
        "module_path": "qlib.contrib.model.pytorch_gru",
        "kwargs": {
            "d_feat": 6,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 200,
            "lr": 1e-3,
            "early_stop": 20,
            "batch_size": 800,
            "metric": "loss",
            "loss": "mse",
            "GPU": 0,
        },
    },
    "transformer": {
        "class": "Transformer",
        "module_path": "qlib.contrib.model.pytorch_transformer",
        "kwargs": {
            "d_feat": 6,
            "d_model": 32,
            "nhead": 2,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 100,
            "lr": 1e-4,
            "early_stop": 20,
            "batch_size": 800,
            "metric": "loss",
            "loss": "mse",
            "GPU": 0,
        },
    },
}

# ─── データセット設定（Alpha158: 日本株向け簡略版） ──────────────────────────

def build_dataset_config(
    ticker: str,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    handler_type: str = "alpha158",
) -> dict:
    """
    Qlibデータセット設定を生成する。
    handler_type: "alpha158" or "alpha360"
    """
    stock_id = ticker.replace(".T", "").upper()

    if handler_type == "alpha158":
        handler_cfg = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": train_start,
                "end_time": test_end,
                "fit_start_time": train_start,
                "fit_end_time": train_end,
                "instruments": [stock_id],
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                ],
                "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
            },
        }
    else:  # alpha360
        handler_cfg = {
            "class": "Alpha360",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": train_start,
                "end_time": test_end,
                "fit_start_time": train_start,
                "fit_end_time": train_end,
                "instruments": [stock_id],
            },
        }

    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_cfg,
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
        },
    }


# ─── メイン学習・予測クラス ────────────────────────────────────────────────────

class QlibModelTrainer:
    """
    Qlib モデルの学習・予測スコア取得を担うクラス。

    使い方:
        trainer = QlibModelTrainer("7203.T", "lightgbm")
        trainer.setup_dataset("2018-01-01", "2022-12-31",
                              "2023-01-01", "2023-06-30",
                              "2023-07-01", "2024-12-31")
        pred_df = trainer.train_and_predict()
    """

    def __init__(self, ticker: str, model_key: str = "lightgbm"):
        self.ticker = ticker
        self.model_key = model_key
        self.model = None
        self.dataset = None
        self.pred_df: Optional[pd.DataFrame] = None
        self._is_trained = False

    def setup_dataset(
        self,
        train_start: str, train_end: str,
        valid_start: str, valid_end: str,
        test_start: str, test_end: str,
        handler_type: str = "alpha158",
    ):
        """データセット設定を準備する"""
        from qlib.utils import init_instance_by_config

        ds_cfg = build_dataset_config(
            self.ticker,
            train_start, train_end,
            valid_start, valid_end,
            test_start, test_end,
            handler_type=handler_type,
        )
        self.dataset = init_instance_by_config(ds_cfg)
        logger.info(f"データセット準備完了: {self.ticker} / {handler_type}")

    def train_and_predict(self) -> pd.DataFrame:
        """
        モデルを学習し、テスト期間の予測スコアを返す。
        Returns:
            DataFrame with columns: ['score']  index: (date, instrument)
        """
        if self.dataset is None:
            raise RuntimeError("setup_dataset() を先に実行してください。")

        from qlib.utils import init_instance_by_config

        cfg = MODEL_CONFIGS.get(self.model_key)
        if cfg is None:
            raise ValueError(f"未対応モデル: {self.model_key}")

        logger.info(f"モデル学習開始: {cfg['class']}")
        self.model = init_instance_by_config({
            "class": cfg["class"],
            "module_path": cfg["module_path"],
            "kwargs": cfg["kwargs"],
        })

        self.model.fit(self.dataset)
        self._is_trained = True
        logger.info("モデル学習完了")

        pred = self.model.predict(self.dataset)
        if isinstance(pred, pd.Series):
            self.pred_df = pred.to_frame("score")
        else:
            self.pred_df = pred.rename(columns={pred.columns[0]: "score"})

        logger.info(f"予測スコア生成完了: {len(self.pred_df)} 件")
        return self.pred_df

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """LightGBM/XGBoost の特徴量重要度を返す（非対応モデルはNone）"""
        if not self._is_trained or self.model is None:
            return None
        try:
            import lightgbm as lgb
            if hasattr(self.model, "model") and isinstance(self.model.model, lgb.Booster):
                imp = pd.DataFrame({
                    "特徴量": self.model.model.feature_name(),
                    "重要度": self.model.model.feature_importance(importance_type="gain"),
                }).sort_values("重要度", ascending=False).head(20)
                return imp
        except Exception:
            pass
        return None


# ─── ローリング再学習 ─────────────────────────────────────────────────────────

class RollingTrainer:
    """
    Rolling Retraining（定期的な再学習）を管理するクラス。

    Qlib の RollingExp を薄くラップし、Streamlit から呼びやすくしている。
    """

    def __init__(
        self,
        ticker: str,
        model_key: str = "lightgbm",
        rolling_window_days: int = 252,   # 学習期間（営業日）
        retrain_freq_days: int = 60,       # 再学習頻度
        horizon_days: int = 5,             # 予測ホライゾン
    ):
        self.ticker = ticker
        self.model_key = model_key
        self.rolling_window = rolling_window_days
        self.retrain_freq = retrain_freq_days
        self.horizon = horizon_days
        self.all_preds: list[pd.DataFrame] = []

    def run(
        self,
        full_start: str,
        full_end: str,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        全期間にわたってローリング学習を実行し、予測スコアを結合して返す。
        progress_callback: Streamlit の st.progress に渡す関数（0.0 〜 1.0）
        """
        dates = pd.bdate_range(full_start, full_end, freq="C",
                               holidays=_jp_holidays())
        windows = _split_rolling_windows(dates, self.rolling_window, self.retrain_freq)
        total = len(windows)

        all_preds = []
        for i, (train_s, train_e, test_s, test_e) in enumerate(windows):
            logger.info(f"[{i+1}/{total}] 学習: {train_s}~{train_e} | テスト: {test_s}~{test_e}")

            trainer = QlibModelTrainer(self.ticker, self.model_key)
            valid_s = pd.Timestamp(train_e) - pd.offsets.BDay(int(self.rolling_window * 0.1))
            trainer.setup_dataset(
                train_s, str(valid_s.date()),
                str((valid_s + pd.offsets.BDay(1)).date()), train_e,
                test_s, test_e,
            )
            try:
                pred = trainer.train_and_predict()
                all_preds.append(pred)
            except Exception as e:
                logger.warning(f"ウィンドウ {i+1} スキップ: {e}")

            if progress_callback:
                progress_callback((i + 1) / total)

        if not all_preds:
            return pd.DataFrame()

        return pd.concat(all_preds).sort_index()


# ─── ユーティリティ ───────────────────────────────────────────────────────────

def _jp_holidays() -> list:
    """簡易日本祝日リスト（2020〜2025）"""
    return pd.to_datetime([
        "2020-01-01", "2020-01-13", "2020-02-11", "2020-02-23", "2020-02-24",
        "2021-01-01", "2021-01-11", "2021-02-11", "2021-02-23",
        "2022-01-01", "2022-01-10", "2022-02-11", "2022-02-23",
        "2023-01-01", "2023-01-02", "2023-01-09", "2023-02-11", "2023-02-23",
        "2024-01-01", "2024-01-08", "2024-02-11", "2024-02-12", "2024-02-23",
        "2025-01-01", "2025-01-13", "2025-02-11", "2025-02-23", "2025-02-24",
    ])


def _split_rolling_windows(
    dates: pd.DatetimeIndex,
    window: int,
    freq: int,
) -> list[Tuple[str, str, str, str]]:
    """ローリングウィンドウを分割する"""
    windows = []
    start_idx = 0
    while start_idx + window < len(dates):
        train_end_idx = start_idx + window
        test_end_idx = min(train_end_idx + freq, len(dates) - 1)
        windows.append((
            str(dates[start_idx].date()),
            str(dates[train_end_idx].date()),
            str(dates[train_end_idx + 1].date()),
            str(dates[test_end_idx].date()),
        ))
        start_idx += freq
    return windows
