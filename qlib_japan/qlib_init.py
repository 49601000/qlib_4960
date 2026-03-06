"""
qlib_init.py
Qlib の初期化・データ検証・ユーティリティ
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# デフォルトの日本株データディレクトリ
DEFAULT_QLIB_DIR = Path.home() / ".qlib" / "qlib_data" / "jp_data"

_qlib_initialized = False


def init_qlib(provider_uri: Optional[str] = None) -> bool:
    """
    Qlib を初期化する。
    Returns:
        True  … 初期化成功（実 Qlib）
        False … Qlib 未インストール or データなし（フォールバックモード）
    """
    global _qlib_initialized
    if _qlib_initialized:
        return True

    uri = provider_uri or str(DEFAULT_QLIB_DIR)

    try:
        import qlib
        from qlib.config import REG_CN  # CN設定をベースに JP 向けカスタマイズ

        if not Path(uri).exists():
            logger.warning(f"Qlibデータディレクトリが見つかりません: {uri}")
            logger.warning("data_collector_japan.py を先に実行してください。")
            return False

        # カレンダーファイルの存在確認
        cal_file = Path(uri) / "calendars" / "day.txt"
        if not cal_file.exists():
            logger.warning("カレンダーファイルが見つかりません。データ変換が必要です。")
            return False

        qlib.init(provider_uri=uri, region=REG_CN)
        _qlib_initialized = True
        logger.info(f"Qlib 初期化成功: {uri}")
        return True

    except ImportError:
        logger.warning("pyqlib が未インストールです。pip install pyqlib を実行してください。")
        return False
    except Exception as e:
        logger.warning(f"Qlib 初期化失敗: {e}")
        return False


def check_data_availability(ticker: str, provider_uri: Optional[str] = None) -> dict:
    """
    指定銘柄のQlibデータ可用性を確認する。
    Returns: dict with keys: available, start_date, end_date, n_days
    """
    uri = Path(provider_uri or DEFAULT_QLIB_DIR)
    stock_id = ticker.replace(".T", "").lower()
    feature_dir = uri / "features" / stock_id

    result = {"available": False, "start_date": None, "end_date": None, "n_days": 0}

    if not feature_dir.exists():
        return result

    close_bin = feature_dir / "close.day.bin"
    if not close_bin.exists():
        return result

    # カレンダーから日付範囲を推定
    cal_file = uri / "calendars" / "day.txt"
    if cal_file.exists():
        dates = pd.to_datetime(cal_file.read_text().strip().split("\n"))
        data_size = os.path.getsize(close_bin) // 4  # float32 = 4 bytes
        if data_size > 0 and len(dates) >= data_size:
            result["start_date"] = dates[0].strftime("%Y-%m-%d")
            result["end_date"] = dates[data_size - 1].strftime("%Y-%m-%d")
            result["n_days"] = data_size

    result["available"] = True
    return result


def get_available_tickers(provider_uri: Optional[str] = None) -> list:
    """Qlibデータが存在する銘柄リストを返す"""
    uri = Path(provider_uri or DEFAULT_QLIB_DIR)
    features_dir = uri / "features"

    if not features_dir.exists():
        return []

    tickers = []
    for d in features_dir.iterdir():
        if d.is_dir() and (d / "close.day.bin").exists():
            tickers.append(d.name.upper() + ".T")
    return sorted(tickers)


def load_price_data_from_qlib(
    ticker: str,
    start_date: str,
    end_date: str,
    provider_uri: Optional[str] = None,
) -> pd.DataFrame:
    """
    Qlibのデータプロバイダー経由で株価データを読み込む。
    init_qlib() が成功している必要がある。
    """
    try:
        from qlib.data import D

        stock_id = ticker.replace(".T", "").upper()
        fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap", "$factor"]
        col_names = ["始値", "高値", "安値", "終値", "出来高", "VWAP", "調整係数"]

        df = D.features(
            instruments=[stock_id],
            fields=fields,
            start_time=start_date,
            end_time=end_date,
            freq="day",
        )

        if df.empty:
            return pd.DataFrame()

        # MultiIndex → 単一インデックスに変換
        df = df.droplevel(0)
        df.columns = col_names
        df.index.name = "日付"
        df["日次リターン"] = df["終値"].pct_change() * 100
        df["20日MA"] = df["終値"].rolling(20).mean()
        df["60日MA"] = df["終値"].rolling(60).mean()
        df["ボラティリティ"] = df["日次リターン"].rolling(20).std()
        return df

    except Exception as e:
        logger.error(f"Qlibデータ読み込みエラー ({ticker}): {e}")
        return pd.DataFrame()
