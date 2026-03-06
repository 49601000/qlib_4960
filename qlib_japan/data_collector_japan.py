"""
data_collector_japan.py
yfinance → Qlib形式（bin）への日本株データ変換スクリプト

使い方:
    python data_collector_japan.py --tickers 7203.T 6758.T 9984.T --start 2018-01-01
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# ─── 設定 ────────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    "7003.T",  # 三井E&S
    "7011.T",  # 三菱重工業
    "7936.T",  # アシックス
    "8306.T",  # 三菱UFJ
    "5333.T",  # NGK
    "9023.T",  # 東京地下鉄
    "4183.T",  # 三井化学
    "5355.T",  # 日本坩堝
    "2801.T",  # キッコーマン
    "1719.T",  # 安藤ハザマ
]

QLIB_DATA_DIR = Path.home() / ".qlib" / "qlib_data" / "jp_data"
FEATURES = ["open", "high", "low", "close", "volume", "vwap", "factor"]


def download_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    """yfinanceで株価データ取得・前処理"""
    print(f"  取得中: {ticker}")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        print(f"  ⚠️  {ticker}: データなし")
        return pd.DataFrame()

    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

    # vwap近似（高値+安値+終値の平均）
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3

    # factor（調整係数 = 1.0 として設定）
    df["factor"] = 1.0

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume", "vwap", "factor"]]


def save_qlib_bin(df: pd.DataFrame, ticker: str, output_dir: Path):
    """Qlib bin形式で保存"""
    stock_id = ticker.replace(".T", "").lower()
    stock_dir = output_dir / "features" / stock_id
    stock_dir.mkdir(parents=True, exist_ok=True)

    for col in FEATURES:
        if col not in df.columns:
            continue
        data = df[col].astype(np.float32).values
        # Qlib bin形式: float32バイナリ + カレンダー日付インデックス
        bin_path = stock_dir / f"{col}.day.bin"
        data.tofile(str(bin_path))

    print(f"  💾 保存: {stock_dir} ({len(df)}日分)")


def save_calendar(dates: pd.DatetimeIndex, output_dir: Path):
    """営業日カレンダー保存"""
    cal_dir = output_dir / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_path = cal_dir / "day.txt"

    date_strs = [d.strftime("%Y-%m-%d") for d in sorted(set(dates))]
    cal_path.write_text("\n".join(date_strs))
    print(f"📅 カレンダー保存: {len(date_strs)}日")


def save_instruments(tickers: list, output_dir: Path):
    """銘柄リスト保存"""
    inst_dir = output_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    inst_path = inst_dir / "all.txt"

    with open(inst_path, "w") as f:
        for ticker in tickers:
            stock_id = ticker.replace(".T", "").lower()
            # Qlib instruments形式: <id>\t<start>\t<end>
            f.write(f"{stock_id}\t2000-01-01\t2030-12-31\n")

    print(f"📋 銘柄リスト保存: {len(tickers)}銘柄")


def main():
    parser = argparse.ArgumentParser(description="日本株データをQlib形式に変換")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="ティッカーリスト（例: 7203.T 6758.T）")
    parser.add_argument("--start", default="2018-01-01", help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"),
                        help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--output", default=str(QLIB_DATA_DIR), help="出力ディレクトリ")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 日本株データ取得開始")
    print(f"   対象: {len(args.tickers)}銘柄 / 期間: {args.start} ~ {args.end}")
    print(f"   出力先: {output_dir}\n")

    all_dates = set()
    success_tickers = []

    for ticker in args.tickers:
        df = download_stock(ticker, args.start, args.end)
        if df.empty:
            continue
        save_qlib_bin(df, ticker, output_dir)
        all_dates.update(df.index)
        success_tickers.append(ticker)

    if all_dates:
        save_calendar(pd.DatetimeIndex(all_dates), output_dir)
    if success_tickers:
        save_instruments(success_tickers, output_dir)

    print(f"\n✅ 完了! {len(success_tickers)}/{len(args.tickers)} 銘柄取得成功")
    print(f"\n次のステップ:")
    print(f"  python -c \"import qlib; qlib.init(provider_uri='{output_dir}', region='jp')\"")


if __name__ == "__main__":
    main()
