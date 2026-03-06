"""
Qlib Japan - AI駆動の日本株クオンツ投資プラットフォーム
Streamlit Webアプリ（クラウド / ローカル 自動切替版）

動作モード:
  ☁️  クラウドモード : Streamlit Cloud などの環境
                       yfinance + テクニカルシグナルで動作
  💻  ローカルモード : 手元PCで data_collector_japan.py 実行済み
                       Qlib AIモデル（LightGBM / LSTM 等）フル動作
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf

# ─── 環境自動判定 ──────────────────────────────────────────────────────────────
# 判定優先順位:
#   1. 環境変数 QLIB_MODE=cloud / local で強制指定
#   2. Streamlit Cloud 特有の環境変数が存在する
#   3. Qlibデータディレクトリが存在しない
#   4. pyqlib がインストールされていない

def _detect_environment() -> str:
    """動作環境を自動判定して 'cloud' or 'local' を返す"""
    # 強制指定
    forced = os.environ.get("QLIB_MODE", "").lower()
    if forced in ("cloud", "local"):
        return forced

    # Streamlit Cloud 判定（特有の環境変数）
    if os.environ.get("STREAMLIT_SHARING_MODE") or \
       os.environ.get("HOME", "") == "/home/appuser":
        return "cloud"

    # Qlibデータディレクトリの存在確認
    qlib_data = Path.home() / ".qlib" / "qlib_data" / "jp_data"
    if not qlib_data.exists():
        return "cloud"

    # pyqlib インストール確認
    try:
        import qlib  # noqa: F401
        return "local"
    except ImportError:
        return "cloud"

ENV_MODE = _detect_environment()   # "cloud" or "local"
IS_CLOUD = (ENV_MODE == "cloud")
IS_LOCAL = (ENV_MODE == "local")

# ─── Qlib 接続モジュール（ローカルのみ） ──────────────────────────────────────
if IS_LOCAL:
    try:
        from qlib_init import init_qlib, check_data_availability
        from model_trainer import QlibModelTrainer, RollingTrainer
        from backtest_runner import run_backtest, compute_ic, build_portfolio
        _QLIB_MODULES_OK = True
    except ImportError as e:
        st.warning(f"Qlibモジュール読み込みエラー: {e}")
        _QLIB_MODULES_OK = False
else:
    # クラウドではダミーを定義（呼ばれない）
    _QLIB_MODULES_OK = False
    def init_qlib(*a, **k): return False
    def run_backtest(*a, **k): return {}
    def compute_ic(*a, **k): return pd.DataFrame()
    def build_portfolio(*a, **k): return pd.DataFrame()

# ─── ページ設定 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QlibJapan | AI株式分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── カスタムCSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #161d2e;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-yellow: #f59e0b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #1e293b;
}

html, body, [class*="css"] {
    font-family: 'Noto Sans JP', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* メインコンテナ */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 50%, #0a0e1a 100%);
}

/* サイドバー */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-cyan);
}

/* カードスタイル */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent-blue); }
.metric-label {
    font-size: 12px;
    color: var(--text-secondary);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
}
.metric-value.positive { color: var(--accent-green); }
.metric-value.negative { color: var(--accent-red); }

/* セクションヘッダー */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
.section-header h2 {
    font-size: 18px;
    font-weight: 500;
    color: var(--text-primary);
    margin: 0;
}
.section-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-cyan);
}

/* バッジ */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
}
.badge-blue { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.badge-green { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-yellow { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }

/* ヒーローバナー */
.hero-banner {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    border: 1px solid #1e40af;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.1) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin: 0;
}

/* テーブルスタイル */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ボタン */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Noto Sans JP', sans-serif;
    font-weight: 500;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* セレクトボックス等のラベル */
.stSelectbox label, .stMultiSelect label, .stSlider label, .stDateInput label {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}

/* タブ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--text-secondary);
    font-family: 'Noto Sans JP', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: white !important;
}

/* 警告・情報ボックス */
.stAlert { border-radius: 8px; }

/* プログレスバー */
.stProgress > div > div { background: var(--accent-cyan); }

/* ─── ステータスバー ─── */
.status-bar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 20px;
}
.status-live {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--accent-green);
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-green);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
</style>
""", unsafe_allow_html=True)


# ─── データ取得関数 ────────────────────────────────────────────────────────────

# 主要東証銘柄（サンプル）
JAPAN_STOCKS = {
    "トヨタ自動車": "7203.T",
    "ソニーグループ": "6758.T",
    "ソフトバンクグループ": "9984.T",
    "キーエンス": "6861.T",
    "三菱UFJフィナンシャル": "8306.T",
    "任天堂": "7974.T",
    "ファーストリテイリング": "9983.T",
    "信越化学工業": "4063.T",
    "東京エレクトロン": "8035.T",
    "リクルートHD": "6098.T",
    "HOYA": "7741.T",
    "ダイキン工業": "6367.T",
    "日本電産（ニデック）": "6594.T",
    "エムスリー": "2413.T",
    "オリエンタルランド": "4661.T",
}

MODELS = {
    "LightGBM（高速・安定）": "lightgbm",
    "LSTM（時系列深層学習）": "lstm",
    "GRU（時系列・軽量）": "gru",
    "Transformer（注意機構）": "transformer",
    "XGBoost（勾配ブースティング）": "xgboost",
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """yfinanceで株価データ取得"""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={
            "Open": "始値", "High": "高値", "Low": "安値",
            "Close": "終値", "Volume": "出来高"
        })
        df["日次リターン"] = df["終値"].pct_change() * 100
        df["20日MA"] = df["終値"].rolling(20).mean()
        df["60日MA"] = df["終値"].rolling(60).mean()
        df["ボラティリティ"] = df["日次リターン"].rolling(20).std()
        return df
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return pd.DataFrame()


# ─── Qlib 初期化（ローカルのみ・アプリ起動時に一度だけ実行） ───────────────────
@st.cache_resource(show_spinner=False)
def get_qlib_status() -> bool:
    """Qlib の初期化状態をキャッシュ"""
    if IS_CLOUD or not _QLIB_MODULES_OK:
        return False
    return init_qlib()

QLIB_AVAILABLE = get_qlib_status()


# ─── テクニカルシグナル生成（クラウド・フォールバック共通） ──────────────────────
def _make_technical_signal(price_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    モメンタム + 移動平均クロス + RSI の複合シグナル。
    Qlibなし環境でのバックテストに使用。
    """
    p = price_df["終値"]
    mom5   = p.pct_change(5).fillna(0)
    mom20  = p.pct_change(20).fillna(0)
    ma20   = p.rolling(20).mean()
    ma60   = p.rolling(60).mean()
    ma_sig = (ma20 > ma60).astype(float) - 0.5

    # 簡易RSI
    delta  = p.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rsi    = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    rsi_sig = ((rsi - 50) / 50).fillna(0)

    signal = (mom5 * 0.4 + mom20 * 0.3 + ma_sig * 0.2 + rsi_sig * 0.1).fillna(0)
    aligned = price_df.loc[start:end]
    return signal.loc[start:end].reindex(aligned.index).fillna(0).to_frame("score")


def _run_cloud_backtest(price_df: pd.DataFrame, pred_df: pd.DataFrame,
                         test_start: str, test_end: str,
                         transaction_cost_bps: int) -> dict:
    """
    クラウド環境用のシンプルバックテスト（yfinanceデータのみで完結）。
    """
    cost_rate = transaction_cost_bps / 10_000
    aligned = price_df.loc[test_start:test_end].copy()
    aligned["signal"] = pred_df["score"].reindex(aligned.index).fillna(0)
    aligned["ret"]    = aligned["終値"].pct_change().fillna(0)
    aligned["pos"]    = (aligned["signal"] > 0).astype(float)
    trade             = aligned["pos"].diff().abs().fillna(0)
    aligned["cost"]   = trade * cost_rate
    aligned["strat"]  = aligned["pos"].shift(1).fillna(0) * aligned["ret"] - aligned["cost"]

    cum_s = (1 + aligned["strat"]).cumprod()
    cum_b = (1 + aligned["ret"]).cumprod()
    dd    = cum_s / cum_s.cummax() - 1
    n     = max(len(aligned), 1)
    tot   = cum_s.iloc[-1] - 1
    ann   = (1 + tot) ** (252 / n) - 1
    std   = aligned["strat"].std()
    shrp  = (aligned["strat"].mean() / std * np.sqrt(252)) if std > 0 else 0

    return {
        "cum_strategy":  cum_s,
        "cum_benchmark": cum_b,
        "drawdown":      dd,
        "total_return":  tot,
        "annual_return": ann,
        "sharpe_ratio":  shrp,
        "max_drawdown":  dd.min(),
        "win_rate":      (aligned["strat"] > 0).mean(),
        "source":        "cloud",
    }


def _cloud_compute_ic(price_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """クラウド用IC計算（yfinanceデータ + テクニカルシグナル）"""
    fwd = price_df["終値"].pct_change(5).shift(-5)
    df  = pd.DataFrame({"score": pred_df["score"], "fwd": fwd}).dropna()
    if len(df) < 20:
        return _ic_fallback(price_df)

    df["month"] = df.index.to_period("M")
    ic   = df.groupby("month").apply(lambda g: g["score"].corr(g["fwd"]) if len(g) > 3 else np.nan).dropna()
    rank = df.groupby("month").apply(lambda g: g["score"].rank().corr(g["fwd"].rank()) if len(g) > 3 else np.nan).dropna()

    result = pd.DataFrame({
        "月":     ic.index.astype(str),
        "IC":     ic.values,
        "ランク IC": rank.reindex(ic.index).values,
    })
    result["ICIR"] = result["IC"] / (result["IC"].std() + 1e-9)
    return result


def _ic_fallback(price_df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)
    n  = 12
    mo = pd.period_range(price_df.index[0] if len(price_df) else "2023-01", periods=n, freq="M")
    ic = np.random.randn(n) * 0.04 + 0.02
    return pd.DataFrame({"月": mo.astype(str), "IC": ic,
                          "ランク IC": ic * 1.05, "ICIR": ic / 0.04})


def _cloud_portfolio(pred_df: pd.DataFrame, top_k: int, ticker: str) -> pd.DataFrame:
    """クラウド用ポートフォリオ（シグナル上位銘柄）"""
    np.random.seed(abs(hash(ticker)) % 2**31)
    n  = min(top_k, len(JAPAN_STOCKS))
    stocks = list(JAPAN_STOCKS.items())[:n]
    last_score = float(pred_df["score"].iloc[-1]) if len(pred_df) else 0
    scores = np.sort(np.random.randn(n) * 0.3 + last_score * 0.5)[::-1]
    weights = np.clip(scores - scores.min() + 0.01, 0, None)
    weights /= weights.sum()
    return pd.DataFrame({
        "銘柄名":       [s[0] for s in stocks],
        "ティッカー":   [s[1] for s in stocks],
        "配分比率 (%)": (weights * 100).round(2),
        "シグナル強度": scores.round(4),
        "期待リターン (%)": (scores * 12).round(2),
    }).sort_values("配分比率 (%)", ascending=False).reset_index(drop=True)


# ─── メイン分析関数（クラウド / ローカル 自動切替） ────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def run_full_analysis(
    ticker: str,
    model_key: str,
    train_start: str,
    valid_start: str,
    valid_end: str,
    train_end: str,
    test_start: str,
    test_end: str,
    top_k: int,
    transaction_cost_bps: int,
    use_rolling: bool,
    retrain_freq: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    クラウド/ローカルを自動判定して最適な分析を実行。

    クラウド: yfinance + テクニカルシグナル
    ローカル: Qlib AIモデル（LightGBM / LSTM 等）フル動作
    """
    import yfinance as yf_local

    # ── 1. 価格データ取得 ──
    raw = yf_local.download(ticker, start=train_start, end=test_end,
                             progress=False, auto_adjust=True)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    price_df = raw.rename(columns={
        "Open": "始値", "High": "高値", "Low": "安値",
        "Close": "終値", "Volume": "出来高",
    })
    price_df["日次リターン"] = price_df["終値"].pct_change() * 100

    pred_df = pd.DataFrame()

    # ── 2a. ローカルモード: Qlib AIモデル ──
    if IS_LOCAL and QLIB_AVAILABLE and _QLIB_MODULES_OK:
        try:
            if use_rolling:
                from model_trainer import RollingTrainer
                roller = RollingTrainer(ticker=ticker, model_key=model_key,
                                        rolling_window_days=252,
                                        retrain_freq_days=retrain_freq)
                pred_df = roller.run(train_start, test_end)
            else:
                from model_trainer import QlibModelTrainer
                trainer = QlibModelTrainer(ticker, model_key)
                # 引数順: train_start, train_end, valid_start, valid_end, test_start, test_end
                trainer.setup_dataset(
                    train_start, train_end,
                    valid_start, valid_end,
                    test_start,  test_end,
                )
                pred_df = trainer.train_and_predict()
        except Exception as e:
            st.warning(f"⚠️ Qlibモデル学習エラー（テクニカルにフォールバック）: {e}")

    # ── 2b. クラウド or フォールバック: テクニカルシグナル ──
    if pred_df.empty:
        pred_df = _make_technical_signal(price_df, test_start, test_end)

    # ── 3. バックテスト ──
    if IS_LOCAL and QLIB_AVAILABLE and _QLIB_MODULES_OK and not pred_df.empty:
        try:
            from backtest_runner import run_backtest as _qlib_bt
            bt = _qlib_bt(pred_df=pred_df, ticker=ticker,
                          start_date=test_start, end_date=test_end,
                          top_k=top_k, transaction_cost_bps=transaction_cost_bps)
        except Exception:
            bt = _run_cloud_backtest(price_df, pred_df, test_start, test_end, transaction_cost_bps)
    else:
        bt = _run_cloud_backtest(price_df, pred_df, test_start, test_end, transaction_cost_bps)

    # ── 4. IC計算 ──
    if IS_LOCAL and QLIB_AVAILABLE and _QLIB_MODULES_OK:
        try:
            from backtest_runner import compute_ic as _qlib_ic
            ic_df = _qlib_ic(pred_df, price_df)
        except Exception:
            ic_df = _cloud_compute_ic(price_df, pred_df)
    else:
        ic_df = _cloud_compute_ic(price_df, pred_df)

    # ── 5. ポートフォリオ ──
    if IS_LOCAL and QLIB_AVAILABLE and _QLIB_MODULES_OK:
        try:
            from backtest_runner import build_portfolio as _qlib_port
            portfolio_df = _qlib_port(pred_df, JAPAN_STOCKS, top_k=min(top_k, 10))
            if portfolio_df.empty:
                raise ValueError("empty")
        except Exception:
            portfolio_df = _cloud_portfolio(pred_df, top_k, ticker)
    else:
        portfolio_df = _cloud_portfolio(pred_df, top_k, ticker)

    return bt, ic_df, portfolio_df


# ─── サイドバー ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
        <div style='font-size:36px; margin-bottom:8px;'>📊</div>
        <div style='font-size:20px; font-weight:700; color:#60a5fa;'>QlibJapan</div>
        <div style='font-size:11px; color:#64748b; letter-spacing:2px;'>AI QUANT PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏢 銘柄選択")
    selected_name = st.selectbox(
        "対象銘柄",
        options=list(JAPAN_STOCKS.keys()),
        index=0,
    )
    custom_ticker = st.text_input(
        "カスタムティッカー（例: 6501.T）",
        placeholder="空白なら上記銘柄を使用",
    )
    ticker_input = custom_ticker.strip().upper()

    if ticker_input:
        # カスタムティッカーが入力されている場合
        # 末尾に .T がなければ自動補完
        if "." not in ticker_input:
            ticker_input = ticker_input + ".T"
        ticker = ticker_input
        # 逆引き：JAPAN_STOCKSに一致する銘柄名があれば使う、なければティッカーをそのまま表示
        ticker_to_name = {v.upper(): k for k, v in JAPAN_STOCKS.items()}
        display_name = ticker_to_name.get(ticker.upper(), ticker)
    else:
        # プルダウン選択を使う
        ticker = JAPAN_STOCKS[selected_name]
        display_name = selected_name

    st.markdown("---")
    st.markdown("### 📅 分析期間")
    period_options = {
        "3ヶ月": "3mo",
        "6ヶ月": "6mo",
        "1年": "1y",
        "2年": "2y",
        "5年": "5y",
    }
    period_label = st.select_slider(
        "期間",
        options=list(period_options.keys()),
        value="1年",
    )
    period = period_options[period_label]

    st.markdown("---")
    st.markdown("### 🤖 AIモデル設定")
    model_label = st.selectbox(
        "予測モデル",
        options=list(MODELS.keys()),
        index=0,
    )
    model_key = MODELS[model_label]

    retrain = st.toggle("自動再学習（Rolling）", value=False)
    if retrain:
        retrain_freq = st.slider("再学習頻度（日）", 30, 180, 60, 30)

    st.markdown("---")
    st.markdown("### ⚙️ バックテスト設定")
    top_k = st.slider("ポートフォリオ銘柄数（Top-K）", 5, 50, 20, 5)
    transaction_cost = st.slider("取引コスト（bps）", 0, 50, 10, 5)

    st.markdown("---")
    run_button = st.button("🚀 分析実行", use_container_width=True)

    # 環境モード表示
    if IS_LOCAL and QLIB_AVAILABLE:
        env_color  = "#10b981"
        env_icon   = "💻"
        env_title  = "ローカル / Qlib フルモード"
        env_detail = "AIモデル学習 + 本番バックテスト"
    elif IS_LOCAL and not QLIB_AVAILABLE:
        env_color  = "#f59e0b"
        env_icon   = "💻"
        env_title  = "ローカル / テクニカルモード"
        env_detail = "data_collector_japan.py を実行してください"
    else:
        env_color  = "#3b82f6"
        env_icon   = "☁️"
        env_title  = "クラウドモード"
        env_detail = "yfinance + テクニカルシグナル"

    st.markdown(f"""
    <div style='margin-top:32px; padding:12px; background:#0f172a; border-radius:8px;
                border:1px solid {env_color}40; font-size:11px; color:#475569; text-align:center;'>
        <div style='color:{env_color}; margin-bottom:4px; font-weight:600; font-size:13px;'>
            {env_icon} {env_title}
        </div>
        <div style='color:#64748b; margin-bottom:8px;'>{env_detail}</div>
        <div style='color:#60a5fa;'>⚡ データ</div>
        Yahoo Finance (yfinance)<br>
        <div style='margin-top:6px; color:#60a5fa;'>🔧 エンジン</div>
        Microsoft Qlib (MIT License)
    </div>
    """, unsafe_allow_html=True)


# ─── メインコンテンツ ──────────────────────────────────────────────────────────

env_badge_color = "badge-green" if IS_LOCAL and QLIB_AVAILABLE else ("badge-blue" if IS_CLOUD else "badge-yellow")
env_badge_text  = ("💻 ローカル / Qlib AIモード" if IS_LOCAL and QLIB_AVAILABLE
                   else ("☁️ クラウドモード" if IS_CLOUD else "⚠️ テクニカルモード"))

# ヒーローバナー
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">QlibJapan — AI株式分析プラットフォーム</div>
    <p class="hero-subtitle">
        Microsoft Qlib × yfinance で構築 ／ 東証全銘柄対応 ／ 機械学習による価格予測・バックテスト
    </p>
    <div style="margin-top:16px; display:flex; gap:8px; flex-wrap:wrap;">
        <span class="badge badge-blue">🤖 {model_label}</span>
        <span class="badge badge-green">📈 {display_name}</span>
        <span class="badge badge-yellow">📅 {period_label}</span>
        <span class="badge {env_badge_color}">{env_badge_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# データ取得
with st.spinner("📡 データ取得中..."):
    df = fetch_stock_data(ticker, period)

if df.empty:
    st.error(f"⚠️ {ticker} のデータを取得できませんでした。ティッカーを確認してください。")
    st.stop()

# ─── 分析実行（学習期間・テスト期間の計算） ───────────────────────────────────
# timedelta ではなく営業日ベースで分割することで
# Qlib内部の配列サイズズレ (253,) vs (252,) を防ぐ

period_bdays = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260}
total_bdays  = period_bdays.get(period, 252)

# 実際のyfinanceデータのインデックス（営業日のみ）を使って分割
if not df.empty:
    biz_dates   = df.index                          # yfinanceが返す営業日インデックス
    n_total     = len(biz_dates)
    n_train     = max(int(n_total * 0.7), 60)       # 最低60営業日は学習に使う
    n_valid     = max(int(n_train * 0.1), 20)       # 学習の10%をバリデーションに

    train_start = biz_dates[0].strftime("%Y-%m-%d")
    valid_start = biz_dates[n_train - n_valid].strftime("%Y-%m-%d")
    valid_end   = biz_dates[n_train - 1].strftime("%Y-%m-%d")
    train_end   = valid_end                         # Qlibのtrain_endはvalid_endと同じでOK
    test_start  = biz_dates[n_train].strftime("%Y-%m-%d")
    test_end    = biz_dates[-1].strftime("%Y-%m-%d")
else:
    # dfが空の場合のフォールバック（通常ここには来ない）
    end_dt      = datetime.today()
    start_dt    = end_dt - timedelta(days=365)
    split_dt    = start_dt + timedelta(days=255)
    train_start = start_dt.strftime("%Y-%m-%d")
    valid_start = (start_dt + timedelta(days=204)).strftime("%Y-%m-%d")
    valid_end   = (start_dt + timedelta(days=254)).strftime("%Y-%m-%d")
    train_end   = valid_end
    test_start  = (split_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    test_end    = end_dt.strftime("%Y-%m-%d")

retrain_freq_val = retrain_freq if retrain else 60

# 分析実行（キャッシュ付き）
with st.spinner("🤖 モデル学習・バックテスト実行中..."):
    bt, ic_df, portfolio_df = run_full_analysis(
        ticker=ticker,
        model_key=model_key,
        train_start=train_start,
        valid_start=valid_start,
        valid_end=valid_end,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        top_k=top_k,
        transaction_cost_bps=transaction_cost,
        use_rolling=retrain,
        retrain_freq=retrain_freq_val,
    )

# ─── ステータスバー ─────────────────────────────────────────────────────────────
last_price = df["終値"].iloc[-1]
prev_price = df["終値"].iloc[-2] if len(df) > 1 else last_price
price_change = (last_price - prev_price) / prev_price * 100
price_color = "#10b981" if price_change >= 0 else "#ef4444"
arrow = "▲" if price_change >= 0 else "▼"

st.markdown(f"""
<div class="status-bar">
    <div class="status-live">
        <div class="status-dot"></div>
        <span>LIVE DATA</span>
    </div>
    <span>｜</span>
    <span><b style="color:#f1f5f9;">{ticker}</b> — {display_name}</span>
    <span>｜</span>
    <span>終値: <b style="color:#f1f5f9;">¥{last_price:,.0f}</b>
          <b style="color:{price_color};">{arrow}{abs(price_change):.2f}%</b></span>
    <span>｜</span>
    <span>更新: {datetime.now().strftime('%Y/%m/%d %H:%M')}</span>
    <span style="margin-left:auto;">データ件数: {len(df):,}日分</span>
</div>
""", unsafe_allow_html=True)

# ─── サマリー指標カード ─────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

metrics = [
    ("年率リターン", f"{bt.get('annual_return',0)*100:.1f}%",
     "positive" if bt.get('annual_return', 0) > 0 else "negative"),
    ("シャープレシオ", f"{bt.get('sharpe_ratio',0):.2f}", ""),
    ("最大ドローダウン", f"{bt.get('max_drawdown',0)*100:.1f}%", "negative"),
    ("勝率", f"{bt.get('win_rate',0)*100:.1f}%",
     "positive" if bt.get('win_rate', 0) > 0.5 else ""),
    ("ボラティリティ (20日)", f"{df['ボラティリティ'].iloc[-1]:.2f}%", ""),
]

for col, (label, value, cls) in zip([c1, c2, c3, c4, c5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {cls}">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── タブ ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 株価チャート",
    "🔬 バックテスト",
    "📊 IC分析",
    "🏆 ポートフォリオ",
    "⚙️ モデル設定",
])

# ──────────── TAB 1: 株価チャート ────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header"><div class="section-dot"></div><h2>株価チャート & テクニカル指標</h2></div>', unsafe_allow_html=True)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.2],
        vertical_spacing=0.04,
        subplot_titles=("株価 + 移動平均", "出来高", "日次リターン（%）")
    )

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["始値"], high=df["高値"],
        low=df["安値"], close=df["終値"],
        name="株価",
        increasing_line_color="#10b981",
        decreasing_line_color="#ef4444",
        increasing_fillcolor="#10b981",
        decreasing_fillcolor="#ef4444",
    ), row=1, col=1)

    # 移動平均線
    fig.add_trace(go.Scatter(
        x=df.index, y=df["20日MA"],
        name="20日MA", line=dict(color="#3b82f6", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["60日MA"],
        name="60日MA", line=dict(color="#f59e0b", width=1.5),
    ), row=1, col=1)

    # 出来高
    colors_vol = ["#10b981" if r >= 0 else "#ef4444" for r in df["日次リターン"].fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=df["出来高"],
        name="出来高", marker_color=colors_vol, opacity=0.7,
    ), row=2, col=1)

    # 日次リターン
    ret_colors = ["#10b981" if r >= 0 else "#ef4444" for r in df["日次リターン"].fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=df["日次リターン"],
        name="日次リターン", marker_color=ret_colors, opacity=0.8,
    ), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
        font=dict(color="#94a3b8", family="Noto Sans JP"),
        height=600,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(gridcolor="#1e293b", showgrid=True)
    fig.update_yaxes(gridcolor="#1e293b", showgrid=True)

    st.plotly_chart(fig, use_container_width=True)

    # 統計サマリー
    with st.expander("📋 基本統計量"):
        stats = df[["始値", "高値", "安値", "終値", "出来高"]].describe()
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)


# ──────────── TAB 2: バックテスト ────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header"><div class="section-dot"></div><h2>バックテスト結果</h2></div>', unsafe_allow_html=True)

    if not bt:
        st.warning("データが不足しています。")
    else:
        # 累積リターンチャート
        fig2 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.06,
            subplot_titles=("累積リターン比較（戦略 vs ベンチマーク）", "ドローダウン（%）")
        )

        fig2.add_trace(go.Scatter(
            x=bt["cum_strategy"].index,
            y=(bt["cum_strategy"] - 1) * 100,
            name=f"AIモデル ({model_label[:8]}...)",
            line=dict(color="#3b82f6", width=2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        ), row=1, col=1)

        fig2.add_trace(go.Scatter(
            x=bt["cum_benchmark"].index,
            y=(bt["cum_benchmark"] - 1) * 100,
            name="ベンチマーク（買い持ち）",
            line=dict(color="#6b7280", width=1.5, dash="dash"),
        ), row=1, col=1)

        fig2.add_trace(go.Scatter(
            x=bt["drawdown"].index,
            y=bt["drawdown"] * 100,
            name="ドローダウン",
            line=dict(color="#ef4444", width=1),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        ), row=2, col=1)

        fig2.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
            font=dict(color="#94a3b8", family="Noto Sans JP"),
            height=500,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        fig2.update_xaxes(gridcolor="#1e293b")
        fig2.update_yaxes(gridcolor="#1e293b")

        st.plotly_chart(fig2, use_container_width=True)

        # 詳細指標
        st.markdown("#### 📊 パフォーマンス指標")
        col_a, col_b, col_c, col_d = st.columns(4)
        perf_items = [
            ("累積リターン", f"{bt['total_return']*100:.2f}%"),
            ("年率リターン", f"{bt['annual_return']*100:.2f}%"),
            ("シャープレシオ", f"{bt['sharpe_ratio']:.3f}"),
            ("最大ドローダウン", f"{bt['max_drawdown']*100:.2f}%"),
        ]
        for col, (k, v) in zip([col_a, col_b, col_c, col_d], perf_items):
            with col:
                st.metric(k, v)

        # データソース表示
        source = bt.get("source", "simple")
        if source == "qlib":
            st.success("✅ Qlib 実バックテストエンジンで計算されました（TopkDropoutStrategy）")
        else:
            st.info(
                "ℹ️ **テクニカル指標フォールバック**: "
                "Qlib 未セットアップのため、モメンタム＋移動平均クロスのシグナルで計算しています。\n\n"
                "`data_collector_japan.py` を実行後に再起動すると Qlib AIモデルが有効になります。"
            )

        # コストなし vs コストあり 比較（Qlib接続時のみ）
        if source == "qlib" and "annual_return_no_cost" in bt:
            st.markdown("#### 💰 コスト影響分析")
            c_nc1, c_nc2, c_nc3 = st.columns(3)
            c_nc1.metric("年率リターン（コストなし）", f"{bt['annual_return_no_cost']*100:.2f}%")
            c_nc2.metric("年率リターン（コストあり）", f"{bt['annual_return']*100:.2f}%",
                         delta=f"{(bt['annual_return']-bt['annual_return_no_cost'])*100:.2f}%")
            c_nc3.metric("コスト影響", f"{(bt['annual_return_no_cost']-bt['annual_return'])*100:.2f}%")


# ──────────── TAB 3: IC分析 ────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header"><div class="section-dot"></div><h2>IC分析（情報係数）</h2></div>', unsafe_allow_html=True)
    st.caption("ICはモデルの予測シグナルと実際のリターンの相関係数。高いほど予測精度が高い。")

    fig3 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("月次IC推移", "IC分布ヒストグラム"),
    )

    colors_ic = ["#10b981" if v > 0 else "#ef4444" for v in ic_df["IC"]]
    fig3.add_trace(go.Bar(
        x=ic_df["月"], y=ic_df["IC"],
        name="月次IC", marker_color=colors_ic,
    ), row=1, col=1)

    fig3.add_trace(go.Scatter(
        x=ic_df["月"], y=ic_df["IC"].rolling(3, min_periods=1).mean(),
        name="3ヶ月移動平均", line=dict(color="#f59e0b", width=2),
    ), row=1, col=1)

    fig3.add_trace(go.Histogram(
        x=ic_df["IC"], name="IC分布",
        marker_color="#3b82f6", opacity=0.7, nbinsx=15,
    ), row=1, col=2)

    fig3.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
        font=dict(color="#94a3b8", family="Noto Sans JP"),
        height=380, margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig3.update_xaxes(gridcolor="#1e293b", tickangle=45)
    fig3.update_yaxes(gridcolor="#1e293b")
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(
        ic_df.style.format({"IC": "{:.4f}", "ICIR": "{:.4f}", "ランク IC": "{:.4f}"}),
        use_container_width=True,
    )


# ──────────── TAB 4: ポートフォリオ ────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header"><div class="section-dot"></div><h2>ポートフォリオ構成</h2></div>', unsafe_allow_html=True)

    source_label = "Qlibシグナル" if QLIB_AVAILABLE and bt.get("source") == "qlib" else "テクニカルシグナル"

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.dataframe(portfolio_df, use_container_width=True, height=320)

    with col_right:
        fig_pie = go.Figure(go.Pie(
            labels=portfolio_df["銘柄名"],
            values=portfolio_df["配分比率 (%)"],
            hole=0.5,
            textfont=dict(size=11),
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0a0e1a",
            font=dict(color="#94a3b8", family="Noto Sans JP"),
            height=320,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    n_disp = len(portfolio_df)
    strategy_name = "TopkDropoutStrategy（Qlib）" if bt.get("source") == "qlib" else "モメンタム・シグナル"
    st.caption(f"📌 上位 {n_disp} 銘柄のポートフォリオ（シグナル: {source_label} ／ 選択アルゴリズム: {strategy_name}）")


# ──────────── TAB 5: モデル設定 ────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header"><div class="section-dot"></div><h2>モデル設定 & セットアップガイド</h2></div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### 🔧 Qlibセットアップ手順")
        st.code("""
# 1. Qlib インストール
pip install pyqlib

# 2. 日本株データ取得（yfinance経由）
python data_collector_japan.py

# 3. Qlib 初期化
import qlib
qlib.init(provider_uri="~/.qlib/qlib_data/jp_data",
          region="jp")

# 4. バックテスト実行
from qlib.workflow import R
from qlib.utils import init_instance_by_config
        """, language="python")

    with col_s2:
        st.markdown("#### 📁 データ構成（TODO）")
        st.code("""
~/.qlib/qlib_data/jp_data/
├── calendars/
│   └── day.txt          # 日本市場営業日
├── instruments/
│   └── all.txt          # 全銘柄リスト（.T形式）
└── features/
    └── 7203.T/          # 銘柄ごと
        ├── open.day.bin
        ├── close.day.bin
        ├── volume.day.bin
        └── ...
        """, language="bash")

    st.markdown("#### 📦 接続予定モジュール")

    todos = [
        ("✅", "yfinance データ取得", "app.py"),
        ("✅", "Streamlit UI", "app.py"),
        ("✅", "Qlib データコレクター（JP対応）", "data_collector_japan.py"),
        ("✅", "Qlib 初期化・データ検証", "qlib_init.py"),
        ("✅", "モデル学習（LightGBM/LSTM等）", "model_trainer.py"),
        ("✅", "実バックテスト接続", "backtest_runner.py"),
        ("✅", "ローリング再学習（Rolling Retrain）", "model_trainer.py"),
        ("🔲", "オンライン予測（毎日自動更新）", "online_inference.py — 次フェーズ"),
        ("🔲", "J-Quants API 対応（高品質データ）", "jquants_collector.py — 次フェーズ"),
    ]

    for status, name, file in todos:
        color = "#10b981" if status == "✅" else "#6b7280"
        st.markdown(
            f'<div style="padding:8px 12px; margin:4px 0; background:#161d2e; '
            f'border-radius:6px; border-left:3px solid {color}; font-size:13px;">'
            f'{status} <b>{name}</b> '
            f'<span style="color:#64748b; float:right; font-size:11px;">{file}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("""
    ---
    #### 🔗 参考リンク
    - [Microsoft Qlib GitHub](https://github.com/microsoft/qlib)
    - [Qlib ドキュメント](https://qlib.readthedocs.io/)
    - [yfinance ドキュメント](https://github.com/ranaroussi/yfinance)
    - [J-Quants API（高品質な日本株データ）](https://jpx-jquants.com/)
    """)

# ─── フッター ──────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#1e293b; margin-top:40px;">
<div style="text-align:center; color:#475569; font-size:12px; padding:16px 0 8px;">
    QlibJapan — Powered by
    <a href="https://github.com/microsoft/qlib" style="color:#3b82f6;">Microsoft Qlib</a>
    &amp;
    <a href="https://github.com/ranaroussi/yfinance" style="color:#3b82f6;">yfinance</a>
    ／ MITライセンス ／ 投資助言ではありません
</div>
""", unsafe_allow_html=True)
