# QlibJapan 🇯🇵 — AI株式分析プラットフォーム

Microsoft Qlib × yfinance で構築した、東証銘柄対応のAIクオンツ投資プラットフォームです。

---

## 📦 ファイル構成

```
qlib_japan/
├── app.py                    # Streamlit メインアプリ ← ここから起動
├── qlib_init.py              # Qlib 初期化・データ検証
├── model_trainer.py          # AIモデル学習（LightGBM/LSTM等）& ローリング再学習
├── backtest_runner.py        # バックテスト実行・IC計算・ポートフォリオ生成
├── data_collector_japan.py   # yfinance → Qlib形式 変換スクリプト
├── requirements.txt          # 依存ライブラリ
└── README.md
```

---

## 🚀 クイックスタート

### モード1: Qlib なし（即起動）

```bash
pip install streamlit yfinance pandas numpy plotly
streamlit run app.py
```

### モード2: Qlib フル機能（AIモデル）

```bash
pip install -r requirements.txt
python data_collector_japan.py --tickers 7203.T 6758.T 9984.T --start 2018-01-01
streamlit run app.py
```

---

## 🗺️ アーキテクチャ

```
[Streamlit UI (app.py)]
    ├── fetch_stock_data()       ← yfinance チャートデータ
    └── run_full_analysis()
          ├── [qlib_init.py]     ← Qlib 初期化・検証
          ├── [model_trainer.py] ← モデル学習・予測スコア
          └── [backtest_runner.py] ← BT / IC / ポートフォリオ
```

### フォールバック設計

| 状態 | モデル | バックテスト |
|------|--------|-------------|
| Qlib あり | LightGBM / LSTM 等 | TopkDropoutStrategy |
| Qlib なし | モメンタム + MA クロス | シンプルシグナル |

---

## 📊 実装済み機能

| 機能 | 状態 |
|------|------|
| yfinance データ取得 | ✅ |
| Qlib 初期化・検証 | ✅ |
| LightGBM / LSTM / GRU / Transformer / XGBoost | ✅ |
| ローリング再学習 | ✅ |
| Qlib バックテストエンジン | ✅ |
| フォールバックバックテスト | ✅ |
| IC / ICIR 計算 | ✅ |
| ポートフォリオ構成 | ✅ |
| コスト影響分析 | ✅ |
| オンライン予測（毎日自動） | 🔲 次フェーズ |
| J-Quants API 対応 | 🔲 次フェーズ |

---

## ⚠️ 免責事項

投資助言ではありません。投資判断はご自身の責任で。

---

## 🔗 参考

- [Microsoft Qlib](https://github.com/microsoft/qlib)
- [J-Quants API](https://jpx-jquants.com/)
- [Qlib ドキュメント](https://qlib.readthedocs.io/)
