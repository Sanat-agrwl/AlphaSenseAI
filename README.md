# AlphaSense — EDFSA Trading Co-Pilot for Indian Equities

Event-Driven Fundamental & Sentiment AI system for NIFTY 500 stocks.

**Core thesis**: Fundamentally strong stocks systematically overreact to negative news.
When a quality stock's Z-score drops below –2.0 on a sentiment shock, mean reversion trades offer a 60%+ win rate over a 20-day holding window.

---

## Architecture

```
AlphaSense/
├── config/
│   └── settings.py          # All thresholds, paths, secrets loading
├── alphasense/
│   ├── data/
│   │   ├── nse_client.py    # NSE price history + VIX + NIFTY 500 constituents
│   │   ├── bse_client.py    # BSE corporate announcements
│   │   ├── news_client.py   # RSS news (ET, MC, LiveMint) + stock matcher
│   │   └── labeler.py       # Build labeled_events.parquet (news + forward returns)
│   ├── screener/
│   │   └── fundamental.py   # Quality universe: Tier 1 (hard rules) + Tier 2 (score)
│   ├── sentiment/
│   │   ├── text.py          # FinBERT pipeline + fine-tuning on price-labeled data
│   │   └── audio.py         # Whisper ASR + management confidence scoring
│   ├── signal/
│   │   └── engine.py        # Z-score + sentiment BUY/SELL logic + position sizing
│   ├── backtest/
│   │   └── engine.py        # Walk-forward backtest (train/val/test splits)
│   ├── broker/
│   │   └── kite.py          # Paper trading + Zerodha Kite Connect live mode
│   ├── dashboard/
│   │   └── app.py           # Streamlit 5-tab dashboard
│   └── pipeline/
│       └── daily.py         # Scheduled daily orchestrator (pre-market / post-close)
└── scripts/
    ├── fetch_prices.py      # Download NSE price history
    ├── fetch_bse.py         # Download BSE announcements
    ├── fetch_news.py        # Download news articles
    ├── build_universe.py    # Build quality universe.parquet
    ├── build_labels.py      # Build labeled_events.parquet
    ├── score_news.py        # Score news with FinBERT
    ├── score_transcripts.py # Score earnings calls with Whisper
    └── run_backtest.py      # Walk-forward backtest
```

---

## Setup

### 1. Install dependencies

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# ffmpeg is required for audio transcription
brew install ffmpeg   # macOS
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and fill in:
#   KITE_API_KEY, KITE_API_SECRET  — from Zerodha developer console
#   HF_TOKEN                        — HuggingFace token for gated models
#   EXECUTION_MODE=paper            — keep as 'paper' until you have a track record
#   INITIAL_CAPITAL=10000000        — ₹1 crore starting capital
```

### 3. Bootstrap data (one-time)

```bash
# Step 1: Download NIFTY 500 constituents
python scripts/fetch_prices.py --constituents

# Step 2: Backfill price history from 2015 (takes ~30 min)
python scripts/fetch_prices.py --backfill

# Step 3: Backfill India VIX
python scripts/fetch_prices.py --vix

# Step 4: Backfill BSE announcements (optional, improves labels)
python scripts/fetch_bse.py --backfill

# Step 5: Build the quality universe
python scripts/build_universe.py --build

# Step 6: Build labeled events (news events + forward returns)
python scripts/build_labels.py --build

# Step 7: Run the backtest to verify everything works
python scripts/run_backtest.py --period all
```

---

## Daily Operation

Run via cron or a scheduler:

```bash
# 08:30 IST — fetch fresh data before market open
python alphasense/pipeline/daily.py --pre-market

# 15:45 IST — post-close: score sentiment, generate signals, execute
python alphasense/pipeline/daily.py --post-close
```

Or run the full pipeline manually:

```bash
python alphasense/pipeline/daily.py --full
```

---

## Dashboard

```bash
streamlit run alphasense/dashboard/app.py
```

Opens at `http://localhost:8501` with 5 tabs:
1. **Backtest** — equity curve + drawdown chart
2. **Live Signals** — today's BUY/SELL signals with sentiment scores
3. **Universe** — quality universe browser
4. **Earnings Calls** — management confidence scoring
5. **Risk Monitor** — portfolio exposure vs limits

---

## Signal Logic

**BUY conditions** (all must be true):
- Stock in quality universe (Tier 1 or Tier 2)
- Sentiment score < –0.4 (FinBERT, trailing 7 days)
- Rolling Z-score < –2.0 (vs 60-day window)
- India VIX < 28
- No blocklist keywords (fraud, scam, insolvency, etc.)

**Exit conditions** (first trigger wins):
- Sentiment recovers above +0.1
- Z-score recovers above –1.0
- –8% stop-loss hit
- 20-day time stop

**Position sizing** (minimum of three constraints):
- 1% of 20-day average daily volume
- 10% of total capital per position
- 2% risk-based: `floor(capital * 0.02 / (price * 0.08))`

---

## Backtest Results (Walk-Forward)

| Period | Years | Trades | Sharpe | Max DD | Win Rate |
|--------|-------|--------|--------|--------|----------|
| Train  | 2015–2020 | 0 | — | — | — |
| Val    | 2021 | 159 | **4.67** | -6.6% | 54.7% |
| Test   | 2022–2024 | 598 | **3.27** | -15.2% | 53.2% |

Train period has 0 trades because the trial labeled_events dataset starts from 2021.
Run `scripts/build_labels.py --build` after backfilling NSE prices to populate 2015–2020.

Pass criteria (test period): Sharpe > 1.2, Max DD < 18%, Win rate > 50%

---

## Fine-Tuning FinBERT

The sentiment model uses `ProsusAI/finbert` by default. To fine-tune on Indian market data with price-as-label ground truth:

```bash
# Build labeled events first
python scripts/build_labels.py --build

# Fine-tune (GPU recommended, CPU works but slow)
python scripts/score_news.py --finetune
```

The fine-tuned model is saved to `models/finbert_indian/` and used automatically on subsequent runs.

---

## Paper → Live Transition Checklist

- [ ] 4–6 months of paper trading with positive Sharpe
- [ ] Drawdown stays below 18% in live paper conditions
- [ ] Win rate > 50% over 50+ trades
- [ ] Zerodha Kite Connect API key set up
- [ ] Change `EXECUTION_MODE=live` in `.env`
- [ ] Test with a single small-cap trade first

**Never switch to live mode without completing the above checklist.**

---

## Project Structure Decisions

- **No database**: All data stored as Parquet files for simplicity and portability
- **Paper mode default**: `EXECUTION_MODE=paper` in `.env.example` — must explicitly opt into live
- **Point-in-time strict**: Backtest uses `df[df.index < date]` — no look-ahead bias
- **Config singleton**: `cfg` imported from `config.settings` — one source of truth for all thresholds
- **Loguru**: Structured daily log files in `logs/daily_YYYY-MM-DD.log`

---

*Last updated: 2026-04-13*