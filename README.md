# AlphaSense AI

**Event-Driven Fundamental & Sentiment Co-Pilot for Indian Equities**

An autonomous trading signal system built for the NSE/BSE mid-cap universe. It combines price statistics, text sentiment, earnings call audio analysis, BSE event classification, and quarterly fundamentals into a single signal engine that runs fully on a schedule with no manual intervention.

---

## What This Is

The Indian mid-cap market is structurally under-covered and emotionally overreactive. When a stock with strong fundamentals drops sharply on negative news, the price often over-shoots — the market is pricing fear, not reality. AlphaSense identifies that gap (negative sentiment + abnormal price drop + quality fundamentals = potential mean-reversion trade) and sizes a position accordingly.

The core thesis: **Z-score divergence + sentiment shock = temporary mispricing**. The edge is not speed — it is the depth of signal: BSE announcement classification, earnings call audio scoring, and XBRL quarterly fundamentals, all of which most retail desks and many institutional desks are not running systematically on Indian companies.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  NSE prices (yfinance)  │  BSE announcements  │  News (RSS+GNews)│
│  India VIX              │  Earnings audio      │  XBRL (yfinance) │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────┐
│                      SIGNAL LAYER                                │
│  FinBERT + Claude ensemble sentiment  │  5-day Z-score           │
│  BSE event classifier                 │  Management confidence   │
│  XBRL fundamentals scorer             │  Fundamental modifier    │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────┐
│                      SIGNAL ENGINE                               │
│  Quality universe filter → BSE block/boost → Fundamental block  │
│  → Sentiment threshold → Z-score threshold → Volume confirm     │
│  → Position size (1% ADV, 10% capital, 2%-risk)                  │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────┐
│                     EXECUTION LAYER                              │
│  Paper engine (₹10L)  ←→  Real engine (₹30K via Groww)          │
│  Pending → fill at next open │ Intraday stop-loss guard          │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────┐
│                      DASHBOARD (Streamlit)                       │
│  Live PnL  │  Signal log  │  Capital allocation  │  Real trades  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Signal Logic

### BUY — all conditions must be true simultaneously

| Condition | Threshold | Notes |
|---|---|---|
| Stock in quality universe | — | ROE ≥ 15%, D/E ≤ 1.0, promoter ≥ 30% |
| Sentiment score | < -0.4 | FinBERT + Claude ensemble; relaxed/tightened by BSE + fundamentals |
| 5-day return Z-score | < -2.0 | 252-day rolling window |
| Volume confirmation | ≥ 1.5× 20-day avg | Abnormal selling pressure confirmation |
| No BSE hard blocklist | — | Fraud, default, SEBI probe, CEO resignation |
| No fundamental block | — | XBRL score=-1 AND audio confidence < 40 |
| India VIX | < 28 | No new entries in systemic crisis |
| Position cap | < 15 open | Max 15 simultaneous positions |

### SELL — any one condition triggers exit

| Condition | Threshold |
|---|---|
| Sentiment recovery | > +0.1 |
| Z-score recovery | > -1.0 |
| Stop-loss | -8% from entry |
| Time-stop | 20 trading days |
| Profit + pending orders waiting | ≥ +12% gain |

### Threshold modifiers (stacking)

BSE and fundamental signals adjust sentiment and Z-score thresholds per symbol before the entry check:

| Source | Event | Sentiment delta | Z-score delta |
|---|---|---|---|
| BSE | Buyback / dividend / order win | +0.20 (relax) | — |
| BSE | Rating downgrade / profit warning | -0.30 (tighten) | — |
| BSE | Fraud / default / CEO departure | **BLOCK** | — |
| XBRL + Audio | Strong fundamentals + confident mgmt (conf ≥ 70) | +0.15 | +0.30 |
| XBRL + Audio | Deteriorating fundamentals + evasive mgmt (conf < 40) | **BLOCK** | — |
| XBRL | Deteriorating fundamentals only | -0.15 | -0.30 |
| Audio | Low management confidence (conf < 50) | -0.08 | -0.15 |

---

## Repository Structure

```
AlphaSense/
├── alphasense/
│   ├── data/
│   │   ├── nse_client.py          Price + VIX data via yfinance
│   │   ├── bse_client.py          BSE corporate announcements (XML feed)
│   │   ├── bse_signal.py          BSE → blocklist / boost / drag / earnings_due
│   │   ├── news_client.py         RSS + Google News article fetch + symbol tagging
│   │   ├── groww_client.py        Groww broker API (live price + order execution)
│   │   ├── audio_fetcher.py       BSE scan for earnings call audio URLs + download
│   │   ├── xbrl_client.py         Quarterly financials via yfinance + signal scores
│   │   └── labeler.py             Training data labeler for FinBERT fine-tuning
│   ├── screener/
│   │   └── fundamental.py         Quality universe builder (ROE, D/E, FCF, promoter)
│   ├── sentiment/
│   │   ├── text.py                FinBERT pipeline + Indian-finance fine-tuning
│   │   ├── ensemble.py            FinBERT + Claude/OpenAI ensemble scorer
│   │   └── audio.py               Deepgram Nova-2 transcription + Management Confidence Score
│   ├── signal/
│   │   ├── engine.py              Core signal engine (BUY/SELL logic + modifiers)
│   │   └── fundamental_signal.py  XBRL + audio → FundamentalModifier per symbol
│   ├── broker/
│   │   └── kite.py                PaperEngine + RealStateManager (Groww execution)
│   ├── pipeline/
│   │   ├── daily.py               Main orchestrator (pre-market / post-close)
│   │   ├── fill_at_open.py        Fill yesterday's pending orders at today's open
│   │   ├── intraday_monitor.py    12:30 IST stop-loss guard + live BUY scan
│   │   ├── quarterly.py           XBRL refresh + earnings audio pipeline
│   │   ├── feedback_loop.py       Post-trade outcome labeling + model retraining
│   │   └── sell_all_holdings.py   One-time Groww liquidation → seed real capital
│   ├── backtest/
│   │   ├── engine.py              Walk-forward backtesting engine
│   │   └── forward_test.py        Paper vs signal comparison reporter
│   └── dashboard/
│       └── app.py                 Streamlit dashboard (PnL, signals, capital)
├── config/
│   └── settings.py                All thresholds, paths, secrets in one place
├── scripts/
│   └── build_universe.py          One-time universe construction
└── data/
    ├── nse/                       Price parquets per symbol
    ├── bse/                       BSE announcement JSONs per day
    ├── news/                      News article JSONs
    ├── transcripts/
    │   ├── audio/                 Downloaded earnings call audio
    │   └── text/                  Transcripts + confidence score JSONs
    ├── xbrl/                      XBRL quarterly signal JSONs per symbol
    ├── results/                   BSE signal JSONs, backtest results
    └── models/                    Fine-tuned FinBERT checkpoint
```

---

## Cron Schedule (EC2, UTC)

| UTC | IST | Job |
|---|---|---|
| 03:00 Mon–Fri | 08:30 | Pre-market: fetch prices + news |
| 04:00 Mon–Fri | 09:30 | Fill pending orders at today's open price |
| 07:00 Mon–Fri | 12:30 | Intraday stop-loss guard + live BUY scan |
| 10:15 Mon–Fri | 15:45 | Post-close: sentiment → signals → stage BUYs, execute SELLs |
| 17:30 Mon–Fri | 23:00 | Feedback loop (PnL tracking + outcome labeling) |
| 00:30 Sunday | 06:00 | XBRL weekly refresh (full universe) |
| 11:00 Mon–Fri | 16:30 | Audio fetch for earnings-due symbols |

---

## Dual Capital System

Two completely independent engines share the same signal logic but size separately:

| | Paper | Real |
|---|---|---|
| Capital | ₹10,00,000 | ₹30,000 (Groww demat proceeds) |
| State file | `data/paper_state.json` | `data/real_state.json` |
| Per-position budget | 10% of capital | 10% of available cash |
| Execution | Simulated fills at open | Live Groww market orders |
| Sizing formula | min(1% ADV, 10% capital, 2%-risk) | `int(capital × 10% / price)` |

**Safety rules:**
- Real SELL only executes if the symbol exists in `real_state.json` — never falls back to paper qty (prevents accidental sale of pre-existing demat holdings)
- `record_buy()` only called when Groww returns a non-empty `order_id` (prevents ghost positions)
- Proceeds from each SELL add back to `real_state.capital` (available for next trade)

---

## Sentiment Pipeline

### Text (FinBERT + LLM ensemble)

1. Fetch articles from RSS feeds + Google News (tagged to NSE symbols)
2. Score each article with **FinBERT** (fine-tuned on Indian finance headlines)
3. If `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set, run **ensemble**: FinBERT + Claude/GPT vote independently; weighted average
4. Aggregate per symbol: mean score across all articles from last 24h

### Audio (Deepgram Nova-2 + Management Confidence Score)

1. `audio_fetcher.py` scans BSE announcements for conference call URLs (Konfer, EarnCast, direct MP3)
2. Downloads audio to `data/transcripts/audio/{SYMBOL}_{QUARTER}.mp3`
3. **Deepgram Nova-2** transcribes with Hindi-English code-switching, built-in speaker diarization
4. `EarningsCallScorer` computes:
   - **Lexical score**: forward-guidance keyword frequencies (positive/negative)
   - **Evasion score**: hedge phrase density, sentence repetition, disfluency rate
   - **Management Confidence Score** (0–100): `0.45 × lexical + 0.55 × (100 - evasion)`
5. `ConfidenceDeltaDetector` flags QoQ confidence drops as leading indicators

**Confidence interpretation:**
- ≥ 70: management is confident → relax entry thresholds if fundamentals also strong
- 40–69: neutral
- < 40: evasive management → tighten thresholds; block if XBRL also deteriorating

---

## XBRL / Quarterly Fundamentals

Source: yfinance (`{symbol}.NS`) quarterly financials, balance sheet, cash flow.

Signals computed per symbol:

| Signal | How | Score contribution |
|---|---|---|
| Revenue growth QoQ | % change last Q vs prior Q | +1 if > 5%, -1 if < -5% |
| EBITDA margin trend | last Q margin − 4Q rolling avg | +1 if > +1pp, -1 if < -2pp |
| Debt/equity direction | D/E ratio change QoQ | +1 if falling, -1 if rising |
| FCF quality | FCF / Net Income | +1 if > 1.2×, -1 if < 0.5× |
| **Overall score** | Sum of above | +1 (≥3 positives, 0 negatives) / -1 (≥2 negatives) / 0 |

Cached for 7 days in `data/xbrl/{symbol}.json`. Weekly refresh via cron.

---

## BSE Announcement Signals

Daily classification of BSE corporate announcements into four buckets:

| Bucket | Trigger | Effect |
|---|---|---|
| `blocklist_hits` | Fraud, NCLT, SEBI order, CEO/CFO resignation, ED raid | Hard block: no BUY for 5 days |
| `sentiment_boost` | Buyback, special dividend, large order win, rating upgrade | Relax sentiment threshold +0.2 |
| `sentiment_drag` | Rating downgrade, profit warning, loss reported | Tighten threshold -0.3 |
| `earnings_due` | Financial results announcement | Trigger audio fetch pipeline |

Rolling 5-day window: modifiers from the last 5 trading days remain active.

---

## Position Sizing

```
position_size = min(
    int(avg_daily_volume × 1%),       # liquidity cap
    int(capital × 10% / price),       # concentration cap
    int(capital × 2% / (price × 8%)) # risk cap (2% capital at 8% stop)
)
```

---

## Dashboard

Streamlit app (`alphasense/dashboard/app.py`) showing:

- Live paper PnL + open positions
- Signal history (BUY/SELL log with reasons)
- Groww portfolio holdings
- Real capital allocation (available cash, deployed, per-position budget)
- Affordability table: paper qty vs real qty vs real cost for each universe stock
- Open real positions + closed real trades from `real_state.json`

---

## Key Config (`.env` on EC2)

```
ANTHROPIC_API_KEY=...      # enables Claude ensemble sentiment
DEEPGRAM_API_KEY=...       # enables Deepgram Nova-2 transcription
GROWW_API_KEY=...          # live order execution
GROWW_TOTP_SECRET=...
PAPER_CAPITAL=1000000      # ₹10L paper portfolio
ACTUAL_TRADE=true          # set false to disable real execution
EXECUTION_MODE=live        # live | paper
```

---

## What's Working (as of May 2026)

- [x] Quality universe construction (NIFTY 500 filtered by ROE, D/E, FCF, promoter holding)
- [x] NSE price data + India VIX pipeline
- [x] BSE announcement fetch + daily classification
- [x] News fetch (RSS + Google News) with symbol tagging
- [x] FinBERT sentiment scoring (base + fine-tuning pipeline)
- [x] FinBERT + Claude/GPT ensemble scoring
- [x] 5-day Z-score signal with volume confirmation
- [x] Signal engine with BSE event modifiers
- [x] XBRL quarterly fundamentals via yfinance
- [x] Deepgram Nova-2 earnings call transcription
- [x] Management Confidence Score (lexical + evasion)
- [x] Fundamental signal modifier (XBRL + audio → threshold adjustment / block)
- [x] Paper trading engine (₹10L, full pipeline)
- [x] Real trading engine (₹30K via Groww)
- [x] Pending order staging → fill at next-day open
- [x] Intraday stop-loss guard (12:30 IST)
- [x] Full daily pipeline (pre-market / post-close / intraday)
- [x] Quarterly pipeline (XBRL refresh + audio fetch)
- [x] Feedback loop + outcome labeling
- [x] Walk-forward backtester
- [x] Streamlit dashboard
- [x] All cron jobs running on EC2

---

## What Can Be Built Next

### Near-term (next 4–6 weeks)

**1. Outcome-labeled training dataset**
Every closed trade already gets logged. Build a labeler that tags each signal with outcome (reversion occurred within 20d / did not), then use this as training data to fine-tune the FinBERT model on Indian-market-specific patterns. The dataset grows automatically every quarter.

**2. Sector exposure control**
Currently `max_sector_pct=25%` is in config but not enforced in the engine. Wire it up — prevents over-concentration in one sector during sector-wide corrections.

**3. Kelly Criterion position sizing**
Replace the current fixed `10% capital` cap with Kelly-optimal sizing based on historical win rate and average PnL per signal type. Makes position size adaptive to edge strength.

**4. Earnings call audio — full quarter batch**
`audio_fetcher.py` currently monitors daily. Add a script to backfill the last 4 quarters for all universe stocks — build the labeled archive from day one.

**5. Hindi earnings call fine-tuning**
Deepgram transcripts contain Hindi text. Fine-tune a small classifier specifically on Hindi management language to improve confidence scoring accuracy for companies that conduct calls in Hindi.

**6. RAG knowledge layer**
Add a vector database (Weaviate or Chroma) that stores BSE filings, annual reports, and credit rating reports. Queries like "what did management say about capex guidance last 3 quarters?" answered instantly during signal research.

### Medium-term (3–6 months)

**7. Live sentiment feed API**
Expose a REST API (`/api/v1/sentiment/{symbol}`) that returns the latest sentiment score, BSE flags, XBRL score, and audio confidence. This is the data product that prop desks pay ₹1–2L/month for.

**8. Multi-strategy support**
Current strategy: mean-reversion on sentiment shock. Add:
- Momentum continuation (sentiment + price moving together, not diverging)
- Earnings surprise (XBRL revenue beat + audio confidence spike → momentum)
- Buyback arbitrage (BSE buyback announcement + below-buyback-price → entry)

**9. Regulatory-grade reporting**
Trade log, daily PnL, drawdown, Sharpe, max drawdown — formatted as an audited performance report. Required for AIF Category III application (18–24 month track record needed).

**10. SEBI Research Analyst registration**
The signal output is already structured as a research report format. Registering as an RA allows publishing signals commercially without discretionary management burden.

### Long-term (6–18 months)

**11. SaaS dashboard for external clients**
Multi-tenant version of the current Streamlit dashboard. Each prop desk sees their own watchlist's signals, BSE flags, and audio confidence scores.

**12. Options strategy layer**
Current system trades equities. Add a layer that converts directional signals into options structures (e.g., buying ATM puts as protection on SELL signals, or selling OTM puts on strong BUY signals to collect premium while waiting for entry).

**13. AIF Category III fund**
Requires ₹20 crore corpus + 18–24 months of audited track record. The paper trading engine is already building this record. Target: apply for AIF registration in 2027 with 18 months of live performance data.

---

## Business Model Roadmap

| Phase | Timeline | Model | Revenue target |
|---|---|---|---|
| Now | 0–12 months | Proprietary trading (paper + real) + dataset building | — |
| Data product | 6–18 months | Sell sentiment scores as API/feed to 3–5 prop desks | ₹3–10L/month |
| SaaS | 12–24 months | Multi-tenant dashboard for SEBI-registered prop desks + family offices | ₹20–50L/month |
| AIF | 24–36 months | SEBI AIF Category III: 2% mgmt fee + 20% performance | Depends on AUM |

**The defensible moat is not the algorithm — it is the labeled dataset.** 500 earnings calls per year, each tagged with a 20-day outcome label, in Hindi and English, for Indian mid-cap companies. By year two: 4,000+ labeled call-outcome pairs. This archive is not replicable quickly and is the core asset.

---

## Infrastructure

| Component | Where |
|---|---|
| Signal engine + crons | AWS EC2 `t3.medium` (`3.110.115.238`) |
| Dashboard | Streamlit on same EC2, port 8501 |
| Price/fundamentals | yfinance (free) |
| BSE data | BSE India API (free) |
| News | RSS + Google News (free) |
| Sentiment LLM | Anthropic API (Claude Sonnet) |
| Audio transcription | Deepgram Nova-2 |
| Broker | Groww API (live) + paper engine |

Monthly infrastructure cost: ~₹3,000–5,000 (EC2 + API usage at current scale).
