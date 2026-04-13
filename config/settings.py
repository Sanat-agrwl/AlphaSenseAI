"""
Centralized configuration for AlphaSense AI.
All paths, thresholds and secrets in one place.

Usage:
    from config.settings import cfg
    cfg.signal.zscore_threshold  # -2.0
    cfg.nse_dir                  # Path to price data
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

# Try loading .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent
DATA_DIR        = ROOT / "data"
NSE_DIR         = DATA_DIR / "nse"
NEWS_DIR        = DATA_DIR / "news"
BSE_DIR         = DATA_DIR / "bse"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
RESULTS_DIR     = DATA_DIR / "results"
MODELS_DIR      = DATA_DIR / "models"
LOGS_DIR        = ROOT / "logs"

for _d in [NSE_DIR, NEWS_DIR, BSE_DIR, TRANSCRIPTS_DIR / "audio",
           TRANSCRIPTS_DIR / "text", RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclass
class KiteConfig:
    api_key:    str = field(default_factory=lambda: os.getenv("KITE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("KITE_API_SECRET", ""))
    mode:       str = field(default_factory=lambda: os.getenv("EXECUTION_MODE", "paper"))


@dataclass
class IBKRConfig:
    host:      str = field(default_factory=lambda: os.getenv("IBKR_HOST", "127.0.0.1"))
    # 4002 = paper gateway, 4001 = live gateway
    port:      int = field(default_factory=lambda: int(os.getenv("IBKR_PORT", "4002")))
    client_id: int = field(default_factory=lambda: int(os.getenv("IBKR_CLIENT_ID", "1")))
    enabled:   bool = field(default_factory=lambda: os.getenv("IBKR_ENABLED", "false").lower() == "true")


@dataclass
class SignalConfig:
    # Entry
    sentiment_threshold: float = -0.4
    zscore_threshold:    float = -2.0
    vix_max:             float = 28.0
    # Exit
    stop_loss_pct:      float = -0.08
    time_stop_days:     int   = 20
    recovery_sentiment: float = 0.1
    recovery_zscore:    float = -1.0
    # Sizing
    max_pct_adv:        float = 0.01    # max 1% of avg daily volume
    max_pct_capital:    float = 0.08    # max 8% of capital per trade
    max_positions:      int   = 15
    max_sector_pct:     float = 0.25
    risk_per_trade_pct: float = 0.02    # 2% capital at risk → determines shares
    # Rolling window
    rolling_window:     int   = 252
    return_period:      int   = 5       # 5-day return for Z-score
    signal_lag_days:    int   = 1


@dataclass
class ScreenerConfig:
    # Tier 1
    min_roe:              float = 0.15
    min_roe_years:        int   = 3
    max_debt_equity:      float = 1.0
    min_promoter_holding: float = 0.30
    # Tier 2 weights
    w_roce:          float = 0.25
    w_fcf:           float = 0.30
    w_accruals:      float = 0.25
    w_concentration: float = 0.20


@dataclass
class SentimentConfig:
    base_model:    str   = "ProsusAI/finbert"
    tuned_dir:     str   = str(MODELS_DIR / "finbert_indian")
    max_length:    int   = 512
    learning_rate: float = 2e-5
    num_epochs:    int   = 5
    batch_size:    int   = 16
    neg_threshold: float = -0.02   # price reaction → negative label
    pos_threshold: float = 0.02


@dataclass
class AudioConfig:
    whisper_model:  str   = "large-v3"
    device:         str   = "cpu"
    hf_token:       str   = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    lookback_qtrs:  int   = 4
    anomaly_drop:   float = 15.0   # confidence drop > 15 pts = alert


@dataclass
class BacktestConfig:
    train_start: str   = "2015-01-01"
    train_end:   str   = "2020-12-31"
    val_start:   str   = "2021-01-01"
    val_end:     str   = "2021-12-31"
    test_start:  str   = "2022-01-01"
    test_end:    str   = "2024-12-31"
    cost_bps:    int   = 50          # 0.5% round-trip
    capital:     float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "10000000")))
    position_size: float = 1_000_000  # fixed ₹10L per trade


@dataclass
class AppConfig:
    kite:      KiteConfig      = field(default_factory=KiteConfig)
    ibkr:      IBKRConfig      = field(default_factory=IBKRConfig)
    signal:    SignalConfig    = field(default_factory=SignalConfig)
    screener:  ScreenerConfig  = field(default_factory=ScreenerConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    audio:     AudioConfig     = field(default_factory=AudioConfig)
    backtest:  BacktestConfig  = field(default_factory=BacktestConfig)
    # paths
    root:        Path = ROOT
    data_dir:    Path = DATA_DIR
    nse_dir:     Path = NSE_DIR
    news_dir:    Path = NEWS_DIR
    bse_dir:     Path = BSE_DIR
    results_dir: Path = RESULTS_DIR
    models_dir:  Path = MODELS_DIR
    logs_dir:    Path = LOGS_DIR
    transcripts_dir: Path = TRANSCRIPTS_DIR


cfg = AppConfig()