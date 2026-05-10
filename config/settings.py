"""
Centralized configuration for AlphaSense AI.
All paths, thresholds and secrets in one place.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

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


@dataclass
class GrowwConfig:
    api_key:     str = field(default_factory=lambda: os.getenv("GROWW_API_KEY", ""))
    api_secret:  str = field(default_factory=lambda: os.getenv("GROWW_API_SECRET", ""))
    totp_secret: str = field(default_factory=lambda: os.getenv("GROWW_TOTP_SECRET", ""))


@dataclass
class KiteConfig:
    api_key:    str = field(default_factory=lambda: os.getenv("KITE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("KITE_API_SECRET", ""))
    mode:       str = field(default_factory=lambda: os.getenv("EXECUTION_MODE", "paper"))


@dataclass
class SignalConfig:
    # Entry — walk-forward OOS backtest (train=2020-21, val=2022, test=2023-26): best z=-3.0, stop=-0.10
    sentiment_threshold: float = -0.4
    zscore_threshold:    float = -3.0   # locked from walk-forward train sweep; Sharpe 2.85 OOS
    vix_max:             float = 28.0
    # Exit
    stop_loss_pct:       float = -0.10  # locked from walk-forward; recovery-dominant (97% exits)
    time_stop_days:      int   = 20
    recovery_sentiment:  float = 0.1
    recovery_zscore:     float = -1.0
    profit_exit_pct:     float = 0.12    # exit at 12% profit if pending orders waiting
    # Sizing
    max_pct_adv:         float = 0.01
    max_pct_capital:     float = 0.08
    max_positions:       int   = 15
    max_sector_pct:      float = 0.25
    risk_per_trade_pct:  float = 0.02
    # Capital (paper vs real are sized independently)
    paper_capital:       float = field(default_factory=lambda: float(os.getenv("PAPER_CAPITAL", "1000000")))
    # Rolling window
    rolling_window:      int   = 252
    return_period:       int   = 5
    signal_lag_days:     int   = 1
    # Momentum strategy
    momentum_zscore_min:      float = 3.0   # was 1.5; raised to reduce false signals (OOS sweep)
    momentum_sentiment_min:   float = 0.3
    momentum_stop_loss_pct:   float = -0.05
    momentum_profit_exit_pct: float = 0.15
    momentum_time_stop_days:  int   = 15
    # Earnings-surprise strategy
    earnings_sent_delta_min:  float = 0.10  # fmod.sentiment_delta threshold
    earnings_time_stop_days:  int   = 10
    earnings_stop_loss_pct:   float = -0.06
    earnings_profit_exit_pct: float = 0.10
    # Kelly sizing
    kelly_min_trades:    int   = 15     # min history before switching to Kelly
    kelly_max_fraction:  float = 0.25   # hard cap on Kelly fraction


@dataclass
class ScreenerConfig:
    min_roe:              float = 0.15
    min_roe_years:        int   = 3
    max_debt_equity:      float = 1.0
    min_promoter_holding: float = 0.30
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
    neg_threshold: float = -0.02
    pos_threshold: float = 0.02


@dataclass
class AudioConfig:
    whisper_model:  str   = "large-v3"
    device:         str   = "cpu"
    hf_token:       str   = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    lookback_qtrs:  int   = 4
    anomaly_drop:   float = 15.0


@dataclass
class BacktestConfig:
    train_start: str   = "2015-01-01"
    train_end:   str   = "2020-12-31"
    val_start:   str   = "2021-01-01"
    val_end:     str   = "2021-12-31"
    test_start:  str   = "2022-01-01"
    test_end:    str   = "2024-12-31"
    cost_bps:    int   = 50
    capital:     float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "10000000")))
    position_size: float = 1_000_000


@dataclass
class AppConfig:
    groww:     GrowwConfig     = field(default_factory=GrowwConfig)
    kite:      KiteConfig      = field(default_factory=KiteConfig)
    signal:    SignalConfig    = field(default_factory=SignalConfig)
    screener:  ScreenerConfig  = field(default_factory=ScreenerConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    audio:     AudioConfig     = field(default_factory=AudioConfig)
    backtest:  BacktestConfig  = field(default_factory=BacktestConfig)
    # actual_trade: place real orders via Groww when True; paper-only when False
    actual_trade: bool = field(default_factory=lambda: os.getenv("ACTUAL_TRADE", "false").lower() == "true")
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
