# 🇮🇳 Indian Listed NCD, G-Sec & Corporate Bond Screener

<!-- LIVE_STREAMLIT_APP_BANNER_START -->
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ncd-bond-screener.streamlit.app/)

> **🚀 Live Streamlit App:** [**https://ncd-bond-screener.streamlit.app/**](https://ncd-bond-screener.streamlit.app/)
<!-- LIVE_STREAMLIT_APP_BANNER_END -->


A high-performance Python & Streamlit institutional analytics screener for secondary market Non-Convertible Debentures (NCDs), Government Securities (G-Secs), and listed corporate bonds in India.

## Features

- **Live 10Y Sovereign Benchmark Feed**: Dynamically tracks the 10-Year Indian Sovereign Benchmark Yield (with live feeds and FBIL/RBI calibration).
- **Dynamic Accrued Interest & Dirty Price Engine**: Computes exact accrued interest from the last coupon date using Actual/Actual Indian debt day-count conventions, giving true settlement cash outlay ($\text{Dirty Price} = \text{Clean LTP} + \text{Accrued Interest}$).
- **Pre-Tax Equivalent Yield (PTEY)**: Evaluates Tax-Free Bonds (Section 10(15)(iv)(h) - IRFC, PFC, REC, NHAI) against Bank Fixed Deposits and taxable bonds for any tax slab.
- **High-Precision Duration & Convexity**: Vectorized cash flow discounting calculates Macaulay Duration, Modified Duration, and Convexity across thousands of bonds in under 250 milliseconds.
- **RBI Policy Rate Shock Simulator**: Evaluates bond portfolio capital gain or loss under monetary policy shifts using mathematically accurate Taylor-expansion sensitivity:
  $$\frac{\Delta P}{P} \approx -D_{\text{mod}} \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2$$
- **Live Bhavcopy Ingestion**: Direct support for daily official debt market files from **NSE** (`sec_bhavdata_full_DDMMYYYY.csv`) and **BSE** (`DEBT_YYYYMMDD.zip` / CSV) to update real traded prices, turnover, and volume.
- **Yield Curve & Credit Spread Matrix**: Interactive Plotly scatter plot (Yield vs. Tenor) and credit spread breakdowns across sectors and rating tiers.
- **Multi-Factor Buy Score (0-100)**: Quantitative scoring balancing Post-Tax Yield, Rating Safety, Asset Security, Govt/PSU backing, Liquidity, and Tenor.

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/purntripathi-cmd/ncd-bond_screener.git
   cd ncd-bond_screener
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Repository Structure

```
├── app.py                      # Main Streamlit web application & dashboards
├── data_loaders.py             # Bond pricing engine, duration/convexity, accrued interest & Bhavcopy parser
├── requirements.txt            # Python dependencies (Streamlit, Pandas, NumPy, Plotly, SciPy, Requests)
├── bonds_data.csv              # Master bond database
├── data/
│   └── sample_bonds.csv        # Seed bond master with G-Secs, Tax-Free bonds, Tier-2, & Corporate NCDs
└── .streamlit/
    └── config.toml             # Institutional dark-mode theme & server configuration
```

## Secondary Market Debt Data Ingestion

Under the **"📥 Live Bhavcopy & Management"** tab, you can drag and drop any daily debt market Bhavcopy from NSE or BSE. The screener will:
1. Match bonds by **ISIN**.
2. Refresh Clean LTP, Daily Turnover (₹ Cr), and Number of Trades.
3. Dynamically recompute Accrued Interest, Dirty Settlement Price, Credit Spreads, and YTM in real time.
