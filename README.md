# 🇮🇳 Indian Listed NCD, G-Sec & Corporate Bond Screener

<!-- LIVE_STREAMLIT_APP_BANNER_START -->
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ncd-bond-screener.streamlit.app/)

> **🚀 Live Streamlit App:** [**https://ncd-bond-screener.streamlit.app/**](https://ncd-bond-screener.streamlit.app/)
<!-- LIVE_STREAMLIT_APP_BANNER_END -->

A high-performance Python & Streamlit institutional analytics screener for secondary market Non-Convertible Debentures (NCDs), Government Securities (G-Secs), State Development Loans (SDLs), InvIT Bonds, and Tax-Free PSU Bonds in India.

## Key Capabilities & Live Data Architecture

- **100% Live NSE Capital Market Debt API Integration**:
  - Connects directly to the official NSE India Debt & G-Sec feeds (`https://www.nseindia.com/api/liveBonds-traded-on-cm?type=bonds` and `?type=gsec`) to stream real-time Clean LTP, traded unit volume, turnover (₹ Cr), credit ratings, coupon rates, and maturity dates.
  - Supports toggling between the **58 Verified Liquid NSE/BSE Securities** (Sovereign G-Secs/SDLs, Section 10(15)(iv)(h) Tax-Free PSU Bonds, InvIT Bonds, and Corporate/NBFC NCDs) and the **Full 300+ Active NSE Traded Debt Universe (>1Y Tenor)**.
- **Live 10Y Sovereign Benchmark Feed**: Dynamically calculates the 10-Year Indian Sovereign Benchmark Yield directly from the live NSE G-Sec order book (`754GS2036` / `71GS2034` / `718GS2033`), with TradingEconomics and FBIL/RBI fallback calibration.
- **Dynamic Accrued Interest & Fractional-Period Dirty Price YTM Solver**:
  - Computes exact accrued interest from the last coupon date using Actual/Actual Indian debt day-count conventions ($\text{Dirty Price} = \text{Clean LTP} + \text{Accrued Interest}$).
  - Solves exact Yield to Maturity (YTM), Macaulay Duration, Modified Duration, and Convexity using fractional-period cash-flow discounting ($t_j = w + j$).
- **Pre-Tax Equivalent Yield (PTEY)**: Evaluates Tax-Free PSU Bonds (Section 10(15)(iv)(h) — IRFC, PFC, REC, NHAI, HUDCO, NTPC, NHPC, IREDA, IIFCL) against Bank Fixed Deposits and taxable corporate bonds for any user tax slab.
- **RBI Policy Rate Shock Simulator**: Evaluates bond portfolio capital gain or loss under monetary policy shifts using Taylor-expansion duration & convexity sensitivity:
  $$\frac{\Delta P}{P} \approx -D_{\text{mod}} \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2$$
- **Live NSE Lookup, Zerodha Kite Holdings Sync & Bhavcopy Ingestion**:
  - Auto-fill any listed NSE debt security by ISIN or ticker symbol directly from the live NSE API.
  - One-click Zerodha Kite `holdings.csv` / TSV portfolio sync and daily NSE/BSE Debt Bhavcopy ingestion.

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aetherasystemscore/NCD-Bond_Screener.git
   cd NCD-Bond_Screener
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
├── data_loaders.py             # Live NSE Debt/G-Sec API client, fractional-period YTM/duration/convexity solver & Bhavcopy parser
├── requirements.txt            # Python dependencies (Streamlit, Pandas, NumPy, Plotly, SciPy, Requests)
├── bonds_data.csv              # Master verified NSE/BSE bond database (zero synthetic rows)
├── data/
│   └── sample_bonds.csv        # Synchronized verified bond master dataset
└── .streamlit/
    └── config.toml             # Institutional dark-mode theme & server configuration
```
