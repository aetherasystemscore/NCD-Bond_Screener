import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from datetime import datetime, date, timedelta
from data_loaders import (
    load_sample_csv,
    compute_derived_metrics,
    get_live_10y_gsec_yield,
    ingest_bhavcopy,
    calculate_accrued_interest,
    calculate_ytm_from_price,
    BENCHMARK_10Y_GSEC_DEFAULT
)

# Page Setup
st.set_page_config(
    page_title="India NCD, G-Sec & Corporate Bond Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional Dark/Clean Styling
st.markdown("""
    <style>
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 1.2rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(30, 41, 59, 0.4);
        border-radius: 6px;
        padding: 5px 12px !important;
        border: 1px solid rgba(51, 65, 85, 0.6);
    }
    div[data-testid="stMetricLabel"] > p {
        font-size: 0.72rem !important;
        margin-bottom: 0px !important;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stDataFrame"] {
        font-size: 0.80rem !important;
    }
    </style>
""", unsafe_allow_html=True)

CSV_FILE_PATH = "data/sample_bonds.csv"

def save_master_bonds_data(df: pd.DataFrame):
    """Saves updated dataframe to master CSV locations and updates session state."""
    df.to_csv("data/sample_bonds.csv", index=False)
    if os.path.exists("bonds_data.csv"):
        df.to_csv("bonds_data.csv", index=False)
    st.session_state["active_bonds_df"] = df
    st.cache_data.clear()

# 1. Fetch Live Sovereign Benchmark
live_benchmark_yield, benchmark_source = get_live_10y_gsec_yield()

# 2. Master Data Loader with Cache
@st.cache_data(ttl=900)
def load_base_master_data():
    if not os.path.exists(CSV_FILE_PATH):
        if os.path.exists("bonds_data.csv"):
            df = load_sample_csv("bonds_data.csv")
        else:
            st.error(f"Dataset not found at `{CSV_FILE_PATH}`.")
            st.stop()
    else:
        df = load_sample_csv(CSV_FILE_PATH)
    
    # Enrich entity metadata
    psu_keywords = [
        "sidbi", "rec", "pfc", "nabard", "nhai", "irfc", "ntpc", "sbi",
        "government of india", "goi", "g-sec", "gilt", "power grid",
        "state development", "sdl", "t-bill", "treasury bill", "rbi"
    ]
    name_s = df["issuer_name"].astype(str).str.lower() if "issuer_name" in df.columns else pd.Series([""] * len(df))
    sector_s = df["sector"].astype(str).str.lower() if "sector" in df.columns else pd.Series([""] * len(df))
    tax_s = df["tax_status"].astype(str).str.lower() if "tax_status" in df.columns else pd.Series([""] * len(df))
    rating_s = df["rating_current"].astype(str).str.upper() if "rating_current" in df.columns else pd.Series([""] * len(df))

    is_psu = (
        name_s.apply(lambda n: any(k in str(n) for k in psu_keywords)) |
        tax_s.str.contains("54ec", na=False) |
        sector_s.str.contains("psu", na=False) |
        sector_s.str.contains("g-sec", na=False) |
        sector_s.str.contains("sdl", na=False) |
        sector_s.str.contains("t-bill", na=False) |
        sector_s.str.contains("treasury", na=False) |
        rating_s.str.contains("SOVEREIGN", na=False)
    ).to_numpy()
    df["is_govt_psu"] = np.where(is_psu, "Yes", "No")
    return df

# Initialize session state for working master dataset
if "active_bonds_df" not in st.session_state:
    st.session_state["active_bonds_df"] = load_base_master_data()
if "custom_benchmark_yield" not in st.session_state:
    st.session_state["custom_benchmark_yield"] = live_benchmark_yield

raw_data = st.session_state["active_bonds_df"]

# Top Bar Header & Refresh
header_col1, header_col2 = st.columns([4, 1.6])
with header_col1:
    st.title("🇮🇳 Listed NCD, G-Sec & Corporate Bond Screener")
    last_sync_time = st.session_state.get("bond_last_sync", datetime.now().strftime("%d-%b-%Y %H:%M:%S"))
    active_benchmark = st.session_state["custom_benchmark_yield"]
    st.caption(f"Fixed income analytics calibrated against 10Y Indian Sovereign Benchmark: **{active_benchmark:.2f}%** ({benchmark_source}) • 🕒 Last Refreshed: {last_sync_time}")
with header_col2:
    st.write("")
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🔄 Refresh Data", use_container_width=True, help="Flush caches and reload latest quotes"):
            st.cache_data.clear()
            st.session_state["active_bonds_df"] = load_base_master_data()
            st.session_state["custom_benchmark_yield"], _ = get_live_10y_gsec_yield()
            st.session_state["bond_last_sync"] = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
            st.toast("Quotes and Bond Screener data refreshed!", icon="✅")
            st.rerun()
    with c_btn2:
        if st.button("⚡ Live G-Sec", use_container_width=True, help="Fetch latest live 10Y G-Sec yield"):
            live_y, src = get_live_10y_gsec_yield(fallback=BENCHMARK_10Y_GSEC_DEFAULT)
            st.session_state["custom_benchmark_yield"] = live_y
            st.toast(f"Updated benchmark to {live_y:.2f}% from {src}", icon="📈")
            st.rerun()

# Sidebar Filters
st.sidebar.title("🔍 Screener Filters")
tax_slab = st.sidebar.number_input(
    "Your Tax Slab (%)",
    min_value=0.0,
    max_value=45.0,
    value=0.0,
    step=1.0,
    help="Default: 0% (Gross / Pre-Tax Yield). Set to 30% or 39% to evaluate Post-Tax Yield and Pre-Tax Equivalent Yield (PTEY) on Tax-Free bonds."
)

target_yield = st.sidebar.number_input(
    "Target Yield (Pre-Tax YTM %)",
    min_value=0.0,
    max_value=25.0,
    value=7.0,
    step=0.25
)

# Compute Derived Metrics with current tax rate and benchmark yield
data = compute_derived_metrics(raw_data, user_tax_rate=tax_slab, benchmark_yield=st.session_state["custom_benchmark_yield"])

# Multi-Factor Weighted Buy Score (Vectorized 0 - 100)
def compute_vectorized_buy_score(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    score = np.zeros(n, dtype=float)

    # 1. Yield Score (max 30 pts)
    if "post_tax_yield" in df.columns:
        ytm_series = pd.to_numeric(df["post_tax_yield"], errors="coerce").fillna(0.0)
    elif "current_yield_ytm" in df.columns:
        ytm_series = pd.to_numeric(df["current_yield_ytm"], errors="coerce").fillna(0.0)
    else:
        ytm_series = pd.Series(0.0, index=df.index)
    score += np.clip((ytm_series.to_numpy(dtype=float) / 8.0) * 30.0, 0.0, 30.0)

    # 2. Rating Score (max 25 pts)
    rating = df["rating_current"].astype(str).str.upper() if "rating_current" in df.columns else pd.Series([""] * n, index=df.index)
    r_conditions = [
        rating.str.contains("SOVEREIGN", na=False).to_numpy(),
        rating.str.contains("AAA", na=False).to_numpy(),
        rating.str.contains(r"AA\+", na=False).to_numpy(),
        rating.str.contains("AA", na=False).to_numpy(),
        rating.str.contains("A", na=False).to_numpy(),
        rating.str.contains("BBB", na=False).to_numpy(),
    ]
    r_choices = [25.0, 23.0, 19.0, 16.0, 10.0, 7.0, 4.0]
    score += np.select(r_conditions, r_choices, default=2.0)

    # 3. Security & Collateral (max 15 pts)
    sec = df["secured_unsecured"].astype(str).str.lower() if "secured_unsecured" in df.columns else pd.Series([""] * n, index=df.index)
    sec_conditions = [
        sec.str.contains("secured", na=False).to_numpy(),
        sec.str.contains("sovereign", na=False).to_numpy(),
    ]
    sec_choices = [15.0, 15.0]
    score += np.select(sec_conditions, sec_choices, default=5.0)

    # 4. Government / PSU Backing (max 15 pts)
    if "is_govt_psu" in df.columns:
        is_psu = (df["is_govt_psu"].astype(str) == "Yes").to_numpy()
    else:
        is_psu = np.zeros(n, dtype=bool)
    score += np.where(is_psu, 15.0, 5.0)

    # 5. Liquidity Flag (max 10 pts)
    if "liquidity_flag" in df.columns:
        liq = df["liquidity_flag"].astype(str).to_numpy()
    else:
        liq = np.full(n, "Low")
    score += np.where(liq == "High", 10.0, np.where(liq == "Medium", 6.0, 2.0))

    # 6. Tenor Sweet-spot (max 5 pts)
    if "remaining_tenor_years" in df.columns:
        tenor = pd.to_numeric(df["remaining_tenor_years"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    else:
        tenor = np.ones(n, dtype=float)
    score += np.where((tenor >= 1.0) & (tenor <= 5.0), 5.0, np.where((tenor > 5.0) & (tenor <= 10.0), 4.0, 2.0))

    return pd.Series(np.round(score, 1), index=df.index)

data["buy_score"] = compute_vectorized_buy_score(data)

st.sidebar.markdown("---")
search_query = st.sidebar.text_input("Search Ticker, ISIN, or Issuer", "", help="Search by NSE/BSE bond ticker symbol, ISIN, or company name")
sectors = st.sidebar.multiselect("Sector", options=sorted(data['sector'].dropna().unique()))
sec_filter = st.sidebar.multiselect("Security Type", options=sorted(data['secured_unsecured'].dropna().unique()) if 'secured_unsecured' in data.columns else [])
pay_freq = st.sidebar.multiselect("Payout Frequency", options=sorted(data['payment_frequency'].dropna().unique()) if 'payment_frequency' in data.columns else [])
exchanges = st.sidebar.multiselect("Exchange", options=sorted(data['exchange'].dropna().unique()))
rating_current = st.sidebar.multiselect("Rating", options=sorted(data['rating_current'].dropna().unique()))
psu_filter = st.sidebar.multiselect("Govt / PSU Entity", options=["Yes", "No"])
tax_status = st.sidebar.multiselect("Tax Status", options=sorted(data['tax_status'].dropna().unique()))

st.sidebar.markdown("---")
min_ytm = float(data['current_yield_ytm'].min())
max_ytm = float(data['current_yield_ytm'].max())
ytm_range = st.sidebar.slider("Current YTM (%)", min_ytm, max_ytm, (min_ytm, max_ytm), step=0.1)

max_tenor = float(data['remaining_tenor_years'].max())
tenor_range = st.sidebar.slider("Tenor (Years)", 0.0, max_tenor, (0.0, max_tenor), step=0.5)

st.sidebar.markdown("---")
ignore_sub_12m = st.sidebar.checkbox("Ignore bonds maturing in < 12 months", value=True, help="Institutional mandate: Excludes short-dated securities with tenor < 1 year")
min_adv = st.sidebar.slider("Min Avg Daily Volume (₹ Cr)", 0.0, 50.0, 0.0, 1.0, help="Filter by minimum exchange average daily turnover")

# Boolean Mask Filtering (Zero copy overhead)
mask = np.ones(len(data), dtype=bool)
if search_query:
    q = search_query.lower()
    t_match = data["ticker"].astype(str).str.lower().str.contains(q, na=False) if "ticker" in data.columns else pd.Series(False, index=data.index)
    i_match = data["issuer_name"].astype(str).str.lower().str.contains(q, na=False) if "issuer_name" in data.columns else pd.Series(False, index=data.index)
    isin_match = data["isin"].astype(str).str.lower().str.contains(q, na=False) if "isin" in data.columns else pd.Series(False, index=data.index)
    mask &= (t_match | i_match | isin_match).to_numpy(dtype=bool)
if sectors:
    mask &= data['sector'].isin(sectors).to_numpy(dtype=bool)
if sec_filter:
    mask &= data['secured_unsecured'].isin(sec_filter).to_numpy(dtype=bool)
if pay_freq:
    mask &= data['payment_frequency'].isin(pay_freq).to_numpy(dtype=bool)
if exchanges:
    mask &= data['exchange'].isin(exchanges).to_numpy(dtype=bool)
if rating_current:
    mask &= data['rating_current'].isin(rating_current).to_numpy(dtype=bool)
if psu_filter:
    mask &= data['is_govt_psu'].isin(psu_filter).to_numpy(dtype=bool)
if tax_status:
    mask &= data['tax_status'].isin(tax_status).to_numpy(dtype=bool)

mask &= ((data['current_yield_ytm'] >= ytm_range[0]) & (data['current_yield_ytm'] <= ytm_range[1])).to_numpy(dtype=bool)
mask &= ((data['remaining_tenor_years'] >= tenor_range[0]) & (data['remaining_tenor_years'] <= tenor_range[1])).to_numpy(dtype=bool)
if ignore_sub_12m:
    mask &= (data['remaining_tenor_years'] >= 1.0).to_numpy(dtype=bool)
if min_adv > 0.0:
    mask &= (data['daily_value_inr_cr'] >= min_adv).to_numpy(dtype=bool)

filtered = data[mask].copy()

# App Navigation Tabs
tab_screener, tab_yield_curve, tab_stress_test, tab_manage = st.tabs([
    "📊 Bond Screener & Analytics", 
    "📈 Yield Curve & Spread Matrix",
    "⚡ RBI Rate Shock Stress-Tester",
    "📥 Add Bonds & Market Operations"
])

with tab_screener:
    # Summary Metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Screened Bonds", f"{len(filtered)} / {len(data)}", delta="Tenor >= 12M" if ignore_sub_12m else "All Tenors")
    m2.metric("Avg Gross YTM", f"{round(filtered['current_yield_ytm'].mean(), 2)}%" if not filtered.empty else "0%")
    if tax_slab > 0.0:
        m3.metric("Avg PTEY (Tax-Equiv)", f"{round(filtered['pre_tax_equiv_yield'].mean(), 2)}%" if not filtered.empty else "0%", help="Pre-Tax Equivalent Yield for direct comparison with FDs")
    else:
        m3.metric("Avg Daily Volume", f"₹ {round(filtered['daily_value_inr_cr'].mean(), 1)} Cr" if not filtered.empty else "₹ 0 Cr")
    m4.metric("Avg G-Sec Spread", f"{int(filtered['credit_spread_bps'].mean()):+d} bps" if not filtered.empty else "0 bps")
    m5.metric("Avg Mod Duration", f"{round(filtered['modified_duration'].mean(), 2)} yrs" if not filtered.empty else "0")
    top_score_bond = filtered.sort_values(by='buy_score', ascending=False).iloc[0]['issuer_name'][:18] + ".." if not filtered.empty else "-"
    m6.metric("Top Buy Score", top_score_bond)

    st.markdown("---")

    display_cols = [
        "ticker", "issuer_name", "isin",
        "current_yield_ytm", "secured_unsecured", "payment_frequency",
        "coupon_rate", "rating_current", "remaining_tenor_display",
        "last_traded_price", "accrued_interest", "dirty_price",
        "credit_spread_bps", "post_tax_yield", "pre_tax_equiv_yield",
        "issue_date", "maturity_date",
        "volume_indicator", "avg_daily_volume_cr",
        "modified_duration", "convexity", "tax_status", "is_govt_psu", "liquidity_flag", "buy_score"
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].sort_values(by="buy_score", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", help="NSE/BSE Trading Symbol", width="small"),
            "issuer_name": st.column_config.TextColumn("Bond Title / Issue Name", width="large"),
            "isin": st.column_config.TextColumn("ISIN", help="12-digit International Securities Identification Number", width="medium"),
            "current_yield_ytm": st.column_config.NumberColumn("Gross YTM (%)", format="%.2f%%", help="Pre-tax / gross yield to maturity"),
            "secured_unsecured": st.column_config.TextColumn("Security", width="small", help="Whether Secured, Unsecured, or Sovereign Guarantee"),
            "payment_frequency": st.column_config.TextColumn("Payout Freq", width="small", help="Interest payout frequency (Annual, Semi-Annual, Monthly, etc.)"),
            "coupon_rate": st.column_config.NumberColumn("Coupon (%)", format="%.2f%%"),
            "rating_current": st.column_config.TextColumn("Rating", width="small"),
            "remaining_tenor_display": st.column_config.TextColumn("Tenor Left", width="small"),
            "last_traded_price": st.column_config.NumberColumn("Clean Price (₹)", format="₹ %.2f", help="Exchange Traded Clean Price"),
            "accrued_interest": st.column_config.NumberColumn("Accrued Int (₹)", format="₹ %.2f", help="Accrued interest since last coupon date"),
            "dirty_price": st.column_config.NumberColumn("Dirty Price (₹)", format="₹ %.2f", help="Actual settlement price (Clean Price + Accrued Interest)"),
            "credit_spread_bps": st.column_config.NumberColumn("Spread (bps)", help=f"Yield spread over 10Y Sovereign G-Sec ({active_benchmark:.2f}%)"),
            "post_tax_yield": st.column_config.NumberColumn("Post-Tax (%)", format="%.2f%%", help="Yield realized after deducting user tax slab"),
            "pre_tax_equiv_yield": st.column_config.NumberColumn("PTEY (%)", format="%.2f%%", help="Pre-Tax Equivalent Yield (apple-to-apple comparison with taxable FDs)"),
            "issue_date": st.column_config.TextColumn("Issue Date", width="small"),
            "maturity_date": st.column_config.TextColumn("Maturity Date", width="small"),
            "volume_indicator": st.column_config.TextColumn("Volume Indication", width="medium"),
            "avg_daily_volume_cr": st.column_config.NumberColumn("ADV (₹ Cr)", format="₹ %.2f Cr", width="small"),
            "modified_duration": st.column_config.NumberColumn("Mod Dur (Yrs)", format="%.2f"),
            "convexity": st.column_config.NumberColumn("Convexity", format="%.2f"),
            "tax_status": st.column_config.TextColumn("Tax Status", width="small"),
            "is_govt_psu": st.column_config.TextColumn("Govt/PSU", width="small"),
            "liquidity_flag": st.column_config.TextColumn("Liquidity", width="small"),
            "buy_score": st.column_config.NumberColumn("Buy Score", format="%.1f")
        }
    )

    st.download_button(
        "📥 Export Screened Bonds (CSV)",
        data=filtered[display_cols].to_csv(index=False).encode('utf-8'),
        file_name=f"bonds_screener_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with tab_yield_curve:
    st.subheader("📈 India Fixed-Income Yield Curve & Credit Spread Matrix")
    c_y1, c_y2 = st.columns([3, 2])

    with c_y1:
        fig_curve = px.scatter(
            filtered,
            x="remaining_tenor_years",
            y="current_yield_ytm",
            color="risk_bucket",
            size="buy_score",
            hover_name="issuer_name",
            hover_data={"ticker": True, "dirty_price": ":.2f", "modified_duration": ":.2f", "rating_current": True},
            labels={"remaining_tenor_years": "Tenor to Maturity (Years)", "current_yield_ytm": "Yield to Maturity (%)"},
            title="Yield vs Maturity Curve by Risk Bucket"
        )
        fig_curve.add_hline(
            y=active_benchmark,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"10Y G-Sec Benchmark ({active_benchmark:.2f}%)"
        )
        fig_curve.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_curve, use_container_width=True)

    with c_y2:
        spread_by_sector = filtered.groupby("sector")["credit_spread_bps"].mean().reset_index().sort_values(by="credit_spread_bps")
        fig_bar = px.bar(
            spread_by_sector,
            x="credit_spread_bps",
            y="sector",
            orientation="h",
            color="credit_spread_bps",
            color_continuous_scale="Viridis",
            title="Average Credit Spread over G-Sec by Sector (bps)"
        )
        fig_bar.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Rating Tier Spread Analysis
    st.markdown("---")
    st.subheader("📊 Yield Spread Distribution Across Rating Tiers")
    rating_spread = filtered.groupby("rating_current")["credit_spread_bps"].agg(["mean", "min", "max", "count"]).reset_index()
    rating_spread.columns = ["Credit Rating", "Avg Spread (bps)", "Min Spread (bps)", "Max Spread (bps)", "Bond Count"]
    st.dataframe(rating_spread.sort_values(by="Avg Spread (bps)"), use_container_width=True, hide_index=True)

with tab_stress_test:
    st.subheader("⚡ RBI Policy Rate Shock Simulator (Duration & Convexity Engine)")
    st.caption("Evaluates bond capital appreciation or depreciation under monetary policy shifts using Taylor-expansion duration/convexity approximation:")
    st.latex(r"\frac{\Delta P}{P} \approx -D_{\text{mod}} \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2")

    shock_bps = st.slider(
        "Simulated RBI Repo Rate Change (bps)",
        -200, 200, -50, 25,
        help="Negative = Rate Cuts (Bond Prices Rise); Positive = Rate Hikes (Bond Prices Drop)"
    )
    shock_y = shock_bps / 10000.0  # e.g., for +100 bps, shock_y = +0.01

    stress_df = filtered.copy()
    # Mathematically accurate Taylor expansion percentage change
    stress_df["price_change_pct"] = (
        (-stress_df["modified_duration"] * shock_y + 0.5 * stress_df["convexity"] * (shock_y ** 2)) * 100.0
    ).round(2)
    stress_df["duration_impact_pct"] = (-stress_df["modified_duration"] * shock_y * 100.0).round(2)
    stress_df["convexity_cushion_pct"] = (0.5 * stress_df["convexity"] * (shock_y ** 2) * 100.0).round(2)
    stress_df["new_ytm"] = (stress_df["current_yield_ytm"] + (shock_bps / 100.0)).round(2)

    st_col1, st_col2 = st.columns([3, 2])
    with st_col1:
        fig_stress = px.bar(
            stress_df.sort_values(by="price_change_pct", ascending=False).head(15),
            x="price_change_pct",
            y="issuer_name",
            orientation="h",
            color="price_change_pct",
            color_continuous_scale="RdYlGn",
            title=f"Capital Gain / Loss (%) under a {shock_bps:+d} bps Rate Shift"
        )
        fig_stress.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_stress, use_container_width=True)

    with st_col2:
        stress_cols = ["ticker", "issuer_name", "remaining_tenor_years", "modified_duration", "duration_impact_pct", "convexity_cushion_pct", "price_change_pct", "new_ytm"]
        stress_cols = [c for c in stress_cols if c in stress_df.columns]
        st.dataframe(
            stress_df[stress_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker"),
                "issuer_name": st.column_config.TextColumn("Bond"),
                "remaining_tenor_years": st.column_config.NumberColumn("Tenor", format="%.1f"),
                "modified_duration": st.column_config.NumberColumn("Mod Dur", format="%.2f"),
                "duration_impact_pct": st.column_config.NumberColumn("Duration Chg", format="%+.2f%%"),
                "convexity_cushion_pct": st.column_config.NumberColumn("Convexity Cushion", format="+%.2f%%"),
                "price_change_pct": st.column_config.NumberColumn("Net Capital Chg (%)", format="%+.2f%%"),
                "new_ytm": st.column_config.NumberColumn("New YTM (%)", format="%.2f%%")
            }
        )

with tab_manage:
    st.subheader("📥 Add Bonds & Master Database Management")
    st.caption("Add new NCDs, G-Secs, SDLs, or T-Bills (>1 year tenor), upload bulk CSVs, or ingest official daily exchange Bhavcopies.")

    subtab_add, subtab_bulk, subtab_bhavcopy, subtab_directory = st.tabs([
        "➕ Add New Security (NCD / G-Sec / SDL / T-Bill)",
        "📤 Bulk CSV Import & Template",
        "🔄 Daily Bhavcopy & Benchmark",
        "📂 Master Security Directory"
    ])

    with subtab_add:
        st.markdown("#### ➕ Add an Individual Debt Security")
        st.info("💡 **Tenor Policy**: Only securities with **more than 1 year (> 365 days) remaining to maturity** can be added. The screener will automatically compute dynamic Accrued Interest, Dirty Price, YTM, Macaulay/Modified Duration, Convexity, and Buy Score.")

        with st.form("add_bond_single_form", clear_on_submit=False):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                inst_type = st.selectbox(
                    "Instrument Classification",
                    options=["Corporate NCD", "Government of India G-Sec", "State Development Loan (SDL)", "Treasury Bill (T-Bill)", "InvIT / Infra Bond", "Tax-Free PSU Bond"],
                    help="Categorization of the debt instrument"
                )
                ticker = st.text_input("Ticker Symbol *", placeholder="e.g. 1125MML27, 690GOI65, 771UP46-SG", help="NSE / BSE trading symbol").strip().upper()
                exchange = st.selectbox("Exchange", ["NSE", "BSE", "Both"])
                face_value = st.number_input("Face Value (₹)", min_value=10.0, max_value=1000000.0, value=100.0 if "G-Sec" in inst_type or "SDL" in inst_type or "T-Bill" in inst_type else 1000.0, step=100.0)

            with f_col2:
                isin = st.text_input("ISIN (12 Characters) *", placeholder="e.g. INE05F407EC8 or IN0020250018", help="International Securities Identification Number").strip().upper()
                issuer_name = st.text_input("Issuer / Bond Title *", placeholder="e.g. Muthoot Mercantile Ltd 11.25% NCD 2027")
                
                sector = st.selectbox(
                    "Sector",
                    options=["NBFC / Financials", "G-Sec / Sovereign", "State Development Loan (SDL)", "Treasury / Sovereign", "PSU / Railways", "PSU / Power NBFC", "PSU / Infra NBFC", "Banking", "Infrastructure / InvIT", "Housing Finance", "Fintech / Retail", "Corporate / Conglomerate", "Other"],
                    index=1 if "G-Sec" in inst_type else (2 if "SDL" in inst_type else (3 if "T-Bill" in inst_type else 0))
                )
                secured_unsecured = st.selectbox(
                    "Secured or Unsecured *",
                    options=["Secured", "Unsecured", "Sovereign Guarantee"],
                    index=2 if "G-Sec" in inst_type or "SDL" in inst_type or "T-Bill" in inst_type else 0,
                    help="Prominently tracked in screener"
                )

            with f_col3:
                coupon_rate = st.number_input(
                    "Coupon Rate (%) *",
                    min_value=0.0,
                    max_value=25.0,
                    value=0.0 if "T-Bill" in inst_type else 8.50,
                    step=0.05,
                    help="Enter 0.0 for T-Bills or Zero Coupon Bonds"
                )
                payment_frequency = st.selectbox(
                    "Payment Frequency *",
                    options=["Annual", "Semi-Annual", "Quarterly", "Monthly", "Cumulative", "At Maturity (Zero Coupon / T-Bill)"],
                    index=5 if "T-Bill" in inst_type or coupon_rate == 0.0 else (1 if "G-Sec" in inst_type or "SDL" in inst_type else 0),
                    help="Interest payout frequency"
                )
                clean_price = st.number_input("Last Traded Clean Price (₹) *", min_value=1.0, max_value=500000.0, value=float(face_value), step=0.5)
                stated_ytm = st.number_input("Stated YTM (%) [Optional]", min_value=0.0, max_value=30.0, value=0.0, step=0.05, help="Leave 0.0 to auto-calculate exact YTM from clean price and tenor")

            st.markdown("##### Maturity, Dates & Credit Rating")
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            with d_col1:
                issue_date = st.date_input("Issue Date", value=date.today() - timedelta(days=365))
            with d_col2:
                maturity_date = st.date_input("Maturity Date *", value=date.today() + timedelta(days=365 * 3))
            with d_col3:
                rating_default = 0 if "G-Sec" in inst_type or "SDL" in inst_type or "T-Bill" in inst_type else 1
                rating_current = st.selectbox("Rating", ["SOVEREIGN", "AAA", "AA+", "AA", "AA-", "A+", "A", "BBB+", "BBB", "Unrated"], index=rating_default)
                rating_agency = st.selectbox("Rating Agency", ["RBI/Govt", "CRISIL", "ICRA", "CARE", "India Ratings", "BWR", "Other"], index=0 if rating_current == "SOVEREIGN" else 1)
            with d_col4:
                tax_status = st.selectbox("Tax Status", ["Taxable", "Tax-free", "54EC"], index=1 if "Tax-Free" in inst_type else 0)
                seniority = st.selectbox("Seniority", ["Senior", "Subordinated / Tier-II", "Perpetual"])

            st.markdown("##### Liquidity & Volume Estimates")
            l_col1, l_col2, l_col3 = st.columns(3)
            with l_col1:
                daily_volume_cr = st.number_input("Est. Avg Daily Volume (₹ Cr)", min_value=0.0, value=25.0 if "G-Sec" in inst_type else 2.0, step=0.5)
            with l_col2:
                daily_trades = st.number_input("Est. Number of Trades", min_value=0, value=150 if "G-Sec" in inst_type else 25, step=5)
            with l_col3:
                issue_size_cr = st.number_input("Issue Size (₹ Cr)", min_value=1.0, value=5000.0 if "G-Sec" in inst_type else 500.0, step=50.0)

            submit_bond = st.form_submit_button("💾 Validate & Add Security to Screener", use_container_width=True)

        if submit_bond:
            if not ticker or not isin:
                st.error("❌ Both **Ticker** and **ISIN** are required fields.")
            elif len(isin) != 12:
                st.error(f"❌ Invalid ISIN length: '{isin}' is {len(isin)} characters. Standard Indian ISINs must be exactly 12 characters (e.g., INE05F407EC8 or IN0020250018).")
            elif not issuer_name:
                st.error("❌ **Issuer / Bond Title** cannot be empty.")
            else:
                today_d = date.today()
                days_left = (maturity_date - today_d).days
                tenor_years = round(days_left / 365.25, 2)

                if days_left <= 365:
                    st.error(
                        f"❌ **Maturity Policy Violation**: Security matures on {maturity_date.strftime('%d-%b-%Y')} "
                        f"({days_left} days left = {tenor_years:.2f} years). "
                        f"Only securities with **more than 1 year (> 365 days) remaining to mature** can be added. "
                        f"Please choose a maturity date on or after {(today_d + timedelta(days=366)).strftime('%d-%b-%Y')}."
                    )
                else:
                    accrued = calculate_accrued_interest(face_value, coupon_rate, payment_frequency, issue_date, maturity_date, today_d)
                    dirty_price = round(clean_price + accrued, 2)
                    
                    if stated_ytm > 0:
                        solved_ytm = float(stated_ytm)
                    else:
                        solved_ytm = calculate_ytm_from_price(dirty_price, face_value, coupon_rate, tenor_years, payment_frequency)

                    is_t_bill = "t-bill" in inst_type.lower() or "t-bill" in sector.lower() or coupon_rate == 0.0
                    c_type = "Zero" if is_t_bill else "Fixed"

                    new_row = {
                        "ticker": ticker.strip().upper(),
                        "isin": isin.strip().upper(),
                        "issuer_name": issuer_name.strip(),
                        "sector": sector,
                        "exchange": exchange,
                        "listing_status": "Listed",
                        "face_value": float(face_value),
                        "coupon_rate": float(coupon_rate),
                        "coupon_type": c_type,
                        "current_yield_ytm": float(solved_ytm),
                        "last_traded_price": float(clean_price),
                        "accrued_interest": float(accrued),
                        "issue_date": issue_date.strftime("%Y-%m-%d"),
                        "maturity_date": maturity_date.strftime("%Y-%m-%d"),
                        "payment_frequency": payment_frequency,
                        "repayment_mode": "Bullet",
                        "call_put_option": "No",
                        "premature_withdrawal_allowed": "No",
                        "rating_current": rating_current,
                        "rating_agency": rating_agency,
                        "rating_outlook": "Stable",
                        "rating_date": today_d.strftime("%Y-%m-%d"),
                        "secured_unsecured": secured_unsecured,
                        "seniority": seniority,
                        "security_cover_ratio": 1.0 if "sovereign" in secured_unsecured.lower() else (1.2 if "secured" in secured_unsecured.lower() else 0.0),
                        "daily_volume_units": float(daily_volume_cr * 10000000.0 / clean_price) if clean_price > 0 else 1000.0,
                        "daily_value_inr_cr": float(daily_volume_cr),
                        "num_trades": int(daily_trades),
                        "issue_size_inr_cr": float(issue_size_cr),
                        "outstanding_inr_cr": float(issue_size_cr),
                        "tax_status": tax_status,
                        "tds_applicable": "No" if "sovereign" in secured_unsecured.lower() or tax_status == "Tax-free" else "Yes",
                        "min_investment_units": 1 if face_value >= 1000 else 10,
                        "lot_size": 1 if face_value >= 1000 else 10,
                        "last_traded_date": today_d.strftime("%Y-%m-%d")
                    }

                    curr_df = st.session_state["active_bonds_df"].copy()
                    if "isin" in curr_df.columns and (curr_df["isin"] == isin).any():
                        idx = curr_df[curr_df["isin"] == isin].index[0]
                        for k, v in new_row.items():
                            curr_df.at[idx, k] = v
                        action_msg = "Updated existing security"
                    else:
                        new_df = pd.DataFrame([new_row])
                        curr_df = pd.concat([curr_df, new_df], ignore_index=True)
                        action_msg = "Added new security"

                    save_master_bonds_data(curr_df)
                    st.success(
                        f"✅ **{action_msg} Successfully!**\n\n"
                        f"• **Ticker**: `{ticker}` | **ISIN**: `{isin}`\n"
                        f"• **Issuer**: {issuer_name}\n"
                        f"• **Gross YTM**: **{solved_ytm:.2f}%**\n"
                        f"• **Security**: **{secured_unsecured}**\n"
                        f"• **Payment Frequency**: **{payment_frequency}**\n"
                        f"• **Remaining Tenor**: **{tenor_years:.2f} Years** ({days_left} days)\n"
                        f"• **Dirty Price**: ₹ {dirty_price:.2f} (Clean: ₹ {clean_price:.2f} + Accrued: ₹ {accrued:.2f})"
                    )
                    st.rerun()

    with subtab_bulk:
        st.markdown("#### 📤 Bulk Add Securities via CSV")
        st.caption("Upload a CSV file containing multiple NCDs, G-Secs, SDLs, or T-Bills. Securities with tenor <= 1 year will be automatically flagged/skipped.")
        
        template_df = pd.DataFrame([{
            "ticker": "EXAMPLE_NCD28",
            "isin": "INE123A07010",
            "issuer_name": "Example Finance Ltd 9.00% NCD 2028",
            "sector": "NBFC / Financials",
            "exchange": "NSE",
            "listing_status": "Listed",
            "face_value": 1000.0,
            "coupon_rate": 9.0,
            "coupon_type": "Fixed",
            "current_yield_ytm": 8.95,
            "last_traded_price": 1002.5,
            "accrued_interest": 12.0,
            "issue_date": "2024-01-01",
            "maturity_date": (date.today() + timedelta(days=365 * 3)).strftime("%Y-%m-%d"),
            "payment_frequency": "Annual",
            "repayment_mode": "Bullet",
            "call_put_option": "No",
            "premature_withdrawal_allowed": "No",
            "rating_current": "AA+",
            "rating_agency": "CRISIL",
            "rating_outlook": "Stable",
            "rating_date": "2024-01-01",
            "secured_unsecured": "Secured",
            "seniority": "Senior",
            "security_cover_ratio": 1.2,
            "daily_volume_units": 5000,
            "daily_value_inr_cr": 5.0,
            "num_trades": 50,
            "issue_size_inr_cr": 500.0,
            "outstanding_inr_cr": 500.0,
            "tax_status": "Taxable",
            "tds_applicable": "Yes",
            "min_investment_units": 1,
            "lot_size": 1,
            "last_traded_date": date.today().strftime("%Y-%m-%d")
        }])
        st.download_button(
            "📥 Download CSV Template for Bond Additions",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="bonds_import_template.csv",
            mime="text/csv"
        )
        
        uploaded_csv = st.file_uploader("Upload CSV with new securities", type=["csv"], key="bulk_bond_uploader")
        if uploaded_csv is not None:
            try:
                bulk_df = pd.read_csv(uploaded_csv)
                if "isin" not in bulk_df.columns:
                    st.error("Uploaded CSV must have an `isin` column.")
                else:
                    today_ts = pd.to_datetime(date.today())
                    mat_ts = pd.to_datetime(bulk_df["maturity_date"], errors="coerce")
                    days_remaining = (mat_ts - today_ts).dt.days

                    valid_mask = days_remaining > 365
                    dropped_count = int((~valid_mask).sum())
                    valid_df = bulk_df[valid_mask].copy()

                    if dropped_count > 0:
                        st.warning(f"⚠️ Skipped {dropped_count} security/securities with tenor <= 1 year (mandate requires > 1 year to mature).")

                    if not valid_df.empty:
                        curr = st.session_state["active_bonds_df"].copy()
                        combined = pd.concat([curr[~curr["isin"].isin(valid_df["isin"])], valid_df], ignore_index=True)
                        save_master_bonds_data(combined)
                        st.success(f"✅ Successfully ingested {len(valid_df)} securities with > 1 year tenor into master database!")
                        if st.button("Apply Bulk Additions and Rerun"):
                            st.rerun()
                    else:
                        st.error("No valid securities with tenor > 1 year found in the uploaded file.")
            except Exception as ex:
                st.error(f"Error parsing bulk CSV: {ex}")

    with subtab_bhavcopy:
        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            st.markdown("""
            **Update Market Quotes via Official Exchange Bhavcopy**:
            Drop today's official daily debt Bhavcopy from **NSE** (`sec_bhavdata_full_DDMMYYYY.csv`) or **BSE** (`DEBT_YYYYMMDD.zip` / CSV).
            The screener will match ISINs, update Clean LTP, daily trade volumes, recalculate dynamic accrued interest, dirty prices, and re-solve YTM.
            """)
            uploaded_bhav = st.file_uploader("Upload NSE / BSE Debt Bhavcopy (CSV or ZIP)", type=["csv", "zip"])
            if uploaded_bhav is not None:
                updated_master, msg = ingest_bhavcopy(uploaded_bhav, st.session_state["active_bonds_df"])
                save_master_bonds_data(updated_master)
                st.session_state["bond_last_sync"] = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
                st.success(msg)
                if st.button("Apply and Rerun Screener"):
                    st.rerun()

        with col_u2:
            st.markdown("**Sovereign Benchmark Override**")
            custom_bm = st.number_input(
                "10Y Benchmark G-Sec Yield (%)",
                min_value=4.0,
                max_value=12.0,
                value=float(st.session_state["custom_benchmark_yield"]),
                step=0.01,
                help="Override or test different benchmark levels"
            )
            if custom_bm != st.session_state["custom_benchmark_yield"]:
                st.session_state["custom_benchmark_yield"] = custom_bm
                st.toast(f"Benchmark updated to {custom_bm:.2f}%", icon="🎯")
                st.rerun()

    with subtab_directory:
        st.markdown("#### 📂 Master Active Securities Directory")
        st.caption(f"Total securities active in screener: **{len(st.session_state['active_bonds_df'])}**")
        
        del_col1, del_col2 = st.columns([3, 1])
        with del_col1:
            all_bonds_list = [f"{r['ticker']} - {r['isin']} - {r['issuer_name']}" for _, r in st.session_state["active_bonds_df"].iterrows()]
            selected_bond_to_del = st.selectbox("Select Security to Remove (Optional)", options=["None"] + all_bonds_list)
        with del_col2:
            st.write("")
            st.write("")
            if selected_bond_to_del != "None":
                del_isin = selected_bond_to_del.split(" - ")[1]
                if st.button("🗑️ Remove Security", type="secondary"):
                    curr = st.session_state["active_bonds_df"]
                    curr = curr[curr["isin"] != del_isin].reset_index(drop=True)
                    save_master_bonds_data(curr)
                    st.success(f"Removed {selected_bond_to_del} from database.")
                    st.rerun()

        st.dataframe(st.session_state["active_bonds_df"], use_container_width=True)
