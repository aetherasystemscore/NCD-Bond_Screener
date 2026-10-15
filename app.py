import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from datetime import datetime
from data_loaders import (
    load_sample_csv,
    compute_derived_metrics,
    get_live_10y_gsec_yield,
    ingest_bhavcopy,
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

# 1. Fetch Live Sovereign Benchmark
live_benchmark_yield, benchmark_source = get_live_10y_gsec_yield()

# 2. Master Data Loader with Cache
@st.cache_data(ttl=900)
def load_base_master_data():
    if not os.path.exists(CSV_FILE_PATH):
        st.error(f"Dataset not found at `{CSV_FILE_PATH}`.")
        st.stop()
    df = load_sample_csv(CSV_FILE_PATH)
    
    # Enrich entity metadata
    psu_keywords = ["sidbi", "rec", "pfc", "nabard", "nhai", "irfc", "ntpc", "sbi", "government of india", "goi", "g-sec", "gilt", "power grid"]
    name_s = df.get("issuer_name", "").astype(str).str.lower()
    sector_s = df.get("sector", "").astype(str).str.lower()
    tax_s = df.get("tax_status", "").astype(str).str.lower()
    rating_s = df.get("rating_current", "").astype(str).str.upper()

    is_psu = (
        name_s.apply(lambda n: any(k in n for k in psu_keywords)) |
        tax_s.str.contains("54ec") |
        sector_s.str.contains("psu") |
        sector_s.str.contains("g-sec") |
        rating_s.str.contains("SOVEREIGN")
    )
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
    score = np.zeros(len(df), dtype=float)

    # 1. Yield Score (max 30 pts)
    post_tax = df.get("post_tax_yield", df["current_yield_ytm"]).values
    score += np.clip((post_tax / 8.0) * 30.0, 0.0, 30.0)

    # 2. Rating Score (max 25 pts)
    rating = df.get("rating_current", "").astype(str).str.upper().values
    r_score = np.where(
        np.char.find(rating.astype(str), "SOVEREIGN") >= 0, 25.0,
        np.where(np.char.find(rating.astype(str), "AAA") >= 0, 23.0,
                 np.where(np.char.find(rating.astype(str), "AA+") >= 0, 19.0,
                          np.where(np.char.find(rating.astype(str), "AA") >= 0, 16.0,
                                   np.where(np.char.find(rating.astype(str), "A+") >= 0, 10.0,
                                            np.where(np.char.find(rating.astype(str), "A") >= 0, 7.0, 2.0))))))
    score += r_score

    # 3. Security & Collateral (max 15 pts)
    sec = df.get("secured_unsecured", "").astype(str).str.capitalize().values
    score += np.where(np.char.find(sec.astype(str), "Secured") >= 0, 15.0,
                      np.where(np.char.find(sec.astype(str), "Sovereign") >= 0, 15.0, 5.0))

    # 4. Government / PSU Backing (max 15 pts)
    is_psu = (df.get("is_govt_psu", "No") == "Yes").values
    score += np.where(is_psu, 15.0, 5.0)

    # 5. Liquidity Flag (max 10 pts)
    liq = df.get("liquidity_flag", "Low").values
    score += np.where(liq == "High", 10.0, np.where(liq == "Medium", 6.0, 2.0))

    # 6. Tenor Sweet-spot (max 5 pts)
    tenor = df.get("remaining_tenor_years", 1.0).values
    score += np.where((tenor >= 1.0) & (tenor <= 5.0), 5.0, np.where((tenor > 5.0) & (tenor <= 10.0), 4.0, 2.0))

    return np.round(score, 1)

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
    t_match = data["ticker"].astype(str).str.lower().str.contains(q) if "ticker" in data.columns else False
    i_match = data["issuer_name"].astype(str).str.lower().str.contains(q)
    isin_match = data["isin"].astype(str).str.lower().str.contains(q)
    mask &= (t_match | i_match | isin_match)
if sectors:
    mask &= data['sector'].isin(sectors)
if sec_filter:
    mask &= data['secured_unsecured'].isin(sec_filter)
if pay_freq:
    mask &= data['payment_frequency'].isin(pay_freq)
if exchanges:
    mask &= data['exchange'].isin(exchanges)
if rating_current:
    mask &= data['rating_current'].isin(rating_current)
if psu_filter:
    mask &= data['is_govt_psu'].isin(psu_filter)
if tax_status:
    mask &= data['tax_status'].isin(tax_status)

mask &= (data['current_yield_ytm'] >= ytm_range[0]) & (data['current_yield_ytm'] <= ytm_range[1])
mask &= (data['remaining_tenor_years'] >= tenor_range[0]) & (data['remaining_tenor_years'] <= tenor_range[1])
if ignore_sub_12m:
    mask &= (data['remaining_tenor_years'] >= 1.0)
if min_adv > 0.0:
    mask &= (data['daily_value_inr_cr'] >= min_adv)

filtered = data[mask].copy()

# App Navigation Tabs
tab_screener, tab_yield_curve, tab_stress_test, tab_manage = st.tabs([
    "📊 Bond Screener & Analytics", 
    "📈 Yield Curve & Spread Matrix",
    "⚡ RBI Rate Shock Stress-Tester",
    "📥 Live Bhavcopy & Management"
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
        "ticker", "issuer_name", "isin", "issue_date", "maturity_date", "remaining_tenor_display",
        "payment_frequency", "rating_current", "secured_unsecured", "coupon_rate",
        "last_traded_price", "accrued_interest", "dirty_price",
        "current_yield_ytm", "credit_spread_bps", "post_tax_yield", "pre_tax_equiv_yield",
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
            "issue_date": st.column_config.TextColumn("Issue Date", width="small"),
            "maturity_date": st.column_config.TextColumn("Maturity Date", width="small"),
            "remaining_tenor_display": st.column_config.TextColumn("Tenor Left", width="small"),
            "payment_frequency": st.column_config.TextColumn("Payout Freq", width="small"),
            "rating_current": st.column_config.TextColumn("Rating", width="small"),
            "secured_unsecured": st.column_config.TextColumn("Security", width="small"),
            "coupon_rate": st.column_config.NumberColumn("Coupon (%)", format="%.2f%%"),
            "last_traded_price": st.column_config.NumberColumn("Clean Price (₹)", format="₹ %.2f", help="Exchange Traded Clean Price"),
            "accrued_interest": st.column_config.NumberColumn("Accrued Int (₹)", format="₹ %.2f", help="Accrued interest since last coupon date"),
            "dirty_price": st.column_config.NumberColumn("Dirty Price (₹)", format="₹ %.2f", help="Actual settlement price (Clean Price + Accrued Interest)"),
            "current_yield_ytm": st.column_config.NumberColumn("Gross YTM (%)", format="%.2f%%", help="Pre-tax / gross yield to maturity"),
            "credit_spread_bps": st.column_config.NumberColumn("Spread (bps)", help=f"Yield spread over 10Y Sovereign G-Sec ({active_benchmark:.2f}%)"),
            "post_tax_yield": st.column_config.NumberColumn("Post-Tax (%)", format="%.2f%%", help="Yield realized after deducting user tax slab"),
            "pre_tax_equiv_yield": st.column_config.NumberColumn("PTEY (%)", format="%.2f%%", help="Pre-Tax Equivalent Yield (apple-to-apple comparison with taxable FDs)"),
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
    st.subheader("📥 Live Bhavcopy Ingestion & Master Management")
    
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
            st.session_state["active_bonds_df"] = updated_master
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

    st.markdown("---")
    st.subheader("📂 Active Bond Master Database")
    st.dataframe(st.session_state["active_bonds_df"], use_container_width=True)
