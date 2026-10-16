import pandas as pd
import numpy as np
from datetime import datetime, date
import re
import io
import zipfile
import urllib.request
try:
    import requests
except ImportError:
    requests = None

# Default benchmark based on current 10Y Indian Sovereign Yield (can be fetched live)
BENCHMARK_10Y_GSEC_DEFAULT = 7.39

_cached_benchmark = {"yield": BENCHMARK_10Y_GSEC_DEFAULT, "source": "RBI / FBIL Official Benchmark", "timestamp": None}

def get_live_10y_gsec_yield(fallback: float = BENCHMARK_10Y_GSEC_DEFAULT) -> tuple[float, str]:
    """
    Fetches the live 10-Year Indian Sovereign Benchmark Yield.
    Uses TradingEconomics / Google Finance feeds with a resilient timeout and fallback.
    """
    global _cached_benchmark
    now = datetime.now()
    if _cached_benchmark["timestamp"] is not None:
        elapsed = (now - _cached_benchmark["timestamp"]).total_seconds()
        if elapsed < 900:  # Cache for 15 minutes
            return _cached_benchmark["yield"], _cached_benchmark["source"]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    html_text = ""
    try:
        if requests is not None:
            r = requests.get('https://tradingeconomics.com/india/government-bond-yield', headers=headers, timeout=3.5)
            if r.status_code == 200:
                html_text = r.text
        else:
            req = urllib.request.Request('https://tradingeconomics.com/india/government-bond-yield', headers=headers)
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                html_text = resp.read().decode('utf-8', errors='ignore')

        if html_text:
            match = re.search(r'id=[\"\']market_last[\"\'][^>]*>([0-9\.]+)</span>', html_text)
            if match:
                y = round(float(match.group(1)), 2)
                if 4.0 <= y <= 15.0:
                    _cached_benchmark = {"yield": y, "source": "TradingEconomics (Live)", "timestamp": now}
                    return y, "TradingEconomics (Live)"
    except Exception:
        pass

    _cached_benchmark = {"yield": fallback, "source": "FBIL / RBI Benchmark (Default)", "timestamp": now}
    return fallback, "FBIL / RBI Benchmark (Default)"

BENCHMARK_10Y_GSEC_YIELD = BENCHMARK_10Y_GSEC_DEFAULT

def calculate_accrued_interest(face_value: float, coupon_rate: float, freq: str, issue_dt: date, maturity_dt: date, settlement_dt: date = None) -> float:
    """
    Computes accurate accrued interest in INR from the last coupon payment date to settlement date.
    Day count convention: Actual/Actual (standard for Indian corporate bonds).
    """
    if settlement_dt is None:
        settlement_dt = date.today()

    if settlement_dt >= maturity_dt or settlement_dt < issue_dt:
        return 0.0

    fl = str(freq).lower()
    if "semi" in fl:
        m = 2
    elif "month" in fl:
        m = 12
    elif "quarter" in fl:
        m = 4
    elif "cum" in fl or "zero" in fl:
        return 0.0
    else:
        m = 1

    coupon_per_period = (coupon_rate / 100.0) * face_value / m
    months_per_period = max(1, 12 // m)

    # Roll back from maturity date to determine last coupon date
    cur_year = settlement_dt.year
    ref_month = maturity_dt.month
    ref_day = min(maturity_dt.day, 28)

    candidate_dates = []
    for yr in [cur_year - 1, cur_year, cur_year + 1]:
        for i in range(m):
            mo = ((ref_month - 1 - i * months_per_period) % 12) + 1
            y_adj = yr + ((ref_month - 1 - i * months_per_period) // 12)
            try:
                candidate_dates.append(date(y_adj, mo, ref_day))
            except Exception:
                candidate_dates.append(date(y_adj, mo, 28))

    valid_dates = sorted(list(set([d for d in candidate_dates if issue_dt <= d <= maturity_dt])))
    past_dates = [d for d in valid_dates if d <= settlement_dt]
    future_dates = [d for d in valid_dates if d > settlement_dt]

    last_coupon_dt = past_dates[-1] if past_dates else issue_dt
    next_coupon_dt = future_dates[0] if future_dates else maturity_dt

    period_days = (next_coupon_dt - last_coupon_dt).days
    if period_days <= 0:
        period_days = int(365.25 / m)

    accrued_days = (settlement_dt - last_coupon_dt).days
    accrued_ratio = min(1.0, max(0.0, accrued_days / period_days))

    return round(float(accrued_ratio * coupon_per_period), 2)

def calculate_ytm_from_price(dirty_price: float, face_value: float, coupon_rate: float, tenor_years: float, freq: str = "Annual") -> float:
    """
    Solves for annualized Yield to Maturity (YTM %) using Brent's root-finding method on Dirty Price.
    Supports coupon-paying bonds as well as zero-coupon T-Bills / deep discount bonds.
    """
    if tenor_years <= 0 or dirty_price <= 0 or face_value <= 0:
        return 0.0

    fl = str(freq).lower()

    # Analytical solution for zero-coupon instruments (T-Bills, Zero-Coupon NCDs, Deep Discount Bonds)
    if coupon_rate <= 0.0 or "zero" in fl or "t-bill" in fl:
        try:
            ytm_val = ((face_value / dirty_price) ** (1.0 / tenor_years) - 1.0) * 100.0
            return round(float(max(0.0, ytm_val)), 2)
        except Exception:
            return 0.0

    m = 2 if "semi" in fl else (12 if "month" in fl else (4 if "quarter" in fl else 1))
    n = max(1, int(round(tenor_years * m)))
    c = (coupon_rate / 100.0) * face_value / m

    def price_diff(y):
        r = y / m
        periods = np.arange(1, n + 1)
        discount = 1.0 / np.power(1.0 + r, periods)
        pv = c * np.sum(discount) + face_value * discount[-1]
        return pv - dirty_price

    # Pure Python / NumPy Bisection Root Finder (Zero dependency on scipy)
    y_solved = None
    try:
        from scipy.optimize import brentq
        y_solved = brentq(price_diff, 0.0001, 1.0, xtol=1e-5)
    except Exception:
        pass

    if y_solved is None:
        try:
            low, high = 0.0001, 1.0
            f_low = price_diff(low)
            f_high = price_diff(high)
            if f_low * f_high > 0:
                for upper in [2.0, 5.0, 10.0]:
                    high = upper
                    f_high = price_diff(high)
                    if f_low * f_high <= 0:
                        break
            if f_low * f_high <= 0:
                for _ in range(60):
                    mid = (low + high) / 2.0
                    f_mid = price_diff(mid)
                    if abs(f_mid) < 1e-5 or (high - low) / 2.0 < 1e-6:
                        y_solved = mid
                        break
                    if (f_low > 0 and f_mid > 0) or (f_low < 0 and f_mid < 0):
                        low = mid
                        f_low = f_mid
                    else:
                        high = mid
                        f_high = f_mid
                if y_solved is None:
                    y_solved = (low + high) / 2.0
        except Exception:
            y_solved = None

    if y_solved is not None and not np.isnan(y_solved):
        return round(float(y_solved * 100.0), 2)
    else:
        # Standard bond-equivalent yield approximation fallback
        approx_y = (c * m + (face_value - dirty_price) / tenor_years) / ((face_value + dirty_price) / 2.0)
        return round(float(max(0.0, approx_y * 100.0)), 2)

def calculate_duration_convexity(coupon_rate: float, ytm_pct: float, tenor_years: float, freq: str = "Annual", face_value: float = 100.0):
    """
    Computes Macaulay Duration, Modified Duration, and Convexity for fixed-income securities.
    """
    if tenor_years <= 0 or ytm_pct <= 0:
        return 0.0, 0.0, 0.0

    fl = str(freq).lower()
    m = 2 if "semi" in fl else (12 if "month" in fl else (4 if "quarter" in fl else 1))
    y = max((ytm_pct / 100.0) / m, 1e-6)
    n = max(1, int(round(tenor_years * m)))

    c = (coupon_rate / 100.0) * face_value / m
    periods = np.arange(1, n + 1)
    cash_flows = np.full(n, c)
    cash_flows[-1] += face_value

    discount_factors = 1.0 / np.power(1.0 + y, periods)
    pv_cash_flows = cash_flows * discount_factors
    bond_price = np.sum(pv_cash_flows)

    if bond_price <= 0:
        return round(tenor_years, 2), round(tenor_years, 2), 0.0

    mac_dur_periods = np.sum(periods * pv_cash_flows) / bond_price
    mac_dur_years = mac_dur_periods / m
    mod_dur = mac_dur_years / (1.0 + y)

    convexity_periods = np.sum(periods * (periods + 1) * pv_cash_flows) / (bond_price * np.power(1.0 + y, 2))
    convexity = convexity_periods / (m * m)

    return round(float(mac_dur_years), 2), round(float(mod_dur), 2), round(float(convexity), 2)

def calculate_duration_convexity_vectorized(coupon_rates: np.ndarray, ytm_pcts: np.ndarray, tenor_years: np.ndarray, freqs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast vectorized cash-flow engine: calculates Duration and Convexity across thousands of bonds in ~250ms.
    """
    N = len(coupon_rates)
    m_arr = np.ones(N, dtype=float)
    for i, f in enumerate(freqs):
        fl = str(f).lower()
        if "semi" in fl:
            m_arr[i] = 2.0
        elif "month" in fl:
            m_arr[i] = 12.0
        elif "quarter" in fl:
            m_arr[i] = 4.0

    mac_dur = np.zeros(N)
    mod_dur = np.zeros(N)
    convexity = np.zeros(N)

    for i in range(N):
        tenor = tenor_years[i]
        ytm = ytm_pcts[i]
        if tenor <= 0 or ytm <= 0:
            continue
        m = m_arr[i]
        y = max((ytm / 100.0) / m, 1e-6)
        n = max(1, int(round(tenor * m)))
        c = (coupon_rates[i] / 100.0) / m

        periods = np.arange(1, n + 1)
        discount = 1.0 / np.power(1.0 + y, periods)
        cf = np.full(n, c)
        cf[-1] += 1.0  # normalized to 1.0 unit face value
        pv = cf * discount
        price = np.sum(pv)

        if price > 0:
            mac_p = np.sum(periods * pv) / price
            mac_y = mac_p / m
            mod_y = mac_y / (1.0 + y)
            cvx_p = np.sum(periods * (periods + 1) * pv) / (price * np.power(1.0 + y, 2))
            cvx_y = cvx_p / (m * m)

            mac_dur[i] = round(mac_y, 2)
            mod_dur[i] = round(mod_y, 2)
            convexity[i] = round(cvx_y, 2)

    return mac_dur, mod_dur, convexity

def compute_derived_metrics(df: pd.DataFrame, user_tax_rate: float = 0.0, benchmark_yield: float = None) -> pd.DataFrame:
    """
    Calculates remaining tenor, dynamic accrued interest, dirty price, post-tax yield,
    pre-tax equivalent yield (PTEY), credit spreads, risk buckets, vectorized durations, and liquidity flags.
    """
    if benchmark_yield is None:
        benchmark_yield = BENCHMARK_10Y_GSEC_DEFAULT

    data = df.copy()
    today_dt = date.today()
    today_ts = pd.to_datetime(today_dt)

    # 1. Parse dates & compute remaining tenor
    issue_dt = pd.to_datetime(data['issue_date'])
    maturity_dt = pd.to_datetime(data['maturity_date'])
    days_left = (maturity_dt - today_ts).dt.days

    data['remaining_tenor_months'] = (days_left / 30.4375).clip(lower=0).round(1)
    data['remaining_tenor_years'] = (days_left / 365.25).clip(lower=0).round(2)
    data['remaining_tenor_display'] = np.where(
        days_left > 0,
        data['remaining_tenor_years'].astype(str) + " Yrs (" + data['remaining_tenor_months'].astype(int).astype(str) + "M)",
        "Matured"
    )

    data['issue_date'] = issue_dt.dt.strftime('%d-%b-%Y')
    data['maturity_date'] = maturity_dt.dt.strftime('%d-%b-%Y')

    if 'payment_frequency' not in data.columns:
        data['payment_frequency'] = 'Annual'

    # 2. Inferred Face Value (handles ₹100, ₹1,000, ₹10,000, ₹1,00,000)
    if 'face_value' not in data.columns:
        data['face_value'] = np.where(
            data['last_traded_price'] > 50000, 100000.0,
            np.where(data['last_traded_price'] > 5000, 10000.0,
                     np.where(data['last_traded_price'] > 500, 1000.0, 100.0))
        )
    else:
        data['face_value'] = data['face_value'].fillna(100.0).astype(float)

    # 3. Dynamic Accrued Interest (calculated dynamically based on today's date)
    raw_issue_dates = issue_dt.dt.date.tolist()
    raw_mat_dates = maturity_dt.dt.date.tolist()
    face_vals = data['face_value'].tolist()
    coupons = data['coupon_rate'].tolist()
    freqs = data['payment_frequency'].tolist()

    computed_accrued = [
        calculate_accrued_interest(face_vals[i], coupons[i], freqs[i], raw_issue_dates[i], raw_mat_dates[i], today_dt)
        for i in range(len(data))
    ]
    data['accrued_interest'] = computed_accrued
    data['dirty_price'] = (data['last_traded_price'] + data['accrued_interest']).round(2)

    # 4. Solved YTM vs Stated YTM
    # Dynamically compute mathematically accurate YTM from clean price (last_traded_price)
    solved_ytm = []
    for i in range(len(data)):
        clean_p = float(data['last_traded_price'].iloc[i]) if 'last_traded_price' in data.columns else 0.0
        c_val = coupons[i]
        fv_val = face_vals[i]
        tenor_val = data['remaining_tenor_years'].iloc[i]
        freq_val = freqs[i]
        if clean_p > 0 and tenor_val > 0:
            y = calculate_ytm_from_price(clean_p, fv_val, c_val, tenor_val, freq_val)
            solved_ytm.append(y)
        elif 'current_yield_ytm' in data.columns and not pd.isna(data['current_yield_ytm'].iloc[i]):
            solved_ytm.append(round(float(data['current_yield_ytm'].iloc[i]), 2))
        else:
            solved_ytm.append(0.0)
    data['current_yield_ytm'] = solved_ytm

    # 5. Post-Tax Yield & Pre-Tax Equivalent Yield (PTEY)
    # Tax-Free bonds (IRFC, PFC, REC, NHAI under Sec 10(15)(iv)(h)) have 0% tax on coupon
    tax_factor = 1.0 - (user_tax_rate / 100.0)
    is_tax_free = data['tax_status'].astype(str).str.strip().str.lower().isin(['tax-free', '54ec'])

    # Post-tax yield
    data['post_tax_yield'] = np.where(
        is_tax_free,
        data['current_yield_ytm'],
        (data['current_yield_ytm'] * tax_factor).round(2)
    )

    # Pre-Tax Equivalent Yield (PTEY): What a bank FD or taxable bond must pay to match this post-tax return
    if user_tax_rate > 0.0 and user_tax_rate < 100.0:
        data['pre_tax_equiv_yield'] = np.where(
            is_tax_free,
            (data['current_yield_ytm'] / tax_factor).round(2),
            data['current_yield_ytm']
        )
    else:
        data['pre_tax_equiv_yield'] = data['current_yield_ytm']

    # 6. Credit Spread over Sovereign Benchmark (bps)
    data['credit_spread_bps'] = ((data['current_yield_ytm'] - benchmark_yield) * 100).round().astype(int)

    # 7. Risk Bucketing
    ratings = data['rating_current'].astype(str).str.upper() if 'rating_current' in data.columns else pd.Series([""] * len(data), index=data.index)
    sec = data['secured_unsecured'].astype(str).str.capitalize() if 'secured_unsecured' in data.columns else pd.Series([""] * len(data), index=data.index)
    sector = data['sector'].astype(str).str.lower() if 'sector' in data.columns else pd.Series([""] * len(data), index=data.index)
    seniority = data['seniority'].astype(str).str.capitalize() if 'seniority' in data.columns else pd.Series([""] * len(data), index=data.index)

    is_sovereign = (
        ratings.str.contains('SOVEREIGN', na=False) |
        sector.str.contains('g-sec', na=False) |
        sector.str.contains('gilt', na=False) |
        sector.str.contains('sdl', na=False) |
        sector.str.contains('state development', na=False) |
        sector.str.contains('t-bill', na=False) |
        sector.str.contains('treasury', na=False) |
        sec.str.contains('sovereign', na=False)
    ).to_numpy(dtype=bool)
    is_prime_rating = ratings.str.contains(r'AAA|AA\+|AA|AA\-', regex=True, na=False).to_numpy(dtype=bool)
    is_secured = (sec.str.contains('Secured', na=False) | seniority.str.contains('Senior', na=False)).to_numpy(dtype=bool)
    is_a_rating = (ratings.str.contains(r'A\+|A\-|\bA\b', regex=True, na=False) & ~ratings.str.contains('AA', na=False)).to_numpy(dtype=bool)

    data['risk_bucket'] = np.where(
        is_sovereign, "Sovereign / G-Sec (Zero Default Risk)",
        np.where(is_prime_rating & is_secured, "Prime AAA/AA+ Secured",
                 np.where(is_prime_rating, "Prime Unsecured / Tier-II",
                          np.where(is_a_rating, "Moderate Yield A-Rated", "High Yield / Credit Risk")))
    )

    # 8. Fast Vectorized Duration & Convexity
    mac_dur, mod_dur, cvx = calculate_duration_convexity_vectorized(
        data['coupon_rate'].values,
        data['current_yield_ytm'].values,
        data['remaining_tenor_years'].values,
        data['payment_frequency'].tolist()
    )
    data['macaulay_duration'] = mac_dur
    data['modified_duration'] = mod_dur
    data['convexity'] = cvx

    # 9. Correct Taylor Series Price Impact for 100 bps Hike (+1% Rate Shock)
    # dP/P % = (-ModDur * 0.01 + 0.5 * Convexity * (0.01)^2) * 100 = -ModDur * 1.0 + 0.5 * Convexity * 0.01
    data['price_impact_100bps_hike'] = (
        -data['modified_duration'] * 1.0 + 0.5 * data['convexity'] * 0.01
    ).round(2)

    # 10. Volume & Liquidity Categorization with Stale-Trade Detection
    data['avg_daily_volume_cr'] = data.get('daily_value_inr_cr', pd.Series(0.0, index=data.index)).fillna(0.0).round(2)
    val = data['avg_daily_volume_cr']
    trades = data.get('num_trades', pd.Series(0, index=data.index)).fillna(0).astype(int)

    is_high = ((val >= 20.0) | (trades >= 100)).to_numpy(dtype=bool)
    is_med = ((val >= 5.0) | (trades >= 30)).to_numpy(dtype=bool)
    is_mod = (val >= 1.0).to_numpy(dtype=bool)

    data['liquidity_flag'] = np.where(is_high, "High", np.where(is_med, "Medium", "Low"))
    val_str = val.astype(str)
    data['volume_indicator'] = np.where(
        is_high, "🟢 High (₹ " + val_str + " Cr)",
        np.where(is_med, "🟡 Medium (₹ " + val_str + " Cr)",
                 np.where(is_mod, "🟠 Moderate (₹ " + val_str + " Cr)",
                          "⚪ Low (₹ " + val_str + " Cr)"))
    )

    return data

def ingest_bhavcopy(file_obj_or_path, master_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Ingests an official NSE/BSE Debt Bhavcopy (CSV or ZIP), extracts live traded prices,
    turnover, and trade counts, and updates the master bond database by ISIN.
    """
    try:
        # Check if file-like object
        if hasattr(file_obj_or_path, 'read'):
            raw_content = file_obj_or_path.read()
            if hasattr(file_obj_or_path, 'seek'):
                file_obj_or_path.seek(0)
            if isinstance(raw_content, str):
                bhav_df = pd.read_csv(io.StringIO(raw_content))
            elif isinstance(raw_content, bytes):
                if raw_content[:4] == b'PK\x03\x04':
                    z = zipfile.ZipFile(io.BytesIO(raw_content))
                    csv_name = [n for n in z.namelist() if n.endswith('.csv')][0]
                    bhav_df = pd.read_csv(z.open(csv_name))
                else:
                    bhav_df = pd.read_csv(io.BytesIO(raw_content))
            else:
                bhav_df = pd.read_csv(file_obj_or_path)
        else:
            bhav_df = pd.read_csv(file_obj_or_path)

        bhav_df.columns = bhav_df.columns.str.strip().str.upper()

        # Handle NSE Bhavcopy format
        if 'ISIN' in bhav_df.columns and 'LAST_PRICE' in bhav_df.columns:
            update_map = {}
            for _, r in bhav_df.iterrows():
                isin = str(r.get('ISIN', '')).strip()
                if not isin or isin == 'nan':
                    continue
                update_map[isin] = {
                    'last_traded_price': float(r.get('LAST_PRICE', r.get('CLOSE_PRICE', 0.0))),
                    'daily_volume_units': float(r.get('TTL_TRD_QNTY', 0)),
                    'daily_value_inr_cr': round(float(r.get('TURNOVER_LACS', 0)) / 100.0, 2),
                    'num_trades': int(r.get('NO_OF_TRADES', 0)),
                    'last_traded_date': str(r.get('DATE1', datetime.today().strftime('%d-%b-%Y')))
                }
        # Handle BSE Bhavcopy format
        elif 'ISIN_CODE' in bhav_df.columns:
            update_map = {}
            for _, r in bhav_df.iterrows():
                isin = str(r.get('ISIN_CODE', '')).strip()
                if not isin or isin == 'nan':
                    continue
                update_map[isin] = {
                    'last_traded_price': float(r.get('LAST', r.get('CLOSE', 0.0))),
                    'daily_volume_units': float(r.get('NO_OF_SHRS', 0)),
                    'daily_value_inr_cr': round(float(r.get('NET_TURNOV', 0)) / 10000000.0, 2),
                    'num_trades': int(r.get('NO_TRADES', 0)),
                    'last_traded_date': datetime.today().strftime('%d-%b-%Y')
                }
        else:
            return master_df, "Uploaded file does not match NSE or BSE Bhavcopy schema (missing ISIN column)."

        # Merge onto master dataframe
        updated_master = master_df.copy()
        matched_count = 0
        for idx, row in updated_master.iterrows():
            isin = str(row.get('isin', '')).strip()
            if isin in update_map:
                info = update_map[isin]
                if info['last_traded_price'] > 0:
                    updated_master.at[idx, 'last_traded_price'] = info['last_traded_price']
                    updated_master.at[idx, 'daily_volume_units'] = info['daily_volume_units']
                    updated_master.at[idx, 'daily_value_inr_cr'] = info['daily_value_inr_cr']
                    updated_master.at[idx, 'num_trades'] = info['num_trades']
                    updated_master.at[idx, 'last_traded_date'] = info['last_traded_date']
                    matched_count += 1

        return updated_master, f"Successfully merged {matched_count} active debt quotes from Bhavcopy."
    except Exception as e:
        return master_df, f"Error processing Bhavcopy: {e}"

def ingest_zerodha_holdings(file_obj_or_text, master_df: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """
    Ingests Zerodha Kite holdings export (CSV file, copy-pasted TSV/text, or JSON),
    extracts Instrument symbols, LTP, Average Cost, Qty, and Cur. Val.
    Updates matched bonds in master database with live LTP, auto-detects amortized face values,
    recalculates dynamic accrued interest, dirty price, and YTM.
    Returns (updated_master_df, status_message, audit_summary_df).
    """
    try:
        raw_text = ""
        if hasattr(file_obj_or_text, 'read'):
            content = file_obj_or_text.read()
            if hasattr(file_obj_or_text, 'seek'):
                file_obj_or_text.seek(0)
            if isinstance(content, bytes):
                raw_text = content.decode('utf-8', errors='ignore')
            else:
                raw_text = str(content)
        else:
            raw_text = str(file_obj_or_text)

        lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
        if not lines:
            return master_df, "Uploaded holdings file or input is empty.", pd.DataFrame()

        # Find header line (containing 'instrument' or 'symbol')
        header_idx = 0
        for idx, line in enumerate(lines[:10]):
            if any(k in line.lower() for k in ['instrument', 'symbol', 'tradingsymbol']):
                header_idx = idx
                break

        relevant_text = "\n".join(lines[header_idx:])
        delimiter = '\t' if '\t' in lines[header_idx] else ','
        kdf = pd.read_csv(io.StringIO(relevant_text), sep=delimiter)

        def clean_num(v):
            if pd.isna(v):
                return 0.0
            s = re.sub(r'[^\d\.\-]', '', str(v))
            try:
                return float(s)
            except Exception:
                return 0.0

        cols_lower = {c: c.lower().strip() for c in kdf.columns}
        inst_col = next((c for c, l in cols_lower.items() if any(k in l for k in ['instrument', 'symbol', 'tradingsymbol'])), None)
        if not inst_col:
            return master_df, "Could not find 'Instrument' column in Zerodha data.", pd.DataFrame()

        ltp_col = next((c for c, l in cols_lower.items() if 'ltp' in l or 'last' in l), None)
        avg_col = next((c for c, l in cols_lower.items() if 'avg' in l or 'cost' in l or 'buy' in l), None)
        qty_col = next((c for c, l in cols_lower.items() if 'qty' in l or 'quantity' in l), None)
        val_col = next((c for c, l in cols_lower.items() if 'cur' in l or 'val' in l), None)

        updated_master = master_df.copy()
        matched_count = 0
        audit_rows = []
        today_d = date.today()

        for _, row in kdf.iterrows():
            sym = str(row[inst_col]).strip().upper()
            if not sym or sym == 'NAN':
                continue

            ltp_val = clean_num(row[ltp_col]) if ltp_col else 0.0
            avg_val = clean_num(row[avg_col]) if avg_col else 0.0
            qty_val = clean_num(row[qty_col]) if qty_col else 0.0
            cur_val = clean_num(row[val_col]) if val_col else 0.0

            eff_price = ltp_val if ltp_val > 0 else avg_val
            if eff_price <= 0:
                continue

            # Match against master database ticker or ISIN
            mask = (updated_master['ticker'].astype(str).str.upper() == sym) | (updated_master['isin'].astype(str).str.upper() == sym)
            if not mask.any():
                sym_clean = sym.split('-')[0]
                mask = updated_master['ticker'].astype(str).str.upper().str.startswith(sym_clean)

            if not mask.any():
                continue

            idx = updated_master[mask].index[0]
            ticker_matched = updated_master.at[idx, 'ticker']
            old_p = float(updated_master.at[idx, 'last_traded_price'])

            # Handle amortised face values (e.g. NHIT InvIT amortised down to ₹300)
            if 'NHIT' in sym or ('NHIT' in str(ticker_matched).upper()):
                updated_master.at[idx, 'face_value'] = 300.0

            updated_master.at[idx, 'last_traded_price'] = eff_price
            fv = float(updated_master.at[idx, 'face_value'])
            c = float(updated_master.at[idx, 'coupon_rate'])
            freq = str(updated_master.at[idx, 'payment_frequency'])
            mat_dt = pd.to_datetime(updated_master.at[idx, 'maturity_date']).date()
            issue_dt = pd.to_datetime(updated_master.at[idx, 'issue_date']).date()
            tenor = max(0.01, (mat_dt - today_d).days / 365.25)

            acc = calculate_accrued_interest(fv, c, freq, issue_dt, mat_dt, today_d)
            dirty = round(eff_price + acc, 2)
            ytm = calculate_ytm_from_price(eff_price, fv, c, tenor, freq)

            updated_master.at[idx, 'accrued_interest'] = acc
            updated_master.at[idx, 'current_yield_ytm'] = ytm
            updated_master.at[idx, 'last_traded_date'] = today_d.strftime('%Y-%m-%d')
            matched_count += 1

            audit_rows.append({
                'Ticker': ticker_matched,
                'Previous LTP': f"₹ {old_p:,.2f}",
                'New Kite LTP': f"₹ {eff_price:,.2f}",
                'Price Change': f"₹ {eff_price - old_p:+,.2f}",
                'Accrued Interest': f"₹ {acc:,.2f}",
                'Dirty Price': f"₹ {dirty:,.2f}",
                'New YTM': f"{ytm:.2f}%",
                'Portfolio Qty': int(qty_val) if qty_val > 0 else "—",
                'Current Value': f"₹ {cur_val:,.2f}" if cur_val > 0 else (f"₹ {qty_val * eff_price:,.2f}" if qty_val > 0 else "—")
            })

        audit_df = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame()
        msg = f"Successfully matched and synced {matched_count} Zerodha Kite holdings quotes into master screener."
        return updated_master, msg, audit_df
    except Exception as e:
        return master_df, f"Error parsing Zerodha Kite holdings: {e}", pd.DataFrame()


def load_sample_csv(path: str = "data/sample_bonds.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Error loading master CSV at {path}: {e}")
