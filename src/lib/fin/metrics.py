from datetime import date as Date
from dateutil.relativedelta import relativedelta
from typing import cast, Literal, TypedDict

import numpy as np
import polars as pl
import statsmodels.api as sm

from lib.fin.calculation import applier, fin_filters


class BetaRecord(TypedDict):
  date: Date
  beta: float
  market_return: float
  riskfree_rate: float
  months: int


class YieldSpreadRecord(TypedDict):
  capitalization_class: Literal["small", "large"]
  lower: float
  upper: float
  yield_spread: float


def f_score(
  lf: pl.LazyFrame, slices: list[tuple[pl.Expr, pl.Expr, Literal[3, 12]]]
) -> pl.LazyFrame:
  f_score_diff = {
    "return_on_assets": "change_retorn_on_assets",
    "debt": "change_debt",
    "quick_ratio": "change_quick_ratio",
    "weighted_average_shares_outstanding_basic": "change_weighted_average_shares_outstanding_basic",
    "operating_profit_margin": "change_operating_profit_margin",
    "asset_turnover": "change_asset_turnover",
  }

  lf = applier(
    lf,
    column_alias=f_score_diff,
    fn="diff",
    slices=slices,
  )

  f_score_expr = (
    (pl.col("return_on_equity") > 0).cast(pl.Int8)
    + (pl.col("change_return_on_assets") > 0).cast(pl.Int8)
    + (pl.col("cashflow_operating") > 0).cast(pl.Int8)
    + (
      (pl.col("cashflow_operating") / pl.col("assets") - pl.col("return_on_assets")) > 0
    ).cast(pl.Int8)
    + (pl.col("change_debt") < 0).cast(pl.Int8)
    + (pl.col("change_quick_ratio") > 0).cast(pl.Int8)
    + (pl.col("change_weighted_average_shares_outstanding_basic") < 0).cast(pl.Int8)
    + (pl.col("change_operating_profit_margin") > 0).cast(pl.Int8)
    + (pl.col("change_asset_turnover") > 0).cast(pl.Int8)
  )

  return lf.with_columns(f_score_expr.alias("piotroski_f_score")).drop(
    list(f_score_diff.values())
  )


def z_score(lf: pl.LazyFrame) -> pl.LazyFrame:
  assets = pl.col("average_assets")
  liabilities = pl.col("average_liabilities")

  return lf.with_columns(
    (
      1.2 * pl.col("average_working_capital_operating") / assets
      + 1.4 * pl.col("retained_earnings_accumulated_deficit") / assets
      + 3.3 * pl.col("cashflow_operating") / assets
      + 0.6 * pl.col("market_capitalization") / liabilities
      + assets / liabilities
    ).alias("altman_z_score")
  )


def m_score(
  lf: pl.LazyFrame, slices: list[tuple[pl.Expr, pl.Expr, Literal[3, 12]]]
) -> pl.LazyFrame:
  lf = lf.with_columns(
    [
      (pl.col("average_receivables_trade_current") / pl.col("revenue")).alias("dsri"),
      (
        1
        - (
          pl.col("working_capital_operating")
          + pl.col("productive_assets")
          + pl.col("financial_assets_noncurrent")
        )
        / pl.col("assets")
      ).alias("aqi"),
      (
        pl.col("depreciation") / (pl.col("productive_assets") - pl.col("depreciation"))
      ).alias("depi"),
      (pl.col("selling_general_administrative_expense") / pl.col("revenue")).alias(
        "sgai"
      ),
      (pl.col("liabilities") / pl.col("assets")).alias("li"),
    ]
  )

  m_score_shift = {
    "dsri": "shift_dsri",
    "operating_profit_margin": "shift_operating_profit_margin",
    "aqi": "shift_aqi",
    "revenue": "shift_revenue",
    "depi": "shift_depi",
    "sgai": "shift_sgai",
    "li": "shift_li",
  }

  # Apply shift using applier
  lf = applier(
    lf,
    column_alias=m_score_shift,
    fn="shift",
    slices=slices,
  )

  # Compute all ratios
  lf = lf.with_columns(
    [
      (pl.col("dsri") / pl.col("shift_dsri")).alias("dsri"),
      (
        pl.col("shift_operating_profit_margin") / pl.col("operating_profit_margin")
      ).alias("gmi"),
      (pl.col("aqi") / pl.col("shift_aqi")).alias("aqi"),
      (pl.col("revenue") / pl.col("shift_revenue")).alias("sgi"),
      (pl.col("depi") / pl.col("shift_depi")).alias("depi"),
      (pl.col("sgai") / pl.col("shift_sgai")).alias("sgai"),
      (pl.col("li") / pl.col("shift_li")).alias("li"),
      (
        (pl.col("income_loss_operating") - pl.col("cashflow_operating"))
        / pl.col("average_assets")
      ).alias("tata"),
    ]
  )

  # Final Beneish M-Score expression
  m_score_expr = (
    -4.84
    + 0.92 * pl.col("dsri")
    + 0.528 * pl.col("gmi")
    + 0.404 * pl.col("aqi")
    + 0.892 * pl.col("sgi")
    + 0.115 * pl.col("depi")
    - 0.172 * pl.col("sgai")
    + 4.679 * pl.col("tata")
    - 0.327 * pl.col("li")
  ).alias("beneish_m_score")

  drop_cols = [
    "aqi",
    "shift_aqi",
    "depi",
    "shift_depi",
    "dsri",
    "shift_dsri",
    "gmi",
    "li",
    "shift_li",
    "sgi",
    "sgai",
    "shift_sgai",
    "shift_operating_profit_margin",
    "shift_revenue",
  ]
  return lf.with_columns(m_score_expr).drop(drop_cols)


def calculate_beta(
  dates: list[Date], returns_df: pl.DataFrame, months: int, years: int | None = None
) -> pl.LazyFrame:
  delta = relativedelta(months=months) if years is None else relativedelta(years=years)

  min_date = returns_df["date"].min()
  dates = [d for d in dates if d > min_date]

  beta_rows: list[BetaRecord] = []
  for i in range(1, len(dates)):
    start = dates[i - 1] - delta
    end = dates[i]

    window_df = returns_df.filter(
      (pl.col("date") >= pl.lit(start)) & (pl.col("date") < pl.lit(end))
    )

    if window_df.height == 0:
      continue

    temp = window_df.select(["market_return", "equity_return", "riskfree_rate"])

    x = sm.add_constant(temp["market_return"].to_numpy())
    y = temp["equity_return"].to_numpy()

    model = sm.OLS(y, x).fit()
    beta = model.params[-1]
    market_mean = cast(float, temp["market_return"].mean())
    riskfree_mean = cast(float, temp["riskfree_rate"].mean())

    beta_rows.append(
      BetaRecord(
        date=end,
        beta=beta,
        market_return=market_mean * 252.0,
        riskfree_rate=riskfree_mean,
        months=months,
      )
    )

  return pl.LazyFrame(beta_rows)


def beta(
  financials: pl.LazyFrame,
  equity_return: pl.LazyFrame,
  market_return: pl.LazyFrame,
  riskfree_rate: pl.LazyFrame,
  years: int | None = 0,
) -> pl.LazyFrame:
  returns_df = (
    equity_return.join(market_return, on="date", how="outer", coalesce=True)
    .join(riskfree_rate, on="date", how="outer", coalesce=True)
    .sort("date")
    .fill_null(strategy="forward")
    .drop_nulls()
    .collect()
  )
  slices = fin_filters()

  beta_frames: list[pl.LazyFrame] = []
  for period_filter, month_filter, months in slices:
    dates = (
      financials.filter(period_filter & month_filter)
      .select("date")
      .collect()
      .get_column("date")
      .to_list()
    )
    if not dates:
      continue

    beta_frame = calculate_beta(dates, returns_df, months, years)
    beta_frames.append(beta_frame)

  betas = pl.concat(beta_frames)
  return financials.join(betas, on=["date", "months"], how="left")


def weighted_average_cost_of_capital(
  financials: pl.LazyFrame, debt_maturity: int = 10, large_cap_limit: float = 2e9
) -> pl.LazyFrame:
  financials = apply_yield_spread(financials, large_cap_limit)

  financials = financials.with_columns(
    (
      pl.col("beta")
      * (
        1 + (1 - pl.col("tax_rate")) * pl.col("average_debt") / pl.col("average_equity")
      )
    ).alias("beta_levered")
  )
  financials = financials.with_columns(
    (
      pl.col("beta_levered") * (pl.col("market_return") - pl.col("riskfree_rate"))
    ).alias("equity_risk_premium")
  )
  financials = financials.with_columns(
    [
      (pl.col("riskfree_rate") + pl.col("equity_risk_premium")).alias("cost_equity"),
      (pl.col("riskfree_rate") + pl.col("yield_spread")).alias("cost_debt"),
    ]
  )
  financials = financials.with_columns(
    (
      (pl.col("interest_expense") / pl.col("cost_debt"))
      * (1 - (1 / (1 + pl.col("cost_debt")) ** debt_maturity))
      + (pl.col("debt") / (1 + pl.col("cost_debt")) ** debt_maturity)
    ).alias("market_value_debt")
  )
  financials = financials.with_columns(
    (
      pl.col("market_capitalization")
      / (pl.col("market_capitalization") + pl.col("market_value_debt"))
    ).alias("equity_to_capital")
  )
  financials = financials.with_columns(
    (
      pl.col("cost_equity") * pl.col("equity_to_capital")
      + pl.col("cost_debt")
      * (1 - pl.col("tax_rate"))
      * (
        pl.col("market_value_debt")
        / (pl.col("market_capitalization") + pl.col("market_value_debt"))
      )
    ).alias("weighted_average_cost_of_capital"),
  )

  return financials.drop(["market_return", "capitalization_class"])


def apply_yield_spread(
  financial: pl.LazyFrame, large_cap_limit: float = 2e9
) -> pl.LazyFrame:
  spread_table = yield_spread_table()

  financials = financial.with_columns(
    [
      pl.when(pl.col("interest_coverage_ratio").is_infinite())
      .then(pl.when(pl.col("interest_coverage_ratio") > 0).then(0.004).otherwise(0.4))
      .when(pl.col("interest_coverage_ratio").is_nan())
      .then(None)
      .otherwise(pl.col("interest_coverage_ratio"))
      .alias("icr_clean"),
      pl.when(pl.col("market_capitalization") < large_cap_limit)
      .then(pl.lit("small"))
      .otherwise(pl.lit("large"))
      .alias("capitalization_class"),
    ]
  )

  joined = (
    financials.join(
      spread_table,
      left_on=["capitalization_class", "icr_clean"],
      right_on=["capitalization_class", "lower"],
      how="inner",
    )
    .filter(pl.col("icr_clean") < pl.col("upper"))
    .drop(["icr_clean", "upper"])
  )
  return joined


def yield_spread_table() -> pl.LazyFrame:
  small_icr_intervals = (
    -1e5,
    0.5,
    0.8,
    1.25,
    1.5,
    2,
    2.5,
    3,
    3.5,
    4,
    4.5,
    6,
    7.5,
    9.5,
    12.5,
    1e5,
  )
  large_icr_intervals = (
    -1e5,
    0.2,
    0.65,
    0.8,
    1.25,
    1.5,
    1.75,
    2,
    2.25,
    2.5,
    3,
    4.25,
    5.5,
    6.5,
    8.5,
    1e5,
  )
  spreads = (
    0.1512,
    0.1134,
    0.0865,
    0.082,
    0.0515,
    0.0421,
    0.0351,
    0.024,
    0.02,
    0.0156,
    0.0122,
    0.0122,
    0.0108,
    0.0098,
    0.0078,
    0.0063,
  )

  rows: list[YieldSpreadRecord] = []
  for cap_class, intervals in [
    ("small", small_icr_intervals),
    ("large", large_icr_intervals),
  ]:
    for i in range(len(spreads) - 1):
      rows.append(
        YieldSpreadRecord(
          capitalization_class=cast(Literal["small", "large"], cap_class),
          lower=intervals[i],
          upper=intervals[i + 1],
          yield_spread=spreads[i],
        )
      )

  return pl.LazyFrame(rows)


def discounted_cash_flow(
  lf: pl.LazyFrame, fc_period: int = 20, longterm_growth: float = 0.03
) -> pl.LazyFrame:
  if (
    lf.select(pl.col("weighted_average_cost_of_capital").is_not_null().sum()).collect()[
      0, 0
    ]
    == 0
  ):
    return lf.with_columns(pl.lit(None).alias("discounted_cashflow_value"))

  wacc_ema = (
    lf.select(pl.col("weighted_average_cost_of_capital"))
    .collect()
    .to_series()
    .ewm_mean(span=None, adjust=False)
  )
  fcf = lf.select("free_cashflow_firm").collect().to_series()
  fcf_roc = fcf.diff() / fcf.shift(1).abs()
  fcf_growth = fcf_roc.ewm_mean(span=fcf_roc.drop_nulls().shape[0], adjust=False)

  dcf = np.zeros(len(fcf))
  x = np.arange(1, fc_period + 1)

  for i, cfg in enumerate(fcf_growth):
    if cfg is None or np.isnan(cfg) or np.isnan(fcf[i]) or np.isnan(wacc_ema[i]):
      continue

    growth_projection = longterm_growth + (cfg - longterm_growth) / (
      1 + np.exp(np.sign(cfg - longterm_growth) * (x - fc_period / 2))
    )

    cf = fcf[i]
    for j, g in zip(x, growth_projection):
      cf += np.abs(cf) * g
      present = cf / (1 + wacc_ema[i]) ** j
      dcf[i] += present

    if wacc_ema[i] > longterm_growth:
      terminal = (
        np.abs(present) * (1 + longterm_growth) / (wacc_ema[i] - longterm_growth)
      )
    else:
      terminal = (
        np.abs(lf.select("enterprise_value").collect()[i, 0])
        * (1 + longterm_growth) ** fc_period
      )

    terminal /= (1 + wacc_ema[i]) ** fc_period
    dcf[i] += terminal

  liquid_assets = lf.select("liquid_assets").collect().to_series()
  debt = lf.select("debt").collect().to_series()
  shares = (
    lf.select("split_adjusted_weighted_average_shares_outstanding_basic")
    .collect()
    .to_series()
  )

  dcf += liquid_assets - debt
  dcf_per_share = dcf / shares

  return lf.with_columns(
    pl.Series("discounted_cashflow_value", dcf_per_share).alias(
      "dicounted_cashflow_value"
    )
  )


def earnings_power_value_lazyframe(lf: pl.LazyFrame) -> pl.LazyFrame:
  com_value = 0.5

  lf = lf.with_columns(
    pl.when(pl.col("weighted_average_cost_of_capital").is_null())
    .then(None)
    .otherwise(pl.col("weighted_average_cost_of_capital"))
    .alias("wacc"),
  )

  # Compute smoothed WACC
  lf = lf.with_columns(
    [
      pl.col("wacc").ewm_mean(com=com_value).alias("wacc_ewm"),
      pl.col("revenue").ewm_mean(com=com_value).alias("sustainable_revenue"),
      pl.col("revenue").diff().alias("revenue_diff"),
      pl.col("revenue").shift(1).abs().alias("revenue_shifted"),
    ]
  )

  # Revenue growth
  lf = lf.with_columns(
    (pl.col("revenue_diff") / pl.col("revenue_shifted")).alias("revenue_growth_raw")
  )
  lf = lf.with_columns(
    pl.col("revenue_growth_raw").ewm_mean(com=com_value).alias("revenue_growth")
  )

  # Capex and margin
  lf = lf.with_columns(
    pl.col("payment_acquisition_productive_assets")
    .ewm_mean(com=com_value)
    .alias("capex"),
  )
  lf = lf.with_columns(
    [
      (pl.col("capex") / pl.col("sustainable_revenue")).alias("capex_margin"),
      (pl.col("capex_margin") * (1 - pl.col("revenue_growth"))).alias(
        "maintenance_capex"
      ),
    ]
  )

  # Operating margin
  lf = lf.with_columns(
    (
      (pl.col("pretax_income_loss") + pl.col("interest_expense")) / pl.col("revenue")
    ).alias("operating_margin_raw")
  )
  lf = lf.with_columns(
    pl.col("operating_margin_raw").ewm_mean(com=com_value).alias("operating_margin")
  )

  # Adjusted DDAA
  lf = lf.with_columns(
    [
      (
        0.5
        * pl.col("tax_rate")
        * pl.col("depreciation_depletion_amortization_accretion")
      ).alias("adjusted_ddaa")
    ]
  )

  # Adjusted earnings
  lf = lf.with_columns(
    (
      pl.col("sustainable_revenue")
      * pl.col("operating_margin")
      * (1 - pl.col("tax_rate"))
      + pl.col("adjusted_ddaa")
      - (pl.col("maintenance_capex") * pl.col("sustainable_revenue"))
    ).alias("adjusted_earnings")
  )

  # EPV
  lf = lf.with_columns(
    (pl.col("adjusted_earnings") / pl.col("wacc_ewm")).alias("epv_raw"),
    (
      (pl.col("epv_raw") + pl.col("liquid_assets") - pl.col("debt"))
      / pl.col("weighted_average_shares_outstanding_basic")
    ).alias("earnings_power_value"),
  )

  return lf.select(
    [
      *[
        col
        for col in lf.columns
        if not col.startswith("epv_")
        and not col.endswith("_raw")
        and not col.endswith("_diff")
        and not col.endswith("_shifted")
        and not col.endswith("_roc")
      ],
      "earnings_power_value",
    ]
  )


"""
ChatGPT
import polars as pl
import numpy as np


def discounted_cash_flow_lazyframe(
    lf: pl.LazyFrame, fc_period: int = 20, longterm_growth: float = 0.03
) -> pl.LazyFrame:
    x = np.arange(1, fc_period + 1)

    # Estimate WACC using a rolling average
    lf = lf.with_columns([
        pl.col("weight_average_cost_of_capital").ewm_mean(com=0.5).alias("wacc"),
        pl.col("free_cash_flow_firm").alias("fcf"),
    ])

    # Estimate FCF growth rate: diff(fcf) / abs(fcf.shift(1))
    lf = lf.with_columns(
        (pl.col("fcf").diff() / pl.col("fcf").shift(1).abs()).alias("fcf_roc")
    )
    lf = lf.with_columns(
        pl.col("fcf_roc").ewm_mean(com=0.5).alias("fcf_growth")
    )

    # Project future FCFs and discount them to present value
    def project_dcf_expr(fc_period: int, longterm_growth: float):
        # Inner expression to simulate sigmoid smoothing of growth
        return pl.map_batches(
            lambda df: _dcf_batch_calc(df, fc_period, longterm_growth),
            return_dtype=pl.Float64,
        ).alias("discounted_cashflow_value")

    lf = lf.with_columns(
        project_dcf_expr(fc_period, longterm_growth)
    )

    return lf


def _dcf_batch_calc(df: pl.DataFrame, fc_period: int, longterm_growth: float) -> pl.Series:
    x = np.arange(1, fc_period + 1)
    values = []

    for i in range(len(df)):
        fcf = df["fcf"][i]
        wacc = df["wacc"][i]
        g = df["fcf_growth"][i]

        if wacc is None or fcf is None or g is None or np.isnan(wacc):
            values.append(np.nan)
            continue

        # Smooth projected growth curve
        growth_projection = longterm_growth + (g - longterm_growth) / (
            1 + np.exp(np.sign(g - longterm_growth) * (x - fc_period / 2))
        )

        # Project cash flows and discount
        discounted_sum = 0.0
        projected_cf = fcf
        for j, growth in enumerate(growth_projection, start=1):
            projected_cf = projected_cf * (1 + growth)
            discounted_sum += projected_cf / ((1 + wacc) ** j)

        # Add terminal value
        if wacc > longterm_growth:
            terminal_value = abs(projected_cf) * (1 + longterm_growth) / (wacc - longterm_growth)
        else:
            terminal_value = 0.0  # fallback

        terminal_value /= (1 + wacc) ** fc_period
        total = discounted_sum + terminal_value

        values.append(total)

    return pl.Series(name="discounted_cashflow_value", values=values)

Claude
import polars as pl
import numpy as np
from typing import Optional


def discounted_cash_flow(
    lf: pl.LazyFrame, 
    fc_period: int = 20, 
    longterm_growth: float = 0.03,
    min_wacc_threshold: float = 0.001  # Prevent division by zero/negative WACC
) -> pl.LazyFrame:
    
    # Required columns for validation
    required_cols = [
        "weighted_average_cost_of_capital",
        "free_cashflow_firm", 
        "liquid_assets",
        "debt",
        "split_adjusted_weighted_average_shares_outstanding_basic"
    ]
    
    # Collect data once and validate
    try:
        df = lf.select(required_cols + ["enterprise_value"]).collect()
    except pl.ColumnNotFoundError as e:
        raise ValueError(f"Missing required column: {e}")
    
    # Early return if no valid WACC data
    wacc_series = df["weighted_average_cost_of_capital"]
    if wacc_series.null_count() == len(wacc_series):
        return lf.with_columns(pl.lit(None).alias("discounted_cashflow_value"))
    
    # Calculate smoothed metrics
    wacc_ema = _calculate_ema_with_fallback(wacc_series)
    fcf_series = df["free_cashflow_firm"]
    fcf_growth_ema = _calculate_fcf_growth_ema(fcf_series)
    
    # Vectorized DCF calculation
    dcf_values = _calculate_dcf_vectorized(
        fcf_series, fcf_growth_ema, wacc_ema, 
        fc_period, longterm_growth, min_wacc_threshold,
        df["enterprise_value"]
    )
    
    # Adjust for net liquid assets and calculate per-share value
    liquid_assets = df["liquid_assets"].fill_null(0)
    debt = df["debt"].fill_null(0)
    shares = df["split_adjusted_weighted_average_shares_outstanding_basic"]
    
    net_liquid_assets = liquid_assets - debt
    enterprise_values = dcf_values + net_liquid_assets
    
    # Handle division by zero for shares
    dcf_per_share = np.where(
        (shares.is_null()) | (shares == 0) | np.isnan(shares),
        np.nan,
        enterprise_values / shares
    )
    
    return lf.with_columns(
        pl.Series("discounted_cashflow_value", dcf_per_share)
        .alias("discounted_cashflow_value")  # Fixed typo
    )


def _calculate_ema_with_fallback(series: pl.Series, default_span: int = 10) -> np.ndarray:
    
    try:
        valid_data = series.drop_nulls()
        if len(valid_data) == 0:
            return np.full(len(series), np.nan)
        
        span = min(len(valid_data), default_span)
        return series.ewm_mean(span=span, adjust=False).to_numpy()
    except Exception:
        return np.full(len(series), np.nan)


def _calculate_fcf_growth_ema(fcf_series: pl.Series) -> np.ndarray:

    try:
        # Calculate rate of change
        fcf_diff = fcf_series.diff()
        fcf_prev = fcf_series.shift(1)
        
        # Avoid division by zero
        fcf_roc = np.where(
            (fcf_prev.is_null()) | (fcf_prev == 0) | fcf_prev.is_nan(),
            np.nan,
            fcf_diff / fcf_prev.abs()
        )
        
        fcf_roc_series = pl.Series(fcf_roc).drop_nulls()
        if len(fcf_roc_series) == 0:
            return np.full(len(fcf_series), np.nan)
        
        span = min(len(fcf_roc_series), len(fcf_series))
        return pl.Series(fcf_roc).ewm_mean(span=span, adjust=False).to_numpy()
        
    except Exception:
        return np.full(len(fcf_series), np.nan)


def _calculate_dcf_vectorized(
    fcf: pl.Series, 
    fcf_growth: np.ndarray, 
    wacc: np.ndarray,
    fc_period: int, 
    longterm_growth: float,
    min_wacc_threshold: float,
    enterprise_values: pl.Series
) -> np.ndarray:
    
    n = len(fcf)
    dcf_values = np.zeros(n)
    
    # Pre-calculate time periods for efficiency
    periods = np.arange(1, fc_period + 1)
    midpoint = fc_period / 2
    
    for i in range(n):
        current_fcf = fcf[i]
        current_growth = fcf_growth[i]
        current_wacc = wacc[i]
        
        # Skip invalid entries
        if (pd.isna(current_fcf) or pd.isna(current_growth) or 
            pd.isna(current_wacc) or current_wacc < min_wacc_threshold):
            dcf_values[i] = np.nan
            continue
        
        # Calculate sigmoid-based growth projection
        sigmoid_factor = np.sign(current_growth - longterm_growth) * (periods - midpoint)
        growth_projection = longterm_growth + (
            (current_growth - longterm_growth) / (1 + np.exp(sigmoid_factor))
        )
        
        # Project cash flows and discount to present value
        projected_cf = current_fcf
        present_values = []
        
        for period, growth_rate in zip(periods, growth_projection):
            projected_cf *= (1 + growth_rate)
            present_value = projected_cf / ((1 + current_wacc) ** period)
            present_values.append(present_value)
        
        dcf_values[i] = sum(present_values)
        
        # Calculate terminal value
        if current_wacc > longterm_growth:
            # Gordon growth model
            final_cf = present_values[-1] * (1 + current_wacc) ** fc_period
            terminal_value = (
                final_cf * (1 + longterm_growth) / 
                (current_wacc - longterm_growth)
            )
        else:
            # Fallback to enterprise value based terminal
            ev = enterprise_values[i]
            if not pd.isna(ev):
                terminal_value = ev * ((1 + longterm_growth) ** fc_period)
            else:
                terminal_value = 0
        
        # Discount terminal value to present
        terminal_pv = terminal_value / ((1 + current_wacc) ** fc_period)
        dcf_values[i] += terminal_pv
    
    return dcf_values
"""
