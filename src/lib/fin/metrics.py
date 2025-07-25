from datetime import date as Date
from dateutil.relativedelta import relativedelta
from typing import Literal, cast, TypedDict

import numpy as np
import polars as pl
import statsmodels.api as sm

from lib.fin.calculation import fin_filters


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


def f_score(lf: pl.LazyFrame) -> pl.LazyFrame:
  def diff_col(col: str) -> pl.Expr:
    exprs = [
      pl.col(col).diff().over(["period", "months"]).filter(per & mon)
      for per, mon, _ in slices
    ]
    return pl.coalesce(*exprs)

  slices = fin_filters()

  return lf.with_columns(
    (
      (pl.col("return_on_equity") > 0).cast(pl.Int8)
      + (diff_col("return_on_assets") > 0).cast(pl.Int8)
      + (pl.col("cashflow_operating") > 0).cast(pl.Int8)
      + (
        (pl.col("cashflow_operating") / pl.col("assets")) > pl.col("return_on_assets")
      ).cast(pl.Int8)
      + (-diff_col("debt") > 0).cast(pl.Int8)
      + (diff_col("quick_ratio") > 0).cast(pl.Int8)
      + (-diff_col("weighted_average_shares_outstanding_basic") > 0).cast(pl.Int8)
      + (diff_col("operating_profit_margin") > 0).cast(pl.Int8)
      + (diff_col("asset_turnover") > 0).cast(pl.Int8)
    ).alias("piotroski_f_score")
  )


def z_score(lf: pl.LazyFrame) -> pl.LazyFrame:
  assets = pl.col("average_assets")
  liabilities = pl.col("average_liabilities")

  return lf.with_columns(
    (
      1.2 * pl.col("average_operating_working_capital") / assets
      + 1.4 * pl.col("retained_earnings_accumulated_deficit") / assets
      + 3.3 * pl.col("cashflow_operating") / assets
      + 0.6 * pl.col("market_capitalization") / liabilities
      + assets / liabilities
    ).alias("altman_z_score")
  )


def m_score(lf: pl.LazyFrame) -> pl.LazyFrame:
  receivables_col = next(
    (
      col
      for col in [
        "average_receivables_trade_current",
        "average_receivables_trade",
        "average_receivables",
      ]
      if col in lf.collect_schema().names()
    ),
    None,
  )
  dsri = (
    pl.when(receivables_col is not None)
    .then(
      (pl.col(receivables_col) / pl.col("revenue"))
      / (pl.col(receivables_col).shift(1) / pl.col("revenue").shift(1))
    )
    .otherwise(0.0)
    .alias("dsri")
  )
  gmi = (
    pl.col("operating_profit_margin").shift(1) / pl.col("operating_profit_margin")
  ).alias("gmi")

  aqi_val = 1 - (
    pl.col("operating_working_capital")
    + pl.col("productive_assets")
    + pl.col("financial_assets_noncurrent")
  ) / pl.col("assets")
  aqi = (aqi_val / aqi_val.shift(1)).alias("aqi")

  sgi = pl.col("revenue") / pl.col("revenue").shift(1).alias("sgi")

  productive_minus_depreciation = pl.col("productive_assets") - pl.col("depreciation")
  depi = (pl.col("depreciation") / productive_minus_depreciation) / (
    pl.col("depreciation").shift(1) / productive_minus_depreciation.shift(1)
  ).alias("depi")

  li_val = pl.col("liabilities") / pl.col("assets")
  li = (li_val / li_val.shift(1)).alias("li")

  tata = (
    (pl.col("income_loss_operating") - pl.col("cashflow_operating"))
    / pl.col("average_assets")
  ).alias("tata")
  beneish = (
    -4.84
    + 0.92 * pl.col("dsri")
    + 0.528 * pl.col("gmi")
    + 0.404 * pl.col("aqi")
    + 0.892 * pl.col("sgi")
    + 0.115 * pl.col("depi")
    - 0.172 * pl.col("sgai")
    + 4.679 * pl.col("tata")
    - 0.327 * pl.col("li")
  ).alias("beneish")

  return (
    lf.with_columns([dsri, gmi, aqi, sgi, depi, li, tata])
    .with_columns(beneish)
    .drop(["dsri", "gmi", "aqi", "sgi", "depi", "li", "tata"])
  )


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
  equity_return: pl.DataFrame,
  market_return: pl.DataFrame,
  riskfree_rate: pl.DataFrame,
  years: int | None = 0,
) -> pl.LazyFrame:
  returns_df = (
    equity_return.join(market_return, on="date", how="outer")
    .join(riskfree_rate, on="date", how="outer")
    .sort("date")
    .fill_null(strategy="forward")
    .drop_nulls()
  )
  slices = fin_filters()

  beta_frames: list[pl.LazyFrame] = []
  for period, _, months in slices:
    dates = (
      financials.filter(pl.col("period") == period)
      .filter(pl.col("months") == months)
      .select("date")
      .collect()
      .get_column("date")
      .to_list()
    )
    beta_frame = calculate_beta(dates, returns_df, months, years)
    beta_frames.append(beta_frame)

  betas = pl.concat(beta_frames)
  return financials.join(betas, on=["date", "months"], how="left")


def weighted_average_cost_of_capital(
  financials: pl.LazyFrame, debt_maturity: int = 10, large_cap_limit: float = 2e9
) -> pl.LazyFrame:
  financials = apply_yield_spread(financials, large_cap_limit)

  financials = financials.with_columns(
    [
      # Levered beta
      (
        pl.col("beta")
        * (
          1
          + (1 - pl.col("tax_rate")) * pl.col("average_debt") / pl.col("average_equity")
        )
      ).alias("beta_levered"),
      # Cost of equity
      (
        pl.col("beta_levered") * (pl.col("market_return") - pl.col("riskfree_rate"))
      ).alias("equity_risk_premium"),
      (pl.col("riskfree_rate") + pl.col("equity_risk_premium")).alias("cost_equity"),
      # Cost of debt
      (pl.col("riskfree_rate") + pl.col("yield_spread")).alias("cost_debt"),
      # Market value of debt
      (
        (pl.col("interest_expense") / pl.col("cost_debt"))
        * (1 - (1 / (1 + pl.col("cost_debt")) ** debt_maturity))
        + (pl.col("debt") / (1 + pl.col("cost_dbet")) ** debt_maturity)
      ).alias("market_value_debt"),
      # Capital weights
      (
        pl.col("market_capitalization")
        / (pl.col("market_capitalization") + pl.col("market_value_debt")).alias(
          "equity_to_capital"
        )
      ),
      # Weighted average cost of capital
      (
        pl.col("cost_equity") * pl.col("equity_to_capital")
        + pl.col("cost_debt")
        * (1 - pl.col("tax_rate"))
        * (
          pl.col("market_value_debt")
          / (pl.col("market_capitalization") + pl.col("market_value_debt"))
        )
      ).alias("weighted_average_cost_of_capital"),
    ]
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
      .then("small")
      .otherwise("large")
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
    .drop(["icr_clean", "lower", "upper"])
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
    for i in range(len(spreads)):
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
    [
      pl.when(pl.col("weighted_average_cost_of_capital").is_null())
      .then(None)
      .otherwise(pl.col("weighted_average_cost_of_capital"))
      .alias("wacc"),
    ]
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
    [(pl.col("revenue_diff") / pl.col("revenue_shifted")).alias("revenue_growth_raw")]
  )
  lf = lf.with_columns(
    [pl.col("revenue_growth_raw").ewm_mean(com=com_value).alias("revenue_growth")]
  )

  # Capex and margin
  lf = lf.with_columns(
    [
      pl.col("payment_acquisition_productive_assets")
      .ewm_mean(com=com_value)
      .alias("capex"),
    ]
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
    [
      (
        (pl.col("pretax_income_loss") + pl.col("interest_expense")) / pl.col("revenue")
      ).alias("operating_margin_raw")
    ]
  )
  lf = lf.with_columns(
    [pl.col("operating_margin_raw").ewm_mean(com=com_value).alias("operating_margin")]
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
    [
      (
        pl.col("sustainable_revenue")
        * pl.col("operating_margin")
        * (1 - pl.col("tax_rate"))
        + pl.col("adjusted_ddaa")
        - (pl.col("maintenance_capex") * pl.col("sustainable_revenue"))
      ).alias("adjusted_earnings")
    ]
  )

  # EPV
  lf = lf.with_columns(
    [
      (pl.col("adjusted_earnings") / pl.col("wacc_ewm")).alias("epv_raw"),
      (
        (pl.col("epv_raw") + pl.col("liquid_assets") - pl.col("debt"))
        / pl.col("weighted_average_shares_outstanding_basic")
      ).alias("earnings_power_value"),
    ]
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
