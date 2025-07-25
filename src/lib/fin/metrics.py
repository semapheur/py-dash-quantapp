import polars as pl

from lib.fin.calculation import fin_filters


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
