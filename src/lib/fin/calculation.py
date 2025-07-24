from typing import Literal

import polars as pl

from lib.utils.dataframe import df_time_difference


def fin_filters() -> list[tuple[pl.Expr, pl.Expr, Literal[3, 12]]]:
  periods = ["FY", "TTM1", "TTM2", "TTM3"]

  filters: list[tuple[pl.Expr, pl.Expr, Literal[3, 12]]] = []
  for p in periods:
    filters.append((pl.col("period") == p, pl.col("months") == 12, 12))

  filters.append((pl.lit(True), pl.col("months") == 3, 3))

  return filters


def applier(
  lf: pl.LazyFrame,
  column: str,
  fn: Literal["avg", "diff", "shift"],
  slices: list[tuple[pl.Expr, pl.Expr, Literal[3, 12]]],
) -> pl.LazyFrame:
  result_parts = []

  month_diff_expr = df_time_difference("date", 30, "D")

  for period_filter, months_filter, month_diff in slices:
    lf_transformed = lf.filter(period_filter & months_filter).sort("date")

    lf_transformed = lf_transformed.with_columns(month_diff_expr.alias("month_diff"))

    if fn == "diff":
      lf_transformed = lf_transformed.with_columns(
        pl.col(column).diff().alias(f"{column}_transformed")
      )
    elif fn == "avg":
      lf_transformed = lf_transformed.with_columns(
        pl.col(column)
        .rolling_mean(window_size=2, min_samples=2)
        .alias(f"{column}_transformed")
      )
    elif fn == "shift":
      lf_transformed = lf_transformed.with_columns(
        pl.col(column).shift(1).alias(f"{column}_transformed")
      )

    lf_transformed = lf_transformed.filter(pl.col("month_diff") == month_diff)
    result_parts.append(
      lf_transformed.select(["date", "period", "months", f"{column}_transformed"])
    )

  if not result_parts:
    return lf

  updates = pl.concat(result_parts)

  return (
    lf.join(
      updates,
      on=["date", "period", "months"],
      how="left",
    )
    .with_columns(
      pl.when(pl.col(f"{column}_transformed").is_not_null())
      .then(pl.col(f"{column}_transformed"))
      .otherwise(pl.col(column))
      .alias(column)
    )
    .drop(f"{column}_transformed")
  )
