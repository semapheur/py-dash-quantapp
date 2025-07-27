import ast
from typing import Literal, cast

import polars as pl

from lib.db.lite import polars_from_sqlite_filter
from lib.fin.taxonomy import TaxonomyCalculation
from lib.utils.dataframe import df_time_difference


class AllTransformer(ast.NodeTransformer):
  def __init__(self) -> None:
    self.names: set[str] = set()

  def reset_names(self):
    self.names = set()

  def visit_Name(self, node):
    self.names.add(node.id)

    return ast.Call(
      func=ast.Attribute(
        value=ast.Name(id="pl", ctx=ast.Load()),
        attr="col",
        ctx=ast.Load(),
      ),
      args=[ast.Constant(value=node.id)],
      keywords=[],
    )


class AnyTransformer(ast.NodeTransformer):
  def __init__(self) -> None:
    self.df_columns: set[str] = set()
    self.names: set[str] = set()

  def reset_names(self):
    self.names = set()

  def set_columns(self, columns: set[str]):
    self.df_columns = columns

  def visit_Name(self, node):
    if node.id not in self.df_columns:
      return ast.Call(
        func=ast.Attribute(
          value=ast.Name(id="pl", ctx=ast.Load()),
          attr="lit",
          ctx=ast.Load(),
        ),
        args=[ast.Constant(value=0.0)],
        keywords=[],
      )

    self.names.add(node.id)
    return ast.Call(
      func=ast.Attribute(
        value=ast.Call(
          func=ast.Attribute(
            value=ast.Name(id="pl", ctx=ast.Load()), attr="col", ctx=ast.Load()
          ),
          args=[ast.Constant(value=node.id)],
        ),
        attr="fill_nan",
        ctx=ast.Load(),
      ),
      args=[ast.Constant(value=None)],
    )


def fin_filters() -> list[tuple[pl.Expr, pl.Expr, Literal[3, 12]]]:
  periods = ["FY", "TTM1", "TTM2", "TTM3"]

  filters: list[tuple[pl.Expr, pl.Expr, Literal[3, 12]]] = []
  for p in periods:
    filters.append((pl.col("period") == p, pl.col("months") == 12, 12))

  filters.append((pl.lit(True), pl.col("months") == 3, 3))

  return filters


def fiscal_days() -> pl.Expr:
  is_leap = (
    (pl.col("date").dt.year() % 4 == 0) & (pl.col("date").dt.year() % 100 != 0)
  ) | (pl.col("date").dt.year() % 400 == 0).cast(pl.Float64)
  fy_days = 365.0 + is_leap

  qtr_days = (
    pl.when(pl.col("period") == "Q1")
    .then(90.0)
    .when(pl.col("period") == "Q2")
    .then(91.0)
    .when(pl.col("period") == "Q3")
    .then(92.0)
    .when(pl.col("period") == "Q4")
    .then(92.0)
    .otherwise(91.0)
  ) + is_leap
  return (
    pl.when(pl.col("months") == 12)
    .then(fy_days)
    .when(pl.col("months") == 3)
    .then(qtr_days)
    .otherwise(pl.col("months").cast(pl.Float64) * 30.0)
  )


def trailing_twelve_months(lf: pl.LazyFrame) -> pl.LazyFrame:
  lf_q = lf.filter(pl.col("months") == 3).sort("date")

  lf_q = lf_q.with_columns(df_time_difference("date", 30, "D").alias("month_diff"))

  lf_q = lf_q.with_columns(
    pl.col("month_diff")
    .rolling_sum(window_size=3, min_samples=3)
    .alias("month_diff_sum")
  )

  sum_items_df = polars_from_sqlite_filter(
    db_name="taxonomy",
    table="items",
    match_column="item",
    filter_values=lf_q.collect_schema().names(),
    select_columns=["items"],
    where_clause="aggregate = 'sum'",
  )
  sum_items = sum_items_df["item"].to_list()

  for col in sum_items:
    lf_q = lf_q.with_columns(
      pl.col(col).rolling_sum(window_size=4, min_samples=4).alias(col)
    )

  lf_q = lf_q.filter(pl.col("month_diff_sum") == 9)
  lf_q = lf_q.filter(pl.col("period") != "Q4")

  lf_q = lf_q.with_columns(
    [
      pl.col("period").str.replace("Q", "TTM").alias("period"),
      pl.lit(12).alias("months"),
    ]
  )
  lf_q.drop(["month_diff", "month_diff_sum"])

  return pl.concat([lf, lf_q], how="vertical")


def tail_trailing_twelve_months(lf: pl.LazyFrame) -> pl.LazyFrame:
  lf = lf.sort("date")
  last_date = lf.select("date").max().unique().collect().item()

  last_fy_date = (
    lf.filter((pl.col("period") == "FY") & (pl.col("months") == 12))
    .select("date")
    .max()
    .collect()
    .item()
  )

  if last_fy_date == last_date:
    ttm = (
      lf.filter((pl.col("period") == "FY") & (pl.col("months") == 12))
      .sort("date")
      .tail(2)
    )
  else:
    columns = lf.collect_schema().names()
    sum_items_df = polars_from_sqlite_filter(
      db_name="taxonomy",
      table="items",
      match_column="item",
      filter_values=columns,
      select_columns=["item"],
      where_clause="aggregate = 'sum'",
    )
    sum_items = sum_items_df["item"].to_list()

    trail = (
      lf.filter(pl.col("months") == 3).select(["date"] + sum_items).sort("date").tail(8)
    )
    trail_df = trail.collect()
    head4 = trail_df.head(4).select(sum_items).sum()
    tail4 = trail_df.tail(4).select(sum_items).sum()

    ttm_sum = pl.concat(
      [
        pl.DataFrame(head4.to_dict(as_series=False)).with_columns(
          [
            pl.lit("TTM3").alias("period"),
            pl.lit(12).alias("months"),
            pl.lit(trail_df["date"][3]).alias("date"),
          ]
        ),
        pl.DataFrame(tail4.to_dict(as_series=False)).with_columns(
          [
            pl.lit("TTM4").alias("period"),
            pl.lit(12).alias("months"),
            pl.lit(trail_df["date"][7]).alias("date"),
          ]
        ),
      ]
    ).lazy()

    rest_cols = list(set(columns).difference(sum_items))
    trail_rest = (
      lf.filter(pl.col("months") == 3).select(rest_cols).sort("date").tail(5).collect()
    )
    row1 = trail_rest[-5]
    row2 = trail_rest[-1]
    rest_df = pl.DataFrame([row1, row2]).lazy()
    ttm = ttm_sum.join(rest_df, on=["date", "period", "months"], how="inner")

  return pl.concat([lf, ttm], how="vertical")


def update_trailing_twelve_months(lf: pl.LazyFrame, new_price: float) -> pl.LazyFrame:
  lf_ttm = lf.filter((pl.col("period") == "TTM") & (pl.col("months") == 12)).sort(
    "date"
  )
  last_ttm_date = lf_ttm.select("date").max().collect().item()

  ttm_filter = (
    (pl.col("date") == last_ttm_date)
    & (pl.col("period") == "TTM")
    & (pl.col("months") == 12)
  )

  old_price = lf.filter(ttm_filter).select("share_price_close").collect().item()

  columns = lf.collect_schema().names()
  items_df = polars_from_sqlite_filter(
    db_name="taxonomy",
    table="items",
    match_column="item",
    filter_values=columns,
    select_columns=["items"],
    where_clause="unit = 'price_ratio'",
  )
  items = items_df["item"].to_list() + ["market_capitalization"]
  price_ratio = old_price / new_price

  lf = lf.with_columns(
    pl.when(ttm_filter)
    .then(pl.lit(new_price))
    .otherwise(pl.col("share_price_close"))
    .alias("share_price_close")
  )

  for item in items:
    lf = lf.with_columns(
      pl.when(ttm_filter)
      .then(pl.col(item) * price_ratio)
      .otherwise(pl.col(item))
      .alias(item)
    )

  return lf


def get_days(
  lf: pl.LazyFrame, filters: list[tuple[pl.Expr, pl.Expr, Literal[3, 12]]]
) -> pl.LazyFrame:
  lf = lf.with_columns(fiscal_days().alias("days"))
  all_updates: list[pl.LazyFrame] = []

  for period_filter, months_filter, month_diff in filters:
    sub = lf.filter(period_filter & months_filter).sort("date")

    if isinstance(period_filter, pl.Expr) and "FY" in str(period_filter):
      sub = sub.filter(
        ~((pl.col("date").dt.month() == 12) & (pl.col("date").dt.day() == 31))
      )

    sub = sub.with_columns(df_time_difference("date", 30, "D").alias("month_diff"))
    sub = sub.filter(pl.col("month_diff") == month_diff)
    sub = sub.with_columns(
      pl.col("date").diff().dt.days().cast(pl.Float64).alias("days")
    )

    all_updates.append(sub.select(["date", "period", "months", "days"]))

  if not all_updates:
    return lf

  updates = pl.concat(all_updates)
  return lf.join(
    updates, on=["date", "period", "months"], how="left", suffix="_updated"
  ).with_columns(pl.coalesce([pl.col("days_updated"), pl.col("days")]).alias("days"))


def calculate_stock_splits(lf: pl.LazyFrame) -> pl.LazyFrame:
  basic = "weighted_average_shares_outstanding_basic"
  shifted = "weighted_average_shares_outstanding_shift"

  lf = lf.filter(pl.col("months") == 3).sort("date")
  lf = lf.with_columns(pl.col(basic).shift(1).alias(shifted))
  lf = lf.with_columns(
    pl.when(pl.col(basic) >= pl.col(shifted))
    .then("split")
    .otherwise("reverse_split")
    .alias("split_type")
  )
  lf = lf.with_columns(
    (
      pl.max_horizontal(pl.col(basic), pl.col(shifted))
      / pl.min_horizontal(pl.col(basic), pl.col(shifted))
    )
    .round(0)
    .alias("stock_split_ratio")
  )
  lf = lf.filter(pl.col("split_type") > 1)
  lf = lf.with_columns(
    pl.when(pl.col("split_type") == "reverse")
    .then(1.0 / pl.col("stock_split_ratio"))
    .otherwise(pl.col("stock_split_ratio"))
    .alias("stock_split_ratio")
  )
  return lf.select("stock_split_ratio")


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

  updates = pl.concat(result_parts, how="vertical")

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


def lower_bound(
  lf: pl.LazyFrame, lf_cols: set[str], value: str | float, calculee: str
) -> pl.LazyFrame:
  if isinstance(value, float):
    return lf.with_columns(
      pl.when(pl.col(calculee) < value)
      .then(value)
      .otherwise(pl.col(calculee))
      .alias(calculee)
    )

  elif value in lf_cols:
    return lf.with_columns(
      pl.when(pl.col(calculee) < pl.col(value))
      .then(pl.col(value))
      .otherwise(pl.col(calculee))
      .alias(calculee)
    )

  return lf


def upper_bound(
  lf: pl.LazyFrame, lf_cols: set[str], value: str | float, calculee: str
) -> pl.LazyFrame:
  if isinstance(value, float):
    return lf.with_columns(
      pl.when(pl.col(calculee) > value)
      .then(value)
      .otherwise(pl.col(calculee))
      .alias(calculee)
    )

  elif value in lf_cols:
    return lf.with_columns(
      pl.when(pl.col(calculee) > pl.col(value))
      .then(pl.col(value))
      .otherwise(pl.col(calculee))
      .alias(calculee)
    )

  return lf


def calculate_items(
  financials: pl.LazyFrame,
  schemas: dict[str, TaxonomyCalculation],
  recalc: bool = False,
) -> pl.LazyFrame:
  def insert_to_lf(
    lf: pl.LazyFrame, lf_cols: set[str], insert_data: pl.Expr, insert_name: str
  ) -> pl.LazyFrame:
    if insert_name in lf_cols:
      if recalc:
        lf = lf.with_columns(insert_data.alias(insert_name))
      else:
        lf = lf.with_columns(
          pl.coalesce(
            [pl.when(pl.col(insert_name) != 0.0).then(pl.col(insert_name)), insert_data]
          ).alias(insert_name)
        )

    else:
      lf = lf.with_columns(insert_data.alias(insert_name))
      lf_cols.add(insert_name)

    return lf

  def apply_formula(
    lf: pl.LazyFrame, lf_cols: set[str], col_name: str, expression: ast.Expression
  ) -> pl.LazyFrame:
    code = compile(expression, "<string>", "eval")
    result = eval(code)

    if isinstance(result, pl.Expr):
      lf = insert_to_lf(lf, lf_cols, result, col_name)

    return lf

  def handle_all_formulas(
    lf: pl.LazyFrame, lf_cols: set[str], col_name: str, formulas: list[str]
  ) -> pl.LazyFrame:
    all_visitor = AllTransformer()

    for formula in formulas:
      all_visitor.reset_names()
      expression = ast.parse(formula, mode="eval")
      expression = cast(
        ast.Expression, ast.fix_missing_locations(all_visitor.visit(expression))
      )

      if all_visitor.names.issubset(lf_cols):
        lf = apply_formula(lf, lf_cols, col_name, expression)

    return lf

  def handle_any_formulas(
    lf: pl.LazyFrame, lf_cols: set[str], col_name: str, formulas: list[str]
  ) -> pl.LazyFrame:
    any_visitor = AnyTransformer()

    for formula in formulas:
      any_visitor.reset_names()
      any_visitor.set_columns(lf_cols)
      expression = ast.parse(formula, mode="eval")
      expression = cast(
        ast.Expression, ast.fix_missing_locations(any_visitor.visit(expression))
      )

      if any_visitor.names:
        lf = apply_formula(financials, lf_cols, col_name, expression)
        break

    return lf

  slices = fin_filters()

  financials = financials.sort("date")
  financials = get_days(financials, slices)
  lf_cols = set(financials.collect_schema().names())

  for calculee, schema in schemas.items():
    lf_cols.update(financials.collect_schema().names())

    fns = cast(
      set[Literal["avg", "diff", "shift"]],
      {"avg", "diff", "shift"}.intersection(set(schema.keys())),
    )
    calculated = False

    for fn in fns:
      calculer = cast(str, schema[fn])
      if calculer not in lf_cols:
        continue

      if fn == "avg":
        result_expr = pl.col(calculer).mean()
      elif fn == "diff":
        result_expr = pl.col(calculer).diff()
      elif fn == "shift":
        result_expr = pl.col(calculer).shift(1)

      financials = insert_to_lf(financials, lf_cols, result_expr, calculee)
      calculated = True

    if calculated:
      continue

    if (formula := schema.get("all")) is not None:
      financials = handle_all_formulas(financials, lf_cols, calculee, formula)

    if (formula := schema.get("any")) is not None:
      financials = handle_any_formulas(financials, lf_cols, calculee, formula)

    if (value := schema.get("min")) is not None:
      financials = lower_bound(financials, lf_cols, value, calculee)

    if (value := schema.get("max")) is not None:
      financials = upper_bound(financials, lf_cols, value, calculee)

  return financials
