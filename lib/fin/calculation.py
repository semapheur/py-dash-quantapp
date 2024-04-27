import ast
from datetime import datetime as dt
from functools import cache
import json
from typing import cast, Literal

import numpy as np
import pandas as pd
from pandera.typing import DataFrame, Series

from lib.db.lite import read_sqlite
from lib.fin.models import FiscalPeriod
from lib.fin.taxonomy import TaxonomyCalculation, TaxonomyCalculationItem
from lib.utils import df_time_difference


class AllTransformer(ast.NodeTransformer):
  def __init__(self, df_name: str):
    self.df_name = df_name
    self.names: set[str] = set()

  def reset_names(self):
    self.names = set()

  def visit_Name(self, node):
    self.names.add(node.id)

    subscript = ast.Subscript(
      value=ast.Name(id=self.df_name, ctx=ast.Load()),
      slice=ast.Index(value=ast.Constant(value=node.id)),
      ctx=node.ctx,
    )
    return subscript


class AnyTransformer(ast.NodeTransformer):
  def __init__(self, df_name: str):
    self.df_name = df_name
    self.df_columns: set[str] = set()

  def set_columns(self, columns: set[str]):
    self.df_columns = columns

  def visit_Name(self, node):
    if node.id not in self.df_columns:
      return ast.Constant(value=0)

    call = ast.Call(
      func=ast.Attribute(
        value=ast.Subscript(
          value=ast.Name(id=self.df_name, ctx=ast.Load()),
          slice=ast.Constant(value=node.id),
          ctx=node.ctx,
        ),
        attr='fillna',
        ctx=ast.Load(),
      ),
      args=[ast.Constant(value=0)],
      keywords=[],
    )
    return call


def fin_slices(
  ix: pd.MultiIndex,
) -> list[tuple[slice, slice | str, Literal[3, 12]]]:
  slices: list[tuple[slice, slice | str, Literal[3, 12]]] = []
  periods = {'FY', 'TTM1', 'TTM2', 'TTM3'}.intersection(ix.levels[1])
  for p in periods:
    slices.append((slice(None), p, 12))

  if 3 in ix.levels[2]:
    slices.append((slice(None), slice(None), 3))

  return slices


def trailing_twelve_months(financials: DataFrame) -> DataFrame:
  financials.sort_index(level='date', inplace=True)

  mask = (slice(None), slice(None), 3)
  ttm = financials.loc[mask, :].copy()

  ttm['month_difference'] = df_time_difference(
    cast(pd.DatetimeIndex, ttm.index.get_level_values('date')), 30, 'D'
  )
  ttm.loc[:, 'month_difference'] = ttm['month_difference'].rolling(3, 3).sum()

  query = f"""SELECT item FROM items
    WHERE aggregate = 'sum' AND item IN {str(tuple(financials.columns))}
  """
  sum_items = read_sqlite('taxonomy.db', query)
  if sum_items is None:
    raise ValueError(
      f'Could not load taxonomy for the following items: {json.dumps(list(financials.columns), indent=2)}'
    )

  ttm.loc[:, sum_items['item']] = ttm[sum_items['item']].rolling(4, 4).sum()
  ttm = ttm.loc[ttm.loc[:, 'month_difference'] == 9]

  ttm.reset_index(level=('period', 'months'), inplace=True)
  ttm = ttm.loc[ttm['period'] != 'Q4']

  ttm.loc[:, 'period'] = ttm['period'].str.replace('Q', 'TTM')

  ttm.loc[:, 'months'] = 12
  ttm.set_index(['period', 'months'], append=True, inplace=True)
  ttm = ttm.reorder_levels(['date', 'period', 'months'])

  ttm.drop('month_difference', axis=1, inplace=True)

  return cast(DataFrame, pd.concat((financials, ttm), axis=0))


def tail_trailing_twelve_months(df: DataFrame) -> DataFrame:
  df.sort_index(level='date', inplace=True)

  last_date = cast(pd.MultiIndex, df.index).levels[0].max()
  mask = (slice(None), slice('FY'), 12)
  last_annual: dt = cast(pd.MultiIndex, df.loc[mask, :].index).levels[0].max()

  if last_annual == last_date:
    ttm = df.loc[(slice(None), 'FY', 12), :].tail(2)
  else:
    query = f"""SELECT item FROM items
      WHERE aggregate = 'sum' AND item IN {str(tuple(df.columns))}
    """
    items = read_sqlite('taxonomy.db', query)
    if items is None:
      raise ValueError('Could not load taxonomy!')

    sum_cols = items['item']

    mask = (slice(None), slice(None), 3)
    trail = df.loc[mask, sum_cols.to_list()].tail(8)
    sums = pd.concat((trail.head(4).sum(), trail.tail(4).sum()), axis=1).T
    rest_cols = list(set(df.columns).difference(sum_cols))
    rest = pd.concat(
      (df.loc[mask, rest_cols].iloc[-5], df.loc[mask, rest_cols].iloc[-1]), axis=1
    ).T

    sums.index = rest.index
    ttm = pd.concat((sums, rest), axis=1)

  return cast(DataFrame, pd.concat((df, ttm), axis=0))


def update_trailing_twelve_months(df: pd.DataFrame, new_price: float) -> pd.DataFrame:
  mask = (slice(None), 'TTM', 12)
  ttm_date = cast(pd.MultiIndex, df.loc[mask, :].tail(1).index).levels[0]
  ttm_mask = (ttm_date, 'TTM', 12)

  old_price = df.at[ttm_mask, 'share_price_close']
  df.loc[ttm_mask, 'share_price_close'] = new_price

  query = """
    SELECT item FROM items WHERE 
      value = 'price_fundamental' AND 
      item IN :columns
    UNION ALL
    SELECT 'market_capitalization' AS item
  """
  items = read_sqlite('taxonomy.db', query, {'columns': str(tuple(df.columns))})
  if items is None:
    raise ValueError(
      f'Could not load taxonomy items: {json.dumps(list(df.columns), indent=2)}'
    )

  df.loc[:, items['item'].to_list()] *= old_price / new_price

  return df


@cache
def fiscal_days(year: int, period: FiscalPeriod, months: int) -> float:
  from calendar import isleap

  leap = isleap(year)

  if months == 12:
    return 365.0 + leap

  quarter_days = {'Q1': 90.0, 'Q2': 91.0, 'Q3': 92.0, 'Q4': 92.0}
  return quarter_days.get(period, 91.0) + leap if months == 3 else months * 30.0


def get_days(
  ix: pd.MultiIndex,
  slices: list[tuple[slice, slice | str, Literal[3, 12]]],
):
  days = pd.DataFrame(
    np.array(
      [fiscal_days(cast(pd.Timestamp, i[0]).year, i[1], i[2]) for i in ix]
    ).astype(np.float64),
    index=ix,
    columns=['days'],
  )

  for s in slices:
    days_ = days.loc[s, :].copy()
    dates = cast(pd.DatetimeIndex, days_.index.get_level_values('date'))
    if s[1] == 'FY':
      mask = (dates.month == 12) & (dates.day == 31)
      days_ = days_.loc[~mask, :]

    days_['month_difference'] = df_time_difference(
      cast(pd.DatetimeIndex, days_.index.get_level_values('date')), 30, 'D'
    )
    mask_ = days_['month_difference'] == s[2]
    days_.loc[mask_, 'days'] = df_time_difference(
      cast(pd.DatetimeIndex, days_.loc[mask_, 'days'].index.get_level_values('date')),
      1,
      'D',
    )
    days_.dropna(inplace=True)

    days.loc[days_.index, 'days'] = days_.loc[days_.index, 'days']

  return days['days']


def calculate_stock_splits(df: pd.DataFrame) -> pd.Series:
  cols = [
    'weighted_average_shares_outstanding_basic',
    'weighted_average_shares_outstanding_basic_shift',
  ]

  df_ = cast(pd.DataFrame, df.xs(3, level='months'))
  df_.reset_index(level='period', drop=True, inplace=True)
  df_.sort_index(level='date', inplace=True)

  df_.loc[:, cols[1]] = df_[cols[0]].shift()
  df_ = df_[cols]

  df_.loc[:, 'action'] = 'split'
  mask = df_[cols[0]] >= df_[cols[1]]
  df_.loc[:, 'action'] = df_.where(mask, 'reverse')

  df_.loc[:, 'stock_split_ratio'] = np.round(
    df_[cols].max(axis=1) / df_[cols].min(axis=1)
  )
  df_ = df_.loc[df_['stock_split_ratio'] > 1]

  mask = df_['action'] == 'reverse'
  df_.loc[mask, 'stock_split_ratio'] = 1 / df_.loc[mask, 'split_ratio']

  return df_['stock_split_ratio']


def applier(
  s: Series[float],
  fn: Literal['avg', 'diff', 'shift'],
  slices: list[tuple[slice, slice | str, Literal[3, 12]]],
) -> pd.Series:
  result = s.copy()
  update: list[Series[float]] = []

  for ix in slices:
    s_ = cast(pd.Series, result.to_frame().loc[ix, :]).copy()
    s_.sort_index(level='date', inplace=True)

    dates = pd.to_datetime(s_.index.get_level_values('date'))
    month_diff = df_time_difference(dates, 30, 'D')
    month_diff.index = s_.index
    # month_diff = pd.Series(df_time_difference(dates, 30, 'D'), index=s_.index)

    if fn == 'diff':
      s_ = s_.diff()
    elif fn == 'avg':
      s_ = s_.rolling(window=2, min_periods=2).mean()
    elif fn == 'shift':
      s_ = s_.shift()

    s_ = s_.loc[month_diff == ix[2]]

    if not s_.empty:
      update.append(cast(Series[float], s_.loc[:, s_.columns[0]]))

  if not update:
    return result

  update_ = pd.concat(update, axis=0)
  nan_index = pd.Index(list(set(result.index).difference(update_.index)))

  result.loc[update_.index] = update_
  result.loc[nan_index] = np.nan

  return result


def lower_bound(
  df: DataFrame, df_cols: set[str], value: str | float, calculee: str
) -> DataFrame:
  if isinstance(value, float):
    df.loc[df[calculee] < value, calculee] = value

  elif value in df_cols:
    df.loc[df[calculee] < df[value], calculee] = df[value]

  return df


def upper_bound(
  df: DataFrame, df_cols: set[str], value: str | float, calculee: str
) -> DataFrame:
  if isinstance(value, float):
    df.loc[df[calculee] > value, calculee] = value

  elif value not in df_cols:
    return df

  df.loc[df[calculee] > df[value], calculee] = df[value]

  return df


def calculate_items(
  financials: DataFrame, schemas: dict[str, TaxonomyCalculation], recalc: bool = False
) -> DataFrame:
  def insert_to_df(
    df: DataFrame, df_cols: set[str], insert_data: pd.Series, insert_name: str
  ) -> DataFrame:
    if insert_name in df_cols:
      df.loc[:, insert_name] = (
        insert_data if recalc else df[insert_name].combine_first(insert_data)
      )
    else:
      new_column = pd.DataFrame({insert_name: insert_data})
      df = cast(DataFrame, pd.concat([df, new_column], axis=1))

    return df

  def apply_formula(
    df: DataFrame,
    df_cols: set[str],
    col_name: str,
    expression: ast.Expression | dict[str, TaxonomyCalculationItem],
  ) -> DataFrame:
    if isinstance(expression, ast.Expression):
      code = compile(expression, '<string>', 'eval')
      result = eval(code)

      if isinstance(result, int):
        return df

    elif isinstance(expression, dict):
      result = pd.Series(0, index=df.index, dtype=float)

      for key, value in expression.items():
        if key not in df_cols:
          continue

        if isinstance(value, int):
          result += df[key] * value

        elif isinstance(value, dict):
          weight = value.get('weight', 1)
          sign = value.get('sign')

          if sign is not None:
            mask = df[key].apply(np.sign) == sign
            result += df[key] * weight * mask
          else:
            result += df[key] * weight

    df = insert_to_df(df, df_cols, result, col_name)
    return df

  schemas = dict(sorted(schemas.items(), key=lambda x: x[1]['order']))
  slices = fin_slices(cast(pd.MultiIndex, financials.index))
  financials.sort_index(level='date', inplace=True)
  financials['days'] = get_days(cast(pd.MultiIndex, financials.index), slices)

  all_visitor = AllTransformer('df')
  any_visitor = AnyTransformer('df')
  col_set = set(financials.columns)

  for calculee, schema in schemas.items():
    col_set = col_set.union(financials.columns)

    fns = cast(
      set[Literal['avg', 'diff', 'shift']],
      {'avg', 'diff', 'shift'}.intersection(schema.keys()),
    )
    calculated = False
    for fn in fns:
      calculer = cast(str, schema[fn])
      if calculer not in col_set:
        continue

      result = applier(cast(Series[float], financials[calculer]), fn, slices)
      financials = insert_to_df(financials, col_set, result, calculee)
      calculated = True

    if calculated:
      continue

    if (formula := schema.get('all')) is not None:
      if isinstance(formula, str):
        all_visitor.reset_names()
        expression = ast.parse(formula, mode='eval')
        expression = cast(
          ast.Expression, ast.fix_missing_locations(all_visitor.visit(expression))
        )

        if all_visitor.names.issubset(col_set):
          financials = apply_formula(financials, col_set, calculee, expression)

      elif isinstance(formula, dict):
        if set(formula.keys()).issubset(col_set):
          financials = apply_formula(financials, col_set, calculee, formula)

    if (formula := schema.get('any')) is not None:
      if isinstance(formula, str):
        any_visitor.set_columns(col_set)
        try:
          expression = ast.parse(formula, mode='eval')
          expression = cast(
            ast.Expression, ast.fix_missing_locations(any_visitor.visit(expression))
          )
          financials = apply_formula(financials, col_set, calculee, expression)
        except Exception as _:
          print(formula)
      elif isinstance(formula, dict):
        formula = {
          key: formula[key] for key in set(formula.keys()).intersection(col_set)
        }
        if formula:
          financials = apply_formula(financials, col_set, calculee, formula)

    if (filler := schema.get('fill')) is not None and filler in col_set:
      financials = insert_to_df(financials, col_set, financials[filler], calculee)

    if (value := schema.get('min')) is not None:
      financials = lower_bound(financials, col_set, value, calculee)

    if (value := schema.get('max')) is not None:
      financials = upper_bound(financials, col_set, value, calculee)

  return financials
