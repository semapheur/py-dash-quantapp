import ast
from datetime import datetime as dt
import json
from typing import cast, Any

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from lib.db.lite import read_sqlite
from lib.utils import df_time_difference

SLICES = (
  (slice(None), 'FY', 12),
  (slice(None), 'TTM1', 12),
  (slice(None), 'TTM2', 12),
  (slice(None), 'TTM3', 12),
  (slice(None), slice(None), 3),
)


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


def trailing_twelve_months(financials: DataFrame) -> DataFrame:
  financials.sort_index(level='date', inplace=True)

  mask = (slice(None), slice(None), 3)
  ttm = financials.loc[mask, :]

  ttm.loc[:, 'month_difference'] = df_time_difference(
    cast(pd.DatetimeIndex, ttm.index.get_level_values('date')), 30, 'D'
  )
  ttm.loc[:, 'month_difference'] = ttm['month_difference'].rolling(3, 3).sum()

  query = f"""SELECT item FROM items
    WHERE aggregate = 'sum' AND item IN {str(tuple(financials.columns))}
  """
  sum_items = read_sqlite('taxonomy.db', query)
  if sum_items is None:
    raise ValueError('Could not load taxonomy!')

  ttm.loc[:, sum_items['item']] = ttm.loc[:, sum_items['item']].rolling(4, 4).sum()
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
      f'Could not load taxonomy items: {json.dumps(list[df.columns], indent=2)}'
    )

  df.loc[:, items['item'].to_list()] *= old_price / new_price

  return df


def day_difference(
  df: pd.DataFrame, slices: tuple[tuple[slice | Any, ...], ...] = SLICES
):
  for ix in slices:
    _df = df.loc[ix, :]
    _df.sort_index(level='date', inplace=True)

    dates = pd.to_datetime(_df.index.get_level_values('date'))
    df.loc[ix, 'days'] = df_time_difference(dates, 1, 'D')

  return df


def stock_split_adjust(df: pd.DataFrame, ratios: pd.Series) -> pd.DataFrame:
  share_cols = {
    'shares_outstanding',
    'weighted_average_shares_outstanding_basic',
    'weighted_average_shares_outstanding_diluted',
  }
  share_cols_ = list(share_cols.intersection(set(df.columns)))

  for i, col in enumerate(share_cols_):
    df.loc[:, f'adjusted_{col}'] = df[col]
    share_cols_[i] = f'adjusted_{col}'

  for date, ratio in ratios.items():
    mask = df.index.get_level_values('date') < date
    df.loc[mask, share_cols_] *= ratio

  return df


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
  s: pd.Series, fn: str, slices: tuple[tuple[slice | Any, ...], ...] = SLICES
) -> pd.Series:
  result = s.copy()
  update = [pd.Series()] * len(slices)

  for i, ix in enumerate(slices):
    _s: pd.Series = result.loc[ix]
    _s.sort_index(level='date', inplace=True)

    dates = pd.to_datetime(_s.index.get_level_values('date'))
    month_diff = pd.Series(df_time_difference(dates, 30, 'D'), index=_s.index)

    if fn == 'diff':
      _s = _s.diff()
    elif fn == 'avg':
      _s = _s.rolling(window=2, min_periods=2).mean()
    elif fn == 'shift':
      _s = _s.shift()

    _s = _s.loc[month_diff == ix[2].stop]
    update[i] = _s

  update_ = pd.concat(update, axis=0)
  nan_index = pd.Index(list(set(result.index).difference(update_.index)))

  result.loc[update_.index] = update_
  result.loc[nan_index] = np.nan

  return result


def calculate_items(
  financials: DataFrame, schemas: dict[str, dict], recalc: bool = False
) -> DataFrame:
  def get_formula(schema: dict[str, dict], key: str):
    formula = schema.get(key)
    if isinstance(formula, list):
      formula = ''.join(formula)

    return formula

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

  def calculate(
    df: DataFrame,
    df_cols: set[str],
    col_name: str,
    expression: ast.Expression | dict[str, int | dict[str, int]],
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
  financials = day_difference(financials)

  all_visitor = AllTransformer('df')
  any_visitor = AnyTransformer('df')
  for calculee, schema in schemas.items():
    col_set = set(financials.columns)

    if 'all' in schema:
      formula = get_formula(schema, 'all')

      if isinstance(formula, str):
        all_visitor.reset_names()
        formula = ast.parse(formula, mode='eval')
        formula = ast.fix_missing_locations(all_visitor.visit(formula))

        if not all_visitor.names.issubset(col_set):
          continue

      elif isinstance(formula, dict):
        if not set(formula.keys()).issubset(col_set):
          continue

      financials = calculate(financials, col_set, calculee, formula)

    elif 'any' in schema:
      formula = get_formula(schema, 'any')

      if isinstance(formula, str):
        any_visitor.set_columns(col_set)
        formula = ast.parse(formula, mode='eval')
        formula = ast.fix_missing_locations(any_visitor.visit(formula))
      elif isinstance(formula, dict):
        formula = {
          key: formula[key] for key in set(formula.keys()).intersection(col_set)
        }
        if not formula:
          continue

      financials = calculate(financials, col_set, calculee, formula)

    elif 'diff' in schema:
      calculer = schema['diff']
      if calculer not in col_set:
        continue

      result = applier(financials[calculer], 'diff')
      financials = insert_to_df(financials, col_set, result, calculee)

    elif 'avg':
      calculer = schema['avg']
      if calculer not in col_set:
        continue

      result = applier(financials[calculer], 'avg')
      financials = insert_to_df(financials, col_set, result, calculee)

  return financials
