import ast
from typing import Any, Iterable

import numpy as np
import pandas as pd

from lib.utils import df_time_difference

SLICES = (
  (slice(None), slice('FY'), slice(12)),
  (slice(None), slice(None), slice(3))
)

class AllTransformer(ast.NodeTransformer):
  def __init__(self, df_name: str):
    self.df_name = df_name
    self.names = set()

  def reset_names(self):
    self.names = set()

  def visit_Name(self, node):
    self.names.add(node.id)

    subscript = ast.Subscript(
      value=ast.Name(id=self.df_name, ctx=ast.Load()),
      slice=ast.Index(value=ast.Constant(value=node.id)),
      ctx=node.ctx
    )
    return subscript
  
class AnyTransformer(ast.NodeTransformer):
  def __init__(self, df_name: str):
    self.df_name = df_name
    self.df_columns = set()

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
          ctx=node.ctx),
        attr='fillna',
        ctx=ast.Load()),
      args=[
        ast.Constant(value=0)],
      keywords=[]
    )
    return call

def day_difference(df: pd.DataFrame, slices = SLICES):
  
  for ix in enumerate(slices):
    _df: pd.DataFrame = df.loc[ix]
    _df.sort_index(level='date', inplace=True)

    dates = pd.to_datetime(_df.index.get_level_values('date'))
    df.loc[ix, 'days'] = df_time_difference(dates, 'M').array

  return df

def applier(
  s: pd.Series, 
  fn: str, 
  slices: Iterable[Iterable[slice|Any]] = SLICES
) -> pd.Series:
  
  update = [pd.Series()] * len(slices)
  
  for i, ix in enumerate(slices):
    _s: pd.Series = s.loc[ix]
    _s.sort_index(level='date', inplace=True)

    dates = pd.to_datetime(_s.index.get_level_values('date'))
    month_diff = pd.Series(df_time_difference(dates, 'M').array, index=_s.index)

    if fn == 'diff':
      _s = _s.diff()
    elif fn == 'avg':
      _s = _s.rolling(window=2, min_periods=2).mean()
    elif fn == 'shift':
      _s = _s.shift()
    
    _s = _s.loc[month_diff == ix[2].stop]
    update[i] = _s
    
  update = pd.concat(update, axis=0)
  nan_index = pd.Index(list(set(s.index).difference(update.index)))

  s.loc[update.index] = update
  s.loc[nan_index] = np.nan

  return s

def calculate_items(
  financials: pd.DataFrame, 
  schemas: dict[str, dict],
  recalc: bool = False
) -> pd.DataFrame:
  
  def get_formula(schema: dict[str, dict], key: str) -> str|dict:
    formula = schema.get(key)
    if isinstance(formula, list):
      formula = ''.join(formula)

    return formula

  def insert_to_df(
    df: pd.DataFrame, 
    df_cols: set[str],
    insert_data: pd.Series,
    insert_name: str
  ) -> pd.DataFrame:
    if insert_name in df_cols:
      df.loc[:, insert_name] = (insert_data if recalc else 
        df[insert_name].combine_first(insert_data)
      )
    else:
      new_column = pd.DataFrame({insert_name: insert_data})
      df = pd.concat([df, new_column], axis=1)

    return df

  def calculate(
    df: pd.DataFrame, 
    df_cols: set[str],
    col_name: str,
    expression: ast.Expression|dict[str,int|dict[str,int]],
  ) -> pd.DataFrame:
        
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
        formula = ast.fix_missing_locations(
          all_visitor.visit(formula)
        )

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
        formula = ast.fix_missing_locations(
          any_visitor.visit(formula)
        )
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