import ast

import pandas as pd

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
  
def calculate_items(
  financials: pd.DataFrame, 
  schemas: dict,
  recalc: bool = False
) -> pd.DataFrame:
  
  def apply_calculation(
    df: pd.DataFrame, 
    df_cols: set[str],
    col_name: str,
    expression: ast.Expression,
  ) -> pd.DataFrame:
    
    code = compile(expression, '<string>', 'eval')
    result = eval(code, globals(), {'financials' : df})

    if isinstance(result, int):
      return df
    
    if col_name in df_cols:
      df.loc[:, col_name] = (result if recalc else 
        df[col_name].combine_first(result)
      )
    else:
      new_column = pd.DataFrame({col_name: result})
      df = pd.concat([df, new_column], axis=1)

    return df

  schemas = dict(sorted(schemas.items(), key=lambda x: x[1]['order']))

  all_visitor = AllTransformer('financials')
  any_visitor = AnyTransformer('financials')
  for calculee, schema in schemas.items():
    col_set = set(financials.columns)

    if 'all' in schema:
      all_schema = schema.get('all')
      if isinstance(all_schema, str):
        all_visitor.reset_names()
        expression = ast.parse(all_schema, mode='eval')
        expression = ast.fix_missing_locations(
          all_visitor.visit(expression)
        )
        
        if not all_visitor.names.issubset(col_set):
          continue

        financials = apply_calculation(
          financials, col_set, calculee, expression
        )

    elif 'any' in schema:
      any_schema = schema.get('any')
      if isinstance(any_schema, str):
        any_visitor.set_columns(col_set)
        expression = ast.parse(any_schema, mode='eval')
        expression = ast.fix_missing_locations(
          any_visitor.visit(expression)
        )

        financials = apply_calculation(
          financials, col_set, calculee, expression
        )

  return financials