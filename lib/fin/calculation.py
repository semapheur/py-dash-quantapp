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
      value=ast.Name(id=self.name, ctx=ast.Load()),
      slice=ast.Index(value=ast.Constant(value=node.id)),
      ctx=node.ctx
    )
    return subscript
  
class AnyVisitor(ast.NodeVisitor):
  def __init__(self):
    self.terms = {}

  def reset_terms(self):
    self. terms = {}

  def visit_BinOp(self, node):
    if (
      isinstance(node.left, ast.Constant) and 
      isinstance(node.right, ast.Name)
    ):
      self.terms[node.right.id] = {'weight': node.left.value}

    elif (isinstance(node.left, ast.Name)):
      self.terms[node.left.id] = {'weight': 1}

    elif (isinstance(node.right, ast.Name)):
      if isinstance(node.op, ast.Add):
        weight = 1
      elif isinstance(node.op, ast.Sub):
        weight = -1

      self.terms[node.right.id] = {'weight': weight}
    
    self.generic_visit(node)
  
def calculate_items(
  financials: pd.DataFrame, 
  schemas: dict,
  recalc: bool = False
) -> pd.DataFrame:
  
  schemas = dict(sorted(schemas.items(), key=lambda x: x[1]['order']))

  vis = AllTransformer('financials')
  for calculee, schema in schemas.items():
    col_set = set(financials.columns)

    if 'all' in schema:
      expression = ast.parse(schema['all'], mode='eval')
      expression = ast.fix_missing_locations(vis.visit(expression))

      if not vis.names.issubset(col_set):
        continue

      code = compile(expression, '<string>', 'eval')
      financials[calculee] = eval(code)
      vis.reset_names()

  return financials