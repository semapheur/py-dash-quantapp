import json

from lib.fin.taxonomy import Taxonomy


def load_taxonomy(json_path='lex/fin_taxonomy.json') -> dict:
  with open(json_path, 'r') as f:
    tax_data: dict = json.load(f)

  return tax_data


def fix_taxonomy():
  taxonomy_raw = load_taxonomy()
  taxonomy = Taxonomy(data=taxonomy_raw)
  taxonomy.add_missing_items()
  taxonomy.resolve_calculation_order()
  taxonomy_dict = taxonomy.model_dump(exclude_none=True)

  with open('lex/fin_taxonomy.json', 'w') as f:
    json.dump(taxonomy_dict, f, indent=2)


def seed_taxonomy(db_name='taxonomy.db'):
  taxonomy_raw = load_taxonomy()
  taxonomy = Taxonomy(data=taxonomy_raw)
  taxonomy.to_sql(db_name)
