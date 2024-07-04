import pandas as pd


def get_mics() -> pd.DataFrame:
  rename = {
    'MIC': 'mic',
    'OPERATING MIC': 'operating_mic',
    'OPRT/SGMT': 'oprt/sgmt',
    'MARKET NAME-INSTITUTION DESCRIPTION': 'market_name',
    'LEGAL ENTITY NAME': 'legal_entity_name',
    'LEI': 'lei',
    'MARKET CATEGORY CODE': 'market_category_code',
    'ACRONYM': 'acronym',
    'ISO COUNTRY CODE (ISO 3166)': 'country',
    'CITY': 'city',
    'WEBSITE': 'url',
    'STATUS': 'status',
    'CREATION DATE': 'creation_date',
    'EXPIRY DATE': 'expiry_date',
    'COMMENTS': 'comments',
  }

  url = 'https://www.iso20022.org/sites/default/files/ISO10383_MIC/ISO10383_MIC.csv'
  df = pd.read_csv(url, sep=',')
  df = df[list(rename.keys())]
  df.rename(columns=rename, inplace=True)

  return df
