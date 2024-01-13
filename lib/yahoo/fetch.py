import re
import textwrap

import bs4 as bs
import httpx
import numpy as np
import pandas as pd

from lib.const import HEADERS


def crumb_cookies() -> tuple[str, httpx.Cookies]:
  url = 'https://finance.yahoo.com/lookup'
  with httpx.Client() as client:
    rs = client.get(url, headers=HEADERS)
    soup = bs.BeautifulSoup(rs.text, 'lxml')

  crumb = re.findall('"CrumbStore":{"crumb":"(.+?)"}', str(soup))

  return crumb[0], rs.cookies


def get_tickers() -> pd.DataFrame:
  def json_params(size, offset, region, price=0):
    region_query = f'{{operator:EQ,operands:[region,{region}]}}'

    if np.isinf(price):
      price = ''

    data = textwrap.dedent(
      f"""
      {{size:{size}, offset:{offset}, sortField:intradayprice, 
      sortType:asc, quoteType:EQUITY, topOperator:AND,
      query: {{operator:AND, operands:[
        {{operator:or,operands:[{region_query}]}},
        {{operator:gt,operands:[intradayprice,{price}]}}
      ]}},userId:"",userIdType:guid}}
    """
    )

    return data

  def parse_json(
    crumb: str,
    cookies: httpx.Cookies,
    size: int,
    offset: int,
    region: str,
    price: int = 0,
  ) -> dict:
    params = {
      'crumb': crumb,
      'lang': 'en-US',
      'region': 'US',
      'formatted': 'true',
      'corsDomain': 'finance.yahoo.com',
    }
    url = 'https://query1.finance.yahoo.com/v1/finance/screener'

    with httpx.Client() as client:
      rs = client.post(
        url,
        headers=HEADERS,
        params=params,
        cookies=cookies,
        data=json_params(size, offset, region, price),
      )
      parse = rs.json()

    return parse['finance']['result'][0]

  def scrap_data(quotes: dict) -> list[dict]:
    scrap = []
    for i in quotes:
      scrap.append(
        {
          'ticker': i['symbol'],
          'name': i.get('longName', i.get('shortName', '').capitalize()),
          'exchange': i['exchange'],
          'exchange_name': i['fullExchangeName'],
        }
      )

    return scrap

  crumb, cookies = crumb_cookies()

  regions = (
    'ar',
    'at',
    'au',
    'be',
    'bh',
    'br',
    'ca',
    'ch',
    'cl',
    'cn',
    'cz',
    'de',
    'dk',
    'eg',
    'es',
    'fi',
    'fr',
    'gb',
    'gr',
    'hk',
    'hu',
    'id',
    'ie',
    'il',
    'in',
    'it',
    'jo',
    'jp',
    'kr',
    'kw',
    'lk',
    'lu',
    'mx',
    'my',
    'nl',
    'no',
    'nz',
    'pe',
    'ph',
    'pk',
    'pl',
    'pt',
    'qa',
    'ru',
    'se',
    'sg',
    'sr',
    'tf',
    'th',
    'tl',
    'tn',
    'tr',
    'tw',
    'us',
    've',
    'vn',
    'za',
  )
  num_res = 1000
  size = 250
  limit = 10000

  scrap = []
  for r in regions:  # tqdm(regions):
    offset = 0

    parse = parse_json(crumb, cookies, size, offset, r)
    offset += size

    if not parse['quotes']:
      continue

    scrap.extend(scrap_data(parse['quotes']))

    num_res = parse['total']

    if num_res > limit:
      price = 0
      scrap_count = len(scrap)
      flag = False

      while scrap_count <= num_res:
        while offset < limit:
          parse = parse_json(crumb, cookies, size, offset, r, price)

          if parse['quotes']:
            scrap_batch = scrap_data(parse['quotes'])
            scrap.extend(scrap_batch)
            scrap_count += len(scrap_batch)

            offset += size

            if offset >= limit:
              price = parse['quotes'][-1]['regularMarketPrice']['raw']
          else:
            flag = True
            break
        if flag:
          break

        offset = 0
    else:
      while offset < num_res:
        parse = parse_json(crumb, cookies, size, offset, r)
        scrap.extend(scrap_data(parse['quotes']))
        offset += size

    df = pd.DataFrame.from_records(scrap)
    df.drop_duplicates(inplace=True)
    df = df[df['name'].astype(bool)]  # Remove empty names

    # Remove options
    patterns = {
      'ASX': r'^Eqt xtb',
      'CCS': r'^(P|O)\.\s?c.',
      'VIE': r'^(Rc|Eg?)b\b',
      'OSL': r'^B(ull|ear)\b|\bpro$',
    }
    for k, v in patterns.items():
      mask = (df['exchange'] == k) & df['name'].str.contains(v, regex=True)
      df = df[~mask]

    patterns = {'SAO': r'F.SA$', 'OSL': r'-PRO.OL$'}
    for k, v in patterns.items():
      mask = (df['exchange'] == k) & df['ticker'].str.contains(v, regex=True)
      df = df[~mask]

  # Remove warrants
  pattern = r'-W(R|T)?[A-DU]?(\.[A-Z]{1,3})?$'
  mask = df['ticker'].str.contains(pattern, regex=True)
  df = df[~mask]

  # Add security type
  df['type'] = 'stock'
  patterns = {
    'fund': r'((?<!reit)|(?<!estate)|(?<!property)|(?<!exchange traded)).+\bfundo?\b',
    'etf': r'(\b(et(c|f|n|p)(s)?)\b)|(\bxtb\b)|(\bexch(ange)? traded\b)',
    'bond': (
      r'(\s\d(\.\d+)?%\s)|'
      r'(t-bill snr)|((\bsnr )?\b(bnd|bds|bonds)\b)|'
      r'(\bsub\.? )?\bno?te?s( due/exp(iry)?)?(?! asa)'
    ),
    'warrant': (
      r'(\bwt|warr(na|an)ts?)\b(?!.+\b(co(rp)?|l(imi)?te?d|asa|ag|plc|spa)\.?\b)'
    ),
    'reit': (
      r'(\breit\b)|(\b(estate|property)\b.+\b(fund|trust)\b)|'
      r'(\bfii\b|\bfdo( de)? inv imob\b)|'
      r'(\bfundos?( de)? in(f|v)est(imento)? (?=imob?i?li(a|á)?ri(a|o)\b))'
    ),
  }
  for k, v in patterns.items():
    mask = df['name'].str.contains(v, regex=True, flags=re.I)
    df.loc[mask, 'type'] = k
  # Remove bonds
  df = df[df['type'] != 'bond']

  # Fix name
  replacements = (
    (r'(d(k|l)|eo|hd|i(l|s)|ls|nk|rc|s(d|f|k)|yc)?\s?\d?(-,|,-)(\d+)?', ''),
    (r'(^\s|["])', ''),
    (r'\s+', ' '),
    (r'&amp;', '&'),
  )
  for old, new in replacements:
    df.loc[:, 'name'] = df['name'].str.replace(old, new, regex=True)

  # Trim tickers
  df['ticker_trim'] = df['ticker'].str.lower().split('.')[0]

  patterns = {'TLV': r'-(l|m)$'}
  for k, v in patterns.items():
    mask = df['exchange'] == k
    df.loc[mask, 'tickerTrim'] = df.loc[mask, 'tickerTrim'].str.replace(
      v, '', regex=True
    )

  # Remove suffixes and special characters
  pattern = r'(\s|-|_|/|\'|^0+(?=[1-9]\d*)|[inxp]\d{4,5}$)'
  pattern = (
    r'((?<=\.p)r(?=\.?[a-z]$))|'  # .PRx
    r'((?<=\.w)i(?=\.?[a-z]$))|'  # .WIx
    r'((?<=\.(r|u|w))t(?=[a-z]?$))|'  # .RT/UT/WT
    r'((\.|/)?[inpx][047]{4,5}$)|'
    r'\s|\.|_|-|/|\')'
  )
  df['ticker_trim'] = df['ticker_trim'].str.replace(pattern, '', regex=True)

  # Change exchange of XLON tickers starting with 0
  mask = (df['exchange'] == 'LSE') & df['ticker'].str.contains(r'^0')
  df.loc[mask, 'exchange'] = 'IOB'

  return df
