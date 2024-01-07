import asyncio
from datetime import date as Date, datetime as dt
from enum import Enum
from functools import partial
import re
from typing import cast, Literal, TypeAlias
import xml.etree.ElementTree as et

import aiometer
import hishel
import httpx
import bs4 as bs
import pandas as pd
from pandera.typing import DataFrame, Series
from tinydb import TinyDB

# Local
from lib.const import DB_DIR, HEADERS
from lib.edgar.models import (
  CikEntry,
  CikFrame,
  RawFinancials,
  Instant,
  Interval,
  Item,
  Member,
  Scope,
  FiscalPeriod,
  FinData,
)
from lib.utils import (
  insert_characters,
  month_difference,
  fiscal_quarter,
  validate_currency,
)


class ScopeEnum(Enum):
  quarterly = 3
  annual = 12


Docs: TypeAlias = Literal['cal', 'def', 'htm', 'lab', 'pre']


async def parse_xbrl_urls(cik: int, doc_ids: list[str], doc_type: Docs) -> Series[str]:
  tasks = [
    asyncio.create_task(parse_xbrl_url(cik, doc_id, doc_type)) for doc_id in doc_ids
  ]
  urls = await asyncio.gather(*tasks)

  result = pd.Series(urls, index=doc_ids)
  result = result.loc[result.notnull()]

  return result


async def parse_xbrl_url(cik: int, doc_id: str, doc_type: Docs = 'htm') -> str:
  url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{doc_id.replace('-', '')}/{doc_id}-index.html"
  async with httpx.AsyncClient() as client:
    rs = await client.get(url, headers=HEADERS)
    if rs.status_code != 200:
      raise httpx.RequestError(f'Error: {rs.text}')
    parse = bs.BeautifulSoup(rs.text, 'lxml')

  data_files = cast(bs.Tag, parse.find('table', {'summary': 'Data Files'}))
  if doc_type == 'htm':
    pattern = r'(?<!_(cal|def|lab|pre)).xml$'
  else:
    pattern = rf'_{doc_type}.xml$'

  a_node = data_files.find('a', href=re.compile(pattern))
  assert a_node is not None, f'a tag containing XBRL href not found from {url}'

  href = cast(bs.Tag, a_node).get('href')
  return f'https://www.sec.gov{href}'


async def parse_statements(urls: list[str]) -> list[RawFinancials]:
  tasks = [partial(parse_statement, url) for url in urls]
  financials = await aiometer.run_all(tasks, max_per_second=5)

  return financials


async def parse_statement(url: str) -> RawFinancials:
  def fix_data(data: FinData) -> FinData:
    fixed: FinData = {}

    for k in sorted(data.keys()):
      temp: list[Item] = []

      for item in data[k]:
        if 'value' in item or (members := item.get('members')) is None:
          temp.append(item)
          continue

        if len(members) == 1:
          member = next(iter(members.values()))
          temp.append(
            Item(
              period=item['period'],
              value=cast(float | int, member.get('value')),
              unit=cast(str, member.get('unit')),
            )
          )
          continue

        value = 0.0
        units = set()
        for m in members.values():
          value += m.get('value', 0)
          if (unit := m.get('unit')) is not None:
            units.add(unit)

        if len(units) == 1:
          temp.append(
            Item(period=item['period'], value=value, unit=units.pop(), members=members)
          )

      fixed[k] = temp

    return fixed

  def parse_period(period: et.Element) -> Instant | Interval:
    def parse_date(date_text: str):
      m = re.search(r'\d{4}-\d{2}-\d{2}', date_text)
      if m is None:
        raise ValueError(f'"{date_text}" does not match format "%Y-%m-%d"')

      return dt.strptime(m.group(), '%Y-%m-%d').date()

    if (el := period.find('./{*}instant')) is not None:
      instant_date = parse_date(cast(str, el.text))
      return Instant(instant=instant_date)

    start_date = parse_date(
      cast(str, cast(et.Element, period.find('./{*}startDate')).text)
    )
    end_date = parse_date(cast(str, cast(et.Element, period.find('./{*}endDate')).text))

    months = month_difference(start_date, end_date)

    return Interval(
      start_date=start_date.date(), end_date=end_date.date(), months=months
    )

  def parse_unit(unit: str) -> str:
    if re.match(r'^Unit\d+$', unit) is not None:
      unit_el = cast(et.Element, root.find(f'.{{*}}unit[@id="{unit}"]'))

      if unit_el is None:
        print(url)

      if (measure_el := unit_el.find('.//{*}measure')) is not None:
        unit_ = cast(str, measure_el.text).split(':')[-1].lower()

      elif (divide := unit_el.find('.//{*}divide')) is not None:
        numerator = (
          cast(str, cast(et.Element, divide.find('.//{*}unitNumerator/measure')).text)
          .split(':')[-1]
          .lower()
        )
        denominator = (
          cast(str, cast(et.Element, divide.find('.//{*}unitDenominator/measure')).text)
          .split(':')[-1]
          .lower()
        )
        unit_ = f'{numerator}/{denominator}'

    else:
      pattern = r'^Unit_(Standard|Divide)_(\w+)_[A-Za-z0-9_-]{22}$'
      if (m := re.search(pattern, unit)) is not None:
        unit_ = m.group(2).lower()
      else:
        unit_ = unit.split('_')[-1].lower()

    pattern = r'(?<=^iso4217)?[a-z]{3}$'
    m = re.search(pattern, unit_, flags=re.I)
    if m is not None:
      if validate_currency(m.group()):
        currency.add(m.group())

    return unit_

  def parse_member(item: et.Element, segment: et.Element) -> dict[str, Member]:
    def parse_name(name: str) -> str:
      name = re.sub(r'(Segment)?Member', '', name)
      return name.split(':')[-1]

    unit = parse_unit(item.attrib['unitRef'])

    return {
      parse_name(cast(str, segment.text)): Member(
        dim=segment.attrib['dimension'].split(':')[-1],
        value=float(cast(str, item.text)),
        unit=unit,
      )
    }

  async def fetch(url: str) -> et.Element:
    # cache_transport = hishel.AsyncCacheTransport(transport=httpx.AsyncHTTPTransport())
    async with hishel.AsyncCacheClient() as client:
      rs = await client.get(url, headers=HEADERS)
      return et.fromstring(rs.content)

  root = await fetch(url)
  if root.tag == 'Error':
    cik, doc_id = url.split('/')[6:8]
    doc_id = insert_characters(doc_id, {'-': [10, 12]})
    url = await parse_xbrl_url(int(cik), doc_id)
    root = await fetch(url)

  form = {'10-K': 'annual', '10-Q': 'quarterly'}

  scope = cast(
    Scope, form[cast(str, cast(et.Element, root.find('.{*}DocumentType')).text)]
  )
  date = dt.strptime(
    cast(str, cast(et.Element, root.find('.{*}DocumentPeriodEndDate')).text), '%Y-%m-%d'
  )
  fiscal_end = cast(
    str, cast(et.Element, root.find('.{*}CurrentFiscalYearEndDate')).text
  )[1:]

  if (el := root.find('.{*}DocumentFiscalPeriodFocus')) is not None:
    fiscal_period = cast(FiscalPeriod, el.text)
  elif scope == 'annual':
    fiscal_period = 'FY'
  else:
    pattern = r'(\d{2})-(\d{2})'

    match = re.search(pattern, fiscal_end)
    month = int(cast(re.Match[str], match).group(1))
    day = int(cast(re.Match[str], match).group(2))

    fiscal_period = cast(FiscalPeriod, fiscal_quarter(date, month, day))

  doc_id = url.split('/')[-2]
  currency: set[str] = set()
  data: FinData = {}

  for item in root.findall('.//*[@unitRef]'):
    if item.text is None:
      continue

    scrap = Item()

    ctx = item.attrib['contextRef']
    period_el = cast(
      et.Element,
      cast(et.Element, root.find(f'./{{*}}context[@id="{ctx}"]')).find('./{*}period'),
    )

    scrap['period'] = parse_period(period_el)

    segment = cast(et.Element, root.find(f'./{{*}}context[@id="{ctx}"]')).find(
      './/{*}segment/{*}explicitMember'
    )

    if segment is not None:
      scrap['members'] = parse_member(item, segment)
    else:
      scrap['value'] = float(item.text)
      unit = parse_unit(item.attrib['unitRef'])
      scrap['unit'] = unit

    item_name = item.tag.split('}')[-1]
    if item_name not in data:
      data[item_name] = [scrap]
      continue

    try:
      entry = next(i for i in data[item_name] if i['period'] == scrap['period'])

      if 'members' in scrap:
        cast(dict[str, Member], entry.setdefault('members', {})).update(
          cast(dict[str, Member], scrap['members'])
        )
      else:
        entry.update(scrap)
    except Exception:
      data[item_name].append(scrap)

  # Sort items
  data = fix_data(data)
  return RawFinancials(
    url=url,
    date=date.date(),
    scope=scope,
    period=fiscal_period,
    fiscal_end=fiscal_end,
    currency=currency,
    data=data,
  )


async def parse_taxonomy(url: str) -> pd.DataFrame:
  namespace = {
    'link': 'http://www.xbrl.org/2003/linkbase',
    'xlink': 'http://www.w3.org/1999/xlink',
  }

  def rename_sheet(txt: str) -> str:
    pattern = r'income|balance|cashflow'
    m = re.search(pattern, txt, flags=re.I)
    if m:
      txt = m.group().lower()

    return txt

  async with httpx.AsyncClient() as client:
    rs = await client.get(url, headers=HEADERS)
    root = et.fromstring(rs.content)

  url_pattern = r'^https?://www\..+/'
  el_pattern = r'(?<=_)[A-Z][A-Za-z]+(?=_)?'

  taxonomy = []
  for sheet in root.findall('.//link:calculationLink', namespaces=namespace):
    sheet_label = re.sub(url_pattern, '', sheet.attrib[f'{{{namespace["xlink"]}}}role'])
    sheet_label = rename_sheet(sheet_label)

    for el in sheet.findall('.//link:calculationArc', namespaces=namespace):
      taxonomy.append(
        {
          'sheet': sheet_label,
          'gaap': cast(
            re.Match[str],
            re.search(el_pattern, el.attrib[f'{{{namespace["xlink"]}}}to']),
          ).group(),
          'parent': cast(
            re.Match[str],
            re.search(el_pattern, el.attrib[f'{{{namespace["xlink"]}}}from']),
          ).group(),
        }
      )

  df = pd.DataFrame.from_records(taxonomy)
  df.set_index('item', inplace=True)
  df.drop_duplicates(inplace=True)
  return df


def statement_to_df(financials: RawFinancials) -> pd.DataFrame:
  def parse_date(period: Instant | Interval) -> Date:
    if isinstance(period, Interval):
      return period.end_date

    return period.instant

  fin_date = financials.date
  fin_scope = financials.scope
  fin_period = financials.period

  df_data: dict[tuple[Date, FiscalPeriod, int], dict[str, int | float]] = {}

  for item, entries in financials.data.items():
    for entry in entries:
      date = parse_date(entry['period'])
      if date != fin_date:
        continue

      if isinstance(entry['period'], Interval):
        months = entry['period'].months
      else:
        months = ScopeEnum[fin_scope].value

      period = fin_period
      if fin_period == 'FY' and months < 12:
        period = 'Q4'

      if value := entry.get('value'):
        df_data.setdefault((fin_date, period, months), {})[item] = value

        if fin_period == 'FY' and (
          isinstance(entry['period'], Instant) or entry.get('unit') == 'shares'
        ):
          df_data.setdefault((fin_date, 'Q4', 3), {})[item] = value

      if (members := entry.get('members')) is None:
        continue

      for member, m_entry in members.items():
        if (m_value := m_entry.get('value')) is None:
          continue

        dim = '.' + d if (d := m_entry.get('dim')) else ''
        key = f'{item}{dim}.{member}'
        df_data.setdefault((fin_date, period, months), {})[key] = m_value

        if fin_period == 'FY' and (
          isinstance(entry['period'], Instant) or m_entry.get('unit') == 'shares'
        ):
          df_data.setdefault((fin_date, 'Q4', 3), {})[key] = m_value

  df = pd.DataFrame.from_dict(df_data, orient='index')
  df.index = pd.MultiIndex.from_tuples(df.index)
  df.index.names = ['date', 'period', 'months']
  return df


def get_ciks() -> DataFrame[CikFrame]:
  rnm = {'cik_str': 'cik', 'title': 'name'}

  url = 'https://www.sec.gov/files/company_tickers.json'

  with httpx.Client() as s:
    rs = s.get(url, headers=HEADERS)
    parse: dict[str, CikEntry] = rs.json()

  df = pd.DataFrame.from_dict(parse, orient='index')
  df.rename(columns=rnm, inplace=True)
  df.set_index('cik', inplace=True)

  return cast(DataFrame[CikFrame], df)


def all_gaap_items() -> set[str]:
  db_path = DB_DIR / 'edgar.json'
  db = TinyDB(db_path)

  items: set[str] = set()
  for t in db.tables():
    data = db.table(t).all()
    for i in data:
      items.update(i['data'].keys())

  return items
