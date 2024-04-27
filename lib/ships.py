from datetime import datetime as dt, date as Date
from io import StringIO
import json
import re
from typing import cast, TypedDict

import hishel
import httpx
import pandas as pd
from parsel import Selector, SelectorList
from tqdm import tqdm

HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
  'Accept-Language': 'en-US,en;q=0.5',
  # 'Accept-Encoding': 'gzip, deflate, br',
  'DNT': '1',
  'Sec-GPC': '1',
  'Connection': 'keep-alive',
  'Upgrade-Insecure-Requests': '1',
  'Sec-Fetch-Dest': 'document',
  'Sec-Fetch-Mode': 'navigate',
  'Sec-Fetch-Site': 'same-origin',
  'Sec-Fetch-User': '?1',
}

cyrillic_traps = str.maketrans(
  {
    'А': 'A',
    'Е': 'E',
    'К': 'K',
    'М': 'M',
    'Т': 'T',
    'С': 'S',
    'Д': 'D',
    'Р': 'R',
    'У': 'U',
  }
)

month_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)'

fleet_pattern = r'(army|(corps of )?eng(e|i)neers( corps)?|(Air( Defence)?|Satellite) Forces?|customs|(Internal|Railway|Signal) Troops|FS(B|O)|GUSS|KGB|PSKA|PVO|RVSN|VKS|VVS|Ministry of Internal Affairs|MP?ChVV?|(Coast|(Maritime )?Border) Guard|KВФ|БФ|((Baltic|Black Sea|Nor?thern|Pacifi?ci?) Fleet|(Amu(-| )Darya|Amur|Baykal|Caspian|Danube|Enisey|Irtysh|Iss?yk Kul|Kolyma|Krasnoyarsk|Moscow|Novosibirsk|Ob|Ozyorsk|Perm|Pskov|Rostov-on-Don|Sevan|Snezhinsk|Valday|Volga|Zheleznjgorsk|Yakutsk)( Flotilia)?))'
fleet_typos = {
  'БФ': 'Baltic Fleet',
  'Isyk Kul': 'Issyk Kul',
  'KВФ': 'Caspian Flotilla',
  'Nothern Fleet': 'Northern Fleet',
  'Pacifc Fleet': 'Pacific Fleet',
  'Pacifci Fleet': 'Pacific Fleet',
  'Engeneers': 'Engineers',
  'Corps Of Engineers': 'Engineers Corps',
}


def scrap_ships() -> tuple[pd.DataFrame, pd.DataFrame]:
  url = 'http://russianships.info/eng/'
  with httpx.Client() as client:
    rs = client.get(url, headers=HEADERS)
    dom = Selector(rs.text)

  category_urls = dom.xpath('//*[@id="sidebarmenu1"]/li/a/@href').getall()
  categories = {i.split('/')[-2] for i in category_urls}

  ship_dfs: list[pd.DataFrame] = []
  spec_dfs: list[pd.DataFrame] = []

  failed = {}

  for c in category_urls:
    with httpx.Client(default_encoding='windows-1251') as client:
      rs = client.get(c, headers=HEADERS)
      dom = Selector(rs.text.translate(cyrillic_traps))

    anchors = dom.xpath(
      '/html/body/table/tr/td[3]/table/tr/td/div/center/table/tr/td/div/table[2]/tr[2]/td[2]/table/tr/td[1]//a'
    )
    roles = dom.xpath(
      '/html/body/table/tr/td[3]/table/tr/td/div/center/table/tr/td/div/table[2]/tr[2]/td[2]/table/tr[not(@bgcolor)]/td[2]/div/text()'
    ).getall()

    for a, role in zip(tqdm(anchors), roles):
      slug = a.xpath('./@href').get()
      slug = slug.replace('../', '')

      if slug.split('/')[0] in categories:
        project_url = url + slug
      else:
        project_url = c + slug
      project = ' '.join(a.xpath('.//text()').get().replace('\r\n', '').split())
      role = role.replace('\r\n', '')

      try:
        ship_df, spec_df = scrap_ship_project(project_url, project, role)
        ship_dfs.append(ship_df)
        spec_dfs.append(spec_df)

      except Exception as e:
        print(f'{project}({project_url}) failed: {e}')
        failed[project] = {'url': project_url, 'error': str(e)}

  with open('log.json', 'w') as f:
    json.dump(failed, f, indent=2)

  return (
    pd.concat(ship_dfs, ignore_index=True),
    pd.concat(spec_dfs, ignore_index=True),
  )


def ship_table(
  table_dom: SelectorList[Selector],
  url: str,
  project: str,
  role: str,
  project_role: dict[str, str],
) -> pd.DataFrame:
  table_text = StringIO(table_dom.get())
  df = pd.read_html(table_text, header=0, decimal=',')[0]
  mask = df.apply(lambda r: r.nunique() == 1, axis=1)
  df = df[~mask]
  df.columns = pd.Index(
    ['name', 'yard', 'laid_down', 'launched', 'commissioned', 'note']
  )
  df['project'] = project
  df['role'] = project_role.get(project, role)
  df['reference'] = url
  df = df.astype(str)
  return df


def unknown_shipyard(unknown_text: str, projects: list[str]) -> pd.DataFrame:
  class ShipRecord(TypedDict):
    name: str
    project: str
    commissioned: pd.Timestamp
    fleet: str

  unknown_text = unknown_text.split(':')[-1].strip()
  ship_infos = re.split(r',(?![^()]*\)) ', unknown_text)

  project_pattern = r'\b|'.join(projects)

  unknown_data: list[ShipRecord] = []
  project = ''
  for ship_info in ship_infos:
    record = ShipRecord(name='', project='', commissioned=pd.to_datetime(''), fleet='')
    info = re.search(r'\((.+)\)?$', ship_info)
    record['name'] = re.sub(r' \(.+\)?$', '', ship_info).strip()

    if info is not None:
      info_text = info.group()
      fleet_match = re.search(fleet_pattern, info_text)
      if fleet_match is not None:
        fleet = fleet_match.group().title().replace('Flotilia', 'Flotilla')
        if fleet in fleet_typos:
          fleet = fleet_typos[fleet]

        record['fleet'] = fleet

      project_match = re.search(project_pattern, info_text)
      if project_match is not None:
        project = project_match.group()

      record['project'] = project

      commission_match = re.search(
        r'(?<=comm\.)\s?(\d{1,2}\.)?(\d{2}\.)?\d{4}', info_text
      )
      if commission_match:
        commission_date = [int(i) for i in commission_match.group().split('.')]
        if (x := 3 - len(commission_date)) > 0:
          commission_date = [1] * x + commission_date

        record['commissioned'] = pd.to_datetime(Date(*commission_date[::-1]))

    unknown_data.append(record)

  return pd.DataFrame.from_records(unknown_data)


def get_fleet_data(fleet_texts: list[str], ship_df: pd.DataFrame) -> dict[str, str]:
  fleet_data: dict[str, str] = {}
  stop_words = ('<p>---</p>', '<p>In Sovyet Army.</p>')

  if fleet_texts[0] in stop_words:
    return fleet_data

  for fleet_text in fleet_texts:
    fleet_text = re.sub(r'<[^>]*>', '', fleet_text).strip()
    fleet_text = re.sub(r'\s+', ' ', fleet_text)
    if fleet_text == '':
      continue

    if ':' not in fleet_text:
      continue

    fleet, ship_text = fleet_text.split(':', 1)
    fleet = fleet.replace('Flotilia', 'Flotilla')
    ship_text = ship_text.strip()
    if ship_text == '':
      continue

    ships = re.split(r',(?![^()]*\)) ', ship_text)

    prefix = ''
    for ship in ships:
      info = fleet
      change = re.search(r'\((.+)\)', ship)
      ship = re.sub(r' \(.+\)', '', ship).split('/')[0].strip().replace('--', '-')

      if re.search(r'\b[A-Z]+(-[A-Z\d]+)( №\d+)?\b', ship) is not None:
        prefix = ship.split('-')[0]

      elif ship.startswith('№'):
        prefix = '№'

      ship_list = [ship]

      interval = re.search(r'\b(?P<start>\d+)-(?P<end>\d+)\b', ship)
      if interval is not None:
        start = int(interval.group('start'))
        end = int(interval.group('end'))
        ship_list = list(str(x) for x in range(start, end + 1))

      if change is not None:
        change_texts = change.group(1).replace('з', '').split(',')

        if (yard := re.search(r'^№\d+$', change_texts[0])) is not None:
          ship_list = [yard.group()]

        else:
          for change_text in change_texts:
            date_search = re.search(r'(\d{1,2}\.)?(\d{2}\.)?\d{4}', change_text)
            date = ''
            if date_search is not None:
              date = date_search.group()

            fleet_search = re.search(fleet_pattern, change_text, flags=re.I)
            fleet_ = '?'
            if fleet_search is not None:
              fleet_ = fleet_search.group().title()

            if fleet_ in fleet_typos:
              fleet_ = fleet_typos[fleet_]
            info += f',{fleet_}({date})'

      for s in ship_list:
        if prefix != '' and s.isdigit():
          s = f'{prefix}-{s}' if prefix != '№' else f'{prefix}{s}'

        if s not in ship_df['name'] or s not in ship_df['yard']:
          if (notes := ship_df['note'].str.contains(s, regex=False)).any():
            ix = notes.idxmax()
            s = ship_df.at[ix, 'name']

        fleet_data[s] = info

  return fleet_data


def get_decommission_data(
  decom_texts: list[str], ship_df: pd.DataFrame
) -> dict[str, Date]:
  decom_data: dict[str, Date] = {}
  if decom_texts[0] == '<p>---</p>':
    return decom_data

  year = 0
  for decom_text in decom_texts:
    decom_text = re.sub(r'<[^>]*>', '', decom_text).strip()
    ship_text = decom_text
    if (delim := re.match('\d{4}\s?(-|–)', decom_text)) is not None:
      year_, ship_text = re.split(rf'(?<=^\d{{4}})\s?{delim.group(1)}', decom_text)
      ship_text = ship_text.strip()
      year = int(year_.strip())

    if ship_text.startswith('As civil'):
      continue

    if year == 0:
      continue

    ship_infos = re.split(r',(?![^()]*\)) ', ship_text)
    for ship_info in ship_infos:
      info = re.search(r'\((.+)\)', ship_info)
      ship = re.sub(r' \(.+\)', '', ship_info).strip().split('/')[0].replace('?', '')

      day = 1
      month = 1
      if info is not None:
        info_text = info.group(1)
        date_search = re.search(
          r'\b(?P<day>\d{1,2})\.(?P<month>(0[1-9]|1[0-2]))(?!\.)', info_text
        )

        if date_search is not None:
          day = int(date_search.group('day'))
          month = int(date_search.group('month'))

        elif (month_ := re.search(month_pattern, info_text)) is not None:
          month = dt.strptime(month_.group(), '%B').month

      if ship not in ship_df['name'] or ship not in ship_df['yard']:
        if (notes := ship_df['note'].str.contains(ship, regex=False)).any():
          ix = notes.idxmax()
          ship = ship_df.at[ix, 'name']

      decom_data[ship] = Date(year, month, day)

  return decom_data


def get_export_data(
  export_doms: SelectorList[Selector], ship_df: pd.DataFrame
) -> dict[str, Date]:
  export_data: dict[str, Date] = {}

  if export_doms[0].get() == '<p>---</p>':
    return export_data

  export_pattern = r'\((till (?P<date>(\d{1,2}\.)?(\d{2}\.)?\d{4}) (?P<names>.+))\)'

  for p in export_doms:
    # country = p.xpath('./i/text()').get()
    export_text = p.get().split('<br>')[-1]
    ship_infos = re.split(r',(?![^()]*\)) ', export_text)
    for ship_info in ship_infos:
      info = re.search(export_pattern, ship_info)

      if info is None:
        continue

      date = [int(i) for i in info.group('date').split('.')]
      if (x := 3 - len(date)) > 0:
        date = [1] * x + date

      if date[1] > 12 and date[0] <= 12:
        date[1], date[0] = date[0], date[1]
      else:
        date[1] = min(12, date[1])

      export_date = Date(*date[::-1])
      ship = info.group('names').strip().replace('?', '')

      ship_match = re.search(r'\b[A-Z]+-\d+\b', ship, flags=re.I)
      if ship_match is not None:
        ship = ship_match.group()

      if ship not in ship_df['name'] or ship not in ship_df['yard']:
        if (notes := ship_df['note'].str.contains(ship, regex=False)).any():
          ix = notes.idxmax()
          ship = ship_df.at[ix, 'name']

      export_data[ship] = export_date

  return export_data


def scrap_ship_project(
  url: str, project: str, role: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
  def get_specification(headers: SelectorList[Selector]) -> pd.DataFrame:
    spec_dfs: list[pd.DataFrame] = []

    for spec in headers:
      spec_project = spec.xpath('./text()').get().split(' - ')[1]

      table = spec.xpath('./parent::node()/following-sibling::table[1]').get()

      spec_df = pd.read_html(StringIO(table), decimal=',')[0]
      mask = spec_df.apply(lambda r: r.nunique() == 1, axis=1)
      spec_df = spec_df[~mask]

      spec_df.columns = pd.Index(['attribute', 'value'])

      data = [('Role', project_role.get(spec_project, role))]

      if len(project_role) > 1:
        data.append(
          ('Variants', ', '.join(set(project_role.keys()).difference({spec_project})))
        )

      nato_value = ''
      if nato_classes and nato_classes[0] != '---':
        nato_value = ', '.join(nato_classes)

      data.append(('NATO classes', nato_value))

      df_ = pd.DataFrame(data=data, columns=['attribute', 'value'])
      spec_df = pd.concat((df_, spec_df), ignore_index=True)

      spec_df['unit'] = spec_df['attribute'].str.extract(r'\((.+)\)')
      spec_df.loc[:, 'attribute'] = spec_df['attribute'].str.replace(
        r'( \(.+\))?:$', '', regex=True
      )

      spec_df.loc[spec_df['attribute'].isin({'Standard', 'Full load'}), 'unit'] = 'tons'
      spec_df.loc[spec_df['attribute'].isin({'Length', 'Beam', 'Draft'}), 'unit'] = 'm'
      spec_df.loc[spec_df['attribute'] == 'Range', 'unit'] = 'nmi'

      spec_df['entity'] = project
      spec_df['reference'] = url
      spec_df = spec_df[['entity', 'attribute', 'unit', 'value', 'reference']]
      spec_dfs.append(spec_df)

    return pd.concat(spec_dfs, ignore_index=True)

  with hishel.CacheClient(default_encoding='windows-1251') as client:
    rs = client.get(url, headers=HEADERS)
    if rs.status_code != 200:
      raise ValueError(f'Unable to retrieve data for {project} at {url}')

    dom = Selector(
      rs.text.replace('\r\n', '').replace('ав.', '').translate(cyrillic_traps)
    )

  project_prefix = project.split(' ')[0]
  name_dom = dom.xpath(
    '/html/body/table/tr/td[3]/table/tr/td/div/center/table/tr/td/div/table[2]/tr[2]/td[2]/h3',
  )
  if not name_dom:
    name_dom = dom.xpath(
      '/html/body/table/tr/td[3]/table/tr/td/div/center/table/tr/td/div/table[2]/tr[3]/td[2]/h3'
    )

  project_role = {project: role}

  for h in name_dom[:-1]:
    role = h.xpath('./text()').get()

    keys: list[str] = []
    for b in h.xpath('./b'):
      for name in b.get().split('<br>'):
        name = re.sub(r'<[^>]*>', '', name).strip()
        if name == '':
          continue

        keys.append(name)

    # keys = [
    #  x.strip() + ' ' + ' '.join(n.xpath('./font/text()').getall()).strip()
    #  for n in names
    #  if (x := n.xpath('./text()').get()) is not None and x.strip() != ''
    # ]

    project_role.update({k: role for k in keys})

  nato_dom = name_dom[-1].xpath('./b/font')

  nato_classes = [
    i.xpath('./text()').get().strip().replace(' Class', '') for i in nato_dom
  ]

  spec_headers = dom.xpath('//b[starts-with(text(), "General characteristics")]')
  spec_df = get_specification(spec_headers)

  ship_doms = dom.xpath(f'//p[starts-with(text(), "{project_prefix}")]')
  ship_header = ship_doms.xpath('./preceding-sibling::p[1]/b/text()').get()
  if ship_header != 'Ships' or not ship_doms:
    table_dom = dom.xpath(
      '//b[starts-with(text(), "Ships")]/parent::node()/following-sibling::table[1]'
    )
    ship_df = ship_table(table_dom, url, project, role, project_role)

  else:
    ship_dfs: list[pd.DataFrame] = []
    for ship_dom in ship_doms:
      project_text = ship_dom.xpath('./text()').get()
      project_text = re.sub(r'«|»', '', project_text)
      if f'{project_prefix} ?' in project_text:
        project_ = project
      else:
        project_ = cast(
          re.Match[str], re.match(rf'{project_prefix} [A-Za-z\d/\-]+\b', project_text)
        ).group()

      table_dom = ship_dom.xpath('./following-sibling::table[1]')
      if not table_dom:
        continue

      ship_dfs.append(ship_table(table_dom, url, project_, role, project_role))

    ship_df = pd.concat(ship_dfs, ignore_index=True)

  for col in ('laid_down', 'launched', 'commissioned'):
    ship_df.loc[:, col] = ship_df[col].str.replace(
      r'^(\d{2})\.(\d{4})$', r'01.\1.\2', regex=True
    )
    ship_df.loc[:, col] = ship_df[col].str.replace(
      r'^(\d{4})$', r'01.01.\1', regex=True
    )
    ship_df.loc[:, col] = pd.to_datetime(
      ship_df[col], format='%d.%m.%Y', errors='coerce'
    ).dt.date

  unknown_ships = dom.xpath(
    '//p[starts-with(text(), "Unknown shipyard:")]/text()'
  ).get()
  if unknown_ships is not None:
    unknown_df = unknown_shipyard(unknown_ships, list(project_role.keys()))
    ship_df = pd.concat((ship_df, unknown_df), ignore_index=True)

  fleet_dom = dom.xpath(
    '//b[contains(text(), "Fleets")]/parent::node()/following-sibling::p[1]'
  )
  fleet_texts = fleet_dom.get().split('<br>')
  fleet_data = get_fleet_data(fleet_texts, ship_df)

  ship_df['fleet'] = pd.NA
  for k, v in fleet_data.items():
    ship_df.loc[(ship_df['name'] == k) | (ship_df['yard'] == k), 'fleet'] = v

  hull_dom = dom.xpath(
    '//b[contains(text(), "Hull Numbers")]/parent::node()/following-sibling::p[1]'
  )
  hull_texts = hull_dom.get().split('<br>')

  hull_data = {}
  decom_texts = []
  if ' – ' in hull_texts[0]:
    decom_texts = hull_texts

  elif hull_texts[0] not in ('<p>---</p>', '<p><i>---</i></p>'):
    for h in hull_texts:
      h = re.sub(r'<[^>]+>', '', h).strip()
      if ':' not in h:
        continue
      ship, numbers = h.rsplit(':', 1)
      ship = ship.split(':')[0].strip()
      hull_data[ship] = numbers.strip()

  ship_df['hulls'] = pd.NA
  for k, v in hull_data.items():
    ship_df.loc[(ship_df['name'] == k) | (ship_df['yard'] == k), 'hulls'] = v

  if not decom_texts:
    decom_dom = dom.xpath(
      '//b[contains(text(), "Decommissioned")]/parent::node()/following-sibling::p[1]'
    )
    decom_texts = decom_dom.get().split('<br>')

  decom_data = get_decommission_data(decom_texts, ship_df)

  ship_df['decommissioned'] = pd.NaT
  for k, v in decom_data.items():
    ship_df.loc[
      (ship_df['name'] == k) | (ship_df['yard'] == k), 'decommissioned'
    ] = pd.to_datetime(v)

  export_doms = dom.xpath(
    '//b[contains(text(), "Export")]/parent::node()/following-sibling::p'
  )
  if export_doms:
    export_data = get_export_data(export_doms, ship_df)

    ship_df['exported'] = pd.NaT
    for k, v in export_data.items():
      ship_df.loc[
        (ship_df['name'] == k) | (ship_df['yard'] == k), 'exported'
      ] = pd.to_datetime(v)
  else:
    print(f'No export data: {project}')

  return ship_df, spec_df
