from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent.parent
DB_DIR = WORK_DIR / 'data'
STATIC_DIR = WORK_DIR / 'assets'

HEADERS = {
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
  'Accept-Encoding': 'gzip, deflate, br, zstd',
  'Accept-Language': 'en-US,en;q=0.5',
  'DNT': '1',
  'Connection': 'keep-alive',
  #'Sec-Fetch-Dest': 'empty',
  #'Sec-Fetch-Mode': 'cors',
  #'Sec-Fetch-Site': 'none',
  #'Sec-GPC': '1',
  #'Cache-Control': 'no-cache',
  'Upgrade-Insecure-Requests': '1',
  'User-Agent': (
    'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0'
  ),
}
