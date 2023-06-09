from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path().cwd() / '.env')

WORK_DIR = Path(__file__).resolve().parent.parent
DB_DIR = WORK_DIR / os.getenv('DB_DIR')
STATIC_DIR = WORK_DIR / os.getenv('STATIC_DIR')

HEADERS = {
  'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0',
  'Accept': 'application/json, text/plain, */*',
  'Accept-Language': 'en-US,en;q=0.5',
  #'Accept-Encoding': 'gzip, deflate, br',
  #'DNT': '1',
  #'Connection': 'keep-alive',
  #'Sec-Fetch-Dest': 'empty',
  #'Sec-Fetch-Mode': 'cors',
  #'Sec-Fetch-Site': 'same-site',
  #'Sec-GPC': '1',
  #'Cache-Control': 'max-age=0',
}