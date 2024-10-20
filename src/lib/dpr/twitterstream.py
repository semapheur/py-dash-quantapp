import pandas as pd

from collections import Counter

# Utils
import json
import unidecode
import pickle
import re
import itertools
import string
from threading import Lock, Timer
import time

# Database
import sqlite3

# Sentiment analysis
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Twitter
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

stopWords = [
  "i",
  "me",
  "my",
  "myself",
  "we",
  "our",
  "ours",
  "ourselves",
  "you",
  "your",
  "yours",
  "yourself",
  "yourselves",
  "he",
  "him",
  "his",
  "himself",
  "she",
  "her",
  "hers",
  "herself",
  "it",
  "its",
  "itself",
  "they",
  "them",
  "their",
  "theirs",
  "themselves",
  "what",
  "which",
  "who",
  "whom",
  "this",
  "that",
  "these",
  "those",
  "am",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
  "have",
  "has",
  "had",
  "having",
  "do",
  "does",
  "did",
  "doing",
  "a",
  "an",
  "the",
  "and",
  "but",
  "if",
  "or",
  "because",
  "as",
  "until",
  "while",
  "of",
  "at",
  "by",
  "for",
  "with",
  "about",
  "against",
  "between",
  "into",
  "through",
  "during",
  "before",
  "after",
  "above",
  "below",
  "to",
  "from",
  "up",
  "down",
  "in",
  "out",
  "on",
  "off",
  "over",
  "under",
  "again",
  "further",
  "then",
  "once",
  "here",
  "there",
  "when",
  "where",
  "why",
  "how",
  "all",
  "any",
  "both",
  "each",
  "few",
  "more",
  "most",
  "other",
  "some",
  "such",
  "no",
  "nor",
  "not",
  "only",
  "own",
  "same",
  "so",
  "than",
  "too",
  "very",
  "s",
  "t",
  "can",
  "will",
  "just",
  "don",
  "should",
  "now",
  "https",
  "'s",
  "...",
  "whats'",
  "rt",
  "whats",
  "n't",
  "de",
  "'m",
  "un",
  "en",
  "``",
  "dedic",
  "twittermoments",
  "amp",
  "e",
  "y",
  "o",
  "ce",
  "retweet",
  "sur",
  "na",
  "el",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "0",
  "ca",
  "nao",
  "se",
  "com",
  "los",
  "u",
  "des",
  "-",
  "--",
  "'",
  "''",
  "la",
  "como",
  "con",
  "segundo",
  "de",
  "la",
  "que",
  "el",
  "en",
  "y",
  "a",
  "los",
  "del",
  "se",
  "las",
  "por",
  "un",
  "para",
  "con",
  "no",
  "una",
  "su",
  "al",
  "lo",
  "como",
  "más",
  "pero",
  "sus",
  "le",
  "ya",
  "o",
  "este",
  "sí",
  "porque",
  "esta",
  "entre",
  "cuando",
  "muy",
  "sin",
  "sobre",
  "también",
  "me",
  "hasta",
  "hay",
  "donde",
  "quien",
  "desde",
  "todo",
  "nos",
  "durante",
  "todos",
  "uno",
  "les",
  "ni",
  "contra",
  "otros",
  "ese",
  "eso",
  "ante",
  "ellos",
  "e",
  "esto",
  "mí",
  "antes",
  "algunos",
  "qué",
  "unos",
  "yo",
  "otro",
  "otras",
  "otra",
  "él",
  "tanto",
  "esa",
  "estos",
  "mucho",
  "quienes",
  "nada",
  "muchos",
  "cual",
  "poco",
  "ella",
  "estar",
  "estas",
  "algunas",
  "algo",
  "nosotros",
  "mi",
  "mis",
  "tú",
  "te",
  "ti",
  "tu",
  "tus",
  "ellas",
  "nosotras",
  "vosostros",
  "vosostras",
  "os",
  "mío",
  "mía",
  "míos",
  "mías",
  "tuyo",
  "tuya",
  "tuyos",
  "tuyas",
  "suyo",
  "suya",
  "suyos",
  "suyas",
  "nuestro",
  "nuestra",
  "nuestros",
  "nuestras",
  "vuestro",
  "vuestra",
  "vuestros",
  "vuestras",
  "esos",
  "esas",
  "estoy",
  "estás",
  "está",
  "estamos",
  "estáis",
  "están",
  "esté",
  "estés",
  "estemos",
  "estéis",
  "estén",
  "estaré",
  "estarás",
  "estará",
  "estaremos",
  "estaréis",
  "estarán",
  "estaría",
  "estarías",
  "estaríamos",
  "estaríais",
  "estarían",
  "estaba",
  "estabas",
  "estábamos",
  "estabais",
  "estaban",
  "estuve",
  "estuviste",
  "estuvo",
  "estuvimos",
  "estuvisteis",
  "estuvieron",
  "estuviera",
  "estuvieras",
  "estuviéramos",
  "estuvierais",
  "estuvieran",
  "estuviese",
  "estuvieses",
  "estuviésemos",
  "estuvieseis",
  "estuviesen",
  "estando",
  "estado",
  "estada",
  "estados",
  "estadas",
  "estad",
  "he",
  "has",
  "ha",
  "hemos",
  "habéis",
  "han",
  "haya",
  "hayas",
  "hayamos",
  "hayáis",
  "hayan",
  "habré",
  "habrás",
  "habrá",
  "habremos",
  "habréis",
  "habrán",
  "habría",
  "habrías",
  "habríamos",
  "habríais",
  "habrían",
  "había",
  "habías",
  "habíamos",
  "habíais",
  "habían",
  "hube",
  "hubiste",
  "hubo",
  "hubimos",
  "hubisteis",
  "hubieron",
  "hubiera",
  "hubieras",
  "hubiéramos",
  "hubierais",
  "hubieran",
  "hubiese",
  "hubieses",
  "hubiésemos",
  "hubieseis",
  "hubiesen",
  "habiendo",
  "habido",
  "habida",
  "habidos",
  "habidas",
  "soy",
  "eres",
  "es",
  "somos",
  "sois",
  "son",
  "sea",
  "seas",
  "seamos",
  "seáis",
  "sean",
  "seré",
  "serás",
  "será",
  "seremos",
  "seréis",
  "serán",
  "sería",
  "serías",
  "seríamos",
  "seríais",
  "serían",
  "era",
  "eras",
  "éramos",
  "erais",
  "eran",
  "fui",
  "fuiste",
  "fue",
  "fuimos",
  "fuisteis",
  "fueron",
  "fuera",
  "fueras",
  "fuéramos",
  "fuerais",
  "fueran",
  "fuese",
  "fueses",
  "fuésemos",
  "fueseis",
  "fuesen",
  "sintiendo",
  "sentido",
  "sentida",
  "sentidos",
  "sentidas",
  "siente",
  "sentid",
  "tengo",
  "tienes",
  "tiene",
  "tenemos",
  "tenéis",
  "tienen",
  "tenga",
  "tengas",
  "tengamos",
  "tengáis",
  "tengan",
  "tendré",
  "tendrás",
  "tendrá",
  "tendremos",
  "tendréis",
  "tendrán",
  "tendría",
  "tendrías",
  "tendríamos",
  "tendríais",
  "tendrían",
  "tenía",
  "tenías",
  "teníamos",
  "teníais",
  "tenían",
  "tuve",
  "tuviste",
  "tuvo",
  "tuvimos",
  "tuvisteis",
  "tuvieron",
  "tuviera",
  "tuvieras",
  "tuviéramos",
  "tuvierais",
  "tuvieran",
  "tuviese",
  "tuvieses",
  "tuviésemos",
  "tuvieseis",
  "tuviesen",
  "teniendo",
  "tenido",
  "tenida",
  "tenidos",
  "tenidas",
  "tened",
  "ve",
  "dia",
  "algun",
  "ningun",
  "pregunta",
  "segunda",
  "bugun",
  "mas",
  "da",
  "alguna",
  "si",
  "bur",
  "bu",
  "icin",
  "bir",
  "um",
  "know",
  "mais",
  "pra",
  "time",
  "q",
  "em",
  "re",
  "11",
  "isnt",
  "wan",
  "ver",
  "like",
  "'re",
  "m",
  "'ve",
  "bec",
  "n",
  "twt",
  "kca",
  "c",
  "a",
  "b",
  "d",
  "e",
  "f",
  "g",
  "h",
  "i",
  "j",
  "k",
  "l",
  "m",
  "n",
  "o",
  "p",
  "q",
  "r",
  "s",
  "t",
  "u",
  "v",
  "w",
  "x",
  "y",
  "z",
]


# sqlite-based cache
class CacheSqlite:
  connection = None
  cursor = None
  tables = []

  def __init__(self):
    # in-memory sqlite based cache
    self.connection = sqlite3.connect(
      ":memory:", check_same_thread=False, isolation_level=None
    )
    self.cursor = self.connection.cursor()
    self.cursor.execute("PRAGMA journal_mode=wal")
    self.cursor.execute("PRAGMA wal_checkpoint=TRUNCATE")

    # start cache cleaning
    self.cleanCache()

  # cleans older than 60 seconds cache elements (those will be regenerated either way by update_hist_graph_scatter)
  def cleanCache(self):
    # Run again in 30 seconds
    Timer(30, self.cleanCache).start()

    # Clean old entries
    for table in self.tables:
      self.cursor.execute(f"DELETE FROM {table} WHERE expires < ?", (int(time.time()),))

  # Get cache element
  def get(self, pool, key):
    # Table doesn't exist, so key can't as well
    if pool not in self.tables:
      return None

    # Get data from cache
    result = self.cursor.execute(
      f"SELECT value FROM {pool} WHERE key = ?", (key,)
    ).fetchone()

    # No result
    if not result:
      return None

    # Load pickle
    return pickle.loads(result[0])

  # Set element in cache
  def set(self, pool, key, value, ttl=0):
    # If new pool, create table
    if pool not in self.tables:
      query = f"""
                CREATE TABLE IF NOT EXISTS 
                {pool}(key TEXT PRIMARY KEY, value TEXT, expires INTEGER)
            """
      self.cursor.execute(query)

      query = f"CREATE INDEX expires_{pool} ON {pool} (expires ASC)"
      self.cursor.execute(query)
      self.tables.append(pool)

    # Store value with key
    self.cursor.execute(
      f"REPLACE INTO {pool} VALUES (?, ?, ?)",
      (
        key,
        pickle.dumps(value),
        int(time.time() + ttl) if ttl > 0 and ttl <= 2592000 else ttl,
      ),
    )


class Listener(StreamListener):
  data = []
  lock = None

  def __init__(self, lock, analyzer, cursor):
    self.lock = lock  # Create lock
    self.analyzer = analyzer  # Sentiment analyzer
    self.c = cursor  # SQLite cursor

    # self.engine = create_engine('sqlite:///tweets.sqlite')

    self.saveDb()  # Init timer for database save

    # Call __init__ of super class
    super().__init__()

  def saveDb(self):
    # Set a timer (1 second)
    Timer(1, self.saveDb).start()

    # With lock, if there's data, save in transaction using one bulk query
    with self.lock:
      if len(self.data):
        self.c.execute("BEGIN TRANSACTION")
        try:
          self.c.executemany(
            "INSERT INTO sentiment (unix, tweet, sentiment) VALUES (?, ?, ?)", self.data
          )
        except:
          pass
        self.c.execute("COMMIT")

        self.data = []

  def on_data(self, data):
    try:
      # print('data')
      data = json.loads(data)
      # Record format:
      # {'limit': {'track': 14667, 'timestamp_ms': '1520216832822'}}
      if "truncated" not in data:
        # print(data)
        return True
      if data["truncated"]:
        tweet = unidecode(data["extended_tweet"]["full_text"])
      else:
        tweet = unidecode(data["text"])
      timeMs = data["timestamp_ms"]
      vs = self.analyzer.polarity_scores(tweet)
      sentiment = vs["compound"]
      # print(time_ms, tweet, sentiment)

      # append to data list (to be saved every 1 second)
      with self.lock:
        self.data.append((timeMs, tweet, sentiment))

    except KeyError as e:
      # print(data)
      print(str(e))
    return True

  def on_error(self, status):
    print(status)


def dbTruncate(days=3):
  path = r"C:\Users\danfy\OneDrive\FinAnly\data\twitter.db"
  conn = sqlite3.connect(path, check_same_thread=False)
  c = conn.cursor()

  currentTime = time.time()
  delTo = int((currentTime - (days * 86400)) * 1e3)

  query = f"""
        DELETE FROM sentiment_fts WHERE rowid 
        IN (SELECT id FROM sentiment WHERE unix <  {delTo})
    """
  c.execute(query)

  query = f"DELETE FROM sentiment WHERE unix < {delTo}"
  c.execute(query)

  # Bulk delete
  # c.execute('VACUUM')
  conn.commit()
  conn.close()


def mapNouns(col):
  return [word[0] for word in TextBlob(col).tags if word[1] == "NNP"]


# Trending terms
def generateTrending(lock, conn, c, splitRegex, blacklistCounter):
  try:
    # Select last 10k tweets
    query = """
            SELECT * FROM sentiment 
            ORDER BY id DESC, unix DESC LIMIT 10000
        """

    df = pd.read_sql(query, conn)
    df["nouns"] = list(map(mapNouns, df["tweet"]))

    # Make tokens
    tokens = splitRegex.split(
      " ".join(list(itertools.chain.from_iterable(df["nouns"].values.tolist()))).lower()
    )
    # Clean and get top 10
    trend = (Counter(tokens) - blacklistCounter).most_common(10)

    # Get sentiments
    trendSentiment = {}
    for term, count in trend:
      query = """
                SELECT sentiment.* FROM  sentiment_fts 
                fts LEFT JOIN sentiment ON fts.rowid = sentiment.id 
                WHERE fts.sentiment_fts MATCH ? 
                ORDER BY fts.rowid DESC LIMIT 1000
            """

      df = pd.read_sql(query, conn, params=(term,))
      trendSentiment[term] = [df["sentiment"].mean(), count]

    # Save in a database
    with lock:
      c.execute("BEGIN TRANSACTION")
      try:
        c.execute(
          'REPLACE INTO misc (key, value) VALUES ("trending", ?)',
          (pickle.dumps(trendSentiment),),
        )
      except:
        pass
      c.execute("COMMIT")

  except Exception as e:
    with open("errors.txt", "a") as f:
      f.write(str(e))
      f.write("\n")
  finally:
    Timer(5, generateTrending).start()


def twitterStream():
  def createTable(c):
    try:
      # Allows concurrent write and reads http://www.sqlite.org/pragma.html#pragma_journal_mode
      c.execute("PRAGMA journal_mode=wal")
      c.execute("PRAGMA wal_checkpoint=TRUNCATE")
      # c.execute('PRAGMA journal_mode=PERSIST')

      # Changed unix to INTEGER (it is integer, sqlite can use up to 8-byte long integers)
      c.execute("""
                CREATE TABLE IF NOT EXISTS 
                sentiment(id INTEGER PRIMARY KEY AUTOINCREMENT, unix INTEGER, tweet TEXT, sentiment REAL)
            """)
      # Key-value table for random stuff
      c.execute("CREATE TABLE IF NOT EXISTS misc(key TEXT PRIMARY KEY, value TEXT)")
      # Id on index, both as DESC (as you are sorting in DESC order)
      c.execute("CREATE INDEX id_unix ON sentiment (id DESC, unix DESC)")
      # out full-text search table, i choosed creating data from external (content) table - sentiment
      # instead of directly inserting to that table, as we are saving more data than just text
      # https://sqlite.org/fts5.html - 4.4.2
      c.execute("""
                CREATE VIRTUAL TABLE sentiment_fts 
                USING fts5(tweet, content=sentiment, content_rowid=id, prefix=1, prefix=2, prefix=3)
            """)
      # that trigger will automagically update out table when row is interted
      # (requires additional triggers on update and delete)
      c.execute("""
                CREATE TRIGGER sentiment_insert AFTER INSERT ON sentiment 
                BEGIN
                    INSERT INTO sentiment_fts(rowid, tweet) VALUES (new.id, new.tweet);
                END
            """)
    except Exception as e:
      print(str(e))

  # Twitter API keys
  ckey = "FpVpgILWD8UhzzvuIMxDOTYs3"
  csecret = "oIXr0dskMhMCFerzWgb6mmqzzOVIyWHlGBaIk4Yg8reM1DiOzm"
  atoken = "1257725628020936707-DydLSUTCLjJqStSUX3vWHwExRU4ptK"
  asecret = "HtQ9yeZibBhY0T9PcEXxDxcWRUZ4yRRqs1OnYkC0Eilvm"

  # Create database
  path = r"C:\Users\danfy\OneDrive\FinAnly\data\twitter.db"
  conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
  c = conn.cursor()
  createTable(c)

  # Sentimental analyzer
  analyzer = SentimentIntensityAnalyzer()

  # Make a counter with blacklist words and empty word with some big value - we'll use it later to filter counter
  stopWords.append("")
  blacklistCounter = Counter(dict(zip(stopWords, [1e6] * len(stopWords))))

  # Compile a regex for split operations (punctuation list, plus space and new line)
  punctuation = [str(i) for i in string.punctuation]
  splitRegex = re.compile("[ \n" + re.escape("".join(punctuation)) + "]")

  # Lock and timer
  lock = Lock()
  args = (lock, conn, c, splitRegex, blacklistCounter)
  Timer(1, generateTrending, args=args).start()  # deamon = True

  while True:
    try:
      auth = OAuthHandler(ckey, csecret)
      auth.set_access_token(atoken, asecret)
      twitterStream = Stream(auth, Listener(lock, analyzer, c))
      twitterStream.filter(track=["a", "e", "i", "o", "u"], is_async=True)

    except Exception as e:
      with open("errors.txt", "a") as f:
        f.write(str(e))
        f.write("\n")
