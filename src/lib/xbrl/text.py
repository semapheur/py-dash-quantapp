import html
import re
from typing import cast, TypedDict

from selectolax.parser import HTMLParser, Node

from lib.utils.string import is_title

TABLE_PREFIX = "__table_"


class TableData(TypedDict):
  unit: str
  values: list[str | int | float]


class TableInfo(TypedDict):
  caption: str
  unit: str
  data: dict[str, TableData]


class ElementData(TypedDict):
  text: str
  font_size: float
  is_bold: bool
  is_heading: bool


class ParserResult(TypedDict):
  text: dict[str, list[str]]
  tables: list[TableInfo]


class HTMLTextParser:
  def __init__(self) -> None:
    self.elements: list[ElementData] = []
    self.tables: list[TableInfo] = []
    self.font_sizes: set[float] = set()

  def parse(self, html_content: str) -> ParserResult:
    tree = HTMLParser(html_content)

    self._extract_elements_and_tables(tree)
    self._analyze_hierarchy_patterns()
    text = self._build_hierarchy()

    return ParserResult(text=text, tables=self.tables)

  def _extract_elements_and_tables(self, tree: HTMLParser) -> None:
    table_index = 0

    body_node = tree.css_first("body")
    if body_node is None:
      raise ValueError(f"Could not find body element in HTML: {tree.html}")

    toplevel_text = body_node.text(deep=False)

    if toplevel_text:
      self.elements.append(
        ElementData(
          text=toplevel_text,
          font_size=0.0,
          is_bold=False,
          is_heading=False,
        )
      )

    for node in body_node.traverse():
      tag = node.tag.lower()

      if not tag:
        continue

      if tag == "table":
        table_info = self._parse_single_table(node)
        if not table_info["data"]:
          continue

        self.elements.append(
          ElementData(
            text=f"{TABLE_PREFIX}{table_index}",
            font_size=0.0,
            is_bold=False,
            is_heading=False,
          )
        )
        self.tables.append(table_info)
        table_index += 1
        continue

      if tag == "span" and not self._is_inside_table(node):
        text = node.text(deep=False)
        if not text.strip():
          continue

        span_style = cast(str, node.attributes.get("style", ""))
        element_data = self._extract_element_data(text, span_style)
        self.elements.append(element_data)

  def _extract_tables(self, tree: HTMLParser) -> list[TableInfo]:
    tables = []

    table_elements = tree.css("table")

    for table_element in table_elements:
      table_info = self._parse_single_table(table_element)
      if table_info["data"]:
        tables.append(table_info)

    return tables

  def _parse_single_table(self, table_element: Node) -> TableInfo:
    # unit_patterns = {"%", "$", "€", "£", "¥"}

    table_rows = table_element.css("tr")
    if not table_rows:
      return TableInfo(caption="", unit="", data={})

    unit = self._extract_unit(table_element.text(strip=True))

    headers: list[str] = []
    data_rows: list[list[str | int | float]] = []
    first_non_empty_row = True

    for row_index, row in enumerate(table_rows):
      cells = row.css("td, th")
      if not cells:
        continue

      cell_texts: list[str | int | float] = []
      for cell in cells:
        colspan = int(cell.attributes.get("colspan", 1))
        cell_text = self._extract_cell_text(cell)
        cell_texts.extend([cell_text] + [""] * (colspan - 1))

      if not any(text for text in cell_texts):
        continue

      if first_non_empty_row:
        first_non_empty_row = False
        if not self._is_header_row(cell_texts, row_index, row):
          continue

        headers = [str(text) for text in cell_texts]
        continue

      data_rows.append(cell_texts)

    if not data_rows:
      return TableInfo(caption="", unit="", data={})

    del_rows: list[int] = []
    for i in range(len(data_rows) - 1):
      data_row = data_rows[i]
      if data_row[0] and not any(c for c in data_row[1:]):
        next_row = data_rows[i + 1]
        next_row[0] = f"{data_row[0]} {next_row[0]}"
        del_rows.append(i)

    if del_rows:
      for i in reversed(del_rows):
        del data_rows[i]

    table_data: dict[str, TableData] = {}
    data_columns = list(map(list, zip(*data_rows)))

    columns = len(data_columns)
    if not headers:
      headers = [f"column_{i + 1}" for i in range(columns)]

    current_header = ""
    current_unit = ""
    for i, header in enumerate(headers):
      if header:
        current_header = header

      if not current_header:
        continue

      if not any(text for text in data_columns[i]):
        continue

      values = set(data_columns[i]).difference(("",))
      if len(values) == 1:
        value = values.pop()
        if isinstance(value, str):
          current_unit = value

        continue

      table_data[current_header] = TableData(unit=current_unit, values=data_columns[i])
      current_unit = ""

    return TableInfo(caption="", unit=unit, data=table_data)

  def _is_inside_table(self, element: Node) -> bool:
    current: Node | None = element
    table_tags = {"table", "tbody", "thead", "tfoot", "tr", "td", "th"}
    while current is not None:
      if hasattr(current, "tag") and current.tag and current.tag.lower() in table_tags:
        return True

      current = getattr(current, "parent", None)

    return False

  def _extract_unit(self, text: str) -> str:
    text_lower = text.lower()

    unit_pattern = r"\((in .*?(?:thousands|millions|billions|trillions)?.*?)\)"
    unit_match = re.search(unit_pattern, text_lower)
    if unit_match is not None:
      return unit_match.group(0)

    return ""

  def _extract_cell_text(self, cell_element: Node) -> str | int | float:
    text = cell_element.text(strip=True)

    if not text:
      return ""

    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = self._normalize_unicode(text)

    return self._process_cell_value(text)

  def _is_header_row(
    self, cell_texts: list[str | int | float], row_index: int, row_element: Node
  ) -> bool:
    non_empty_texts = [str(text).strip() for text in cell_texts if str(text).strip()]

    if not non_empty_texts:
      return False

    if any(isinstance(text, float) for text in cell_texts):
      return False

    if len(non_empty_texts) == 1 and len(cell_texts) > 1:
      return False

    th_elements = row_element.css("th")
    if th_elements:
      return True

    has_bold_cells = any(self._has_bold_styling(cell) for cell in row_element.css("td"))
    if has_bold_cells:
      return True

    header_patterns = [
      r"^year$",
      r"^period$",
      r"^amount$",
      r"^value$",
      r"^note$",
      r"^total(?:\s+\w+)*$",  # "total", "total assets", etc.
      r"^description$",
      r"^category$",
      r"^type$",
      r"^name$",
      r"^item$",
      r"^\d{4}$",  # Years like "2024", "2025"
      r"^q[1-4]$",  # Quarters like "Q1", "Q2"
      r"^[a-z\s]{2,20}$",  # Short descriptive text (2-20 chars)
    ]

    has_header_pattern = any(
      re.search(pattern, str(text).lower(), re.IGNORECASE)
      for text in non_empty_texts
      for pattern in header_patterns
    )
    if has_header_pattern:
      return True

    if row_index > 5:
      return False

    if len(non_empty_texts) >= 1:
      all_short = all(len(text) <= 50 for text in non_empty_texts)

      numeric_pattern = r"^[\d,.\s$%-]+$"
      mostly_non_numeric = (
        sum(
          1 for text in non_empty_texts if not re.match(numeric_pattern, text.strip())
        )
        >= len(non_empty_texts) * 0.7
      )

      has_text_content = any(re.search(r"[a-zA-Z]", text) for text in non_empty_texts)

      if all_short and has_text_content and (mostly_non_numeric or row_index <= 2):
        return True

    if (
      len(non_empty_texts) == 1
      and row_index <= 2
      and len(non_empty_texts[0]) > 5
      and re.search(r"[a-zA-Z]", non_empty_texts[0])
    ):
      return True

    return False

  def _has_bold_styling(self, element: Node) -> bool:
    def is_bold(style: str) -> bool:
      if re.search(r"font-weight:\s*bold|text-decoration:\s*underline", style):
        return True

      weight_match = re.search(r"font-weight:\s*(\d+)", style)
      return bool(weight_match and int(weight_match.group(1)) >= 700)

    style = element.attributes.get("style", "")

    if not style:
      return False

    if is_bold(style):
      return True

    for child in element.css("*"):
      child_style = child.attributes.get("style", "")

      if not child_style:
        continue

      if is_bold(child_style):
        return True

    return False

  def _extract_element_data(self, text: str, span_style: str) -> ElementData:
    text = html.unescape(text)
    text = self._normalize_unicode(text)

    font_size = self._extract_font_size(span_style)
    if font_size > 0.0:
      self.font_sizes.add(font_size)

    font_weight = self._extract_font_weight(span_style)

    return ElementData(
      text=text,
      font_size=font_size,
      is_bold=font_weight >= 700,
      is_heading=False,
    )

  def _normalize_unicode(self, text: str) -> str:
    replacements = {
      "\u2019": "'",  # Right single quotation mark
      "\u2018": "'",  # Left single quotation mark
      "\u201c": '"',  # Left double quotation mark
      "\u201d": '"',  # Right double quotation mark
      "\u2013": "-",  # En dash
      "\u2014": "—",  # Em dash
      "\u00a0": " ",  # Non-breaking space
    }

    for unicode_char, replacement in replacements.items():
      text = text.replace(unicode_char, replacement)

    return text

  def _clean_headers(self, headers: list[str]) -> list[str]:
    cleaned: list[str] = []
    for header in headers:
      clean_header = header.strip()

      if not clean_header:
        clean_header = f"column_{len(cleaned) + 1}"

      clean_header = re.sub(r"[^\w\s-]", "", clean_header)
      clean_header = re.sub(r"\s+", "_", clean_header)

      original_header = clean_header
      counter = 1
      while clean_header in cleaned:
        clean_header = f"{original_header}_{counter}"
        counter += 1

      cleaned.append(clean_header)

    return cleaned

  def _process_cell_value(self, value: str) -> str | int | float:
    if not value or not value.strip():
      return ""

    value = value.strip()
    numeric_value = re.sub(r"[,$\s]", "", value)

    if re.match(r"^-?\d+$", numeric_value):
      return int(numeric_value)
    elif re.match(r"^-?\d*\.\d+$", numeric_value):
      return float(numeric_value)
    elif re.match(r"^-?\d+,\d+$", numeric_value):
      return float(numeric_value.replace(",", "."))

    return value

  def _extract_font_size(self, style: str) -> float:
    match = re.search(r"font-size:\s*(\d+(?:\.\d+)?)pt", style)
    return float(match.group(1)) if match else 0.0

  def _extract_font_weight(self, style: str) -> int:
    match = re.search(r"font-weight:\s*(\d+)", style)
    if match:
      return int(match.group(1))
    if re.search(r"font-weight:\s*bold|text-decoration:\s*underline", style):
      return 700
    return 400

  def _extract_top_position(self, style: str) -> float | None:
    match = re.search(r"top:\s*(\d+(?:\.\d+)?)pt", style)
    return float(match.group(1)) if match else None

  def _extract_left_position(self, style: str) -> float | None:
    match = re.search(r"left:\s*(\d+(?:\.\d+)?)pt", style)
    return float(match.group(1)) if match else None

  def _extract_margin_top(self, style: str) -> float | None:
    match = re.search(r"margin-top:\s*(\d+(?:\.\d+)?)pt", style)
    return float(match.group(1)) if match else None

  def _analyze_hierarchy_patterns(self):
    if not self.elements:
      return

    mean_front_size = 0.0
    if self.font_sizes:
      mean_front_size = sum(self.font_sizes) / len(self.font_sizes)

    heading_phrases = ("Notes to", "Summary of")

    sentence_words = {
      " although ",
      " because ",
      " but ",
      " hence ",
      " however ",
      " if ",
      " likewise ",
      " nor ",
      " since ",
      " so ",
      " that ",
      " therefore ",
      " though ",
      " thus ",
      " which ",
      " when ",
      " where ",
      " whereas ",
      " whether ",
      " while ",
      " unless ",
      " until ",
      " yet ",
    }

    verb_patterns = (
      r"\b(?:is|are|was|were|has|have|had|does|do|did|will|would|can|could|should|must|may|might)\b",
      r"\b\w+(ify|fies)\b",
    )

    for element in self.elements:
      text = element["text"]
      text_lower = text.lower()
      font_size = element["font_size"]
      is_bold = element["is_bold"]
      words = len(text.split())

      heading_score = 0

      if (font_size > 0 and mean_front_size > 0) and font_size > mean_front_size:
        heading_score += 3

      if is_bold:
        heading_score += 3

      if words < 6:
        heading_score += 1

      if words > 1 and is_title(text):
        heading_score += 1

      for phrase in heading_phrases:
        if re.match(rf"{phrase}\b", text):
          heading_score += 3

      if text.endswith(" "):
        heading_score -= 4

      if any(word in text_lower for word in sentence_words) > 0:
        heading_score -= 4

      if any(re.search(pattern, text, flags=re.I) for pattern in verb_patterns) > 0:
        heading_score -= 5

      if text[0].islower():
        heading_score -= 4

      if text.count(".") >= 1:
        heading_score -= 3

      if text.count(",") >= 2:
        heading_score -= 3

      if "?" in text:
        heading_score -= 3

      element["is_heading"] = heading_score >= 3 and not text.endswith(",")

  def _build_hierarchy(self) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}

    has_any_heading = any(element["is_heading"] for element in self.elements)

    if not has_any_heading:
      all_content = [element["text"] for element in self.elements]
      if all_content:
        paragraphs = self._combine_content_fragments(all_content)
        result[""] = paragraphs

      return result

    current_heading: str | None = None
    current_content: list[str] = []
    content_before_first_heading: list[str] = []

    for element in self.elements:
      text = element["text"]
      if not text.strip():
        continue

      is_heading = element.get("is_heading", False)

      if is_heading:
        if current_heading is None and content_before_first_heading:
          paragraphs = self._combine_content_fragments(content_before_first_heading)
          result[""] = paragraphs
          content_before_first_heading = []

        if current_heading is not None:
          paragraphs = self._combine_content_fragments(current_content)
          result[current_heading] = paragraphs
          current_content = []

        current_heading = text

      else:
        if current_heading is None:
          content_before_first_heading.append(text)
        else:
          if not current_content and text[0].islower():
            current_heading += " " + text
            continue

          current_content.append(text)

    if current_heading is not None:
      paragraphs = self._combine_content_fragments(current_content)
      result[current_heading] = paragraphs

    elif content_before_first_heading:
      paragraphs = self._combine_content_fragments(content_before_first_heading)
      result[""] = paragraphs

    return result

  def _combine_content_fragments(self, content_fragments: list[str]) -> list[str]:
    if not content_fragments:
      return []

    table_pattern = re.compile(rf"{TABLE_PREFIX}(\d+)$")
    paragraphs = []
    current_paragraph = ""
    end_punctuations = (".", "!", "?", ":")
    start_punctuations = (".", ",", "!", "?", ":", "'")

    for fragment in content_fragments:
      fragment = fragment
      if not fragment:
        continue

      table_match = table_pattern.match(fragment)
      if table_match:
        if current_paragraph:
          paragraphs.append(current_paragraph)
          table_index = int(table_match.group(1))
          table = self.tables[table_index]
          table["caption"] = current_paragraph
          if not table["unit"]:
            unit = self._extract_unit(current_paragraph)
            if unit:
              table["unit"] = unit

          current_paragraph = ""

        paragraphs.append(fragment)
        continue

      if not current_paragraph:
        current_paragraph = fragment
        continue

      first_upper = fragment[0].isupper()
      ends_with_punctuation = current_paragraph.rstrip().endswith(end_punctuations)
      starts_with_punctuation = fragment.startswith(start_punctuations)
      ends_with_space = current_paragraph.endswith(" ")

      should_combine = (
        not first_upper
        or (first_upper and not ends_with_punctuation)
        or starts_with_punctuation
        or ends_with_space
      )

      if should_combine:
        current_paragraph += fragment
        continue

      if current_paragraph:
        paragraphs.append(current_paragraph)
        current_paragraph = fragment

    if current_paragraph:
      paragraphs.append(current_paragraph)

    return paragraphs
