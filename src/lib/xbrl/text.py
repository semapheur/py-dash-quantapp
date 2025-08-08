from collections import OrderedDict
import html
import re
from typing import cast, TypedDict

from selectolax.parser import HTMLParser, Node

from lib.utils.string import is_title


class TableData(TypedDict):
  unit: str
  values: list[list[str | int | float]]


class TableInfo(TypedDict):
  caption: str
  unit: str
  data: dict[str, TableData]


class ElementData(TypedDict):
  text: str
  font_size: float
  top_position: float
  left_position: float
  is_bold: bool
  char_length: int
  is_heading: bool


class ParserResult(TypedDict):
  hierarchy: OrderedDict[str, list[str]]
  tables: list[TableInfo]


class HTMLTextParser:
  def __init__(self) -> None:
    self.elements: list[ElementData] = []
    self.tables: list[TableInfo] = []
    self.font_sizes: set[float] = set()

  def parse(self, html_content: str) -> ParserResult:
    tree = HTMLParser(html_content)

    self.elements = self._extract_elements(tree)
    self.tables = self._extract_tables(tree)
    self.elements.sort(key=lambda x: (x["top_position"], x["left_position"]))

    self._analyze_hierarchy_patterns()

    hierarchy = self._build_hierarchy()

    return ParserResult(hierarchy=hierarchy, tables=self.tables)

  def _extract_tables(self, tree: HTMLParser) -> list[TableInfo]:
    tables = []

    table_elements = tree.css("table")

    for table_element in table_elements:
      table_info = self._parse_single_table(table_element)
      if table_info["data"]:
        tables.append(table_info)

    return tables

  def _parse_single_table(self, table_element: Node) -> TableInfo:
    unit_patterns = {"%", "$", "€", "£", "¥"}

    table_rows = table_element.css("tr")
    if not table_rows:
      return TableInfo(caption="", unit="", data={})

    caption = self._extract_caption(table_element)
    unit = ""
    if caption:
      unit = self._extract_unit(caption)

    if not unit:
      unit = self._extract_unit(table_element.text(strip=True))

    headers: list[str] = []
    data_rows: list[list[str]] = []
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

        headers = cell_texts
        continue

      data_rows.append(cell_texts)

    if not data_rows:
      return TableInfo(caption="", unit="", data={})

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

      values = set(data_columns[i]).difference(set(""))
      if len(values) == 1:
        current_unit = values.pop()
        continue

      table_data[current_header] = TableData(unit=current_unit, values=data_columns[i])
      current_unit = ""

    return TableInfo(caption=caption, unit=unit, data=table_data)

  def _extract_caption(self, table_element: Node) -> str:
    caption = ""

    parent = table_element.parent
    if parent:
      for sibling in parent.css("*"):
        if sibling == table_element:
          break
        text = sibling.text(strip=True) if hasattr(sibling, "text") else ""
        if text and len(text) > 10:
          caption = re.sub(r"\s+", " ", text).strip()
          caption = re.sub(r":$", "", caption)
          break

    return caption

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
    self, cell_texts: list[str], row_index: int, row_element: Node
  ) -> bool:
    non_empty_texts = [text.strip() for text in cell_texts if text.strip()]

    if not non_empty_texts:
      return False

    th_elements = row_element.css("th")
    if th_elements:
      return True

    if len(non_empty_texts) == 1 and len(cell_texts) > 1:
      return False

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
      re.search(pattern, text.lower(), re.IGNORECASE)
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
      if "font-weight:bold" in style or "font-weight: bold" in style:
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

  def _extract_elements(self, tree: HTMLParser) -> list[ElementData]:
    def is_inside_table(element: Node) -> bool:
      current: Node | None = element
      table_tags = {"table", "tbody", "thead", "tfoot", "tr", "td", "th"}
      while current is not None:
        if (
          hasattr(current, "tag") and current.tag and current.tag.lower() in table_tags
        ):
          return True

        current = getattr(current, "parent", None)

      return False

    elements: list[ElementData] = []
    seen_texts: set[str] = set()

    text_pattern = r"(?<=<body>)[^<]+?(?=<)"
    toplevel_match = re.search(text_pattern, str(tree.html), re.DOTALL)
    if toplevel_match:
      text = toplevel_match.group(0).strip()
      elements.append(
        ElementData(
          text=text,
          font_size=0.0,
          top_position=0.0,
          left_position=0.0,
          is_bold=False,
          is_heading=False,
          char_length=len(text),
        )
      )

    span_selector = (
      "span:not(table span):not(tbody span):not(thead span):not(tfoot span)"
      ":not(tr span):not(td span):not(th span)"
    )

    for span in tree.css(span_selector):
      if is_inside_table(span):
        continue

      text = span.text(strip=True)
      if not text or text in seen_texts:
        continue

      element_data = self._extract_element_data(span, "span")
      if element_data:
        elements.append(element_data)
        seen_texts.add(text)

    div_selector = (
      "div:not(table div):not(tbody div):not(thead div):not(tfoot div)"
      ":not(tr div):not(td div):not(th div)"
    )

    for div in tree.css(div_selector):
      span_child = div.css_first("span")
      if span_child is None:
        continue

      text = span_child.text(strip=True)
      if not text or text in seen_texts:
        continue

      element_data = self._extract_element_data(span, "div_span", div)
      if element_data:
        elements.append(element_data)
        seen_texts.add(text)

    return elements

  def _extract_element_data(
    self, span: Node, element_type: str, parent_div: Node | None = None
  ) -> ElementData | None:
    text = span.text(strip=True)
    if not text:
      return None

    text = html.unescape(text)
    text = self._normalize_unicode(text)

    span_style = cast(str, span.attributes.get("style", ""))

    font_size = self._extract_font_size(span_style)
    if font_size > 0.0:
      self.font_sizes.add(font_size)

    font_weight = self._extract_font_weight(span_style)

    if element_type == "span":
      top_position = self._extract_top_position(span_style)
      left_position = self._extract_left_position(span_style)
    else:
      div_style = cast(
        str, parent_div.attributes.get("style", "") if parent_div is not None else ""
      )
      top_position = self._extract_top_position(div_style) or self._extract_margin_top(
        div_style
      )
      left_position = self._extract_left_position(span_style) or 0

    return ElementData(
      text=text,
      font_size=font_size,
      top_position=top_position or 0.0,
      left_position=left_position or 0.0,
      is_bold=font_weight >= 700,
      is_heading=False,
      char_length=len(text),
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
    if "font-weight:bold" in style or "font-weight: bold" in style:
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
        heading_score += 1

      if is_bold:
        heading_score += 1

      if words < 6:
        heading_score += 1

      if words > 1 and is_title(text):
        heading_score += 1

      for phrase in heading_phrases:
        if re.match(rf"^{phrase}\b", text):
          heading_score += 3

      if any(word in text_lower for word in sentence_words) > 0:
        heading_score -= 4

      if any(re.search(pattern, text, flags=re.I) for pattern in verb_patterns) > 0:
        heading_score -= 4

      if text.count(".") >= 1:
        heading_score -= 3

      if text.count(",") >= 2:
        heading_score -= 3

      if "?" in text:
        heading_score -= 3

      element["is_heading"] = heading_score >= 3 and not text.endswith(",")

  def _build_hierarchy(self) -> OrderedDict[str, list[str]]:
    result = OrderedDict()

    has_any_heading = any(element["is_heading"] for element in self.elements)

    if not has_any_heading:
      all_content = []
      for element in self.elements:
        text = element["text"].strip()
        if text:
          all_content.append(text)

        if all_content:
          paragraphs = self._combine_content_fragments(all_content)
          result[""] = paragraphs

      return result

    current_heading: str | None = None
    current_content: list[str] = []
    content_before_first_heading: list[str] = []

    for element in self.elements:
      text = element["text"].strip()
      if not text:
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

    paragraphs = []
    current_paragraph = ""

    for fragment in content_fragments:
      fragment = fragment.strip()
      if not fragment:
        continue

      if not current_paragraph:
        current_paragraph = fragment
      else:
        should_combine = (
          # Previous doesn't end with sentence punctuation
          not current_paragraph.rstrip().endswith((".", "!", "?"))
          or
          # Current fragment doesn't start with capital letter (continuation)
          (fragment and not fragment[0].isupper())
          or
          # Both fragments are short (likely parts of same sentence)
          (len(current_paragraph) < 100 and len(fragment) < 100)
        )

        if should_combine and len(current_paragraph + " " + fragment) < 500:
          current_paragraph += " " + fragment
        else:
          if current_paragraph:
            paragraphs.append(current_paragraph)
          current_paragraph = fragment

    if current_paragraph:
      paragraphs.append(current_paragraph)

    return paragraphs
