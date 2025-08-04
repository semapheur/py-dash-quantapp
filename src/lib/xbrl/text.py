from collections import OrderedDict
import html
import re
from typing import TypedDict

from selectolax.parser import HTMLParser

type TableData = dict[str, list[str | int | float]]


class ElementData(TypedDict):
  text: str
  font_size: float
  font_weight: int
  position: float
  left_pos: float
  element_type: str
  is_bold: bool
  char_length: int


class ParserResult(TypedDict):
  hierarchy: OrderedDict[str, list[str]]
  tables: list[TableData]


class HTMLTextParser:
  def __init__(self) -> None:
    self.elements: list[ElementData] = []
    self.hierarchy_levels: dict[str, int] = {}
    self.tables: list[TableData] = []

  def parse(self, html_content: str, debug: bool = False) -> ParserResult:
    tree = HTMLParser(html_content)

    self.elements = self._extract_elements(tree)

    self.tables = self._extract_tables(tree, debug)

    if debug:
      print(f"Extracted {len(self.elements)} elements:")
      for i, elem in enumerate(self.elements):
        print(
          f"  {i}: '{elem['text'][:50]}...' (size: {elem['font_size']}, weight: {elem['font_weight']}, pos: {elem['position']}, bold: {elem['is_bold']})"
        )
      print(f"\nExtracted {len(self.tables)} tables")

    self.elements.sort(key=lambda x: (x["position"], x["left_pos"]))

    self._analyze_hierarchy_patterns()

    if debug:
      print("\nHierarchy levels assigned:")
      for i, elem in enumerate(self.elements):
        print(
          f"  {i}: '{elem['text'][:30]}...' → level {elem.get('hierarchy_level', 'UNSET')}, is_heading: {elem.get('is_heading', 'UNSET')}"
        )

    hierarchy = self._build_hierarchy()

    if debug:
      print(f"\nFinal hierarchy: {hierarchy}")
      for i, table in enumerate(self.tables):
        print(f"\nTable {i + 1}:")
        for col_name, col_data in table.items():
          print(f"  {col_name}: {col_data}")

    return {"hierarchy": hierarchy, "tables": self.tables}

  def _extract_tables(self, tree: HTMLParser, debug: bool = False) -> list[TableData]:
    tables = []

    table_elements = tree.css("table")

    for table_index, table_element in enumerate(table_elements):
      if debug:
        print(f"\n--- Processing Table {table_index + 1} ---")

      table_data = self._parse_single_table(table_element, debug)
      if table_data:
        tables.append(table_data)
        if debug:
          print(
            f"Table {table_index + 1} extracted with columns: {list(table_data.keys())}"
          )
          for col_name, col_data in table_data.items():
            print(f"  {col_name}: {len(col_data)} values")

    return tables

  def _parse_single_table(self, table_element, debug: bool = False) -> TableData:
    rows = table_element.css("tr")

    if not rows:
      return {}

    headers: list[str] = []
    data_rows = []

    for row_index, row in enumerate(rows):
      cells = row.css("td, th")

      if not cells:
        continue

      cell_texts = []
      for cell in cells:
        cell_text = self._extract_cell_text_selectolax(cell)
        cell_texts.append(cell_text)

      if debug:
        print(f"Row {row_index}: {cell_texts}")

      if self._is_header_row_selectolax(cell_texts, row_index, row):
        if not headers:
          headers = self._clean_headers(cell_texts)
          if debug:
            print(f"Headers identified: {headers}")
        continue

      if not any(text.strip() for text in cell_texts):
        continue

      data_rows.append(cell_texts)

    if not headers and data_rows:
      max_cols = max(len(row) for row in data_rows)
      headers = [f"Column_{i + 1}" for i in range(max_cols)]
      if debug:
        print(f"Generated headers: {headers}")

    if not headers:
      return {}

    table_data = {}
    for i, header in enumerate(headers):
      column_values = []
      for row in data_rows:
        value = row[i] if i < len(row) else ""
        processed_value = self._process_cell_value(value)
        column_values.append(processed_value)
      table_data[header] = column_values

    return table_data

  def _extract_cell_text_selectolax(self, cell_element) -> str:
    text = cell_element.text(strip=True)

    if not text:
      return ""

    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = self._normalize_unicode(text)

    return text

  def _is_header_row_selectolax(
    self, cell_texts: list[str], row_index: int, row_element
  ) -> bool:
    th_elements = row_element.css("th")
    if th_elements:
      return True

    for cell in row_element.css("td"):
      if self._has_bold_styling(cell):
        return True

    if all(len(text.strip()) < 50 and text.strip() for text in cell_texts):
      has_numbers = any(re.search(r"\d+[,.]?\d*", text) for text in cell_texts)
      has_long_text = any(len(text.strip()) > 20 for text in cell_texts)

      if not has_numbers and not has_long_text:
        return True

    if row_index == 0:
      return True

    return False

  def _has_bold_styling(self, element) -> bool:
    style = element.attributes.get("style", "")
    if "font-weight:bold" in style or "font-weight: bold" in style:
      return True

    weight_match = re.search(r"font-weight:\s*(\d+)", style)
    if weight_match and int(weight_match.group(1)) >= 700:
      return True

    for child in element.css("*"):
      child_style = child.attributes.get("style", "")
      if "font-weight:bold" in child_style or "font-weight: bold" in child_style:
        return True

      child_weight_match = re.search(r"font-weight:\s*(\d+)", child_style)
      if child_weight_match and int(child_weight_match.group(1)) >= 700:
        return True

    return False

  def _extract_elements(self, tree: HTMLParser) -> list[ElementData]:
    elements = []
    seen_texts = set()

    for span in tree.css("span"):
      text = span.text(strip=True)
      if not text or text in seen_texts:
        continue

      element_data = self._extract_element_data_selectolax(span, "span")
      if element_data:
        elements.append(element_data)
        seen_texts.add(text)

    for div in tree.css("div"):
      span = div.css_first("span")
      if span:
        text = span.text(strip=True)
        if not text or text in seen_texts:
          continue

        element_data = self._extract_element_data_selectolax(span, "div_span", div)
        if element_data:
          elements.append(element_data)
          seen_texts.add(text)

    return elements

  def _extract_element_data_selectolax(
    self, span, element_type: str, parent_div=None
  ) -> ElementData | None:
    text = span.text(strip=True)
    if not text:
      return None

    text = html.unescape(text)
    text = self._normalize_unicode(text)

    span_style = span.attributes.get("style", "")

    font_size = self._extract_font_size(span_style)
    font_weight = self._extract_font_weight(span_style)

    if element_type == "span":
      position = self._extract_top_position(span_style)
      left_pos = self._extract_left_position(span_style)
    else:
      div_style = parent_div.attributes.get("style", "") if parent_div else ""
      position = self._extract_top_position(div_style) or self._extract_margin_top(
        div_style
      )
      left_pos = self._extract_left_position(span_style) or 0

    return ElementData(
      text=text,
      font_size=font_size,
      font_weight=font_weight,
      position=position or 0,
      left_pos=left_pos or 0,
      element_type=element_type,
      is_bold=font_weight >= 700,
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
        clean_header = f"Column_{len(cleaned) + 1}"

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
    return float(match.group(1)) if match else 9.0

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

    for elem in self.elements:
      text = elem["text"]
      font_size = elem["font_size"]
      is_bold = elem["is_bold"]

      heading_score = 0

      if font_size >= 13:
        heading_score += 4
      elif font_size >= 10:
        heading_score += 2
      elif font_size >= 9.5:
        heading_score += 1

      if is_bold:
        heading_score += 3

      if len(text) <= 30:
        heading_score += 2
      elif len(text) <= 50:
        heading_score += 1

      if len(text) < 80 and any(
        word in text.lower()
        for word in [
          "policy",
          "policies",
          "recognition",
          "accounting",
          "revenue",
          "compensation",
          "equivalents",
          "securities",
          "inventories",
          "property",
          "plant",
          "equipment",
          "derivative",
          "instruments",
          "income",
          "taxes",
          "leases",
          "presentation",
          "preparation",
        ]
      ):
        heading_score += 2

      strong_content_patterns = [
        "presents",
        "recognised",
        "recognized",
        "includes",
        "requires",
        "when a",
        "which for",
        "based on the",
        "represents a",
        "determined using",
        "measured using",
        "is recognized",
        "are recognized",
        "records",
        "combines and accounts",
        "conformity with",
        "in accordance",
        "the company",
        "the preparation",
        "financial statements",
        "amounts have been",
        "fiscal year",
        "straight-line basis",
        "fair value",
        "deferred tax",
        "lease component",
      ]

      if any(pattern in text.lower() for pattern in strong_content_patterns):
        heading_score -= 5

      if text.endswith(","):
        heading_score -= 4

      if text.endswith("."):
        heading_score -= 3

      if len(text) > 60 and (" and " in text or " which " in text or " when " in text):
        heading_score -= 4

      if " is " in text.lower() or " are " in text.lower():
        heading_score -= 2

      if text.count(",") >= 2:
        heading_score -= 3

      elem["is_heading"] = heading_score >= 4 and not text.endswith(",")
      elem["heading_score"] = heading_score

      if font_size >= 13:
        elem["hierarchy_level"] = 0  # Main title
      elif font_size >= 9.5 and is_bold:
        elem["hierarchy_level"] = 1  # Subheading
      elif font_size >= 9.5:
        elem["hierarchy_level"] = 2  # Sub-subheading or content
      else:
        elem["hierarchy_level"] = 3  # Content

  def _build_hierarchy(self) -> OrderedDict[str, list[str]]:
    """Build the final hierarchical structure."""
    result = OrderedDict()
    current_heading = None
    current_content: list[str] = []

    for elem in self.elements:
      text = elem["text"].strip()
      is_heading = elem.get("is_heading", False)

      if is_heading:
        if current_heading and current_content:
          paragraphs = self._combine_content_fragments(current_content)
          result[current_heading] = paragraphs
          current_content = []
        elif current_heading and current_heading not in result:
          result[current_heading] = []

        current_heading = text
        current_content = []

      else:
        if current_heading:
          current_content.append(text)

    if current_heading:
      if current_content:
        paragraphs = self._combine_content_fragments(current_content)
        result[current_heading] = paragraphs
      elif current_heading not in result:
        result[current_heading] = []

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
