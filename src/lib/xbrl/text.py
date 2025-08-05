from collections import OrderedDict
import html
import re
from typing import cast, TypedDict

from selectolax.parser import HTMLParser, Node

type TableData = dict[str, list[str | int | float]]


class TableInfo(TypedDict):
  caption: str
  unit: str
  data: TableData


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
  tables: list[TableInfo]


class HTMLTextParser:
  def __init__(self) -> None:
    self.elements: list[ElementData] = []
    self.hierarchy_levels: dict[str, int] = {}
    self.tables: list[TableInfo] = []

  def parse(self, html_content: str) -> ParserResult:
    tree = HTMLParser(html_content)

    self.elements = self._extract_elements(tree)
    self.tables = self._extract_tables(tree)
    self.elements.sort(key=lambda x: (x["position"], x["left_pos"]))

    self._analyze_hierarchy_patterns()

    hierarchy = self._build_hierarchy()

    return {"hierarchy": hierarchy, "tables": self.tables}

  def _extract_tables(self, tree: HTMLParser) -> list[TableInfo]:
    tables = []

    table_elements = tree.css("table")

    for table_index, table_element in enumerate(table_elements):
      table_info = self._parse_single_table(table_element)
      if table_info["data"]:
        tables.append(table_info)

    return tables

  def _parse_single_table(self, table_element: Node) -> TableInfo:
    table_rows = table_element.css("tr")
    if not table_rows:
      return TableInfo(caption="", unit="", data={})

    caption, unit = self._extract_caption_and_unit(table_element)

    headers: list[str] = []
    data_rows: list[list[str]] = []
    header_found = False

    for row_index, row in enumerate(table_rows):
      cells = row.css("td, th")
      if not cells:
        continue

      cell_texts: list[str] = []
      for cell in cells:
        cell_text = self._extract_cell_text(cell)
        cell_texts.append(cell_text)

      if not any(text.strip() for text in cell_texts):
        continue

      if not header_found and self._is_header_row(cell_texts, row_index, row):
        if not unit:
          unit = self._extract_unit_from_header(cell_texts)
        header_found = True
        continue

      if self._is_separator_row(cell_texts, row):
        continue

      data_rows.append(cell_texts)

    if data_rows:
      max_cols = max(len(row) for row in data_rows) if data_rows else 0
      headers = [f"Column_{i + 1}" for i in range(max_cols)]

    if not headers:
      return TableInfo(caption="", unit="", data={})

    table_data: TableData = {}
    for i, header in enumerate(headers):
      column_values = []
      for data_row in data_rows:
        value = data_row[i] if i < len(row) else ""
        processed_value = self._process_cell_value(value)
        column_values.append(processed_value)
      table_data[header] = column_values

    return {"caption": caption, "unit": unit, "data": table_data}

  def _extract_caption_and_unit(self, table_element: Node) -> tuple[str, str]:
    caption = ""
    unit = ""

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

    if caption and not unit:
      unit = self._extract_unit_from_text(caption)

    return caption, unit

  def _extract_unit_from_header(self, header_texts: list[str]) -> str:
    for text in header_texts:
      unit = self._extract_unit_from_text(text)
      if unit:
        return unit
    return ""

  def _extract_unit_from_text(self, text: str) -> str:
    text_lower = text.lower()

    unit_pattern = r"\(in .*?(thousands|millions|billions|trillions)?.*?\)"
    unit_match = re.search(unit_pattern, text_lower)
    if unit_match is not None:
      return unit_match.group()

    return ""

  def _extract_cell_text(self, cell_element) -> str:
    text = cell_element.text(strip=True)

    if not text:
      return ""

    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = self._normalize_unicode(text)

    return text

  def _is_separator_row(self, cell_texts: list[str], row_element) -> bool:
    if not any(text.strip() for text in cell_texts):
      return True

    row_style = row_element.attrs.get("style", "")
    cells = row_element.css("td, th")

    if (
      all(
        "border" in cell.attrs.get("style", "") and not cell.text(strip=True)
        for cell in cells
      )
      and "border" in row_style
    ):
      return True

    return False

  def _is_header_row(self, cell_texts: list[str], row_index: int, row_element) -> bool:
    non_empty_texts = [text.strip() for text in cell_texts if text.strip()]

    if not non_empty_texts:
      return False

    th_elements = row_element.css("th")
    if th_elements:
      return True

    has_bold_cells = any(self._has_bold_styling(cell) for cell in row_element.css("td"))
    if has_bold_cells:
      return True

    header_patterns = [
      r"\(in\s+[\w\s]+\)",  # "(in USD million)", "(in thousands)", etc.
      r"^year$",
      r"^period$",
      r"^amount$",
      r"^value$",
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
    elements = []
    seen_texts = set()

    for span in tree.css("span"):
      text = span.text(strip=True)
      if not text or text in seen_texts:
        continue

      element_data = self._extract_element_data(span, "span")
      if element_data:
        elements.append(element_data)
        seen_texts.add(text)

    for div in tree.css("div"):
      span = div.css_first("span")
      if span:
        text = span.text(strip=True)
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
    font_weight = self._extract_font_weight(span_style)

    if element_type == "span":
      position = self._extract_top_position(span_style)
      left_pos = self._extract_left_position(span_style)
    else:
      div_style = cast(
        str, parent_div.attributes.get("style", "") if parent_div is not None else ""
      )
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
