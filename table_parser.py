"""
table_parser.py - Structured table parser for engineering drawing extraction.

Reconstructs table grid structure from detected lines, supports merged cells
(colspan/rowspan), multi-level headers, and spatial neighbor lookups.
"""

import cv2
import numpy as np
import pytesseract
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class GridCell:
    """A single cell in the table grid, with position, span, and content."""
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    text: str = ""
    text_lines: List[str] = field(default_factory=list)
    is_header: bool = False
    parent_header: Optional['GridCell'] = None
    children: List['GridCell'] = field(default_factory=list)

    @property
    def end_row(self):
        return self.row + self.rowspan - 1

    @property
    def end_col(self):
        return self.col + self.colspan - 1

    @property
    def area(self):
        return self.w * self.h


class TableGrid:
    """A table represented as a grid of cells with merged-cell and hierarchy support."""

    def __init__(self):
        self.cells: List[GridCell] = []
        self.row_boundaries: List[int] = []
        self.col_boundaries: List[int] = []
        self.num_rows: int = 0
        self.num_cols: int = 0
        self._grid_map: Dict[Tuple[int, int], GridCell] = {}

    def get_cell(self, row: int, col: int) -> Optional[GridCell]:
        """Get the cell at a grid position, accounting for merged cells."""
        return self._grid_map.get((row, col))

    def get_neighbor(self, cell: GridCell, direction: str) -> Optional[GridCell]:
        """Get the neighboring cell in a direction: 'up', 'down', 'left', 'right'."""
        if direction == "right":
            target_col = cell.col + cell.colspan
            if target_col < self.num_cols:
                return self._grid_map.get((cell.row, target_col))
        elif direction == "left":
            if cell.col > 0:
                return self._grid_map.get((cell.row, cell.col - 1))
        elif direction == "down":
            target_row = cell.row + cell.rowspan
            if target_row < self.num_rows:
                return self._grid_map.get((target_row, cell.col))
        elif direction == "up":
            if cell.row > 0:
                return self._grid_map.get((cell.row - 1, cell.col))
        return None

    def get_row(self, row_idx: int) -> List[GridCell]:
        """Get all unique cells in a specific row (deduplicated for merged cells)."""
        seen = set()
        row_cells = []
        for col in range(self.num_cols):
            cell = self._grid_map.get((row_idx, col))
            if cell and id(cell) not in seen:
                seen.add(id(cell))
                row_cells.append(cell)
        return row_cells

    def get_column(self, col_idx: int) -> List[GridCell]:
        """Get all unique cells in a specific column."""
        seen = set()
        col_cells = []
        for row in range(self.num_rows):
            cell = self._grid_map.get((row, col_idx))
            if cell and id(cell) not in seen:
                seen.add(id(cell))
                col_cells.append(cell)
        return col_cells

    def build_grid_map(self):
        """Build the (row, col) -> cell lookup, filling merged-cell spans."""
        self._grid_map.clear()
        for cell in self.cells:
            for r in range(cell.row, cell.row + cell.rowspan):
                for c in range(cell.col, cell.col + cell.colspan):
                    self._grid_map[(r, c)] = cell

    def detect_header_hierarchy(self):
        """Detect multi-level headers: cells spanning multiple columns/rows."""
        if self.num_rows < 2:
            return

        for cell in self.cells:
            if cell.colspan > 1 or cell.rowspan > 1:
                cell.is_header = True

            if cell.colspan > 1:
                # Find child cells directly below within the column span
                for sub_row in range(cell.row + cell.rowspan, self.num_rows):
                    for sub_col in range(cell.col, cell.col + cell.colspan):
                        sub_cell = self._grid_map.get((sub_row, sub_col))
                        if (sub_cell and sub_cell != cell
                                and sub_cell.parent_header is None):
                            sub_cell.parent_header = cell
                            if sub_cell not in cell.children:
                                cell.children.append(sub_cell)

    def to_nested_dict(self) -> List[Dict]:
        """Convert the grid to row dicts with compound header keys for multi-level tables.

        For a table like:
            | Revision History (colspan=3) |
            | REV | DATE | BY              |
            | A   | 2024 | JD              |

        Returns: [{"Revision History > REV": "A",
                    "Revision History > DATE": "2024",
                    "Revision History > BY": "JD"}]
        """
        if self.num_rows < 2:
            return []

        # Identify header rows (contiguous top rows where all cells look like headers)
        num_header_rows = 0
        for row_idx in range(self.num_rows):
            row_cells = self.get_row(row_idx)
            non_empty = [c for c in row_cells if c.text.strip()]
            if non_empty and all(_looks_like_header(c) for c in non_empty):
                num_header_rows += 1
            else:
                break
        num_header_rows = max(num_header_rows, 1)

        # Build compound column headers by tracing through header rows
        column_headers = {}
        for col in range(self.num_cols):
            parts = []
            for row_idx in range(num_header_rows):
                cell = self._grid_map.get((row_idx, col))
                if cell:
                    text = cell.text.strip()
                    if text and text not in parts:
                        parts.append(text)
            column_headers[col] = " > ".join(parts) if parts else f"Column {col}"

        # Extract data rows
        results = []
        for row_idx in range(num_header_rows, self.num_rows):
            row_dict = {}
            row_cells = self.get_row(row_idx)
            for cell in row_cells:
                header = column_headers.get(cell.col, f"Column {cell.col}")
                row_dict[header] = cell.text.strip()
            if any(v for v in row_dict.values()):
                results.append(row_dict)
        return results


def _looks_like_header(cell: GridCell) -> bool:
    """Heuristic: short, mostly non-numeric text or a spanning cell."""
    text = cell.text.strip()
    if not text:
        return False
    if cell.colspan > 1 or cell.rowspan > 1:
        return True
    if len(text) > 60:
        return False
    if text.replace(" ", "").replace("-", "").replace(".", "").isdigit():
        return False
    return True


class TableParser:
    """Parses complex multi-level tables from engineering drawing images.

    Pipeline:
      1. Detect horizontal and vertical line segments
      2. Find intersection points to establish a grid
      3. Cluster intersections into row/column boundaries
      4. Map detected contours to grid cells (with rowspan/colspan)
      5. OCR each cell
      6. Detect header hierarchy for multi-level tables
    """

    def __init__(self, tesseract_config: str = "--psm 6"):
        self.tesseract_config = tesseract_config

    # ------------------------------------------------------------------
    # Line segment detection
    # ------------------------------------------------------------------

    def detect_line_segments(self, binary_img: np.ndarray,
                             min_line_length: int = 50
                             ) -> Tuple[List[Tuple], List[Tuple]]:
        """Detect horizontal and vertical line segments from a binary image.

        Args:
            binary_img: Binary image (white foreground on black background,
                        i.e. lines are white / 255).
            min_line_length: Minimum pixel length for a line segment.

        Returns:
            (h_segments, v_segments) where
            h_segments = [(y_center, x_start, x_end), ...]
            v_segments = [(x_center, y_start, y_end), ...]
        """
        nrow, ncol = binary_img.shape

        # --- Horizontal lines ---
        h_kernel_len = max(ncol // 30, min_line_length)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, h_kernel, iterations=1)
        h_mask = cv2.dilate(
            h_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1
        )

        # --- Vertical lines ---
        v_kernel_len = max(nrow // 30, min_line_length)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, v_kernel, iterations=1)
        v_mask = cv2.dilate(
            v_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1
        )

        h_segments = self._segments_from_mask(h_mask, "horizontal")
        v_segments = self._segments_from_mask(v_mask, "vertical")
        return h_segments, v_segments

    @staticmethod
    def _segments_from_mask(mask: np.ndarray,
                            orientation: str) -> List[Tuple]:
        """Extract line-segment coordinates from a binary mask via connected components."""
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        segments = []
        for i in range(1, num_labels):  # skip background
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if orientation == "horizontal" and w > h * 3:
                segments.append((y + h // 2, x, x + w))
            elif orientation == "vertical" and h > w * 3:
                segments.append((x + w // 2, y, y + h))
        return segments

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def find_intersections(self, h_segments: List[Tuple],
                           v_segments: List[Tuple],
                           tolerance: int = 10) -> List[Tuple[int, int]]:
        """Find (x, y) intersection points between h- and v-segments."""
        intersections = []
        for h_y, h_x1, h_x2 in h_segments:
            for v_x, v_y1, v_y2 in v_segments:
                if (h_x1 - tolerance <= v_x <= h_x2 + tolerance
                        and v_y1 - tolerance <= h_y <= v_y2 + tolerance):
                    intersections.append((v_x, h_y))
        return intersections

    @staticmethod
    def cluster_values(values: List[int], min_gap: int = 15) -> List[int]:
        """Cluster nearby coordinate values; return the median of each cluster."""
        if not values:
            return []
        sorted_vals = sorted(set(values))
        clusters: List[List[int]] = [[sorted_vals[0]]]
        for val in sorted_vals[1:]:
            if val - clusters[-1][-1] <= min_gap:
                clusters[-1].append(val)
            else:
                clusters.append([val])
        return [c[len(c) // 2] for c in clusters]

    def build_grid_boundaries(self, h_segments: List[Tuple],
                              v_segments: List[Tuple],
                              intersections: List[Tuple[int, int]],
                              min_gap: int = 15
                              ) -> Tuple[List[int], List[int]]:
        """Derive sorted row- and column-boundary lists from intersections."""
        if intersections:
            xs = [p[0] for p in intersections]
            ys = [p[1] for p in intersections]
        else:
            ys = [seg[0] for seg in h_segments]
            xs = [seg[0] for seg in v_segments]
        row_b = self.cluster_values(ys, min_gap)
        col_b = self.cluster_values(xs, min_gap)
        return row_b, col_b

    # ------------------------------------------------------------------
    # Contour → grid-cell mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_boundary_idx(value: int, boundaries: List[int],
                              tolerance: int) -> Optional[int]:
        """Return the boundary index closest to *value* within *tolerance*."""
        best_idx, best_dist = None, tolerance + 1
        for i, b in enumerate(boundaries):
            dist = abs(value - b)
            if dist < best_dist:
                best_idx, best_dist = i, dist
        if best_idx is not None:
            return best_idx
        # Fall back: find which interval the value is inside
        for i in range(len(boundaries) - 1):
            if boundaries[i] < value < boundaries[i + 1]:
                if value - boundaries[i] < boundaries[i + 1] - value:
                    return i
                return i + 1
        return None

    def map_rect_to_grid(self, rect: Tuple[int, int, int, int],
                         row_b: List[int], col_b: List[int],
                         tolerance: int = 15
                         ) -> Optional[Tuple[int, int, int, int]]:
        """Map a bounding rect (x,y,w,h) to (row, col, rowspan, colspan)."""
        x, y, w, h = rect
        r0 = self._nearest_boundary_idx(y, row_b, tolerance)
        r1 = self._nearest_boundary_idx(y + h, row_b, tolerance)
        c0 = self._nearest_boundary_idx(x, col_b, tolerance)
        c1 = self._nearest_boundary_idx(x + w, col_b, tolerance)
        if r0 is None or c0 is None:
            return None
        if r1 is None:
            r1 = r0 + 1
        if c1 is None:
            c1 = c0 + 1
        return (r0, c0, max(1, r1 - r0), max(1, c1 - c0))

    # ------------------------------------------------------------------
    # Cell image cleaning for OCR
    # ------------------------------------------------------------------

    @staticmethod
    def clean_cell_for_ocr(cell_img: np.ndarray) -> np.ndarray:
        """Remove table gridlines from a cell image before OCR."""
        if cell_img.size == 0:
            return cell_img
        result = cell_img.copy()
        _, thresh = cv2.threshold(cell_img, 150, 255, cv2.THRESH_BINARY_INV)
        h, w = thresh.shape[:2]

        # Remove horizontal lines
        if w > 40:
            hk = cv2.getStructuringElement(cv2.MORPH_RECT, (min(40, w // 2), 1))
            h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hk, iterations=2)
            cnts, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                cv2.drawContours(result, [c], -1, 255, 5)

        # Remove vertical lines
        if h > 30:
            vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min(30, h // 2)))
            v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vk, iterations=2)
            cnts, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                cv2.drawContours(result, [c], -1, 255, 5)
        return result

    # ------------------------------------------------------------------
    # Main entry point: parse a table region into a TableGrid
    # ------------------------------------------------------------------

    def parse_table_region(self, img_gray: np.ndarray,
                           binary_lines: np.ndarray = None) -> TableGrid:
        """Parse a table region into a structured TableGrid.

        Args:
            img_gray: Original grayscale image region (for OCR).
            binary_lines: Binary image where table lines are white (255).
                          If None, one is derived from *img_gray*.

        Returns:
            A populated TableGrid (empty grid if structure is insufficient).
        """
        if binary_lines is None:
            _, binary_lines = cv2.threshold(img_gray, 127, 255,
                                            cv2.THRESH_BINARY_INV)

        nrow, ncol = binary_lines.shape

        # 1. Detect line segments
        h_segs, v_segs = self.detect_line_segments(binary_lines)
        if not h_segs or not v_segs:
            return TableGrid()

        # 2. Find intersections
        intersections = self.find_intersections(h_segs, v_segs)
        if len(intersections) < 4:
            return TableGrid()

        # 3. Grid boundaries
        row_b, col_b = self.build_grid_boundaries(h_segs, v_segs, intersections)
        if len(row_b) < 2 or len(col_b) < 2:
            return TableGrid()

        # 4. Detect cell contours
        padded = cv2.copyMakeBorder(binary_lines, 5, 5, 5, 5,
                                    cv2.BORDER_CONSTANT, value=0)
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(padded, k3, iterations=1)
        contours, hierarchy = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # 5. Build grid
        grid = TableGrid()
        grid.row_boundaries = row_b
        grid.col_boundaries = col_b
        grid.num_rows = len(row_b) - 1
        grid.num_cols = len(col_b) - 1

        max_area = (nrow // 2) * (ncol // 2)
        occupied: Dict[Tuple[int, int], GridCell] = {}

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x -= 5  # undo padding
            y -= 5
            area = w * h
            if area > max_area or area < 100:
                continue

            pos = self.map_rect_to_grid((x, y, w, h), row_b, col_b)
            if pos is None:
                continue
            r, c, rspan, cspan = pos
            if r >= grid.num_rows or c >= grid.num_cols:
                continue

            # Deduplicate: keep the tightest-fitting contour per grid slot
            key = (r, c)
            if key in occupied:
                if occupied[key].area <= area:
                    continue

            # OCR the cell
            cy1 = max(0, y)
            cy2 = min(img_gray.shape[0], y + h)  # use original h from contour
            cx1 = max(0, x)
            cx2 = min(img_gray.shape[1], x + w)
            cell_img = img_gray[cy1:cy2, cx1:cx2]
            if cell_img.size == 0:
                continue
            cleaned = self.clean_cell_for_ocr(cell_img)
            text = pytesseract.image_to_string(
                cleaned, config=self.tesseract_config
            ).strip()

            cell = GridCell(
                row=r, col=c, rowspan=rspan, colspan=cspan,
                x=x, y=y, w=w, h=h,  # use original contour dims
                text=text, text_lines=text.splitlines(),
            )
            grid.cells.append(cell)
            occupied[key] = cell

        # Remove cells that were superseded by tighter fits
        valid_ids = {id(c) for c in occupied.values()}
        grid.cells = [c for c in grid.cells if id(c) in valid_ids]

        grid.build_grid_map()

        # 6. Header hierarchy
        grid.detect_header_hierarchy()
        return grid

    # ------------------------------------------------------------------
    # Higher-level helpers
    # ------------------------------------------------------------------

    def extract_labeled_values(self, grid: TableGrid,
                               keywords: List[str]) -> List[Dict]:
        """Extract keyword→value pairs from the grid.

        For each cell whose text matches a keyword, the value is taken from:
          1. The remainder of the same cell text (after the keyword line).
          2. The cell to the right   (horizontal label | value layout).
          3. The cell below          (vertical   label / value layout).

        Returns:
            [{"keyword": str, "value": str, "cell": GridCell}, ...]
        """
        results = []
        matched_ids: set = set()
        sorted_kw = sorted(keywords, key=len, reverse=True)

        for cell in grid.cells:
            if id(cell) in matched_ids:
                continue
            upper = cell.text.upper()
            for kw in sorted_kw:
                if kw.upper() not in upper:
                    continue
                matched_ids.add(id(cell))

                value = self._value_from_same_cell(cell, kw)

                if not value:
                    right = grid.get_neighbor(cell, "right")
                    if right and id(right) not in matched_ids:
                        value = right.text.strip()
                        matched_ids.add(id(right))

                if not value:
                    below = grid.get_neighbor(cell, "down")
                    if below and id(below) not in matched_ids:
                        value = below.text.strip()
                        matched_ids.add(id(below))

                results.append({
                    "keyword": kw,
                    "value": value or "",
                    "cell": cell,
                })
                break  # first (longest) keyword wins per cell
        return results

    @staticmethod
    def _value_from_same_cell(cell: GridCell, keyword: str) -> str:
        """Try to pull a value from lines in the same cell after the keyword."""
        kw_upper = keyword.upper()
        for line in cell.text_lines:
            if kw_upper in line.upper():
                remainder = line[line.upper().index(kw_upper) + len(kw_upper):]
                remainder = remainder.strip().lstrip(":").strip()
                if remainder and len(remainder) > 1:
                    return remainder
            elif line.strip() and kw_upper not in line.upper():
                return line.strip()
        return ""

    def parse_sub_table(self, grid: TableGrid,
                        anchor_keyword: str) -> Optional[TableGrid]:
        """Find a keyword cell that heads a sub-table and parse just that region.

        Useful for multi-level sections like an Amendments/Revision History
        block that sits inside a larger title-block grid.
        """
        for cell in grid.cells:
            if anchor_keyword.upper() in cell.text.upper():
                # The sub-table starts at the row below and spans the cell's columns
                start_row = cell.row + cell.rowspan
                start_col = cell.col
                end_col = cell.col + cell.colspan

                sub_cells = []
                for c in grid.cells:
                    if (c.row >= start_row
                            and c.col >= start_col
                            and c.end_col <= end_col - 1):
                        # Re-number relative to sub-table origin
                        sc = GridCell(
                            row=c.row - start_row,
                            col=c.col - start_col,
                            rowspan=c.rowspan, colspan=c.colspan,
                            x=c.x, y=c.y, w=c.w, h=c.h,
                            text=c.text, text_lines=list(c.text_lines),
                        )
                        sub_cells.append(sc)

                if not sub_cells:
                    return None

                sub = TableGrid()
                sub.cells = sub_cells
                max_r = max(c.row + c.rowspan for c in sub_cells)
                max_c = max(c.col + c.colspan for c in sub_cells)
                sub.num_rows = max_r
                sub.num_cols = max_c
                sub.build_grid_map()
                sub.detect_header_hierarchy()
                return sub
        return None
