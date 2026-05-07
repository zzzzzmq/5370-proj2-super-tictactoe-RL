from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =========================
# Basic type aliases
# =========================
Level = int
Block = int
Row = int
Col = int
Index = int

LBRC = Tuple[Level, Block, Row, Col]
Geom = Tuple[Level, Row, Col]
DisplayCoord = Tuple[int, int]   # (board_row, board_col)


# =========================
# Board constants
# =========================
LEVELS: Tuple[int, ...] = (1, 2, 3)
ROWS_PER_BLOCK = 4
COLS_PER_BLOCK = 4

# Number of 4x4 blocks in each level
BLOCKS_PER_LEVEL: Dict[int, int] = {
    1: 1,
    2: 2,
    3: 3,
}

# Flat index offsets
LEVEL_OFFSETS: Dict[int, int] = {
    1: 0,
    2: 16,
    3: 48,
}

LEVEL_ROW_OFFSETS: Dict[int, int] = {
    1: 0,
    2: 4,
    3: 8,
}

# Same-level 8-neighborhood in fixed order
NEIGHBOR_DIRS_8: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

@dataclass(frozen=True)
class GeometryData:
    """Container for all static geometry artifacts."""
    index_to_lbrc: Dict[Index, LBRC]
    lbrc_to_index: Dict[LBRC, Index]
    index_to_geom: Dict[Index, Geom]
    geom_to_index: Dict[Geom, Index]

    index_to_display: Dict[Index, DisplayCoord]
    display_to_index: Dict[DisplayCoord, Index]

    neighbors: Dict[Index, List[Optional[Index]]]

    row_lines: List[List[Index]]
    diag_lines: List[List[Index]]
    column_lines: List[List[Index]]
    winning_lines: List[List[Index]]

# =========================
# Basic helpers
# =========================
def is_valid_level(level: int) -> bool:
    return level in LEVELS


def blocks_in_level(level: int) -> int:
    if level not in BLOCKS_PER_LEVEL:
        raise ValueError(f"Invalid level: {level}")
    return BLOCKS_PER_LEVEL[level]


def level_width(level: int) -> int:
    """Number of global columns occupied by a level."""
    return blocks_in_level(level) * COLS_PER_BLOCK


def is_valid_lbrc(level: int, block: int, row: int, col: int) -> bool:
    if not is_valid_level(level):
        return False
    if not (0 <= block < blocks_in_level(level)):
        return False
    if not (0 <= row < ROWS_PER_BLOCK):
        return False
    if not (0 <= col < COLS_PER_BLOCK):
        return False
    return True


def is_valid_geom(level: int, gr: int, gc: int) -> bool:
    if not is_valid_level(level):
        return False
    if not (0 <= gr < ROWS_PER_BLOCK):
        return False
    if not (0 <= gc < level_width(level)):
        return False
    return True


# =========================
# Coordinate conversions
# =========================
def lbrc_to_geom(lbrc: LBRC) -> Geom:
    level, block, row, col = lbrc
    if not is_valid_lbrc(level, block, row, col):
        raise ValueError(f"Invalid LBRC coordinate: {lbrc}")
    gr = row
    gc = block * COLS_PER_BLOCK + col
    return (level, gr, gc)


def geom_to_lbrc(geom: Geom) -> LBRC:
    level, gr, gc = geom
    if not is_valid_geom(level, gr, gc):
        raise ValueError(f"Invalid geometric coordinate: {geom}")
    block = gc // COLS_PER_BLOCK
    col = gc % COLS_PER_BLOCK
    row = gr
    return (level, block, row, col)


def lbrc_to_flat_index(lbrc: LBRC) -> Index:
    level, block, row, col = lbrc
    if not is_valid_lbrc(level, block, row, col):
        raise ValueError(f"Invalid LBRC coordinate: {lbrc}")
    return LEVEL_OFFSETS[level] + 16 * block + 4 * row + col


def flat_index_to_lbrc(index: int) -> LBRC:
    if not (0 <= index < 96):
        raise ValueError(f"Index out of range: {index}")

    if index < 16:
        level = 1
        local = index
    elif index < 48:
        level = 2
        local = index - 16
    else:
        level = 3
        local = index - 48

    block = local // 16
    rem = local % 16
    row = rem // 4
    col = rem % 4
    return (level, block, row, col)


def flat_index_to_geom(index: int) -> Geom:
    return lbrc_to_geom(flat_index_to_lbrc(index))

def lbrc_to_display(lbrc: LBRC) -> DisplayCoord:
    """
    Map logical board coordinate to stacked 2D display coordinate.

    Rows are stacked by level:
      - level 1 -> rows 0..3
      - level 2 -> rows 4..7
      - level 3 -> rows 8..11

    Columns remain left-aligned:
      bc = 4 * block + col
    """
    level, block, row, col = lbrc
    if not is_valid_lbrc(level, block, row, col):
        raise ValueError(f"Invalid LBRC coordinate: {lbrc}")

    br = LEVEL_ROW_OFFSETS[level] + row
    bc = block * COLS_PER_BLOCK + col
    return (br, bc)

# =========================
# Mapping builders
# =========================
def build_mappings() -> Tuple[
    Dict[Index, LBRC],
    Dict[LBRC, Index],
    Dict[Index, Geom],
    Dict[Geom, Index],
    Dict[Index, DisplayCoord],
    Dict[DisplayCoord, Index],
]:
    """Build all coordinate/index mapping tables."""
    index_to_lbrc: Dict[Index, LBRC] = {}
    lbrc_to_index: Dict[LBRC, Index] = {}
    index_to_geom: Dict[Index, Geom] = {}
    geom_to_index: Dict[Geom, Index] = {}
    index_to_display: Dict[Index, DisplayCoord] = {}
    display_to_index: Dict[DisplayCoord, Index] = {}

    for level in LEVELS:
        for block in range(blocks_in_level(level)):
            for row in range(ROWS_PER_BLOCK):
                for col in range(COLS_PER_BLOCK):
                    lbrc = (level, block, row, col)
                    idx = lbrc_to_flat_index(lbrc)
                    geom = lbrc_to_geom(lbrc)
                    display = lbrc_to_display(lbrc)

                    index_to_lbrc[idx] = lbrc
                    lbrc_to_index[lbrc] = idx
                    index_to_geom[idx] = geom
                    geom_to_index[geom] = idx
                    index_to_display[idx] = display
                    display_to_index[display] = idx

    if len(index_to_lbrc) != 96:
        raise RuntimeError(f"Expected 96 playable squares, got {len(index_to_lbrc)}")

    return (
        index_to_lbrc,
        lbrc_to_index,
        index_to_geom,
        geom_to_index,
        index_to_display,
        display_to_index,
    )


# =========================
# Neighborhood builder
# =========================
def build_neighbors(
    geom_to_index_map: Dict[Geom, Index],
) -> Dict[Index, List[Optional[Index]]]:
    """
    Build same-level 8-neighborhood lookup table.

    Returns:
        neighbors[index] = list of length 8
        each entry is either a neighbor index or None if out of board.
    """
    neighbors: Dict[Index, List[Optional[Index]]] = {}

    for geom, idx in geom_to_index_map.items():
        level, gr, gc = geom
        nbrs: List[Optional[Index]] = []

        for dr, dc in NEIGHBOR_DIRS_8:
            ngh = (level, gr + dr, gc + dc)
            nbrs.append(geom_to_index_map.get(ngh))

        neighbors[idx] = nbrs

    return neighbors


# =========================
# Winning line builders
# =========================
def build_row_lines(
    geom_to_index_map: Dict[Geom, Index],
) -> List[List[Index]]:
    """
    Row = same level, same gr, consecutive gc of length 4.
    """
    row_lines: List[List[Index]] = []

    for level in LEVELS:
        width = level_width(level)
        for gr in range(ROWS_PER_BLOCK):
            for gc_start in range(width - 4 + 1):
                coords = [(level, gr, gc_start + k) for k in range(4)]
                if all(coord in geom_to_index_map for coord in coords):
                    row_lines.append([geom_to_index_map[c] for c in coords])

    return deduplicate_lines(row_lines)


def build_diag_lines(
    display_to_index_map: Dict[DisplayCoord, Index],
) -> List[List[Index]]:
    """
    Diagonal = length 5 on the stacked 2D board display.

    We use two directions:
      - down-right: (br, bc) -> (br+1, bc+1)
      - down-left : (br, bc) -> (br+1, bc-1)

    A line is valid iff all 5 display coordinates correspond to real playable squares.
    """
    diag_lines: List[List[Index]] = []

    max_br = 12
    max_bc = 12

    for br in range(max_br):
        for bc in range(max_bc):
            # down-right diagonal of length 5
            coords1 = [(br + k, bc + k) for k in range(5)]
            if all(c in display_to_index_map for c in coords1):
                diag_lines.append([display_to_index_map[c] for c in coords1])

            # down-left diagonal of length 5
            coords2 = [(br + k, bc - k) for k in range(5)]
            if all(c in display_to_index_map for c in coords2):
                diag_lines.append([display_to_index_map[c] for c in coords2])

    return deduplicate_lines(diag_lines)


def build_column_lines(
    display_to_index_map: Dict[DisplayCoord, Index],
    index_to_lbrc_map: Dict[Index, LBRC],
) -> List[List[Index]]:
    """
    Column = vertical length-4 line on the stacked 2D display board,
    with the additional requirement that the 4-square pattern must
    involve at least two different levels.

    Stacked display coordinates:
      - Level 1 occupies board rows 0..3
      - Level 2 occupies board rows 4..7
      - Level 3 occupies board rows 8..11

    A candidate column line is:
      (br, bc), (br+1, bc), (br+2, bc), (br+3, bc)

    It is valid iff:
      1) all 4 display coordinates correspond to real playable squares
      2) the corresponding 4 squares are not all from the same level
    """
    column_lines: List[List[Index]] = []

    max_br = 12
    max_bc = 12

    for br in range(max_br):
        for bc in range(max_bc):
            coords = [(br + k, bc) for k in range(4)]

            # all 4 positions must be real playable squares
            if not all(c in display_to_index_map for c in coords):
                continue

            line = [display_to_index_map[c] for c in coords]
            levels = [index_to_lbrc_map[idx][0] for idx in line]

            # must involve at least two different levels
            if len(set(levels)) >= 2:
                column_lines.append(line)

    return deduplicate_lines(column_lines)


# =========================
# Utility helpers
# =========================
def deduplicate_lines(lines: List[List[Index]]) -> List[List[Index]]:
    """
    Remove duplicate winning lines while preserving deterministic ordering.

    Two lines are treated as the same if they contain the same indices
    in the same order.
    """
    seen = set()
    unique_lines: List[List[Index]] = []

    for line in lines:
        key = tuple(line)
        if key not in seen:
            seen.add(key)
            unique_lines.append(line)

    return unique_lines


def build_winning_lines(
    row_lines: List[List[Index]],
    diag_lines: List[List[Index]],
    column_lines: List[List[Index]]
) -> List[List[Index]]:
    """
    Combine row_lines, diag_lines, and column_lines into one winning_lines list.
    """
    return deduplicate_lines(row_lines + diag_lines + column_lines)


# =========================
# High-level factory
# =========================
def build_geometry_data() -> GeometryData:
    """
    Build all static geometry artifacts needed by the environment.
    """
    (
        index_to_lbrc,
        lbrc_to_index,
        index_to_geom,
        geom_to_index_map,
        index_to_display,
        display_to_index_map,
    ) = build_mappings()

    neighbors = build_neighbors(geom_to_index_map)

    row_lines = build_row_lines(geom_to_index_map)
    diag_lines = build_diag_lines(display_to_index_map)
    column_lines = build_column_lines(display_to_index_map, index_to_lbrc)
    winning_lines = build_winning_lines(row_lines, diag_lines, column_lines)

    return GeometryData(
        index_to_lbrc=index_to_lbrc,
        lbrc_to_index=lbrc_to_index,
        index_to_geom=index_to_geom,
        geom_to_index=geom_to_index_map,
        index_to_display=index_to_display,
        display_to_index=display_to_index_map,
        neighbors=neighbors,
        row_lines=row_lines,
        diag_lines=diag_lines,
        column_lines=column_lines,
        winning_lines=winning_lines,
    )


# =========================
# Optional debug helpers
# =========================
def pretty_print_level_indices(index_to_geom: Dict[Index, Geom]) -> None:
    """
    Print index layout level by level for debugging.
    """
    for level in LEVELS:
        print(f"\nLevel {level}")
        width = level_width(level)
        grid = [["   ." for _ in range(width)] for _ in range(ROWS_PER_BLOCK)]

        for idx, (lv, gr, gc) in index_to_geom.items():
            if lv == level:
                grid[gr][gc] = f"{idx:4d}"

        for row in grid:
            print(" ".join(row))


if __name__ == "__main__":
    geom = build_geometry_data()

    print(f"Total playable squares: {len(geom.index_to_lbrc)}")
    print(f"Total row lines       : {len(geom.row_lines)}")
    print(f"Total diag lines      : {len(geom.diag_lines)}")
    print(f"Total column lines    : {len(geom.column_lines)}")
    print(f"Total winning lines   : {len(geom.winning_lines)}")
    print(f"Total diag lines      : {len(geom.diag_lines)}")

    print("\nFirst 10 row lines:")
    for line in geom.row_lines[:10]:
        print(line)

    print("\nFirst 10 diag lines:")
    for line in geom.diag_lines[:10]:
        print(line)

    print("\nAll column lines:")
    for line in geom.column_lines:
        print(line)

    pretty_print_level_indices(geom.index_to_geom)