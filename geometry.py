from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# Basic type aliases
# ============================================================

Index = int
LBRC = Tuple[int, int, int, int]      # (level, block, row, col)
GeomCoord = Tuple[int, int, int]     # (level, gr, gc)
DisplayCoord = Tuple[int, int]       # (br, bc)
Line = List[Index]


# ============================================================
# Board constants
# ============================================================

BOARD_SIZE = 96
BLOCK_SIZE = 4
GLOBAL_WIDTH = 12
GLOBAL_HEIGHT = 12

LEVELS = (1, 2, 3)

# Number of 4x4 blocks in each level
LEVEL_NUM_BLOCKS: Dict[int, int] = {
    1: 1,
    2: 2,
    3: 3,
}

# Offset in flat indexing
# Level 1: 0--15
# Level 2: 16--47
# Level 3: 48--95
LEVEL_INDEX_OFFSET: Dict[int, int] = {
    1: 0,
    2: 16,
    3: 48,
}

# Centered pyramid layout in a global width-12 coordinate system.
#
# Level 1 occupies columns 4--7.
# Level 2 occupies columns 2--9.
# Level 3 occupies columns 0--11.
#
# This matches the assignment figure:
#
#         Level 1
#       Level 2 Level 2
#   Level 3 Level 3 Level 3
#
LEVEL_COL_OFFSET: Dict[int, int] = {
    1: 4,
    2: 2,
    3: 0,
}


# ============================================================
# Helper functions
# ============================================================

def level_width(level: int) -> int:
    """Return the number of columns occupied by a level."""
    return LEVEL_NUM_BLOCKS[level] * BLOCK_SIZE


def valid_level(level: int) -> bool:
    return level in LEVELS


def valid_lbrc(level: int, block: int, row: int, col: int) -> bool:
    """Check whether a logical coordinate is valid."""
    if level not in LEVEL_NUM_BLOCKS:
        return False

    if block < 0 or block >= LEVEL_NUM_BLOCKS[level]:
        return False

    if row < 0 or row >= BLOCK_SIZE:
        return False

    if col < 0 or col >= BLOCK_SIZE:
        return False

    return True


def logical_to_index(level: int, block: int, row: int, col: int) -> int:
    """
    Convert logical coordinate (level, block, row, col) to flat index.

    This indexing scheme is storage-oriented and does not depend on
    the centered pyramid geometry.
    """
    if not valid_lbrc(level, block, row, col):
        raise ValueError(f"Invalid LBRC coordinate: {(level, block, row, col)}")

    return LEVEL_INDEX_OFFSET[level] + 16 * block + 4 * row + col


def logical_to_geom(level: int, block: int, row: int, col: int) -> GeomCoord:
    """
    Convert logical coordinate to centered level-wise geometric coordinate.

    gr = row
    gc = LEVEL_COL_OFFSET[level] + 4 * block + col
    """
    if not valid_lbrc(level, block, row, col):
        raise ValueError(f"Invalid LBRC coordinate: {(level, block, row, col)}")

    gr = row
    gc = LEVEL_COL_OFFSET[level] + BLOCK_SIZE * block + col
    return level, gr, gc


def logical_to_display(level: int, block: int, row: int, col: int) -> DisplayCoord:
    """
    Convert logical coordinate to stacked display coordinate.

    The three levels are stacked vertically:
    Level 1: br = 0--3
    Level 2: br = 4--7
    Level 3: br = 8--11

    The horizontal coordinate uses the centered pyramid offset.
    """
    if not valid_lbrc(level, block, row, col):
        raise ValueError(f"Invalid LBRC coordinate: {(level, block, row, col)}")

    br = BLOCK_SIZE * (level - 1) + row
    bc = LEVEL_COL_OFFSET[level] + BLOCK_SIZE * block + col
    return br, bc


# ============================================================
# Main geometry class
# ============================================================

@dataclass
class BoardGeometry:
    """
    Geometry container for the centered pyramid super tic-tac-toe board.

    The class builds:
    - flat index <-> logical coordinate mappings;
    - flat index <-> level-wise geometric coordinate mappings;
    - flat index <-> stacked display coordinate mappings;
    - same-level 8-neighborhood table;
    - row, diagonal, and column winning lines.
    """

    index_to_lbrc: Dict[Index, LBRC] = field(default_factory=dict)
    lbrc_to_index: Dict[LBRC, Index] = field(default_factory=dict)

    index_to_geom: Dict[Index, GeomCoord] = field(default_factory=dict)
    geom_to_index: Dict[GeomCoord, Index] = field(default_factory=dict)

    index_to_display: Dict[Index, DisplayCoord] = field(default_factory=dict)
    display_to_index: Dict[DisplayCoord, Index] = field(default_factory=dict)

    neighbors: Dict[Index, List[Optional[Index]]] = field(default_factory=dict)

    row_lines: List[Line] = field(default_factory=list)
    diag_lines: List[Line] = field(default_factory=list)
    column_lines: List[Line] = field(default_factory=list)
    winning_lines: List[Line] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.build_mappings()
        self.build_neighbors()
        self.row_lines = self.build_row_lines()
        self.diag_lines = self.build_diag_lines()
        self.column_lines = self.build_column_lines()
        self.winning_lines = self._deduplicate_lines(
            self.row_lines + self.diag_lines + self.column_lines
        )

    # --------------------------------------------------------
    # Mapping construction
    # --------------------------------------------------------

    def build_mappings(self) -> None:
        """Build all coordinate mappings."""
        self.index_to_lbrc.clear()
        self.lbrc_to_index.clear()

        self.index_to_geom.clear()
        self.geom_to_index.clear()

        self.index_to_display.clear()
        self.display_to_index.clear()

        for level in LEVELS:
            for block in range(LEVEL_NUM_BLOCKS[level]):
                for row in range(BLOCK_SIZE):
                    for col in range(BLOCK_SIZE):
                        idx = logical_to_index(level, block, row, col)
                        lbrc = (level, block, row, col)
                        geom = logical_to_geom(level, block, row, col)
                        display = logical_to_display(level, block, row, col)

                        self.index_to_lbrc[idx] = lbrc
                        self.lbrc_to_index[lbrc] = idx

                        self.index_to_geom[idx] = geom
                        self.geom_to_index[geom] = idx

                        self.index_to_display[idx] = display
                        self.display_to_index[display] = idx

        if len(self.index_to_lbrc) != BOARD_SIZE:
            raise RuntimeError(
                f"Expected {BOARD_SIZE} playable squares, "
                f"got {len(self.index_to_lbrc)}."
            )

    # --------------------------------------------------------
    # Neighbor construction
    # --------------------------------------------------------

    def build_neighbors(self) -> None:
        """
        Build same-level 8-neighborhood table.

        Each index maps to a list of length 8. If a neighbor direction
        is outside the playable board, the entry is None.

        The direction order is:
        (-1,-1), (-1,0), (-1,1),
        ( 0,-1),         ( 0,1),
        ( 1,-1), ( 1,0), ( 1,1).
        """
        self.neighbors.clear()

        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        for idx in range(BOARD_SIZE):
            level, gr, gc = self.index_to_geom[idx]
            cur_neighbors: List[Optional[Index]] = []

            for dr, dc in directions:
                candidate = (level, gr + dr, gc + dc)
                neighbor_idx = self.geom_to_index.get(candidate, None)
                cur_neighbors.append(neighbor_idx)

            self.neighbors[idx] = cur_neighbors

    # --------------------------------------------------------
    # Winning line construction
    # --------------------------------------------------------

    def build_row_lines(self) -> List[Line]:
        """
        Build all length-4 horizontal row lines within each level.

        Centering does not change the count:
        Level 1 width = 4  -> 1 window per row
        Level 2 width = 8  -> 5 windows per row
        Level 3 width = 12 -> 9 windows per row

        Total = 4 * (1 + 5 + 9) = 60.
        """
        lines: List[Line] = []

        for level in LEVELS:
            start_gc = LEVEL_COL_OFFSET[level]
            width = level_width(level)
            end_gc = start_gc + width - 1

            for gr in range(BLOCK_SIZE):
                for gc_start in range(start_gc, end_gc - 4 + 2):
                    coords = [
                        (level, gr, gc_start + k)
                        for k in range(4)
                    ]

                    if all(coord in self.geom_to_index for coord in coords):
                        line = [self.geom_to_index[coord] for coord in coords]
                        lines.append(line)

        return self._deduplicate_lines(lines)

    def build_diag_lines(self) -> List[Line]:
        """
        Build all length-5 diagonal lines in stacked display coordinates.

        The centered pyramid layout changes the diagonal set compared
        with the previous left-aligned layout. With centered offsets,
        this should produce 60 diagonal lines.
        """
        lines: List[Line] = []

        # Diagonal directions in stacked display coordinate
        directions = [
            (1, 1),
            (1, -1),
        ]

        for br in range(GLOBAL_HEIGHT):
            for bc in range(GLOBAL_WIDTH):
                for dr, dc in directions:
                    coords = [
                        (br + k * dr, bc + k * dc)
                        for k in range(5)
                    ]

                    if all(coord in self.display_to_index for coord in coords):
                        line = [self.display_to_index[coord] for coord in coords]
                        lines.append(line)

        return self._deduplicate_lines(lines)

    def build_column_lines(self) -> List[Line]:
        """
        Build all length-4 vertical column lines in stacked display coordinates.

        A valid column line must:
        1. contain four vertically consecutive playable squares;
        2. involve at least two different levels.

        This excludes ordinary within-level vertical lines.
        """
        lines: List[Line] = []

        for br_start in range(GLOBAL_HEIGHT - 4 + 1):
            for bc in range(GLOBAL_WIDTH):
                coords = [
                    (br_start + k, bc)
                    for k in range(4)
                ]

                if not all(coord in self.display_to_index for coord in coords):
                    continue

                line = [self.display_to_index[coord] for coord in coords]

                involved_levels = {
                    self.index_to_lbrc[idx][0]
                    for idx in line
                }

                if len(involved_levels) >= 2:
                    lines.append(line)

        return self._deduplicate_lines(lines)

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    @staticmethod
    def _deduplicate_lines(lines: List[Line]) -> List[Line]:
        """
        Deduplicate lines while preserving order.

        Lines are direction-sensitive only during construction.
        For winner checking, [a,b,c,d] and [d,c,b,a] are equivalent,
        so we use sorted tuple as the uniqueness key.
        """
        seen = set()
        unique: List[Line] = []

        for line in lines:
            key = tuple(sorted(line))
            if key not in seen:
                seen.add(key)
                unique.append(line)

        return unique

    def is_valid_index(self, idx: int) -> bool:
        return 0 <= idx < BOARD_SIZE

    def get_neighbors(self, idx: int) -> List[Optional[Index]]:
        if not self.is_valid_index(idx):
            raise ValueError(f"Invalid index: {idx}")
        return self.neighbors[idx]

    def print_layout(self) -> None:
        """
        Print the centered pyramid layout using indices.

        Empty spaces are printed before Level 1 and Level 2 so that
        the visual output matches the assignment figure.
        """
        for level in LEVELS:
            print(f"\nLevel {level}")
            offset_spaces = LEVEL_COL_OFFSET[level]
            width = level_width(level)

            for row in range(BLOCK_SIZE):
                row_items: List[str] = []

                # left padding for centered pyramid layout
                for _ in range(offset_spaces):
                    row_items.append("    ")

                for gc in range(LEVEL_COL_OFFSET[level],
                                LEVEL_COL_OFFSET[level] + width):
                    idx = self.geom_to_index[(level, row, gc)]
                    row_items.append(f"{idx:4d}")

                print("".join(row_items))

    def print_summary(self) -> None:
        print(f"Total playable squares: {len(self.index_to_lbrc)}")
        print(f"Total row lines       : {len(self.row_lines)}")
        print(f"Total diag lines      : {len(self.diag_lines)}")
        print(f"Total column lines    : {len(self.column_lines)}")
        print(f"Total winning lines   : {len(self.winning_lines)}")

    def debug_print(self) -> None:
        self.print_summary()
        self.print_layout()


# ============================================================
# Backward-compatible aliases
# ============================================================

GeometryData = BoardGeometry


def build_geometry_data() -> BoardGeometry:
    """
    Backward-compatible factory function.
    """
    return BoardGeometry()


# ============================================================
# Module-level default geometry object
# ============================================================

GEOMETRY = BoardGeometry()

index_to_lbrc = GEOMETRY.index_to_lbrc
lbrc_to_index = GEOMETRY.lbrc_to_index

index_to_geom = GEOMETRY.index_to_geom
geom_to_index = GEOMETRY.geom_to_index

index_to_display = GEOMETRY.index_to_display
display_to_index = GEOMETRY.display_to_index

neighbors = GEOMETRY.neighbors

row_lines = GEOMETRY.row_lines
diag_lines = GEOMETRY.diag_lines
column_lines = GEOMETRY.column_lines
winning_lines = GEOMETRY.winning_lines


# ============================================================
# Compatibility helper functions
# ============================================================

def get_geometry() -> BoardGeometry:
    return GEOMETRY


def get_neighbors(idx: int) -> List[Optional[Index]]:
    return GEOMETRY.get_neighbors(idx)


def get_winning_lines() -> List[Line]:
    return GEOMETRY.winning_lines


def get_row_lines() -> List[Line]:
    return GEOMETRY.row_lines


def get_diag_lines() -> List[Line]:
    return GEOMETRY.diag_lines


def get_column_lines() -> List[Line]:
    return GEOMETRY.column_lines


def print_board_indices() -> None:
    GEOMETRY.print_layout()


def print_geometry_summary() -> None:
    GEOMETRY.print_summary()


# ============================================================
# Debug entry point
# ============================================================

if __name__ == "__main__":
    GEOMETRY.debug_print()

    print("\nFirst 10 row lines:")
    for line in row_lines[:10]:
        print(line)

    print("\nFirst 10 diag lines:")
    for line in diag_lines[:10]:
        print(line)

    print("\nFirst 10 column lines:")
    for line in column_lines[:10]:
        print(line)