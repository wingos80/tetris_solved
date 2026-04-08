"""Core Tetris game logic — no rendering dependencies."""

import random
from collections import deque

BLOCK_NAMES = ["I", "Z", "S", "J", "L", "T", "O"]
BLOCK_COLORS = [
    (0, 255, 255),   # I - cyan
    (255, 0, 0),     # Z - red
    (0, 255, 0),     # S - green
    (0, 0, 255),     # J - blue
    (255, 165, 0),   # L - orange
    (128, 0, 128),   # T - purple
    (255, 255, 0),   # O - yellow
]

# Each block type: list of rotations.
# Each rotation: 4 (row, col) cell offsets within a 4x4 bounding box.
BLOCK_SHAPES = [
    # I
    [[(0, 1), (1, 1), (2, 1), (3, 1)],
     [(1, 0), (1, 1), (1, 2), (1, 3)]],
    # Z
    [[(1, 0), (1, 1), (2, 1), (2, 2)],
     [(0, 2), (1, 1), (1, 2), (2, 1)]],
    # S
    [[(1, 2), (1, 3), (2, 1), (2, 2)],
     [(0, 1), (1, 1), (1, 2), (2, 2)]],
    # J
    [[(0, 1), (0, 2), (1, 1), (2, 1)],
     [(0, 0), (1, 0), (1, 1), (1, 2)],
     [(0, 1), (1, 1), (2, 0), (2, 1)],
     [(1, 0), (1, 1), (1, 2), (2, 2)]],
    # L
    [[(0, 1), (0, 2), (1, 2), (2, 2)],
     [(1, 1), (1, 2), (1, 3), (2, 1)],
     [(0, 2), (1, 2), (2, 2), (2, 3)],
     [(0, 3), (1, 1), (1, 2), (1, 3)]],
    # T
    [[(0, 1), (1, 0), (1, 1), (1, 2)],
     [(0, 1), (1, 0), (1, 1), (2, 1)],
     [(1, 0), (1, 1), (1, 2), (2, 1)],
     [(0, 1), (1, 1), (1, 2), (2, 1)]],
    # O
    [[(0, 1), (0, 2), (1, 1), (1, 2)]],
]

NUM_BLOCK_TYPES = len(BLOCK_NAMES)


class Block:
    """A tetromino with type, rotation, and grid position."""

    def __init__(self, type_idx=None):
        self.type = type_idx if type_idx is not None else random.randint(0, NUM_BLOCK_TYPES - 1)
        self.rotation = 0
        self.x = 0
        self.y = 0

    @property
    def cells(self):
        """The 4 (row, col) offsets for the current rotation."""
        return BLOCK_SHAPES[self.type][self.rotation]

    @property
    def color(self):
        return BLOCK_COLORS[self.type]

    @property
    def num_rotations(self):
        return len(BLOCK_SHAPES[self.type])

    def rotate_cw(self):
        """Rotate clockwise."""
        self.rotation = (self.rotation + 1) % self.num_rotations

    def rotate_ccw(self):
        """Rotate counter-clockwise."""
        self.rotation = (self.rotation - 1) % self.num_rotations


class Tetris:
    """Core Tetris game state and logic.

    The field is a ROWS x COLS grid. 0 = empty, positive int = block type + 1
    (1-indexed so 0 can mean empty).

    Action methods (move, rotate_cw, rotate_ccw, soft_drop, hard_drop, hold)
    are safe to call at any time — they no-op or revert on collision.
    soft_drop() and hard_drop() return the number of lines cleared (0 if none).
    """

    ROWS = 20
    COLS = 10
    LINE_SCORES = [0, 100, 300, 500, 800]

    def __init__(self):
        self.field = [[0] * self.COLS for _ in range(self.ROWS)]
        self.score = 0
        self.state = "playing"
        self.block: Block = None
        self.hold_block = None
        self.can_hold = True
        self.queue = deque()
        self._fill_queue()

    # ── internal helpers ──────────────────────────────────────────────

    def _fill_queue(self):
        """Keep the preview queue at 5 blocks."""
        while len(self.queue) < 5:
            self.queue.append(Block())

    def _spawn(self, block):
        """Place a block at the top-center spawn point. Sets gameover if blocked."""
        block.x = self.COLS // 2 - 2
        block.y = 0
        self.block = block
        if self._intersects():
            self.state = "gameover"

    def _intersects(self):
        """True if the active block overlaps walls or filled cells."""
        for dr, dc in self.block.cells:
            r, c = dr + self.block.y, dc + self.block.x
            if r < 0 or r >= self.ROWS or c < 0 or c >= self.COLS:
                return True
            if self.field[r][c] > 0:
                return True
        return False

    def _freeze(self):
        """Lock the active block into the field, clear lines, spawn next block.

        Returns the number of lines cleared.
        """
        for dr, dc in self.block.cells:
            self.field[dr + self.block.y][dc + self.block.x] = self.block.type + 1
        lines = self._clear_lines()
        self.score += self.LINE_SCORES[min(lines, 4)]
        self.can_hold = True
        self.next_block()
        return lines

    def _clear_lines(self):
        """Remove completed rows, shift everything above down. Returns count."""
        lines = 0
        for i in range(self.ROWS):
            if all(cell > 0 for cell in self.field[i]):
                lines += 1
                for k in range(i, 0, -1):
                    self.field[k] = self.field[k - 1][:]
                self.field[0] = [0] * self.COLS
        return lines

    # ── public actions ────────────────────────────────────────────────

    def next_block(self):
        """Pop the next block from the queue and spawn it."""
        self._spawn(self.queue.popleft())
        self._fill_queue()

    def hold(self):
        """Swap current block with hold slot. Allowed once per placement."""
        if not self.can_hold or self.block is None:
            return
        self.can_hold = False
        old_type = self.block.type
        if self.hold_block is None:
            self.hold_block = Block(old_type)
            self.next_block()
        else:
            swapped = Block(self.hold_block.type)
            self.hold_block = Block(old_type)
            self._spawn(swapped)

    def move(self, dx):
        """Move block horizontally by dx. Reverts if blocked."""
        self.block.x += dx
        if self._intersects():
            self.block.x -= dx

    def rotate_cw(self):
        """Rotate block clockwise. Reverts if blocked."""
        old = self.block.rotation
        self.block.rotate_cw()
        if self._intersects():
            self.block.rotation = old

    def rotate_ccw(self):
        """Rotate block counter-clockwise. Reverts if blocked."""
        old = self.block.rotation
        self.block.rotate_ccw()
        if self._intersects():
            self.block.rotation = old

    def soft_drop(self):
        """Move block down one row. Freezes if blocked. Returns lines cleared."""
        self.block.y += 1
        if self._intersects():
            self.block.y -= 1
            return self._freeze()
        return 0

    def hard_drop(self):
        """Instant drop to lowest valid row and freeze. Returns lines cleared."""
        while not self._intersects():
            self.block.y += 1
        self.block.y -= 1
        return self._freeze()
