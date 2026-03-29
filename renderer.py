"""Pygame renderer for Tetris — draws game state to a window."""

import pygame
from game import BLOCK_COLORS

CELL = 20
PREVIEW_CELL = 12
BG = (255, 255, 255)
GRID = (128, 128, 128)
TEXT_COLOR = (0, 0, 0)


class TetrisRenderer:
    """Draws a Tetris instance (field, active block, sidebar) each frame."""

    def __init__(self, game):
        self.game = game
        field_w = game.COLS * CELL
        field_h = game.ROWS * CELL
        self.fx = 20                       # field top-left x
        self.fy = 60                       # field top-left y
        self.sx = self.fx + field_w + 20   # sidebar x
        width = self.sx + 100
        height = self.fy + field_h + 20
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Tetris  |  Up/Z=rotate  C=hold  Space=drop  Esc=restart")
        self.font = pygame.font.SysFont("Calibri", 20, True)
        self.font_lg = pygame.font.SysFont("Calibri", 50, True)

    def _cell_rect(self, row, col):
        return [self.fx + col * CELL, self.fy + row * CELL, CELL, CELL]

    def _draw_field(self):
        g = self.game
        for i in range(g.ROWS):
            for j in range(g.COLS):
                pygame.draw.rect(self.screen, GRID, self._cell_rect(i, j), 1)
                if g.field[i][j] > 0:
                    r = self._cell_rect(i, j)
                    pygame.draw.rect(self.screen, BLOCK_COLORS[g.field[i][j] - 1],
                                     [r[0] + 1, r[1] + 1, r[2] - 2, r[3] - 2])

    def _draw_active(self):
        b = self.game.block
        if b is None:
            return
        for dr, dc in b.cells:
            r = self._cell_rect(dr + b.y, dc + b.x)
            pygame.draw.rect(self.screen, b.color,
                             [r[0] + 1, r[1] + 1, r[2] - 2, r[3] - 2])

    def _draw_preview(self, block, px, py):
        """Draw a block's shape at pixel position (px, py) using small cells."""
        if block is None:
            return
        for dr, dc in block.cells:
            pygame.draw.rect(self.screen, block.color,
                             [px + dc * PREVIEW_CELL + 1,
                              py + dr * PREVIEW_CELL + 1,
                              PREVIEW_CELL - 2, PREVIEW_CELL - 2])

    def _draw_sidebar(self):
        y = self.fy
        self.screen.blit(self.font.render(f"Score: {self.game.score}", True, TEXT_COLOR),
                         (self.sx, y))
        y += 35
        self.screen.blit(self.font.render("Hold:", True, TEXT_COLOR), (self.sx, y))
        self._draw_preview(self.game.hold_block, self.sx, y + 22)

        y += 90
        self.screen.blit(self.font.render("Next:", True, TEXT_COLOR), (self.sx, y))
        for i, block in enumerate(self.game.queue):
            self._draw_preview(block, self.sx, y + 22 + i * 55)

    def draw(self):
        """Render one frame."""
        self.screen.fill(BG)
        self._draw_field()
        self._draw_active()
        self._draw_sidebar()

        if self.game.state == "gameover":
            cx = self.fx + self.game.COLS * CELL // 2
            for txt, color, dy in [("Game Over", (255, 125, 0), 200),
                                   ("Press ESC", (255, 215, 0), 260)]:
                surf = self.font_lg.render(txt, True, color)
                self.screen.blit(surf, (cx - surf.get_width() // 2, dy))

        pygame.display.flip()
