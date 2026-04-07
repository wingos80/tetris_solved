"""Human-playable Tetris.

Controls:
    LEFT / RIGHT    move piece
    UP              rotate counter-clockwise
    Z               rotate clockwise
    DOWN (hold)     soft drop
    SPACE           hard drop
    C               hold piece
    ESC             restart game
"""

import pygame
from tetris import Tetris
from tetris.renderer import TetrisRenderer

FPS = 25
DROP_EVERY = 6  # frames between automatic drops


def main():
    pygame.init()
    game = Tetris()
    game.next_block()
    renderer = TetrisRenderer(game)
    clock = pygame.time.Clock()

    counter = 0
    holding_down = False

    running = True
    while running:
        counter = (counter + 1) % 100_000

        if game.state == "playing":
            if counter % DROP_EVERY == 0 or holding_down:
                game.soft_drop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game = Tetris()
                    game.next_block()
                    renderer.game = game
                    holding_down = False
                elif game.state == "playing":
                    if event.key == pygame.K_LEFT:
                        game.move(-1)
                    elif event.key == pygame.K_RIGHT:
                        game.move(1)
                    elif event.key == pygame.K_UP:
                        game.rotate_ccw()
                    elif event.key == pygame.K_z:
                        game.rotate_cw()
                    elif event.key == pygame.K_DOWN:
                        holding_down = True
                    elif event.key == pygame.K_SPACE:
                        game.hard_drop()
                    elif event.key == pygame.K_c:
                        game.hold()

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    holding_down = False

        renderer.draw()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
