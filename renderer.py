"""
renderer.py
===========
Clase Renderer: toda la lógica visual con PyGame.

Responsabilidades:
  - Dibujar el terreno (hierba, hoyos/depredadores, fruta/meta)
  - Dibujar la Q-Table en tiempo real (valores + flechas de mejor acción)
  - Dibujar el agente (presa)
  - Dibujar el HUD con estadísticas del episodio
  - Gestionar eventos de ventana (cierre, ESC)
"""

import sys
import pygame
from config import (
    WIN_W, WIN_H, HUD_H, CELL, GRID, FPS,
    HOLES, GOAL, START,
    C_BG, C_GRASS_L, C_GRASS_D, C_HOLE, C_HOLE_BDR,
    C_GOAL, C_GOAL_BDR, C_START_BDR,
    C_PREY, C_PREY_DK, C_FRUIT, C_FRUIT_SH, C_PRED_ICON,
    C_GRID_LINE, C_HUD_BG, C_HUD_BDR, C_TEXT, C_TEXT_DIM,
    C_QVAL_POS, C_QVAL_NEG, C_QVAL_ZERO, C_ARROW,
    ACTION_LABELS, ACTION_DELTAS,
)
import numpy as np


def _cell_xy(idx: int) -> tuple:
    return (idx % GRID, idx // GRID)


class Renderer:
    """
    Gestiona toda la ventana PyGame y el dibujado.

    Uso:
        renderer = Renderer()
        renderer.draw(Q_matrix, state, stats_dict)
        renderer.close()
    """

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption(
            "V1 · Presa busca Fruta · Q-Learning · FrozenLake-v1")
        self.clock = pygame.time.Clock()
        try:
            self.font_big = pygame.font.SysFont("consolas", 15, bold=True)
            self.font_med = pygame.font.SysFont("consolas", 12)
            self.font_sm  = pygame.font.SysFont("consolas", 10)
        except Exception:
            self.font_big = pygame.font.Font(None, 16)
            self.font_med = pygame.font.Font(None, 13)
            self.font_sm  = pygame.font.Font(None, 11)

    # ── API pública ───────────────────────────────────────────────────────────
    def draw(self, Q: np.ndarray, state: int, stats: dict) -> None:
        """
        Dibuja un frame completo:
          1. Terreno
          2. Valores Q y flechas
          3. Agente (presa)
          4. HUD con estadísticas
        """
        self._draw_terrain()
        self._draw_qvalues(Q)
        self._draw_agent(state)
        self._draw_hud(Q, state, stats)
        pygame.display.flip()
        self.clock.tick(FPS)

    def handle_events(self) -> None:
        """Gestiona cierre de ventana y tecla ESC."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                sys.exit()

    def close(self) -> None:
        pygame.quit()

    # ── Terreno ───────────────────────────────────────────────────────────────
    def _draw_terrain(self) -> None:
        self.screen.fill(C_BG)
        for idx in range(GRID * GRID):
            cx, cy = _cell_xy(idx)
            rx, ry = cx * CELL, cy * CELL
            if idx in HOLES:
                self._draw_hole(rx, ry)
            elif idx == GOAL:
                self._draw_goal(rx, ry)
            else:
                color = C_GRASS_L if (cx + cy) % 2 == 0 else C_GRASS_D
                pygame.draw.rect(self.screen, color, (rx, ry, CELL, CELL))
                if idx == START:
                    pygame.draw.rect(self.screen, C_START_BDR,
                                     (rx, ry, CELL, CELL), 2)
                    lbl = self.font_sm.render("INICIO", True, C_START_BDR)
                    self.screen.blit(
                        lbl, (rx + CELL//2 - lbl.get_width()//2, ry + CELL - 18))
        # Líneas de cuadrícula
        for x in range(0, WIN_W + 1, CELL):
            pygame.draw.line(self.screen, C_GRID_LINE, (x, 0), (x, GRID*CELL), 1)
        for y in range(0, GRID*CELL + 1, CELL):
            pygame.draw.line(self.screen, C_GRID_LINE, (0, y), (WIN_W, y), 1)

    def _draw_hole(self, rx: int, ry: int) -> None:
        pygame.draw.rect(self.screen, C_HOLE, (rx, ry, CELL, CELL))
        pygame.draw.rect(self.screen, C_HOLE_BDR, (rx, ry, CELL, CELL), 3)
        mx, my = rx + CELL//2, ry + CELL//2
        pygame.draw.polygon(self.screen, C_PRED_ICON,
            [(mx, my-18), (mx+14, my), (mx, my+18), (mx-14, my)])
        for ox in (-4, 4):
            pygame.draw.circle(self.screen, (255, 220, 0), (mx+ox, my-3), 3)
            pygame.draw.circle(self.screen, (0, 0, 0),     (mx+ox, my-3), 1)
        lbl = self.font_sm.render("DEPREDADOR", True, C_HOLE_BDR)
        self.screen.blit(lbl, (rx + CELL//2 - lbl.get_width()//2, ry + CELL - 18))

    def _draw_goal(self, rx: int, ry: int) -> None:
        pygame.draw.rect(self.screen, C_GOAL, (rx, ry, CELL, CELL))
        pygame.draw.rect(self.screen, C_GOAL_BDR, (rx, ry, CELL, CELL), 3)
        mx, my = rx + CELL//2, ry + CELL//2
        pygame.draw.line(self.screen, (80, 120, 40), (mx, my-10), (mx, my-20), 3)
        pygame.draw.circle(self.screen, C_FRUIT,   (mx, my),   18)
        pygame.draw.circle(self.screen, C_FRUIT_SH,(mx-5, my-5), 7)
        lbl = self.font_sm.render("FRUTA", True, C_GOAL_BDR)
        self.screen.blit(lbl, (rx + CELL//2 - lbl.get_width()//2, ry + CELL - 18))

    # ── Q-Table visual ────────────────────────────────────────────────────────
    def _draw_qvalues(self, Q: np.ndarray) -> None:
        """
        Por cada celda transitable dibuja:
          - Valor Q máximo (color verde/rojo/gris)
          - Flecha hacia la mejor acción
          - Los 4 valores Q individuales (uno por esquina)
        """
        for idx in range(GRID * GRID):
            if idx in HOLES or idx == GOAL:
                continue
            cx, cy = _cell_xy(idx)
            rx, ry = cx * CELL, cy * CELL
            best  = int(np.argmax(Q[idx]))
            q_max = float(np.max(Q[idx]))

            c = C_QVAL_POS if q_max > 0.01 else (
                C_QVAL_NEG if q_max < -0.01 else C_QVAL_ZERO)
            self.screen.blit(
                self.font_sm.render(f"Q={q_max:.2f}", True, c), (rx+4, ry+6))

            if q_max > 0.001:
                mx, my = rx + CELL//2, ry + CELL//2 + 10
                ddx, ddy = ACTION_DELTAS[best]
                ex, ey   = mx + ddx*22, my + ddy*22
                pygame.draw.line(self.screen, C_ARROW, (mx, my), (ex, ey), 2)
                pygame.draw.circle(self.screen, C_ARROW, (ex, ey), 4)

            offsets = {0: (4, 50), 1: (40, 72), 2: (80, 50), 3: (40, 28)}
            for a in range(4):
                qv  = Q[idx][a]
                col = (C_QVAL_POS if qv > 0.01 else
                       C_QVAL_NEG if qv < -0.01 else C_QVAL_ZERO)
                ox, oy = offsets[a]
                self.screen.blit(
                    self.font_sm.render(f"{ACTION_LABELS[a]}{qv:.1f}", True, col),
                    (rx+ox, ry+oy))

    # ── Agente ────────────────────────────────────────────────────────────────
    def _draw_agent(self, state: int) -> None:
        cx, cy = _cell_xy(state)
        mx     = cx * CELL + CELL//2
        my     = cy * CELL + CELL//2
        pygame.draw.polygon(self.screen, C_PREY_DK,
            [(mx, my-16), (mx-12, my+10), (mx+12, my+10)])
        pygame.draw.polygon(self.screen, C_PREY,
            [(mx, my-13), (mx-9,  my+8),  (mx+9,  my+8)])
        pygame.draw.circle(self.screen, (255, 255, 255), (mx+4, my-2), 3)
        pygame.draw.circle(self.screen, (0,   0,   0),   (mx+4, my-2), 1)

    # ── HUD ───────────────────────────────────────────────────────────────────
    def _draw_hud(self, Q: np.ndarray, state: int, stats: dict) -> None:
        hy = GRID * CELL
        pygame.draw.rect(self.screen, C_HUD_BG, (0, hy, WIN_W, HUD_H))
        pygame.draw.line(self.screen, C_HUD_BDR, (0, hy), (WIN_W, hy), 2)

        title = self.font_big.render(
            "V1 · Presa busca Fruta · FrozenLake-v1 · Q-Learning", True, C_TEXT)
        self.screen.blit(title, (10, hy+6))
        self.screen.blit(
            self.font_sm.render("ESC para salir", True, C_TEXT_DIM),
            (WIN_W - 100, hy+6))

        def stat(label, val, color, x, y):
            self.screen.blit(self.font_sm.render(label, True, C_TEXT_DIM), (x, y))
            self.screen.blit(self.font_med.render(str(val), True, color),
                             (x + len(label)*7, y))

        stat("Episodio:     ", stats.get("episode", 0),         C_TEXT,      10, hy+26)
        stat("Epsilon:      ", f'{stats.get("epsilon", 0):.3f}', C_PREY,     10, hy+44)
        stat("Tasa global:  ", f'{stats.get("win_rate", 0):.1f}%', C_QVAL_POS,10, hy+62)
        stat("Tasa reciente:",
             f'{stats.get("recent_rate", 0):.1f}%',
             (90,210,120) if stats.get("recent_rate",0) > 50 else C_QVAL_NEG,
             10, hy+80)

        cx2, cy2 = _cell_xy(state)
        stat("Celda presa:  ", f"({cx2},{cy2}) idx={state}", C_TEXT_DIM,    280, hy+26)
        stat("Mejor acción: ", ACTION_LABELS[int(np.argmax(Q[state]))], C_ARROW, 280, hy+44)
        stat("Aprendidas:   ",
             f"{int(np.count_nonzero(Q.any(axis=1)))}/16 celdas",
             C_TEXT_DIM, 280, hy+62)

        lx = 430
        pygame.draw.polygon(self.screen, C_PREY,
            [(lx+6,hy+28),(lx,hy+40),(lx+12,hy+40)])
        self.screen.blit(
            self.font_sm.render("= Presa", True, C_TEXT_DIM), (lx+16, hy+30))
        pygame.draw.polygon(self.screen, C_PRED_ICON,
            [(lx+6,hy+50),(lx+12,hy+58),(lx+6,hy+66),(lx,hy+58)])
        self.screen.blit(
            self.font_sm.render("= Depredador", True, C_TEXT_DIM), (lx+16, hy+55))
        pygame.draw.circle(self.screen, C_FRUIT, (lx+6, hy+82), 6)
        self.screen.blit(
            self.font_sm.render("= Fruta", True, C_TEXT_DIM), (lx+16, hy+78))
