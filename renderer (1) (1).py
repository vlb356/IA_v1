"""
renderer.py
===========
Clase Renderer: lógica visual con PyGame — versión mejorada.

Cambios respecto a v1:
  - Se eliminan los 4 valores Q individuales por celda (menos ruido visual)
  - Flecha de mejor acción más grande y clara (con punta triangular real)
  - Fondo de cada celda tintado según el valor Q máximo aprendido
    (verde oscuro = valor alto, neutro = sin aprender)
  - HUD más limpio con iconos más grandes
"""

import sys
import pygame
import numpy as np
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


def _cell_xy(idx: int) -> tuple:
    return (idx % GRID, idx // GRID)


def _lerp_color(c1, c2, t):
    """Interpola linealmente entre dos colores RGB."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


class Renderer:
    """Gestiona la ventana PyGame y el dibujado de la simulación."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption(
            "V1 · Presa busca Fruta · Q-Learning · FrozenLake-v1")
        self.clock = pygame.time.Clock()
        try:
            self.font_title = pygame.font.SysFont("consolas", 14, bold=True)
            self.font_med   = pygame.font.SysFont("consolas", 13, bold=True)
            self.font_sm    = pygame.font.SysFont("consolas", 11)
            self.font_xs    = pygame.font.SysFont("consolas", 10)
        except Exception:
            self.font_title = pygame.font.Font(None, 15)
            self.font_med   = pygame.font.Font(None, 14)
            self.font_sm    = pygame.font.Font(None, 12)
            self.font_xs    = pygame.font.Font(None, 11)

    # ── API pública ───────────────────────────────────────────────────────────
    def draw(self, Q: np.ndarray, state: int, stats: dict) -> None:
        self._draw_terrain(Q)
        self._draw_qvalues(Q)
        self._draw_agent(state)
        self._draw_hud(Q, state, stats)
        pygame.display.flip()
        self.clock.tick(FPS)

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close(); sys.exit()

    def close(self) -> None:
        pygame.quit()

    # ── Terreno con tinte de Q-valor ──────────────────────────────────────────
    def _draw_terrain(self, Q: np.ndarray) -> None:
        self.screen.fill(C_BG)
        q_max_global = float(np.max(Q)) if np.max(Q) > 0 else 1.0

        for idx in range(GRID * GRID):
            cx, cy = _cell_xy(idx)
            rx, ry = cx * CELL, cy * CELL

            if idx in HOLES:
                self._draw_hole(rx, ry)
                continue
            if idx == GOAL:
                self._draw_goal(rx, ry)
                continue

            # Base: hierba alternada
            base_color = C_GRASS_L if (cx + cy) % 2 == 0 else C_GRASS_D

            # Tinte verde según valor Q aprendido (cuanto más alto, más vivo)
            q_val = float(np.max(Q[idx]))
            if q_val > 0.01 and q_max_global > 0:
                t = min(q_val / q_max_global, 1.0) * 0.45
                tinted = _lerp_color(base_color, (60, 160, 60), t)
            else:
                tinted = base_color

            pygame.draw.rect(self.screen, tinted, (rx, ry, CELL, CELL))

            # Borde inicio
            if idx == START:
                pygame.draw.rect(self.screen, C_START_BDR,
                                 (rx, ry, CELL, CELL), 2)
                lbl = self.font_xs.render("INICIO", True, C_START_BDR)
                self.screen.blit(
                    lbl, (rx + CELL//2 - lbl.get_width()//2, ry + CELL - 16))

        # Grid lines
        for x in range(0, WIN_W + 1, CELL):
            pygame.draw.line(self.screen, C_GRID_LINE, (x, 0), (x, GRID*CELL), 1)
        for y in range(0, GRID*CELL + 1, CELL):
            pygame.draw.line(self.screen, C_GRID_LINE, (0, y), (WIN_W, y), 1)

    def _draw_hole(self, rx: int, ry: int) -> None:
        pygame.draw.rect(self.screen, C_HOLE, (rx, ry, CELL, CELL))
        pygame.draw.rect(self.screen, C_HOLE_BDR, (rx, ry, CELL, CELL), 3)
        mx, my = rx + CELL//2, ry + CELL//2
        # Rombo depredador
        pygame.draw.polygon(self.screen, (160, 30, 20),
            [(mx, my-26), (mx+20, my), (mx, my+26), (mx-20, my)])
        pygame.draw.polygon(self.screen, C_PRED_ICON,
            [(mx, my-22), (mx+16, my), (mx, my+22), (mx-16, my)])
        # Ojos
        for ox in (-5, 5):
            pygame.draw.circle(self.screen, (255, 220, 0), (mx+ox, my-4), 4)
            pygame.draw.circle(self.screen, (0, 0, 0),     (mx+ox, my-4), 2)
        lbl = self.font_xs.render("DEPREDADOR", True, C_HOLE_BDR)
        self.screen.blit(lbl, (rx + CELL//2 - lbl.get_width()//2, ry + CELL - 16))

    def _draw_goal(self, rx: int, ry: int) -> None:
        pygame.draw.rect(self.screen, C_GOAL, (rx, ry, CELL, CELL))
        pygame.draw.rect(self.screen, C_GOAL_BDR, (rx, ry, CELL, CELL), 3)
        mx, my = rx + CELL//2, ry + CELL//2
        # Tallo
        pygame.draw.line(self.screen, (70, 130, 40), (mx, my-14), (mx, my-28), 3)
        pygame.draw.line(self.screen, (70, 130, 40), (mx, my-22), (mx+8, my-28), 2)
        # Fruta
        pygame.draw.circle(self.screen, (180, 30, 20), (mx, my), 22)
        pygame.draw.circle(self.screen, C_FRUIT,        (mx, my), 20)
        pygame.draw.circle(self.screen, C_FRUIT_SH,     (mx-6, my-6), 8)
        lbl = self.font_xs.render("FRUTA", True, C_GOAL_BDR)
        self.screen.blit(lbl, (rx + CELL//2 - lbl.get_width()//2, ry + CELL - 16))

    # ── Q-valores: solo valor máximo + flecha ─────────────────────────────────
    def _draw_qvalues(self, Q: np.ndarray) -> None:
        """
        Por cada celda transitable dibuja SOLO:
          1. Valor Q máximo (esquina sup-izq) — limpio, sin los 4 valores
          2. Flecha grande y clara hacia la mejor acción
        """
        for idx in range(GRID * GRID):
            if idx in HOLES or idx == GOAL:
                continue
            cx, cy = _cell_xy(idx)
            rx, ry = cx * CELL, cy * CELL
            q_max  = float(np.max(Q[idx]))
            best   = int(np.argmax(Q[idx]))

            # Valor Q máximo — solo uno, arriba a la izquierda
            if q_max > 0.001:
                col = C_QVAL_POS
            elif q_max < -0.001:
                col = C_QVAL_NEG
            else:
                col = C_QVAL_ZERO
            txt = self.font_sm.render(f"Q={q_max:.2f}", True, col)
            self.screen.blit(txt, (rx + 6, ry + 6))

            # Flecha grande con punta triangular real
            if q_max > 0.001:
                self._draw_arrow(
                    rx + CELL//2, ry + CELL//2 + 12,
                    ACTION_DELTAS[best], length=30, color=C_ARROW)

    def _draw_arrow(self, ox: int, oy: int, delta: tuple,
                    length: int, color: tuple) -> None:
        """Dibuja una flecha con cuerpo y punta triangular."""
        ddx, ddy = delta
        ex = ox + ddx * length
        ey = oy + ddy * length
        # Cuerpo
        pygame.draw.line(self.screen, color, (ox, oy), (ex, ey), 3)
        # Punta triangular perpendicular
        perp_x, perp_y = -ddy, ddx
        tip_size = 8
        p1 = (ex + perp_x * tip_size, ey + perp_y * tip_size)
        p2 = (ex - perp_x * tip_size, ey - perp_y * tip_size)
        p3 = (ex + ddx * tip_size * 1.5, ey + ddy * tip_size * 1.5)
        pygame.draw.polygon(self.screen, color, [p1, p2, p3])

    # ── Agente (presa) ────────────────────────────────────────────────────────
    def _draw_agent(self, state: int) -> None:
        cx, cy = _cell_xy(state)
        mx     = cx * CELL + CELL//2
        my     = cy * CELL + CELL//2
        # Sombra
        pygame.draw.polygon(self.screen, (20, 50, 80),
            [(mx+2, my-12), (mx-10, my+12), (mx+14, my+12)])
        # Cuerpo
        pygame.draw.polygon(self.screen, C_PREY_DK,
            [(mx, my-16), (mx-12, my+10), (mx+12, my+10)])
        pygame.draw.polygon(self.screen, C_PREY,
            [(mx, my-13), (mx-9,  my+8),  (mx+9,  my+8)])
        # Ojo
        pygame.draw.circle(self.screen, (255, 255, 255), (mx+4, my-2), 4)
        pygame.draw.circle(self.screen, (0,   0,   0),   (mx+4, my-2), 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (mx+5, my-3), 1)

    # ── HUD ───────────────────────────────────────────────────────────────────
    def _draw_hud(self, Q: np.ndarray, state: int, stats: dict) -> None:
        hy = GRID * CELL
        pygame.draw.rect(self.screen, C_HUD_BG, (0, hy, WIN_W, HUD_H))
        pygame.draw.line(self.screen, C_HUD_BDR, (0, hy), (WIN_W, hy), 2)

        # Título
        title = self.font_title.render(
            "V1  ·  Presa busca Fruta  ·  FrozenLake-v1  ·  Q-Learning",
            True, C_TEXT)
        self.screen.blit(title, (WIN_W//2 - title.get_width()//2, hy + 5))
        self.screen.blit(
            self.font_xs.render("ESC para salir", True, C_TEXT_DIM),
            (WIN_W - 95, hy + 5))

        # Separador
        pygame.draw.line(self.screen, C_HUD_BDR,
                         (0, hy+22), (WIN_W, hy+22), 1)

        # Columna izquierda — estadísticas episodio
        def stat(label, val, color, x, y):
            self.screen.blit(self.font_xs.render(label, True, C_TEXT_DIM), (x, y))
            self.screen.blit(self.font_med.render(str(val), True, color),
                             (x + len(label) * 7 + 2, y))

        stat("Episodio:",    stats.get("episode", 0),          C_TEXT,     8,  hy+28)
        stat("Epsilon:",     f'{stats.get("epsilon",0):.3f}',  C_PREY,     8,  hy+46)
        stat("Éxito total:", f'{stats.get("win_rate",0):.1f}%', C_QVAL_POS, 8, hy+64)
        rr = stats.get("recent_rate", 0)
        stat("Éxito reciente:",
             f'{rr:.1f}%',
             (90, 210, 120) if rr > 50 else C_QVAL_NEG,
             8, hy+82)

        # Columna central — estado del agente
        cx2, cy2 = _cell_xy(state)
        best_act = ACTION_LABELS[int(np.argmax(Q[state]))]
        learned  = int(np.count_nonzero(Q.any(axis=1)))
        stat("Celda presa:",   f"({cx2},{cy2})",  C_TEXT_DIM, 240, hy+28)
        stat("Mejor acción:",  best_act,           C_ARROW,    240, hy+46)
        stat("Estados aprendidos:", f"{learned}/16", C_QVAL_POS if learned > 8 else C_TEXT_DIM, 240, hy+64)
        init_lbl = self.font_xs.render("Init. optimista Q=0.1", True, C_TEXT_DIM)
        self.screen.blit(init_lbl, (240, hy+82))

        # Leyenda derecha — iconos grandes
        lx = WIN_W - 155
        ly = hy + 28

        # Presa
        pygame.draw.polygon(self.screen, C_PREY,
            [(lx+8, ly+2), (lx, ly+16), (lx+16, ly+16)])
        self.screen.blit(self.font_sm.render("Presa", True, C_TEXT_DIM), (lx+22, ly+4))

        # Depredador
        dy2 = ly + 30
        pygame.draw.polygon(self.screen, C_PRED_ICON,
            [(lx+8,dy2),(lx+16,dy2+10),(lx+8,dy2+20),(lx,dy2+10)])
        self.screen.blit(self.font_sm.render("Depredador", True, C_TEXT_DIM), (lx+22, dy2+6))

        # Fruta
        fy = ly + 62
        pygame.draw.circle(self.screen, C_FRUIT, (lx+8, fy+8), 8)
        pygame.draw.circle(self.screen, C_FRUIT_SH, (lx+5, fy+5), 3)
        self.screen.blit(self.font_sm.render("Fruta (meta)", True, C_TEXT_DIM), (lx+22, fy+3))
