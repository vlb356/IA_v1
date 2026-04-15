"""
qtable.py
=========
Clase QTable: encapsula la tabla Q y toda la lógica de Q-Learning.

Técnica de IA: Q-Learning (TD(0)) con:
  - Política ε-greedy para equilibrar exploración/explotación
  - Reward Shaping basado en potencial para señal densa
  - ε adaptativo que acelera el decaimiento cuando el agente ya aprende bien

Actualización estándar TD(0):
  Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]
"""

import numpy as np
import random
from config import (
    ALPHA, GAMMA, EPSILON_START, EPSILON_END,
    EPSILON_DECAY, ADAPTIVE_THRESHOLD,
    SHAPING_WEIGHT, GRID, GOAL, HOLES,
    ACTION_LABELS,
)


def _cell_xy(idx: int) -> tuple:
    """Convierte índice lineal a coordenadas (col, fila)."""
    return (idx % GRID, idx // GRID)


GOAL_POS = _cell_xy(GOAL)


class QTable:
    """
    Tabla Q de tamaño [n_states × n_actions] con política ε-greedy.

    Atributos públicos:
        epsilon  (float) — nivel actual de exploración
        Q        (ndarray) — tabla de valores Q
    """

    def __init__(self, n_states: int, n_actions: int):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.Q         = np.zeros((n_states, n_actions))
        self.epsilon   = EPSILON_START

    # ── Política ──────────────────────────────────────────────────────────────
    def choose_action(self, state: int) -> int:
        """
        ε-greedy: explora aleatoriamente con prob. ε,
        explota la mejor acción conocida con prob. 1-ε.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def best_action(self, state: int) -> int:
        """Devuelve siempre la mejor acción (ε=0). Usado en evaluación."""
        return int(np.argmax(self.Q[state]))

    # ── Actualización Q-Learning ──────────────────────────────────────────────
    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> None:
        """
        Actualización TD(0) con reward shaping:
          1. Calcula r_shaped = r + Φ(s') - Φ(s)   (reward shaping)
          2. Calcula td_target = r_shaped + γ·max Q(s',·)
          3. Actualiza Q(s,a) += α·(td_target - Q(s,a))
        """
        r_shaped  = self._shape_reward(state, next_state, reward, done)
        best_next = 0.0 if done else float(np.max(self.Q[next_state]))
        td_target = r_shaped + GAMMA * best_next
        self.Q[state][action] += ALPHA * (td_target - self.Q[state][action])

    # ── Reward Shaping ────────────────────────────────────────────────────────
    def _potential(self, state: int) -> float:
        """
        Φ(s) = -(distancia Manhattan a la fruta) / distancia_máxima
        Cuanto más cerca de la fruta, mayor el potencial (menos negativo).
        """
        sx, sy = _cell_xy(state)
        dist   = abs(sx - GOAL_POS[0]) + abs(sy - GOAL_POS[1])
        max_d  = (GRID - 1) * 2
        return -dist / max_d

    def _shape_reward(self, state: int, next_state: int,
                      reward: float, done: bool) -> float:
        """
        Potential-based reward shaping:
          r_shaped = r + w·(γ·Φ(s') - Φ(s))
        Si done, solo devuelve r (no se aplica shaping en el paso terminal).
        """
        if done:
            return reward
        phi_s  = self._potential(state)
        phi_s2 = self._potential(next_state)
        return reward + SHAPING_WEIGHT * (GAMMA * phi_s2 - phi_s)

    # ── ε Adaptativo ─────────────────────────────────────────────────────────
    def decay_epsilon(self, recent_win_rate: float = 0.0) -> None:
        """
        Decaimiento de ε adaptativo:
        - Si la tasa de éxito reciente >= ADAPTIVE_THRESHOLD → decae más rápido
        - Si no → decaimiento normal
        """
        if recent_win_rate >= ADAPTIVE_THRESHOLD:
            decay = EPSILON_DECAY * 0.98
        else:
            decay = EPSILON_DECAY
        self.epsilon = max(EPSILON_END, self.epsilon * decay)

    # ── Estadísticas ──────────────────────────────────────────────────────────
    def learned_states(self) -> int:
        """Número de estados con al menos un valor Q distinto de cero."""
        return int(np.count_nonzero(self.Q.any(axis=1)))

    def print_policy(self) -> None:
        """Imprime en consola la mejor acción por celda (tabla 4×4)."""
        print("  Q-TABLE (mejor acción por celda):")
        print("  ┌──────┬──────┬──────┬──────┐")
        for row in range(GRID):
            line = "  │"
            for col in range(GRID):
                idx = row * GRID + col
                if idx == GOAL:
                    ch = " 🍎 G "
                elif idx in HOLES:
                    ch = " 🔴 H "
                else:
                    ch = f"  {ACTION_LABELS[self.best_action(idx)]}   "
                line += ch + "│"
            print(line)
            if row < GRID - 1:
                print("  ├──────┼──────┼──────┼──────┤")
        print("  └──────┴──────┴──────┴──────┘")
