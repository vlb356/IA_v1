"""
qtable.py
=========
Clase QTable: Q-Learning tabular.

Mejoras respecto a v1:
  - Inicialización OPTIMISTA: Q = 0.1 en vez de 0.0
    Fuerza al agente a explorar todos los estados sistemáticamente
    porque cualquier acción no probada "parece mejor" de lo que es.
    Es una técnica clásica de RL que reduce la dependencia de epsilon
    en las primeras fases del entrenamiento.
  - ε adaptativo afinado

Actualización TD(0):
  Q(s,a) ← Q(s,a) + α·[r_shaped + γ·max Q(s',·) - Q(s,a)]
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
    return (idx % GRID, idx // GRID)


GOAL_POS = _cell_xy(GOAL)


class QTable:
    """
    Tabla Q con inicialización optimista y política ε-greedy.

    Inicialización optimista: Q = 0.1 (no 0.0)
      El agente "asume" que cada estado-acción vale 0.1.
      Al explorar y recibir recompensas reales menores, aprende
      a distinguir buenos de malos estados más rápido.
    """

    def __init__(self, n_states: int, n_actions: int):
        self.n_states  = n_states
        self.n_actions = n_actions
        # Inicialización optimista: 0.1 en lugar de 0.0
        self.Q       = np.full((n_states, n_actions), 0.1)
        # Los hoyos valen 0 desde el inicio (ya sabemos que son malos)
        for h in HOLES:
            self.Q[h] = 0.0
        self.epsilon = EPSILON_START

    def choose_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def best_action(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> None:
        r_shaped  = self._shape_reward(state, next_state, reward, done)
        best_next = 0.0 if done else float(np.max(self.Q[next_state]))
        td_target = r_shaped + GAMMA * best_next
        self.Q[state][action] += ALPHA * (td_target - self.Q[state][action])

    def _potential(self, state: int) -> float:
        sx, sy = _cell_xy(state)
        dist   = abs(sx - GOAL_POS[0]) + abs(sy - GOAL_POS[1])
        return -(dist / ((GRID - 1) * 2))

    def _shape_reward(self, state, next_state, reward, done) -> float:
        if done:
            return reward
        return reward + SHAPING_WEIGHT * (GAMMA * self._potential(next_state)
                                          - self._potential(state))

    def decay_epsilon(self, recent_win_rate: float = 0.0) -> None:
        decay = EPSILON_DECAY * 0.98 if recent_win_rate >= ADAPTIVE_THRESHOLD else EPSILON_DECAY
        self.epsilon = max(EPSILON_END, self.epsilon * decay)

    def learned_states(self) -> int:
        return int(np.count_nonzero((self.Q > 0.11).any(axis=1)))

    def print_policy(self) -> None:
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
