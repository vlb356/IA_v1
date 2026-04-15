import time
import gymnasium as gym
from config import (
    MAP_NAME, IS_SLIPPERY, EPISODES, MAX_STEPS,
    RENDER_EVERY, RENDER_DELAY, GRID,
)
from qtable  import QTable
from renderer import Renderer


class Trainer:
    """
    Orquesta el entrenamiento completo:
      1. Crea el entorno FrozenLake-v1 prediseñado de Gymnasium
      2. Entrena la QTable durante EPISODES episodios
      3. Evalúa el agente con ε=0
      4. Lanza la demo visual final

    El entorno se usa exactamente como sale de gym.make(), sin wrappers.
    """

    def __init__(self):
        # Entorno prediseñado de Gymnasium — sin ninguna modificación
        self.env      = gym.make("FrozenLake-v1", map_name=MAP_NAME,
                                 is_slippery=IS_SLIPPERY, render_mode=None)
        self.qtable   = QTable(
            n_states  = self.env.observation_space.n,
            n_actions = self.env.action_space.n,
        )
        self.renderer = Renderer()

        self._total_wins   = 0
        self._recent_wins  = 0
        self._recent_total = 0
        self._win_rate     = 0.0
        self._recent_rate  = 0.0

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    def train(self) -> None:
        print("=" * 60)
        print("  V1 · Presa busca Fruta · FrozenLake-v1 · Q-Learning")
        print("  Técnicas: Q-Learning + Reward Shaping + ε adaptativo")
        print("=" * 60)
        print(f"  Estados: {self.env.observation_space.n}  "
              f"Acciones: {self.env.action_space.n}  "
              f"Episodios: {EPISODES}")
        print("=" * 60)

        for episode in range(1, EPISODES + 1):
            self.renderer.handle_events()
            do_render = (episode % RENDER_EVERY == 0)
            self._run_episode(episode, do_render)

            if episode % 100 == 0:
                self._win_rate    = self._total_wins / episode * 100
                self._recent_rate = (self._recent_wins /
                                     max(1, self._recent_total) * 100)
                print(f"  Ep {episode:5d} | global {self._win_rate:5.1f}% "
                      f"| reciente {self._recent_rate:5.1f}% "
                      f"| ε={self.qtable.epsilon:.4f} "
                      f"| aprendidas {self.qtable.learned_states()}/16")
                self._recent_wins  = 0
                self._recent_total = 0

    def _run_episode(self, episode: int, do_render: bool) -> None:
        state, _ = self.env.reset()
        done     = False

        for _ in range(MAX_STEPS):
            action                        = self.qtable.choose_action(state)
            next_state, reward, t, u, _   = self.env.step(action)
            done                          = t or u

            self.qtable.update(state, action, reward, next_state, done)
            state = next_state

            if do_render:
                self.renderer.handle_events()
                self.renderer.draw(self.qtable.Q, state, self._stats(episode))
                import pygame; pygame.time.delay(RENDER_DELAY)

            if done:
                break

        won = reward > 0
        if won:
            self._total_wins  += 1
            self._recent_wins += 1
        self._recent_total += 1

        recent_rate = self._recent_wins / max(1, self._recent_total)
        self.qtable.decay_epsilon(recent_rate)

    # ── Evaluación ────────────────────────────────────────────────────────────
    def evaluate(self, n: int = 200) -> float:
        """
        Evalúa el agente con ε=0 (solo explotación) durante n episodios.
        Devuelve la tasa de éxito [0, 1].
        """
        wins = 0
        for _ in range(n):
            state, _ = self.env.reset()
            for _ in range(MAX_STEPS):
                action             = self.qtable.best_action(state)
                state, r, t, u, _  = self.env.step(action)
                if t or u:
                    if r > 0:
                        wins += 1
                    break
        rate = wins / n * 100
        print(f"\n  EVALUACIÓN FINAL (ε=0, {n} episodios): "
              f"{wins}/{n} victorias ({rate:.1f}%)")
        return rate / 100

    # ── Demo visual ───────────────────────────────────────────────────────────
    def demo(self, n: int = 10, delay_ms: int = 120) -> None:
        """Ejecuta n episodios con render PyGame y ε=0."""
        import pygame
        print(f"\n  Demo visual ({n} episodios)...")
        for i in range(1, n + 1):
            state, _ = self.env.reset()
            done     = False
            steps    = 0
            won      = False
            while not done and steps < MAX_STEPS:
                self.renderer.handle_events()
                action             = self.qtable.best_action(state)
                state, r, t, u, _  = self.env.step(action)
                done               = t or u
                won                = r > 0
                steps             += 1
                self.renderer.draw(self.qtable.Q, state, self._stats(EPISODES))
                pygame.time.delay(delay_ms)
            result = "✓ FRUTA" if won else "✗ DEPREDADOR"
            print(f"  Demo {i:2d}/{n} → {result} en {steps} pasos")
            pygame.time.delay(300)

    def close(self) -> None:
        self.env.close()
        self.renderer.close()

    def _stats(self, episode: int) -> dict:
        return {
            "episode":     episode,
            "epsilon":     self.qtable.epsilon,
            "win_rate":    self._win_rate,
            "recent_rate": self._recent_rate,
        }