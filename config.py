"""
config.py
=========
Constantes y parámetros globales de la V1.
Modificar aquí para ajustar el comportamiento sin tocar el resto del código.
"""

# ── Q-Learning ────────────────────────────────────────────────────────────────
ALPHA          = 0.8      # tasa de aprendizaje
GAMMA          = 0.95     # factor de descuento
EPSILON_START  = 1.0      # exploración inicial (100%)
EPSILON_END    = 0.01     # exploración mínima (1%)
EPSILON_DECAY  = 0.995    # decaimiento multiplicativo por episodio
ADAPTIVE_THRESHOLD = 0.7  # umbral de tasa de éxito para ε adaptativo

# ── Entrenamiento ─────────────────────────────────────────────────────────────
EPISODES       = 8_000    # episodios totales
MAX_STEPS      = 100      # pasos máximos por episodio
RENDER_EVERY   = 300      # mostrar ventana cada N episodios
RENDER_DELAY   = 18       # ms de pausa entre pasos cuando renderiza

# ── Reward Shaping ────────────────────────────────────────────────────────────
SHAPING_WEIGHT = 0.3      # peso de la señal de distancia (Φ)

# ── Entorno FrozenLake ────────────────────────────────────────────────────────
MAP_NAME       = "4x4"
IS_SLIPPERY    = False
GRID           = 4
HOLES          = {5, 7, 11, 12}   # índices lineales de los hoyos
GOAL           = 15
START          = 0

# ── PyGame ────────────────────────────────────────────────────────────────────
CELL           = 130
WIN_W          = GRID * CELL
HUD_H          = 110
WIN_H          = GRID * CELL + HUD_H
FPS            = 30

# ── Colores ───────────────────────────────────────────────────────────────────
C_BG        = (34,  45,  28)
C_GRASS_L   = (52,  72,  38)
C_GRASS_D   = (44,  61,  30)
C_HOLE      = (18,  22,  40)
C_HOLE_BDR  = (40,  60, 110)
C_GOAL      = (80,  55,  10)
C_GOAL_BDR  = (180, 130,  20)
C_START_BDR = (60,  100,  50)
C_PREY      = (90,  180, 240)
C_PREY_DK   = (40,  100, 170)
C_FRUIT     = (220,  80,  60)
C_FRUIT_SH  = (240, 140, 110)
C_PRED_ICON = (200,  60,  40)
C_GRID_LINE = (42,   55,  34)
C_HUD_BG    = (22,   28,  18)
C_HUD_BDR   = (70,   90,  50)
C_TEXT      = (210, 220, 190)
C_TEXT_DIM  = (130, 145, 110)
C_QVAL_POS  = (80,  200, 120)
C_QVAL_NEG  = (200,  80,  60)
C_QVAL_ZERO = (80,   90,  70)
C_ARROW     = (220, 200, 100)

ACTION_LABELS = {0: "←", 1: "↓", 2: "→", 3: "↑"}
ACTION_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
