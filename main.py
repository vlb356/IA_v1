"""
main.py
=======
Punto de entrada de la V1.

Uso:
    python main.py

Requisitos:
    pip install gymnasium pygame numpy
"""

from trainer import Trainer


def main():
    trainer = Trainer()
    try:
        trainer.train()
        trainer.qtable.print_policy()
        trainer.evaluate(n=200)
        trainer.demo(n=10)
        print("\n  Simulación completada. Cierra la ventana o pulsa ESC.")
        import pygame, sys
        clock = pygame.time.Clock()
        while True:
            trainer.renderer.handle_events()
            clock.tick(30)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
