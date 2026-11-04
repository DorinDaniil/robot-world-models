"""Управление рукой командами на естественном языке: текст или голос → разбор → контроллер.

    python scripts/40_command_arm.py --text "подними кубик"
    python scripts/40_command_arm.py --text "положи кубик слева" --text "домой"
    python scripts/40_command_arm.py                 # интерактивно: вводить команды, q — выход
    python scripts/40_command_arm.py --voice         # микрофон, 4 с на фразу (pip install -e ".[voice]")

Команды выполняются последовательно в одной сцене; кубик остаётся там, куда его положили.
Разбор — rwm.language.commands.parse, исполнение — rwm.control.execute (IK-контроллер).
"""

from __future__ import annotations

import argparse

from rwm.control import execute
from rwm.envs import PickEnv
from rwm.language import ParseError, parse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", action="append", default=None, help="команда (можно несколько)")
    p.add_argument("--voice", action="store_true")
    p.add_argument("--seconds", type=float, default=4.0, help="длительность записи голоса")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-view", action="store_true", help="без окна (для проверки)")
    args = p.parse_args()

    env = PickEnv(render_mode=None if args.no_view else "human", max_steps=10**6)
    env.reset(seed=args.seed)
    sleep = 0.0 if args.no_view else env.dt

    def run(text: str) -> None:
        try:
            cmd = parse(text)
        except ParseError as e:
            print(f"  ? {e}")
            return
        print(f"  -> {cmd}")
        info = execute(env, cmd, sleep=sleep)
        print(f"  кубик: {env.cube_pos.round(3)}  подъём: {info.get('lift', 0):.3f} м")

    if args.text:
        for t in args.text:
            print(f"> {t}")
            run(t)
    elif args.voice:
        from rwm.language.speech import Transcriber, record

        asr = Transcriber(language=None)
        while True:
            input("Enter — начать запись (Ctrl+C — выход) ")
            text = asr(record(args.seconds))
            print(f"> {text}")
            run(text)
    else:
        while True:
            text = input("> ").strip()
            if text in ("q", "quit", "выход"):
                break
            run(text)
    env.close()


if __name__ == "__main__":
    main()
