"""weather-lab-app: serve the explorer, or build the Hugging Face Space content."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="weather-lab-app", description=__doc__)
    sub = p.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the Gradio explorer (default)")
    b = sub.add_parser("build-space", help="write the Space repository content (code + generated metadata)")
    b.add_argument("--out", default="dist/space")
    args = p.parse_args(argv)
    if args.command == "build-space":
        from weather_lab_app.space import build_space

        build_space(Path(args.out))
    else:
        from weather_lab_app.main import main as serve

        serve()


if __name__ == "__main__":
    main()
