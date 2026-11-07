"""Start with ``python /path/to/wn_a100`` or ``python -m lerobot.yzh.wn_a100``."""

if __package__:
    from .monitor import main
else:
    # Directory / file execution does not require installing the LeRobot package.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from wn_a100.monitor import main

if __name__ == "__main__":
    raise SystemExit(main())
