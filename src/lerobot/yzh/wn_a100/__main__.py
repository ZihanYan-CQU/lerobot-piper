"""Package entry point; the CLI implementation lives in monitor.py."""

from .monitor import main

if __name__ == "__main__":
    raise SystemExit(main())
