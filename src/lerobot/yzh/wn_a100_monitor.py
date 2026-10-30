#!/usr/bin/env python3
"""Legacy launcher only; implementation and configuration live in wn_a100/."""

if __package__:
    from .wn_a100.monitor import main
else:
    from wn_a100.monitor import main

if __name__ == "__main__":
    raise SystemExit(main())
