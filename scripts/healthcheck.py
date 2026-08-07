"""Docker HEALTHCHECK probe.

Exits 0 when the container's liveness endpoint answers healthy, 1 otherwise.

Usage:
    python scripts/healthcheck.py [URL]
"""

from __future__ import annotations

import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/health"


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return 1
    if response.status != 200:
        return 1
    return 0 if payload.get("healthy", False) else 1


if __name__ == "__main__":
    sys.exit(main())
