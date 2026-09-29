"""Small shared HTTP helper: JSON POST with retry/backoff on transient errors, and a thread pool."""
from __future__ import annotations

import concurrent.futures as cf
import json
import random
import time
import urllib.error
import urllib.request

from .base import RerankerError

RETRY = {429, 500, 502, 503, 520, 529}


def post_json(url: str, body: dict, headers: dict, timeout: float = 60, retries: int = 4) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in RETRY and i < retries - 1:
                time.sleep(float(e.headers.get("retry-after") or 2 ** i) + random.random())
                continue
            raise RerankerError(f"HTTP {e.code}: {e.read()[:300]!r}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if i < retries - 1:
                time.sleep(2 ** i + random.random())
                continue
            raise RerankerError(str(e))
    raise RerankerError("retries exhausted")


def pmap(fn, items, workers: int):
    with cf.ThreadPoolExecutor(max(1, workers)) as ex:
        return list(ex.map(fn, items))
