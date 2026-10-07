"""Embedders. Default: fastembed (ONNX, CPU, no PyTorch). `hash:<dim>` is a deterministic
bag-of-words hashing embedder for tests and offline smoke runs (no model download).

`google/embeddinggemma-2` (EG2, 2026-10-06) is REMOTE-FIRST: an OpenAI-compatible `/v1/embeddings`
server on the LAN (the RTX 5080 desktop) embeds; the local ONNX int8 copy is the fallback for queries
(measured cos 0.9999 vs fp32, identical top-3 at 768d and 256d; infrastructure/embeddinggemma2-eval.md).
Index builds refuse the local fallback by default (`remote_required_for_build`): 69.7k chunks is ~80 h
on the N150 CPU vs ~15 min on the GPU, so a silent fallback would be a hang, not a build.

Every embedder has `embed(docs)` and `embed_query(queries)`; EG2 needs the asymmetric task prefixes,
the others ignore the distinction."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from functools import lru_cache


class HashEmbedder:
    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class FastEmbedder:
    # batch_size 16, not 64: ONNX Runtime's CPU arena keeps its peak allocation for the life of the
    # process, and a 64 x 512-token batch of bge-small peaks at ~2.3 GB RSS (measured 2026-09-29;
    # 16 -> ~0.75 GB and no slower). The 64 default was the 2.4 GB process the kernel OOM-killed that night.
    def __init__(self, model: str, batch_size: int = 16):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model)
        self.batch_size = batch_size
        self.dim = len(next(iter(self.model.embed(["probe"]))))

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # sort by length so each batch pads to similar sizes (~2x less compute), then restore order
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vecs = list(self.model.embed([texts[i] for i in order], batch_size=self.batch_size))
        out: list[list[float]] = [[] for _ in texts]
        for i, v in zip(order, vecs):
            out[i] = list(map(float, v))
        return out


EG2_MODELS = {"google/embeddinggemma-2", "embeddinggemma-2", "eg2"}
EG2_QUERY_PREFIX = "task: search result | query: "
# A REAL directory holding model_quantized.onnx + model_quantized.onnx_data + tokenizer.json. Not the HF cache snapshot:
# its files are symlinks into blobs/ and onnxruntime >= 1.30 refuses external data that resolves outside the model dir.
EG2_ONNX_DEFAULT = "~/.local/share/deeprecall/eg2-onnx"


class RemoteDown(RuntimeError):
    """The remote embedding server failed and the local CPU fallback is not allowed for this call (index builds).
    index.py catches it and switches to FTS-first mode (docs stay keyword-findable, vectors pending)."""


class Eg2Embedder:
    """EmbeddingGemma 2: remote `/v1/embeddings` first, local ONNX int8 fallback (queries; builds only if allowed).

    Doc text convention: deeprecall formats docs as "{title}\n{chunk}" and sections as "{title} > {hpath}\n{text}",
    so the first line IS the title. EG2's document prompt is "title: {title} | text: {text}"; we split on the
    first newline to fill it (no newline -> title: none). Queries get "task: search result | query: ".
    `dim` < 768 is a Matryoshka cut: the server truncates+renormalizes (`dimensions`), the local path does the same.
    """
    FULL_DIM = 768

    def __init__(self, opts: dict):
        self.dim = int(opts.get("dim", self.FULL_DIM))
        self.remote_url = (opts.get("remote_url") or "").rstrip("/")
        # Ollama option: point remote_url at http://HOST:11434/v1/embeddings (OpenAI-compatible) and set
        # remote_model = "embeddinggemma-2:270m"; Ollama requires the "model" field, the desktop server ignores it.
        self.remote_model = opts.get("remote_model") or ""
        self.remote_timeout = float(opts.get("remote_timeout", 120))
        self.query_timeout = float(opts.get("query_timeout", 0.75))   # LAN GPU answers in ~50 ms; fall back fast
        # Query-side down-marker shared across processes (each `deeprecall recall` is a fresh process): after a remote
        # query failure, skip the remote for remote_down_secs. 2026-10-06 A/B with the tower off: no marker + 2 retries
        # = 14 s per search; the int8 encode itself is ~0.3 s.
        self.remote_down_secs = float(opts.get("remote_down_secs", 120))
        self.down_marker = os.path.expanduser(str(opts.get("down_marker") or "~/.cache/deeprecall/eg2-remote-down-"
                                                  + hashlib.sha1(self.remote_url.encode()).hexdigest()[:10]))   # per URL
        self.batch_size = int(opts.get("batch_size", 48))
        self.inflight = int(opts.get("inflight", 3))
        self.remote_required_for_build = bool(opts.get("remote_required_for_build", True))
        self.local_onnx = os.path.expanduser(str(opts.get("local_onnx", EG2_ONNX_DEFAULT)))
        self.threads = int(opts.get("threads", 4))
        # Tower-down RAM cap: each process that loads the int8 model holds ~530 MB RSS. The recall log peaks at 4
        # overlapping recalls (2026-10-06, 418 recalls), i.e. ~2.1 GB on a 7.8 GB box that OOM-killed at 2.4 GB in
        # September. At most `local_slots` processes load the model at once (flock slot files); the rest wait.
        self.local_slots = max(1, int(opts.get("local_slots", 2)))
        self.slot_dir = os.path.expanduser(str(opts.get("slot_dir") or "~/.cache/deeprecall"))
        self._slot_fd = None
        self._local = None
        self.last_backend = None
        self.stats = {"remote_calls": 0, "remote_texts": 0, "remote_secs": 0.0, "local_texts": 0, "fallbacks": 0}

    # -- formatting -------------------------------------------------------------------------------------------
    @staticmethod
    def doc_text(t: str) -> str:
        title, nl, body = t.partition("\n")
        if not nl:
            return f"title: none | text: {t}"
        return f"title: {title.strip() or 'none'} | text: {body}"

    @staticmethod
    def query_text(q: str) -> str:
        return EG2_QUERY_PREFIX + q

    # -- remote -----------------------------------------------------------------------------------------------
    def _post(self, texts: list[str], timeout: float) -> list[list[float]]:
        import urllib.request
        body = {"input": texts}
        if self.remote_model:
            body["model"] = self.remote_model
        if self.dim != self.FULL_DIM:
            body["dimensions"] = self.dim
        req = urllib.request.Request(self.remote_url, json.dumps(body).encode(), {"content-type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        self.stats["remote_calls"] += 1; self.stats["remote_texts"] += len(texts); self.stats["remote_secs"] += time.time() - t0
        rows = sorted(d["data"], key=lambda x: x["index"])
        if len(rows) != len(texts):
            raise RuntimeError(f"remote returned {len(rows)} embeddings for {len(texts)} texts")
        # the server's 768d vectors come back with norm ~1.002 (bf16 normalize); renormalize so stored and query
        # vectors are exactly unit (sqlite-vec L2 distance == cosine only then)
        return [normalize(list(map(float, x["embedding"]))) for x in rows]

    def _probe(self, timeout: float = 3.0) -> None:
        """Fail fast when the server is unreachable. A powered-off host (or WSL's mirrored loopback) does not refuse,
        it black-holes: without this, a build waited remote_timeout x (retries+1) = ~6 min to learn the tower is off
        (measured 2026-10-06, T3b)."""
        import socket
        from urllib.parse import urlparse
        u = urlparse(self.remote_url)
        with socket.create_connection((u.hostname, u.port or (443 if u.scheme == "https" else 80)), timeout=timeout):
            pass

    def _remote(self, texts: list[str], timeout: float, retries: int = 2) -> list[list[float]]:
        from concurrent.futures import ThreadPoolExecutor
        if retries:                     # builds: one cheap connect check before committing to long timeouts
            self._probe(min(3.0, timeout))
        # length-sorted batches: the server pads each batch to its longest text, so mixing a 20-token summary with
        # 800-token chunks wastes most of the GPU (measured 2026-10-06: unsorted 2.8 files/s, ~23K tok/s)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        def one(idx):
            for a in range(retries + 1):
                try:
                    return self._post([texts[i] for i in idx], timeout)
                except Exception:
                    if a == retries:
                        raise
                    time.sleep(1.5 * (a + 1))
        out: list[list[float]] = [[] for _ in texts]
        if len(batches) == 1:
            parts = [one(batches[0])]
        else:
            with ThreadPoolExecutor(max_workers=self.inflight) as ex:
                parts = list(ex.map(one, batches))
        for idx, part in zip(batches, parts):
            for i, v in zip(idx, part):
                out[i] = v
        return out

    # -- local ONNX int8 ----------------------------------------------------------------------------------------
    def _acquire_slot(self) -> None:
        """Hold one of `local_slots` flock slots for the life of this process (released on exit)."""
        import fcntl
        if self._slot_fd is not None:
            return
        os.makedirs(self.slot_dir, exist_ok=True)
        fds = []
        for i in range(self.local_slots):
            fd = open(os.path.join(self.slot_dir, f"eg2-local-slot-{i}.lock"), "w")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._slot_fd = fd
                for o in fds:
                    o.close()
                return
            except OSError:
                fds.append(fd)
        # all busy: queue on one slot (spread by pid), blocking
        keep = fds[os.getpid() % len(fds)]
        for o in fds:
            if o is not keep:
                o.close()
        t0 = time.time()
        fcntl.flock(keep, fcntl.LOCK_EX)
        self.stats["slot_wait_secs"] = round(time.time() - t0, 2)
        self._slot_fd = keep

    def _load_local(self):
        if self._local is None:
            self._acquire_slot()
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
            d = self.local_onnx
            if not os.path.isfile(os.path.join(d, "model_quantized.onnx")):
                raise FileNotFoundError(f"EG2 ONNX int8 model not found in {d} (model_quantized.onnx + _data + tokenizer.json)")
            so = ort.SessionOptions(); so.intra_op_num_threads = self.threads
            sess = ort.InferenceSession(os.path.join(d, "model_quantized.onnx"), so, providers=["CPUExecutionProvider"])
            tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
            z = np.zeros((0, 512), dtype=np.float32)
            names = {i.name for i in sess.get_inputs()}
            def enc(t: str) -> list[float]:
                ids = np.array([tk.encode(t).ids[:8192]], dtype=np.int64)
                feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
                for n in ("image_features", "video_features", "audio_features"):
                    if n in names:
                        feed[n] = z
                v = sess.run(["sentence_embedding"], feed)[0][0][: self.dim]
                return normalize(list(map(float, v)))
            self._local = enc
        return self._local

    def _local_embed(self, texts: list[str]) -> list[list[float]]:
        enc = self._load_local()
        self.stats["local_texts"] += len(texts)
        return [enc(t) for t in texts]

    # -- public -------------------------------------------------------------------------------------------------
    def _remote_marked_down(self) -> bool:
        try:
            return time.time() - os.path.getmtime(self.down_marker) < self.remote_down_secs
        except OSError:
            return False

    def _mark_remote_down(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.down_marker), exist_ok=True)
            with open(self.down_marker, "w") as f:
                f.write(f"{self.remote_url}\n")
        except OSError:
            pass

    def _run(self, texts: list[str], timeout: float, allow_local: bool, query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        if self.remote_url and not (query and self._remote_marked_down()):
            try:
                out = self._remote(texts, timeout, retries=0 if query else 2)
                self.last_backend = "remote"
                return out
            except Exception as e:
                if not allow_local:
                    raise RemoteDown(f"EG2 remote embedder {self.remote_url} failed ({e}); local CPU fallback is refused "
                                     f"for index builds (remote_required_for_build). Is the desktop endpoint up?")
                self.stats["fallbacks"] += 1
                if query:
                    self._mark_remote_down()
                print(f"deeprecall: EG2 remote failed ({e}); using local ONNX int8", file=sys.stderr)
        self.last_backend = "local"
        return self._local_embed(texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._run([self.doc_text(t) for t in texts], self.remote_timeout, allow_local=not self.remote_required_for_build)

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self._run([self.query_text(t) for t in texts], self.query_timeout, allow_local=True, query=True)


def _key(opts: dict | None) -> str:
    return json.dumps(opts or {}, sort_keys=True)


@lru_cache(maxsize=4)
def _get(model: str, opts_key: str):
    opts = json.loads(opts_key)
    if model.startswith("hash"):
        return HashEmbedder(int(model.split(":")[1]) if ":" in model else 64)
    if model in EG2_MODELS:
        return Eg2Embedder(opts)
    return FastEmbedder(model)


def get_embedder(model: str, opts: dict | None = None):
    """Embedder for a model name; `opts` is the config's [embedding] table (EG2 reads remote_url/dim/... from it)."""
    return _get(model, _key(opts))


def normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def mean(vs: list[list[float]]) -> list[float]:
    d = len(vs[0])
    return normalize([sum(v[i] for v in vs) / len(vs) for i in range(d)])
