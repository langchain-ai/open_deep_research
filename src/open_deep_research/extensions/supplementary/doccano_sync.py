"""Real-time Doccano synchronization via Infer API.

Provides:
- DoccanoSyncClient: push annotations one-by-one with offline queue + auto-flush.
- DoccanoRewriteMatcher: robust matching for rewrite-LLM mode.
- setup_doccano_project: create-or-find a Doccano project through the Infer API.
"""

import json
import logging
import re
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


def _build_headers(token: str) -> Dict[str, str]:
    """Build Bearer auth headers."""
    return {"Authorization": f"Bearer {token}"}


# ──────────────────────────────────────────────────────
# Project setup helper
# ──────────────────────────────────────────────────────

def setup_doccano_project(
    api_url: str,
    token: str,
    project_name: str,
    labels: Optional[List[str]] = None,
    project_type: str = "DocumentClassification",
) -> int:
    """Create or find an existing Doccano project via Infer API.

    Returns the project_id.
    """
    headers = _build_headers(token)
    base = api_url.rstrip("/")

    # 1. Check if project already exists
    try:
        resp = requests.get(f"{base}/doccano/projects", headers=headers, timeout=15)
        resp.raise_for_status()
        for proj in resp.json():
            if proj.get("name") == project_name:
                logger.info("Reusing existing Doccano project #%d '%s'", proj["id"], project_name)
                return proj["id"]
    except Exception as e:
        logger.warning("Could not list Doccano projects: %s", e)

    # 2. Create new project
    payload: Dict[str, Any] = {
        "name": project_name,
        "description": f"Auto-created by LLM Tool for {project_name}",
        "project_type": project_type,
    }
    if labels:
        payload["labels"] = labels

    resp = requests.post(f"{base}/doccano/projects", headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    project_id = data["project_id"]
    logger.info("Created Doccano project #%d '%s'", project_id, project_name)
    return project_id


def setup_doccano_project_safe(
    api_url: str,
    token: str,
    project_name: str,
    labels: Optional[List[str]] = None,
    project_type: str = "DocumentClassification",
) -> Optional[int]:
    """Like setup_doccano_project but returns None instead of raising."""
    try:
        return setup_doccano_project(api_url, token, project_name, labels, project_type)
    except requests.exceptions.HTTPError as exc:
        logger.error("Doccano project setup failed: %s", exc)
        return None
    except Exception as exc:
        logger.error("Doccano project setup failed: %s", exc)
        return None


# ──────────────────────────────────────────────────────
# Project query helpers (used by sync mode selector)
# ──────────────────────────────────────────────────────

def list_doccano_projects(api_url: str, token: str) -> List[dict]:
    """List all Doccano projects via Infer API."""
    headers = _build_headers(token)
    base = api_url.rstrip("/")
    resp = requests.get(f"{base}/doccano/projects", headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def count_doccano_examples(api_url: str, token: str, project_id: int) -> int:
    """Return the number of examples in a Doccano project."""
    headers = _build_headers(token)
    base = api_url.rstrip("/")
    resp = requests.get(
        f"{base}/doccano/projects/{project_id}/examples/count",
        headers=headers, timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("count", 0)


def clear_doccano_project(api_url: str, token: str, project_id: int) -> int:
    """Delete all examples from a Doccano project. Returns count deleted."""
    headers = _build_headers(token)
    base = api_url.rstrip("/")
    resp = requests.delete(
        f"{base}/doccano/projects/{project_id}/examples",
        headers=headers, timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("deleted", 0)


def list_doccano_examples(
    api_url: str, token: str, project_id: int,
    total: Optional[int] = None,
    on_progress: Optional[Any] = None,
) -> List[dict]:
    """Fetch all examples from a project (auto-paginate).

    Args:
        total: If known, the total number of examples (used for progress).
        on_progress: Optional callback ``(fetched_so_far, total)`` called
            after each page is fetched.

    Returns the full list of example dicts.
    """
    headers = _build_headers(token)
    base = api_url.rstrip("/")
    results: List[dict] = []
    offset = 0
    page_size = 1000

    while True:
        resp = requests.get(
            f"{base}/doccano/projects/{project_id}/examples",
            headers=headers,
            params={"limit": page_size, "offset": offset},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        page = data.get("results", [])
        results.extend(page)
        if on_progress:
            on_progress(len(results), total)
        if len(page) < page_size:
            break
        offset += page_size

    logger.info("Fetched %d examples from project #%d", len(results), project_id)
    return results


def update_doccano_llm_annotation(
    api_url: str,
    token: str,
    project_id: int,
    example_id: int,
    existing_meta: dict,
    new_annotation: dict,
    label_names: List[str],
    extra_meta: Optional[Dict[str, Any]] = None,
    merge_only: bool = False,
) -> bool:
    """PATCH an example's meta to fully rewrite LLM-related fields.

    Preserves human categories but rewrites all metadata including
    inference_time, llm_annotations, and any extra_meta keys.

    When *merge_only* is True the ``llm_annotations`` block is left
    untouched — only *extra_meta* keys are merged.  This is used by
    Stage 3 (trained-model annotation) to add ``model_annotations``
    without overwriting existing LLM data.
    """
    headers = _build_headers(token)
    base = api_url.rstrip("/")

    meta = dict(existing_meta) if existing_meta else {}
    # Merge extra meta (inference_time, identifier, etc.)
    if extra_meta:
        meta.update(extra_meta)
    if not merge_only:
        meta["llm_annotations"] = {
            "raw": new_annotation,
            "labels": label_names,
        }

    try:
        resp = requests.patch(
            f"{base}/doccano/projects/{project_id}/examples/{example_id}/meta",
            headers=headers,
            json={"meta": meta},
            timeout=15,
        )
        return resp.status_code in (200, 201)
    except Exception as e:
        logger.warning("update_doccano_llm_annotation failed for example %d: %s", example_id, e)
        return False


# ──────────────────────────────────────────────────────
# Rewrite matcher
# ──────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Strip, collapse whitespace, lowercase."""
    return re.sub(r"\s+", " ", text.strip()).lower()


class DoccanoRewriteMatcher:
    """Match incoming annotations to existing Doccano examples.

    Uses a double-key index (normalized text + meta values) for robust
    matching.  Falls back to text-only matching for diagnostics.
    """

    def __init__(
        self,
        existing_examples: List[dict],
        match_meta_keys: Optional[List[str]] = None,
    ):
        self._match_keys = match_meta_keys or ["identifier"]
        # Primary index: (norm_text, meta_tuple) → [example, ...]
        self._index: Dict[Tuple, List[dict]] = defaultdict(list)
        # Secondary index: norm_text → [example, ...]
        self._text_index: Dict[str, List[dict]] = defaultdict(list)

        for ex in existing_examples:
            norm = _normalize_text(ex.get("text", ""))
            meta = ex.get("meta") or {}
            meta_vals = tuple(str(meta.get(k, "")) for k in self._match_keys)
            self._index[(norm, meta_vals)].append(ex)
            self._text_index[norm].append(ex)

    def match(
        self, text: str, meta: Optional[dict] = None
    ) -> Tuple[str, Optional[dict], str]:
        """Try to match ``text`` + ``meta`` to an existing example.

        Returns ``(action, example_or_None, reason)``:
        - ``("update", example, ...)`` — single match found
        - ``("create", None, ...)``   — no match, create new
        - ``("skip", None, ...)``     — ambiguous (N>1 matches)
        """
        norm = _normalize_text(text)
        meta = meta or {}
        meta_vals = tuple(str(meta.get(k, "")) for k in self._match_keys)

        matches = self._index.get((norm, meta_vals), [])

        if len(matches) == 1:
            return ("update", matches[0], "exact match (text + meta)")

        if len(matches) > 1:
            logger.warning(
                "Rewrite matcher: %d ambiguous matches for text='%s…' meta=%s — skipping",
                len(matches), text[:40], meta_vals,
            )
            return ("skip", None, f"ambiguous: {len(matches)} matches")

        # No full match — fall back to text-only
        text_matches = self._text_index.get(norm, [])
        if len(text_matches) == 1:
            return ("update", text_matches[0], "text match (meta differs)")
        if len(text_matches) > 1:
            logger.debug(
                "Rewrite matcher: %d text matches but meta differs for '%s…' — ambiguous",
                len(text_matches), text[:40],
            )
            return ("skip", None, f"ambiguous: {len(text_matches)} text matches")

        return ("create", None, "no match")

    def preview_match(
        self,
        rows: List[Dict[str, str]],
        text_key: str,
    ) -> Dict[str, int]:
        """Dry-run matching on sample rows. Returns stats dict."""
        stats = {"matched": 0, "unmatched": 0, "ambiguous": 0}
        for row in rows:
            text = row.get(text_key, "")
            meta = {k: row.get(k, "") for k in self._match_keys}
            action, _, reason = self.match(text, meta)
            if action == "update":
                stats["matched"] += 1
            elif action == "skip":
                stats["ambiguous"] += 1
            else:
                stats["unmatched"] += 1
        return stats


# ──────────────────────────────────────────────────────
# Sync client with offline queue
# ──────────────────────────────────────────────────────

class DoccanoSyncClient:
    """Real-time annotation sync to Doccano via Infer API.

    Features
    --------
    - Push each annotation as it completes
    - Offline queue: if connection fails, queue locally
    - Auto-flush: background thread retries queued items when connection returns
    - Heartbeat: periodic /health check to detect recovery
    """

    FLUSH_INTERVAL = 30      # seconds between flush attempts
    BATCH_CHUNK_SIZE = 50    # max items per batch push

    def __init__(
        self,
        api_url: str,
        token: str,
        project_id: int,
        mode: str = "push",
        match_meta_keys: Optional[List[str]] = None,
        existing_examples: Optional[List[dict]] = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.token = token
        self.project_id = project_id
        self.mode = mode  # "push" or "rewrite"

        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._online = True
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stats = {"pushed": 0, "queued": 0, "flush_errors": 0,
                       "updated": 0, "created": 0, "skipped": 0}

        # Optional push cap — only push items whose sequential index is in
        # this set.  Populated by set_push_sample() for random sampling, or
        # left as None to push everything.
        self.push_limit: Optional[int] = None
        self._push_indices: Optional[set] = None
        self._push_counter: int = 0  # incremented on every push() call

        # Rewrite mode matcher
        self._matcher: Optional[DoccanoRewriteMatcher] = None
        if mode == "rewrite" and existing_examples is not None:
            self._matcher = DoccanoRewriteMatcher(
                existing_examples, match_meta_keys=match_meta_keys,
            )

    # ── Lifecycle ───────────────────────────────────

    def start(self) -> None:
        """Start the background flush thread."""
        self._stop_event.clear()
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="doccano-flush"
        )
        self._flush_thread.start()
        logger.info("DoccanoSyncClient started (project #%d)", self.project_id)

    def stop(self) -> dict:
        """Stop flush thread, attempt final flush, return stats."""
        self._stop_event.set()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=10)

        # Final flush attempt
        if self._queue:
            self._try_flush()

        remaining = len(self._queue)
        stats = {**self._stats, "remaining_queue": remaining}
        logger.info("DoccanoSyncClient stopped. Stats: %s", stats)
        return stats

    # ── Sampling ─────────────────────────────────────

    def set_push_sample(self, total_rows: int, sample_size: int) -> None:
        """Pre-select random indices for representative Doccano sampling.

        Picks ``sample_size`` unique indices from ``[0, total_rows)`` uniformly
        at random.  Only calls to :meth:`push` whose sequential counter falls
        in this set will actually be sent to Doccano.
        """
        import random
        sample_size = min(sample_size, total_rows)
        self._push_indices = set(random.sample(range(total_rows), sample_size))
        self.push_limit = sample_size
        self._push_counter = 0
        logger.info(
            "Doccano push sample: %d / %d indices selected (random)",
            sample_size, total_rows,
        )

    # ── Push ────────────────────────────────────────

    def push(self, text: str, annotation: dict, meta: Optional[dict] = None) -> bool:
        """Push one annotation. Returns True if sent immediately."""
        # Random sampling: check if this row's index is in the pre-selected set
        if self._push_indices is not None:
            idx = self._push_counter
            self._push_counter += 1
            if idx not in self._push_indices:
                self._stats["skipped"] += 1
                return False
        elif self.push_limit is not None and self._stats["pushed"] >= self.push_limit:
            self._stats["skipped"] += 1
            return False
        if self.mode == "rewrite" and self._matcher is not None:
            return self._push_rewrite(text, annotation, meta)

        item = {"text": text, "annotation": annotation}
        if meta:
            item["meta"] = meta

        if self._try_push_single(item):
            self._stats["pushed"] += 1
            self._online = True
            # If we just came back online, try flushing the queue
            if self._queue:
                self._try_flush()
            return True
        else:
            with self._lock:
                self._queue.append(item)
            self._stats["queued"] += 1
            self._online = False
            return False

    def _push_rewrite(self, text: str, annotation: dict, meta: Optional[dict] = None) -> bool:
        """Route a single item through the rewrite matcher."""
        action, example, reason = self._matcher.match(text, meta)

        if action == "update":
            ok = update_doccano_llm_annotation(
                api_url=self.api_url,
                token=self.token,
                project_id=self.project_id,
                example_id=example["id"],
                existing_meta=example.get("meta") or {},
                new_annotation=annotation,
                label_names=self._extract_label_names(annotation),
                extra_meta=meta,
            )
            if ok:
                self._stats["updated"] += 1
            else:
                self._stats["flush_errors"] += 1
            return ok

        if action == "create":
            item = {"text": text, "annotation": annotation}
            if meta:
                item["meta"] = meta
            ok = self._try_push_single(item)
            if ok:
                self._stats["created"] += 1
                self._stats["pushed"] += 1
            else:
                with self._lock:
                    self._queue.append(item)
                self._stats["queued"] += 1
            return ok

        # action == "skip"
        self._stats["skipped"] += 1
        return False

    @staticmethod
    def _extract_label_names(annotation: dict) -> List[str]:
        """Simple label extraction from annotation keys/values."""
        labels = []
        for key, value in annotation.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    labels.append(f"{key}_yes")
            elif isinstance(value, str):
                for part in value.split(","):
                    part = part.strip()
                    if part:
                        labels.append(f"{key}_{part}")
            elif isinstance(value, list):
                for item in value:
                    if item is not None:
                        labels.append(f"{key}_{item}")
            elif isinstance(value, (int, float)):
                labels.append(f"{key}_{value}")
        return labels

    def push_batch(self, items: List[dict]) -> dict:
        """Push a list of annotations in chunks.

        Each item: ``{text, annotation, meta}``.
        On failure the items are enqueued for later flush.
        Returns ``{pushed, queued, errors}``.

        In rewrite mode, each item goes through the matcher individually.
        """
        # Rewrite mode: item-by-item matching
        if self.mode == "rewrite" and self._matcher is not None:
            ok_count = 0
            for item in items:
                ok = self._push_rewrite(
                    text=item["text"],
                    annotation=item.get("annotation", {}),
                    meta=item.get("meta"),
                )
                if ok:
                    ok_count += 1
            return {
                "pushed": self._stats["created"],
                "updated": self._stats["updated"],
                "skipped": self._stats["skipped"],
                "queued": self._stats["queued"],
                "errors": self._stats["flush_errors"],
            }

        pushed = 0
        queued = 0
        errors = 0

        for i in range(0, len(items), self.BATCH_CHUNK_SIZE):
            chunk = items[i : i + self.BATCH_CHUNK_SIZE]
            try:
                resp = requests.post(
                    f"{self.api_url}/doccano/push/batch",
                    headers=_build_headers(self.token),
                    json={"project_id": self.project_id, "items": chunk},
                    timeout=30,
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    n = data.get("pushed", len(chunk))
                    pushed += n
                    self._stats["pushed"] += n
                    errs = data.get("errors", [])
                    if errs:
                        errors += len(errs)
                        self._stats["flush_errors"] += len(errs)
                        logger.warning("push_batch chunk had %d errors: %s", len(errs), errs[:3])
                else:
                    # Enqueue for later
                    with self._lock:
                        self._queue.extend(chunk)
                    queued += len(chunk)
                    self._stats["queued"] += len(chunk)
                    logger.warning("push_batch chunk failed HTTP %d — queued %d items", resp.status_code, len(chunk))
            except Exception as e:
                with self._lock:
                    self._queue.extend(chunk)
                queued += len(chunk)
                self._stats["queued"] += len(chunk)
                logger.warning("push_batch chunk exception: %s — queued %d items", e, len(chunk))

        logger.info("push_batch done: pushed=%d, queued=%d, errors=%d", pushed, queued, errors)
        return {"pushed": pushed, "queued": queued, "errors": errors}

    def _try_push_single(self, item: dict) -> bool:
        """Attempt to POST a single annotation. Returns True on success."""
        try:
            resp = requests.post(
                f"{self.api_url}/doccano/push",
                headers=_build_headers(self.token),
                json={
                    "project_id": self.project_id,
                    "text": item["text"],
                    "annotation": item.get("annotation", {}),
                    "meta": item.get("meta"),
                },
                timeout=10,
            )
            return resp.status_code in (200, 201)
        except Exception:
            return False

    # ── Background flush ────────────────────────────

    def _flush_loop(self) -> None:
        """Periodically check connectivity and flush queue."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.FLUSH_INTERVAL)
            if self._stop_event.is_set():
                break
            if not self._queue:
                continue
            if self._heartbeat():
                self._online = True
                self._try_flush()

    def _try_flush(self) -> None:
        """Batch-flush the offline queue in chunks."""
        while self._queue:
            chunk: List[dict] = []
            with self._lock:
                for _ in range(min(self.BATCH_CHUNK_SIZE, len(self._queue))):
                    chunk.append(self._queue.popleft())

            try:
                resp = requests.post(
                    f"{self.api_url}/doccano/push/batch",
                    headers=_build_headers(self.token),
                    json={"project_id": self.project_id, "items": chunk},
                    timeout=30,
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    pushed = data.get("pushed", len(chunk))
                    self._stats["pushed"] += pushed
                    errors = data.get("errors", [])
                    if errors:
                        self._stats["flush_errors"] += len(errors)
                        logger.warning("Batch flush had %d errors: %s", len(errors), errors[:3])
                    logger.info("Flushed %d items from queue (%d remaining)", pushed, len(self._queue))
                else:
                    # Put items back
                    with self._lock:
                        self._queue.extendleft(reversed(chunk))
                    self._stats["flush_errors"] += 1
                    logger.warning("Batch flush failed: HTTP %d", resp.status_code)
                    break
            except Exception as e:
                # Put items back
                with self._lock:
                    self._queue.extendleft(reversed(chunk))
                self._stats["flush_errors"] += 1
                self._online = False
                logger.warning("Batch flush exception: %s", e)
                break

    def _heartbeat(self) -> bool:
        """Check Infer API connectivity via /health."""
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Properties ──────────────────────────────────

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> dict:
        return {**self._stats, "remaining_queue": len(self._queue), "online": self._online}
