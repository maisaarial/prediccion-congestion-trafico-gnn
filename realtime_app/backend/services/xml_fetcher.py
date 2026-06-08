from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests


@dataclass(frozen=True)
class XmlCapture:
    content: str
    captured_at: datetime
    path: Path
    source_url: str
    from_cache: bool = False


@dataclass(frozen=True)
class XmlDownload:
    content: str
    raw_bytes: bytes
    captured_at: datetime
    source_url: str


def _timestamp_for_filename(moment: datetime) -> str:
    return moment.astimezone().strftime("%Y%m%d_%H%M%S")


def fetch_xml(
    url: str,
    timeout_seconds: int = 20,
    user_agent: str = "traffic-gnn-realtime-demo/0.1",
) -> XmlDownload:
    captured_at = datetime.now(timezone.utc)

    response = requests.get(
        url,
        timeout=timeout_seconds,
        headers={"User-Agent": user_agent},
    )
    response.raise_for_status()

    raw_bytes = response.content
    encoding = response.encoding or "utf-8"
    content = raw_bytes.decode(encoding, errors="replace")

    return XmlDownload(
        content=content,
        raw_bytes=raw_bytes,
        captured_at=captured_at,
        source_url=url,
    )


def store_xml_capture(download: XmlDownload, captures_dir: Path) -> XmlCapture:
    captures_dir.mkdir(parents=True, exist_ok=True)
    path = captures_dir / f"pm_{_timestamp_for_filename(download.captured_at)}.xml"
    path.write_bytes(download.raw_bytes)

    return XmlCapture(
        content=download.content,
        captured_at=download.captured_at,
        path=path,
        source_url=download.source_url,
        from_cache=False,
    )


def fetch_and_store_xml(
    url: str,
    captures_dir: Path,
    timeout_seconds: int = 20,
    user_agent: str = "traffic-gnn-realtime-demo/0.1",
) -> XmlCapture:
    download = fetch_xml(
        url=url,
        timeout_seconds=timeout_seconds,
        user_agent=user_agent,
    )
    return store_xml_capture(download, captures_dir)


def load_latest_xml_capture(captures_dir: Path, source_url: str) -> XmlCapture | None:
    if not captures_dir.exists():
        return None

    captures = sorted(captures_dir.glob("pm_*.xml"), key=lambda p: p.stat().st_mtime)
    if not captures:
        return None

    path = captures[-1]
    captured_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    content = path.read_text(encoding="utf-8", errors="replace")

    return XmlCapture(
        content=content,
        captured_at=captured_at,
        path=path,
        source_url=source_url,
        from_cache=True,
    )
