"""
MkDocs hook: OopsieBench demo video cells → clickable thumbnails (GLightbox).

- Converts legacy text-button anchors to thumbnail links at build time.
- Replaces cells with a muted placeholder when the MP4 is absent on disk.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

# Legacy: <a … href="….mp4" …>Unsafe|Safe</a>
_LEGACY_BUTTON_RE = re.compile(
    r'<a\s([^>]*?)href="(?P<href>[^"#]+\.mp4)"([^>]*)>'
    r"(?P<label>Unsafe|Safe)</a>",
    flags=re.IGNORECASE,
)

# Thumbnail: <a … href="….mp4" …><img src="….jpg" …></a>
_THUMB_ANCHOR_RE = re.compile(
    r'<a\s(?P<attrs>[^>]*?)href="(?P<href>[^"#]+\.mp4)"(?P<attrs2>[^>]*)>\s*'
    r'<img\s(?P<imgattrs>[^>]*?)>\s*</a>',
    flags=re.IGNORECASE | re.DOTALL,
)


def _docs_video_path(href: str, docs_dir: Path) -> Path | None:
    ix = href.find("assets/videos/")
    if ix < 0:
        return None
    return (docs_dir / href[ix:]).resolve()


def _thumb_href_for_video(href: str) -> str:
    return href.replace("/assets/videos/", "/assets/videos/thumbnails/").replace(
        ".mp4", ".jpg"
    )


def _task_slug_from_href(href: str) -> str:
    stem = Path(href.split("/")[-1]).stem
    for suffix in ("_unsafe", "_safe"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _kind_from_label(label: str) -> str:
    return "unsafe" if label.strip().lower() == "unsafe" else "safe"


def _build_thumb_anchor(href: str, kind: str) -> str:
    task = _task_slug_from_href(href)
    thumb = _thumb_href_for_video(href)
    title = html.escape(f"{task} — {kind}")
    alt = html.escape(f"{kind.capitalize()} {task} demo")
    return (
        f'<a href="{href}" class="glightbox oopsiebench-video-thumb '
        f'oopsiebench-video-thumb--{kind}" data-type="video" title="{title}">'
        f'<img src="{thumb}" alt="{alt}" loading="lazy"></a>'
    )


def _missing_placeholder(kind: str, hint: str) -> str:
    tip = html.escape(f"{kind.capitalize()} demo: file not present at {hint}.")
    return (
        f'<span class="oopsiebench-video-missing oopsiebench-video-thumb '
        f'oopsiebench-video-thumb--{kind}" role="note">'
        f'<span class="oopsiebench-video-thumb__placeholder" title="{tip}">'
        f"To be added</span></span>"
    )


def _substitute_if_missing(
    href_raw: str,
    kind: str,
    present_html: str,
    docs_dir: Path,
    missing: list[str],
) -> str:
    resolved = _docs_video_path(href_raw, docs_dir)
    if resolved is not None and resolved.is_file():
        return present_html
    hint = resolved.as_posix() if resolved is not None else href_raw
    missing.append(hint)
    return _missing_placeholder(kind, hint)


def on_page_markdown(markdown: str, *, page, config, **kwargs) -> str:
    if page is None:
        return markdown
    if not str(page.file.src_path).replace("\\", "/").endswith("oopsiebench.md"):
        return markdown

    docs_dir = Path(config["docs_dir"]).resolve()
    missing: list[str] = []

    def legacy_repl(match: re.Match[str]) -> str:
        href = match.group("href")
        kind = _kind_from_label(match.group("label"))
        anchor = _build_thumb_anchor(href, kind)
        return _substitute_if_missing(href, kind, anchor, docs_dir, missing)

    markdown = _LEGACY_BUTTON_RE.sub(legacy_repl, markdown)

    def thumb_repl(match: re.Match[str]) -> str:
        href = match.group("href")
        attrs = match.group("attrs") + match.group("attrs2")
        kind = "unsafe" if "thumb--unsafe" in attrs else "safe"
        if "thumb--safe" in attrs:
            kind = "safe"
        elif "thumb--unsafe" not in attrs:
            kind = "unsafe" if href.endswith("_unsafe.mp4") else "safe"
        return _substitute_if_missing(href, kind, match.group(0), docs_dir, missing)

    markdown = _THUMB_ANCHOR_RE.sub(thumb_repl, markdown)

    if missing:
        print(
            f"[oopsiebench_video_links] {len(missing)} missing MP4(s) on disk"
            " — substituted placeholders on OopsieBench page.",
        )

    return markdown
