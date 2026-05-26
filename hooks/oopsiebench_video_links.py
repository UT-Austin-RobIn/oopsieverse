"""
MkDocs hook: omit GLightbox / button styling for demo MP4s that are absent on disk.

Videos live under docs/assets/videos/*. When clones skip Git LFS (or recordings are
missing), replace the <a> with a muted <span> at build time so the bench page stays
readable without broken lightbox buttons.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

# <a … href="….mp4" …>Unsafe</a> — href may appear anywhere inside the opening tag
_VIDEO_ANCHOR_RE = re.compile(
    r'<a\s([^>]*?)href="(?P<href>[^"#]+\.mp4)"([^>]*)>(?P<label>[^<]*)</a>',
    flags=re.IGNORECASE,
)


def _docs_video_path(href: str, docs_dir: Path) -> Path | None:
    """Map table href to a docs-root path under docs/assets/videos."""
    ix = href.find("assets/videos/")
    if ix < 0:
        return None
    rel = href[ix:]  # assets/videos/…
    return (docs_dir / rel).resolve()


def on_page_markdown(markdown: str, *, page, config, **kwargs) -> str:
    """Strip GLightbox video anchors when their files are absent from docs/."""
    if page is None:
        return markdown
    if not str(page.file.src_path).replace("\\", "/").endswith("oopsiebench.md"):
        return markdown

    docs_dir = Path(config["docs_dir"]).resolve()
    missing: list[str] = []

    def repl(match: re.Match[str]) -> str:
        href_raw = match.group("href")
        kind = match.group("label").strip()
        resolved = _docs_video_path(href_raw, docs_dir)
        if resolved is not None and resolved.is_file():
            return match.group(0)

        hint = resolved.as_posix() if resolved is not None else href_raw
        missing.append(hint)
        tip = html.escape(f"{kind} demo: file not present at {hint} (run git lfs pull if tracked).")
        return (
            f'<span class="oopsiebench-video-missing" role="note">'
            f'<abbr title="{tip}">Unavailable</abbr>'
            f"</span>"
        )

    markdown = _VIDEO_ANCHOR_RE.sub(repl, markdown)

    if missing:
        # Non-fatal: local builds skip LFS frequently; CI with LFS should keep buttons.
        print(
            f"[oopsiebench_video_links] {len(missing)} missing MP4(s) on disk"
            " — substituted muted labels on OopsieBench page.",
        )

    return markdown
