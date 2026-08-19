# -*- coding: utf-8 -*-
"""
wallpaper_recolor.color.pantone
-----------------------------
Offline Pantone code → hex lookup (pantonecolors.net/pantone-to-hex table).

``pantone_hex.json`` is shipped in-repo so the app never scrapes at runtime.
Keys are casefolded with a leading “Pantone” prefix stripped.

Class references (code + name only):
- CAP3321C Data Wrangling
"""

from __future__ import annotations  # tuple[int, int, int] without quoting

import json
import re
from pathlib import Path

import numpy as np

from wallpaper_recolor.color.color_math import rgb_to_lab_array

_JSON_PATH = Path(__file__).resolve().parent / "pantone_hex.json"
_BOOK_SUFFIXES = frozenset({"c", "u", "tcx", "tpx", "tpg", "tpn", "tp", "cp", "up"})
_FASHION_CODE = re.compile(r"\b(\d{2}-\d{4})(?:\s+(tpx|tcx|tpg|tpn|tp))?\b")

_table: dict[str, str] | None = None
_hex_to_code: dict[str, str] | None = None
_alias: dict[str, str] | None = None
_labs: np.ndarray | None = None
_lab_hexes: list[str] | None = None
_display_rows: list[tuple[str, str]] | None = None  # (search haystack, display code)


def normalize_pantone_key(text: str) -> str:
    """Strip a leading “Pantone”, collapse space, casefold — JSON / lookup key."""
    raw = " ".join(text.strip().split())
    if raw.casefold().startswith("pantone "):
        raw = raw[7:].strip()
    return raw.casefold()


def display_pantone_code(key: str) -> str:
    """Pretty-print a normalized key: ``186 c`` → ``186 C``."""
    parts = normalize_pantone_key(key).split()
    if not parts:
        return ""
    if parts[-1] in _BOOK_SUFFIXES:
        body, suf = parts[:-1], parts[-1].upper()
        shown = " ".join(_display_token(p) for p in body)
        return f"{shown} {suf}".strip()
    return " ".join(_display_token(p) for p in parts)


def _display_token(token: str) -> str:
    if token.isalpha() and len(token) <= 3:
        return token.upper()
    if token[0].isalpha():
        return token.title()
    return token


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    raw = hex_color.strip().lstrip("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _alias_keys(key: str) -> list[str]:
    """Extra lookup spellings: no-space codes, fashion ``17-1230`` / ``17-1230 tcx``."""
    out = [key]
    compact = key.replace(" ", "")
    if compact != key:
        out.append(compact)
    match = _FASHION_CODE.search(key)
    if match:
        out.append(match.group(1))
        if match.group(2):
            out.append(f"{match.group(1)} {match.group(2)}")
    return out


def _canonical_rank(key: str) -> tuple[int, int, str]:
    """Prefer numeric book codes (``186 c``) over long names for reverse lookup."""
    if re.fullmatch(r"\d+[a-z]?\s+[cu]", key):
        return (0, len(key), key)
    if re.fullmatch(r"\d{2}-\d{4}(?:\s+\w+)?", key):
        return (1, len(key), key)
    return (2, len(key), key)


def _ensure_loaded() -> None:
    global _table, _hex_to_code, _alias, _labs, _lab_hexes, _display_rows
    if _table is not None:
        return
    raw = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    table: dict[str, str] = {}
    hex_to_code: dict[str, str] = {}
    alias: dict[str, str] = {}
    for name, hex_color in raw.items():
        key = normalize_pantone_key(str(name))
        hex_u = "#" + str(hex_color).strip().lstrip("#").upper()
        if not key:
            continue
        table[key] = hex_u
        prev = hex_to_code.get(hex_u)
        if prev is None or _canonical_rank(key) < _canonical_rank(prev):
            hex_to_code[hex_u] = key
        for extra in _alias_keys(key):
            alias.setdefault(extra, hex_u)
    _table = table
    _hex_to_code = hex_to_code
    _alias = alias
    hexes = list(hex_to_code)
    rgbs = np.array([_hex_to_rgb(h) for h in hexes], dtype=np.uint8).reshape(-1, 1, 3)
    _labs = rgb_to_lab_array(rgbs)[:, 0, :].astype(np.float32)
    _lab_hexes = hexes
    rows: list[tuple[str, str]] = []
    for key in table:
        shown = display_pantone_code(key)
        compact = key.replace(" ", "")
        haystack = f"{key} {compact} {shown.casefold()}"
        rows.append((haystack, shown))
    _display_rows = rows


def lookup_pantone_hex(text: str) -> str | None:
    """``#RRGGBB`` for a Pantone code, or ``None`` if the table has no match."""
    _ensure_loaded()
    assert _table is not None and _alias is not None
    key = normalize_pantone_key(text)
    if not key:
        return None
    return _table.get(key) or _alias.get(key)


def lookup_pantone_rgb(text: str) -> tuple[int, int, int] | None:
    """8-bit RGB for a Pantone code, or ``None`` if unknown."""
    hex_color = lookup_pantone_hex(text)
    if hex_color is None:
        return None
    return _hex_to_rgb(hex_color)


def pantone_code_for_hex(hex_color: str) -> str | None:
    """Canonical table code for an exact ``#RRGGBB``, or ``None``."""
    _ensure_loaded()
    assert _hex_to_code is not None
    hex_u = "#" + hex_color.strip().lstrip("#").upper()
    key = _hex_to_code.get(hex_u)
    if key is None:
        return None
    return display_pantone_code(key)


def pantone_code_for_rgb(
    rgb: tuple[int, int, int], *, closest: bool = True
) -> str | None:
    """Code whose hex matches ``rgb``, or the nearest Lab neighbor when ``closest``."""
    hex_color = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
    exact = pantone_code_for_hex(hex_color)
    if exact is not None or not closest:
        return exact
    _ensure_loaded()
    assert _labs is not None and _lab_hexes is not None
    target = rgb_to_lab_array(
        np.array(((rgb[0], rgb[1], rgb[2]),), dtype=np.uint8).reshape(1, 1, 3)
    )[0, 0]
    delta = _labs - target
    idx = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
    return pantone_code_for_hex(_lab_hexes[idx])


def list_pantone_codes() -> list[str]:
    """Display-form catalog codes (coated book + named), JSON order."""
    _ensure_loaded()
    assert _display_rows is not None
    return [shown for _hay, shown in _display_rows]


def filter_pantone_codes(query: str, *, limit: int = 24) -> list[str]:
    """Prefix, then substring, matches on code names. Empty query → ``[]``."""
    needle = " ".join(query.strip().split()).casefold()
    if not needle or limit <= 0:
        return []
    _ensure_loaded()
    assert _display_rows is not None
    prefix: list[str] = []
    substr: list[str] = []
    for haystack, shown in _display_rows:
        folded = shown.casefold()
        if folded.startswith(needle) or haystack.startswith(needle):
            prefix.append(shown)
            if len(prefix) >= limit:
                return prefix[:limit]
        elif needle in folded or needle in haystack:
            substr.append(shown)
    return (prefix + substr)[:limit]
