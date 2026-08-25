# -*- coding: utf-8 -*-
"""
wallpaper_recolor.color.presets
-------------------------
Named jobs stored in presets.json next to run_app.py. First launch seeds
V6-N plus White/Black, White Gray Black, and Analogous Reds/Greens/Blues.
Every seeded palette is a normal preset: deletable and overwritable.
Generic is the UI no-preset choice (not stored): N k-means colors from
the image.

Class references (code + name only):
- CAP3321C Data Wrangling
- CAP4631C Machine Learning
"""

from __future__ import annotations  # tuple hints on 3.9-style runtimes

from dataclasses import dataclass
from pathlib import Path
import json
import re

from wallpaper_recolor.paths import user_data_dir
from wallpaper_recolor.color.color_ranges import (
    RANGE_BY_COLOR_LABEL,
    RANGE_BY_LAB_A_LABEL,
    RANGE_BY_LAB_B_LABEL,
    RANGE_BY_LAB_C_LABEL,
    RANGE_BY_LUMA_LABEL,
    SPLIT_COLOR_CLOSENESS,
    SPLIT_COLOR_CLOSENESS_LABEL,
    SPLIT_EQUAL_LIGHTNESS_LABEL,
    SPLIT_EQUAL_PIXELS_LABEL,
    ColorRangeMap,
    apply_weights,
    canonicalize_split_method,
    is_color_split,
    is_pixel_bin_split,
    split_axis_channel,
    sync_centers_from_match,
)

# Screen RGB from V6-N/Greens.txt — not a print guarantee (use ICC proof)
V6N_DARK = (0x5E, 0x7F, 0x57)  # Garden Grove — palette target (nearest Lab)
V6N_MID = (0x5F, 0x88, 0x7D)  # Surf Green
V6N_LIGHT = (0x81, 0xA5, 0x95)  # Verdigreen

V6N_ID = "V6-N"
V6N_RANGE_COUNT = 3
# Recolor-to-palette: each pixel → nearest of these three hexes in Lab
# (color closeness), not histogram luma thirds.
V6N_SPLIT = SPLIT_COLOR_CLOSENESS
V6N_PALETTE_RGB: tuple[tuple[int, int, int], ...] = (V6N_DARK, V6N_MID, V6N_LIGHT)

# (index 0 = Garden Grove, 1 = Surf Green, 2 = Verdigreen) — palette order
V6N_BANDS: tuple[tuple[str, tuple[int, int, int], str], ...] = (
    ("Dark / Garden Grove", V6N_DARK, "03_dark_garden_grove"),
    ("Mid / Surf Green", V6N_MID, "02_midtone_surf_green"),
    ("Light / Verdigreen", V6N_LIGHT, "01_light_verdigreen"),
)

PRESETS_FILENAME = "presets.json"
# UI-only no-preset row — not written to JSON
GENERIC_LABEL = "Generic"
# Offered before seeded_ids existed — do not re-add if the user deleted them
LEGACY_SEED_IDS = (V6N_ID,)

# Named seed palettes (match-from = change-to). Screen RGB, not a print guarantee.
WHITE = (0xFF, 0xFF, 0xFF)  # #FFFFFF
GRAY = (0x80, 0x80, 0x80)  # #808080
BLACK = (0x00, 0x00, 0x00)  # #000000
RED_DARK = (0xC6, 0x28, 0x28)  # #C62828
RED_MID = (0xE5, 0x39, 0x35)  # #E53935
RED_LIGHT = (0xEF, 0x9A, 0x9A)  # #EF9A9A
GREEN_DARK = (0x1B, 0x5E, 0x20)  # #1B5E20
GREEN_MID = (0x43, 0xA0, 0x47)  # #43A047
GREEN_LIGHT = (0xA5, 0xD6, 0xA7)  # #A5D6A7
BLUE_DARK = (0x0D, 0x47, 0xA5)  # #0D47A5
BLUE_MID = (0x1E, 0x88, 0xE5)  # #1E88E5
BLUE_LIGHT = (0x90, 0xCA, 0xF9)  # #90CAF9


@dataclass(frozen=True)
class PaletteBand:
    """One named range: change-to RGB, optional match-from (closeness target)."""

    name: str
    rgb: tuple[int, int, int]  # change-to / replacement
    match_rgb: tuple[int, int, int] | None = None  # None → same as rgb when used as centers


@dataclass(frozen=True)
class PalettePreset:
    """A saved palette: band count, split, colors, names, optional weights."""

    id: str
    name: str
    range_count: int
    split_method: str
    bands: tuple[PaletteBand, ...]
    weights: tuple[float, ...] | None = None
    builtin: bool = False  # unused for lock — every JSON preset is deletable
    # True: pixels map to nearest band match RGB in Lab (V6-N), not image k-means
    palette_as_centers: bool = False

    @property
    def split_label(self) -> str:
        return split_label_for(self.split_method)

    @property
    def palette_rgb(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(band.rgb for band in self.bands)

    @property
    def match_palette_rgb(self) -> tuple[tuple[int, int, int], ...]:
        """Lab closeness targets: match-from, or change-to when match was not saved."""
        return tuple(
            (band.match_rgb if band.match_rgb is not None else band.rgb) for band in self.bands
        )


def range_by_label_for(method: str) -> str:
    """Primary Range by: dropdown from a ColorRangeMap split_method."""
    method = canonicalize_split_method(method)
    if is_color_split(method):
        return RANGE_BY_COLOR_LABEL
    ch = split_axis_channel(method)
    if ch == 1:
        return RANGE_BY_LAB_A_LABEL
    if ch == 2:
        return RANGE_BY_LAB_B_LABEL
    if ch == 3:
        return RANGE_BY_LAB_C_LABEL
    return RANGE_BY_LUMA_LABEL


def split_label_for(method: str) -> str:
    """Equal-steps vs even-pixel sub-option (or Color closeness)."""
    if method == SPLIT_COLOR_CLOSENESS:
        return SPLIT_COLOR_CLOSENESS_LABEL
    if is_pixel_bin_split(method):
        return SPLIT_EQUAL_PIXELS_LABEL
    return SPLIT_EQUAL_LIGHTNESS_LABEL


def default_presets_path() -> Path:
    """presets.json beside run_app.py, or beside the .exe when frozen."""
    return user_data_dir() / PRESETS_FILENAME


def v6n_preset() -> PalettePreset:
    """Default V6-N three-green palette — first-run seed; the user may delete it."""
    return PalettePreset(
        id=V6N_ID,
        name=V6N_ID,
        range_count=V6N_RANGE_COUNT,
        split_method=V6N_SPLIT,
        bands=tuple(PaletteBand(name, rgb, match_rgb=rgb) for name, rgb, _slug in V6N_BANDS),
        weights=_even_weights(V6N_RANGE_COUNT),
        builtin=False,
        palette_as_centers=True,
    )


def _even_weights(n: int) -> tuple[float, ...]:
    n = max(1, int(n))
    return tuple(1.0 / n for _ in range(n))


def _seed_palette(
    preset_id: str,
    name: str,
    bands: tuple[tuple[str, tuple[int, int, int]], ...],
) -> PalettePreset:
    """Color-closeness seed: match-from and change-to start as the same hexes."""
    n = len(bands)
    return PalettePreset(
        id=preset_id,
        name=name,
        range_count=n,
        split_method=SPLIT_COLOR_CLOSENESS,
        bands=tuple(PaletteBand(label, rgb, match_rgb=rgb) for label, rgb in bands),
        weights=_even_weights(n),
        builtin=False,
        palette_as_centers=True,
    )


def white_black_preset() -> PalettePreset:
    """2-split white / black."""
    return _seed_palette(
        "white-and-black",
        "White and Black",
        (("White", WHITE), ("Black", BLACK)),
    )


def white_gray_black_preset() -> PalettePreset:
    """3-split white / gray / black."""
    return _seed_palette(
        "white-gray-black",
        "White Gray Black",
        (("White", WHITE), ("Gray", GRAY), ("Black", BLACK)),
    )


def analogous_reds_preset() -> PalettePreset:
    """3 analogous reds (crimson / red / light red)."""
    return _seed_palette(
        "analogous-reds",
        "Analogous Reds",
        (("Dark red", RED_DARK), ("Red", RED_MID), ("Light red", RED_LIGHT)),
    )


def analogous_greens_preset() -> PalettePreset:
    """3 analogous greens (not V6-N)."""
    return _seed_palette(
        "analogous-greens",
        "Analogous Greens",
        (("Dark green", GREEN_DARK), ("Green", GREEN_MID), ("Light green", GREEN_LIGHT)),
    )


def analogous_blues_preset() -> PalettePreset:
    """3 analogous blues."""
    return _seed_palette(
        "analogous-blues",
        "Analogous Blues",
        (("Dark blue", BLUE_DARK), ("Blue", BLUE_MID), ("Light blue", BLUE_LIGHT)),
    )


def default_seed_presets() -> list[PalettePreset]:
    """First-run named palettes (Generic is UI-only and is not in this list)."""
    return [
        v6n_preset(),
        white_black_preset(),
        white_gray_black_preset(),
        analogous_reds_preset(),
        analogous_greens_preset(),
        analogous_blues_preset(),
    ]


def is_generic_label(name: str) -> bool:
    """True for the UI no-preset row (Generic / (None))."""
    raw = name.strip().lower()
    return raw in {GENERIC_LABEL.lower(), "(none)", "none", ""}


def apply_preset_palette(
    range_map: ColorRangeMap,
    preset: PalettePreset,
    *,
    snap_centers: bool | None = None,
) -> bool:
    """Paint preset names + change-to RGBs onto a matching map.

    ``snap_centers`` True (Snap to palette): also write match-from and rebuild
    Lab assignment from those hexes. False (Cluster from image): keep the
    current k-means match-from / centers; only names and replacement colors
    change. None follows ``preset.palette_as_centers``.
    """
    if len(range_map.ranges) != len(preset.bands):
        return False
    use_centers = bool(preset.palette_as_centers) if snap_centers is None else bool(snap_centers)
    for band, spec in zip(range_map.ranges, preset.bands):
        band.name = spec.name
        band.replacement_rgb = spec.rgb
        match = spec.match_rgb if spec.match_rgb is not None else spec.rgb
        if use_centers:
            band.match_rgb = match
    if use_centers and is_color_split(range_map.split_method):
        sync_centers_from_match(range_map)
        if range_map.rgb is not None:
            apply_weights(range_map, range_map.weights())
    return True


def apply_v6n_palette(range_map: ColorRangeMap) -> bool:
    """Paint the V6-N greens onto a 3-band map. False if band count is not 3."""
    return apply_preset_palette(range_map, v6n_preset())


def snapshot_preset(name: str, range_map: ColorRangeMap, *, preset_id: str | None = None) -> PalettePreset:
    """Capture the current ranges as a user preset (match, replace, names, split, weights)."""
    label = name.strip()
    if not label:
        raise ValueError("Preset name is empty")
    if is_generic_label(label):
        raise ValueError("Generic is the no-preset default — pick another name")
    bands = tuple(
        PaletteBand(
            band.name,
            (int(band.replacement_rgb[0]), int(band.replacement_rgb[1]), int(band.replacement_rgb[2])),
            match_rgb=(int(band.match_rgb[0]), int(band.match_rgb[1]), int(band.match_rgb[2])),
        )
        for band in range_map.ranges
    )
    weights = tuple(float(band.weight) for band in range_map.ranges)
    slug = preset_id or _slug_id(label)
    return PalettePreset(
        id=slug,
        name=label,
        range_count=len(bands),
        split_method=range_map.split_method or SPLIT_COLOR_CLOSENESS,
        bands=bands,
        weights=weights if weights else None,
        builtin=False,
        palette_as_centers=is_color_split(range_map.split_method),
    )


def ensure_default_presets(path: Path | None = None) -> None:
    """Seed missing first-run palettes. Do not restore a seed the user already deleted."""
    dest = path or default_presets_path()
    users = _load_user_presets(dest)
    seeded = set(_read_seeded_ids(dest))
    if dest.is_file() and not seeded:
        seeded.update(LEGACY_SEED_IDS)
        seeded.update(item.id for item in users)
    names = {item.name.lower() for item in users}
    ids = {item.id.lower() for item in users}
    changed = not dest.is_file()
    for seed in default_seed_presets():
        already = seed.id in seeded or seed.id.lower() in ids or seed.name.lower() in names
        if already:
            seeded.add(seed.id)
            continue
        users.append(seed)
        seeded.add(seed.id)
        names.add(seed.name.lower())
        ids.add(seed.id.lower())
        changed = True
    if changed or set(_read_seeded_ids(dest)) != seeded:
        _write_user_presets(users, dest, seeded_ids=list(seeded))


def list_presets(path: Path | None = None) -> list[PalettePreset]:
    """Named palettes from presets.json (empty list if none). Generic is UI-only."""
    return [
        preset
        for preset in _load_user_presets(path)
        if not is_generic_label(preset.name)
    ]


def get_preset(key: str, path: Path | None = None) -> PalettePreset | None:
    """Look up by id or display name (case-insensitive). Generic → None."""
    needle = key.strip()
    if not needle or is_generic_label(needle):
        return None
    presets = list_presets(path)
    for preset in presets:
        if preset.id == needle or preset.name == needle:
            return preset
    lower = needle.lower()
    for preset in presets:
        if preset.id.lower() == lower or preset.name.lower() == lower:
            return preset
    return None


def save_user_preset(preset: PalettePreset, path: Path | None = None) -> PalettePreset:
    """Append or replace a preset by name (including V6-N). Caller confirms overwrite."""
    if is_generic_label(preset.name):
        raise ValueError("Generic is the no-preset default — pick another name")
    dest = path or default_presets_path()
    users = _load_user_presets(dest)
    existing_ids = {item.id for item in users}
    replaced = False
    for i, item in enumerate(users):
        same = item.name == preset.name or item.id == preset.id
        if not same:
            same = item.name.lower() == preset.name.lower() or item.id.lower() == preset.id.lower()
        if same:
            users[i] = PalettePreset(
                id=item.id,
                name=preset.name,
                range_count=preset.range_count,
                split_method=preset.split_method,
                bands=preset.bands,
                weights=preset.weights,
                builtin=False,
                palette_as_centers=preset.palette_as_centers,
            )
            replaced = True
            break
    if not replaced:
        slug = preset.id if preset.id not in existing_ids else _unique_id(preset.name, existing_ids)
        users.append(
            PalettePreset(
                id=slug,
                name=preset.name,
                range_count=preset.range_count,
                split_method=preset.split_method,
                bands=preset.bands,
                weights=preset.weights,
                builtin=False,
                palette_as_centers=preset.palette_as_centers,
            )
        )
    _write_user_presets(users, dest)
    return get_preset(preset.name, dest) or users[-1]


def delete_user_preset(key: str, path: Path | None = None) -> bool:
    """Remove a preset by id or name, including V6-N. Returns False if missing."""
    if is_generic_label(key):
        return False
    dest = path or default_presets_path()
    users = _load_user_presets(dest)
    lower = key.strip().lower()
    kept = [
        item
        for item in users
        if item.id != key and item.name != key and item.id.lower() != lower and item.name.lower() != lower
    ]
    if len(kept) == len(users):
        return False
    _write_user_presets(kept, dest)
    return True


def _is_v6n_name(value: str) -> bool:
    return value.strip().lower() in {V6N_ID.lower(), "v6n", "v6-n"}


def _slug_id(name: str) -> str:
    if _is_v6n_name(name):
        return V6N_ID
    raw = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return raw or "preset"


def _unique_id(name: str, existing: set[str]) -> str:
    base = _slug_id(name)
    candidate = base
    n = 2
    while candidate in existing:
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def _clamp_rgb(values: object) -> tuple[int, int, int]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("rgb must be three integers")
    rgb = tuple(max(0, min(255, int(c))) for c in values)
    return rgb[0], rgb[1], rgb[2]


def _preset_from_json(raw: object) -> PalettePreset | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name or is_generic_label(name):
        return None
    bands_raw = raw.get("bands")
    if not isinstance(bands_raw, list) or len(bands_raw) < 2:
        return None
    bands: list[PaletteBand] = []
    for item in bands_raw:
        if not isinstance(item, dict):
            return None
        try:
            rgb = _clamp_rgb(item.get("rgb"))
        except (TypeError, ValueError):
            return None
        match_rgb = None
        if item.get("match_rgb") is not None:
            try:
                match_rgb = _clamp_rgb(item.get("match_rgb"))
            except (TypeError, ValueError):
                match_rgb = None
        bands.append(PaletteBand(str(item.get("name") or ""), rgb, match_rgb=match_rgb))
    split = canonicalize_split_method(raw.get("split_method") or SPLIT_COLOR_CLOSENESS)
    weights = None
    weights_raw = raw.get("weights")
    if isinstance(weights_raw, list) and len(weights_raw) == len(bands):
        try:
            weights = tuple(float(w) for w in weights_raw)
        except (TypeError, ValueError):
            weights = None
    slug = str(raw.get("id") or "").strip() or _slug_id(name)
    return PalettePreset(
        id=slug,
        name=name,
        range_count=len(bands),
        split_method=split,
        bands=tuple(bands),
        weights=weights,
        builtin=False,
        palette_as_centers=bool(raw.get("palette_as_centers")),
    )


def _load_user_presets(path: Path | None) -> list[PalettePreset]:
    dest = Path(path) if path is not None else default_presets_path()
    if not dest.is_file():
        return []
    try:
        payload = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    rows = payload.get("presets") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    out: list[PalettePreset] = []
    seen: set[str] = set()
    for row in rows:
        preset = _preset_from_json(row)
        if preset is None or preset.id in seen:
            continue
        seen.add(preset.id)
        out.append(preset)
    return out


def _read_seeded_ids(path: Path) -> list[str]:
    """Ids of seed palettes already offered (so delete is not undone on next launch)."""
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    raw = payload.get("seeded_ids")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _write_user_presets(
    presets: list[PalettePreset],
    path: Path,
    *,
    seeded_ids: list[str] | None = None,
) -> None:
    offered = list(seeded_ids) if seeded_ids is not None else _read_seeded_ids(path)
    payload = {
        "seeded_ids": offered,
        "presets": [
            {
                "id": item.id,
                "name": item.name,
                "range_count": item.range_count,
                "split_method": item.split_method,
                "bands": [
                    {
                        "name": band.name,
                        "rgb": list(band.rgb),
                        **(
                            {"match_rgb": list(band.match_rgb)}
                            if band.match_rgb is not None
                            else {}
                        ),
                    }
                    for band in item.bands
                ],
                **({"weights": list(item.weights)} if item.weights is not None else {}),
                **({"palette_as_centers": True} if item.palette_as_centers else {}),
            }
            for item in presets
            if not item.builtin
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
