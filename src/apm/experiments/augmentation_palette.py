"""Shared augmentation color palette for consistent experiment visualizations."""

from __future__ import annotations

from typing import Sequence


AUGMENTATION_COLORS: dict[str, str] = {
    "fewshot": "#E15759",
    "fix_ai_artifact": "#4E79A7",
    "back_trans_5langs": "#59A14F",
    "back_trans_3langs": "#F28E2B",
    "back_trans_pol_eng": "#76B7B2",
    "hasty": "#B07AA1",
    "baseline": "#9D9D9D",
}

_FALLBACK_COLORS: tuple[str, ...] = (
    "#EDC948",
    "#FF9DA7",
    "#BAB0AC",
    "#86BCB6",
    "#D37295",
    "#B6992D",
)

_AUGMENTATION_ORDER: tuple[str, ...] = (
    "fewshot",
    "fix_ai_artifact",
    "back_trans_5langs",
    "back_trans_3langs",
    "back_trans_pol_eng",
    "hasty",
)


def ordered_augmentations(augmentations: Sequence[str]) -> list[str]:
    """Return augmentations in fixed project order, then unknown names alphabetically."""

    unique = sorted(set(str(value) for value in augmentations))
    known = [name for name in _AUGMENTATION_ORDER if name in unique]
    unknown = sorted(name for name in unique if name not in set(_AUGMENTATION_ORDER))
    return known + unknown


def augmentation_color(augmentation: str) -> str:
    """Return the stable color for one augmentation name."""

    normalized = str(augmentation)
    if normalized in AUGMENTATION_COLORS:
        return AUGMENTATION_COLORS[normalized]
    return _FALLBACK_COLORS[sum(ord(char) for char in normalized) % len(_FALLBACK_COLORS)]


def augmentation_palette(augmentations: Sequence[str]) -> dict[str, str]:
    """Build palette mapping for plotting libraries (seaborn/matplotlib)."""

    return {name: augmentation_color(name) for name in ordered_augmentations(augmentations)}
