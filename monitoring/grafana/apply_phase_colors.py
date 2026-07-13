#!/usr/bin/env python3
"""Rewrite the phase-lane colors of the verl GPU-phase dashboard.

The dashboard ships a default palette inside its value mappings, so it is usable as-is.
This script produces a recolored copy, letting a run override any phase color without
touching the checked-in JSON. `start_monitoring.sh` calls it when provisioning.

Precedence (later wins):
    dashboard defaults  <  palette file (--palette / VERL_PHASE_COLORS_FILE)  <  VERL_PHASE_COLORS

A palette is a phase -> color map; phases are named (``gen``) or numeric codes (``1``), and
``fallback`` sets the color of values with no mapping. Colors are hex (``#3987e5``),
``rgb()/rgba()``, or Grafana palette names (``semi-dark-blue``).

    # one-off tweak
    VERL_PHASE_COLORS='gen=#e0b400,sleep=#2a2a2a' bash monitoring/start_monitoring.sh

    # persistent palette
    cat > monitoring/grafana/phase_colors.json <<'EOF'
    { "gen": "#e0b400", "update_actor": "purple", "fallback": "#333333" }
    EOF

    # inspect what a dashboard currently uses
    python3 monitoring/grafana/apply_phase_colors.py --print
"""

import argparse
import json
import os
import re
import sys

DEFAULT_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verl_gpu_phase_dashboard.json")
DEFAULT_PALETTE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_colors.json")
PANEL_TYPE = "state-timeline"

# Hex, rgb()/rgba(), or a Grafana named color ("red", "semi-dark-blue", "super-light-green").
_COLOR_RE = re.compile(r"^(#[0-9a-fA-F]{3,8}|rgba?\([\d.,\s%]+\)|[a-z]+(-[a-z]+)*)$")


def _phase_panel(dashboard: dict) -> dict:
    for panel in dashboard.get("panels", []):
        if panel.get("type") == PANEL_TYPE:
            return panel
    raise SystemExit(f"no {PANEL_TYPE} panel found -- is this the verl GPU-phase dashboard?")


def _mapping_entries(panel: dict):
    """Yield (code, name, options_dict) for every value mapping of the phase panel."""
    for mapping in panel["fieldConfig"]["defaults"].get("mappings", []):
        if mapping.get("type") != "value":
            continue
        for code, opts in mapping.get("options", {}).items():
            yield code, opts.get("text", code), opts


def current_colors(panel: dict) -> dict:
    colors = {name: opts.get("color", "") for _, name, opts in _mapping_entries(panel)}
    colors["fallback"] = panel["fieldConfig"]["defaults"].get("color", {}).get("fixedColor", "")
    return colors


def parse_env_colors(raw: str) -> dict:
    """Parse ``VERL_PHASE_COLORS`` -- ``gen=#e0b400,sleep=#2a2a2a``."""
    overrides = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"VERL_PHASE_COLORS: expected phase=color, got {item!r}")
        phase, color = item.split("=", 1)
        overrides[phase.strip()] = color.strip()
    return overrides


def load_overrides(palette_path: str | None) -> dict:
    """Palette file (if any), then the env var on top."""
    overrides = {}
    path = palette_path or os.environ.get("VERL_PHASE_COLORS_FILE") or DEFAULT_PALETTE
    if os.path.exists(path):
        with open(path) as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise SystemExit(f"{path}: expected a JSON object of phase -> color")
        overrides.update({k: str(v) for k, v in loaded.items()})
    elif palette_path:  # explicitly requested but missing: that's an error, not a default
        raise SystemExit(f"palette file not found: {palette_path}")

    if env := os.environ.get("VERL_PHASE_COLORS"):
        overrides.update(parse_env_colors(env))
    return overrides


def _sync_overrides(dashboard: dict, colors: dict) -> None:
    """Push the phase colors into every panel that colors series by phase name.

    The utilization-by-phase panel names its series after the phase (one query per phase) and
    colors them with `byName` field overrides. Driving them from the timeline's value mappings
    keeps one palette across the dashboard instead of two that silently drift apart.
    """
    for panel in dashboard.get("panels", []):
        for override in panel.get("fieldConfig", {}).get("overrides", []):
            matcher = override.get("matcher", {})
            if matcher.get("id") != "byName":
                continue
            color = colors.get(matcher.get("options"))
            if color is None:
                continue
            for prop in override.get("properties", []):
                if prop.get("id") == "color":
                    prop["value"] = {"mode": "fixed", "fixedColor": color}
        fallback = colors.get("fallback")
        if fallback and panel.get("fieldConfig", {}).get("defaults", {}).get("color", {}).get("mode") == "fixed":
            panel["fieldConfig"]["defaults"]["color"]["fixedColor"] = fallback


def apply(panel: dict, overrides: dict) -> list[str]:
    """Apply overrides to the timeline's value mappings; return a log of what changed."""
    known = {}
    for code, name, opts in _mapping_entries(panel):
        known[name] = opts
        known[code] = opts

    changed = []
    for phase, color in overrides.items():
        if not _COLOR_RE.match(color):
            raise SystemExit(f"{phase}: {color!r} is not a hex/rgb()/Grafana-named color")
        if phase == "fallback":
            panel["fieldConfig"]["defaults"].setdefault("color", {})["fixedColor"] = color
            changed.append(f"fallback -> {color}")
            continue
        if phase not in known:
            names = sorted({n for _, n, _ in _mapping_entries(panel)})
            raise SystemExit(f"unknown phase {phase!r}; expected one of {', '.join(names)}, fallback")
        opts = known[phase]
        if opts.get("color") != color:
            changed.append(f"{opts.get('text', phase)} -> {color}")
        opts["color"] = color
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default=DEFAULT_SRC, help="dashboard JSON to read")
    ap.add_argument("dst", nargs="?", help="where to write the recolored copy (default: stdout)")
    ap.add_argument("--palette", help="JSON file of phase -> color (default: phase_colors.json if present)")
    ap.add_argument("--print", dest="show", action="store_true", help="print the effective palette and exit")
    args = ap.parse_args()

    with open(args.src) as f:
        dashboard = json.load(f)
    panel = _phase_panel(dashboard)

    overrides = load_overrides(args.palette)
    changed = apply(panel, overrides)
    # Timeline mappings are the source of truth; mirror them onto the by-phase line colors.
    _sync_overrides(dashboard, current_colors(panel))

    if args.show:
        for phase, color in current_colors(panel).items():
            print(f"{phase:<14} {color}")
        return 0

    out = json.dumps(dashboard, indent=2) + "\n"
    if args.dst:
        with open(args.dst, "w") as f:
            f.write(out)
        if changed:
            print(f"phase colors overridden: {'; '.join(changed)}", file=sys.stderr)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
