#!/usr/bin/env python3
"""Freeze one Grafana dashboard per experiment.

Every verl run calls ``ray.init()`` itself, so one experiment == one Ray session == one
distinct ``SessionName`` label. Ray stamps that label on *both* our ``ray_verl_gpu_phase``
and its own ``ray_node_gpus_utilization`` / ``ray_node_gram_*``, so it is the only
discriminator that scopes every panel of the dashboard uniformly.

This clones the live dashboard into a per-run copy whose every PromQL selector is pinned to
one session, and whose time range is pinned to that run's window. The result keeps rendering
long after the run (and Ray itself) is gone -- as far back as Prometheus retention goes.

    # after experiment A finishes
    make_run_dashboard.py --latest --name exp-a

    # what sessions does Prometheus know about?
    make_run_dashboard.py --list

    # an older run, by session id
    make_run_dashboard.py --session session_2026-07-10_11-44-20_471818_3405251 --name exp-a

Grafana's file provider rescans the dashboards dir every ~10s, so a new dashboard shows up
without restarting anything.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_phase_colors  # noqa: E402  (same directory; reused as a library)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "verl_gpu_phase_dashboard.json")
DEFAULT_PROM = os.environ.get("VERL_PROMETHEUS_URL", "http://localhost:9090")
MONITORING_HOME = os.environ.get("VERL_MONITORING_HOME", os.path.expanduser("~/.verl-monitoring"))
DEFAULT_OUT_DIR = os.path.join(MONITORING_HOME, "grafana", "dashboards")

# Metric names are lowercase+underscore; every one we touch starts with the `ray_` prefix Ray
# adds on Prometheus export. Digits allowed to stay ahead of future metric names.
_METRIC_RE = re.compile(r"\bray_[a-z0-9_]+\b")


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------
def _api(url: str, path: str, **params) -> dict:
    q = urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(f"{url}/api/v1/{path}?{q}", timeout=60) as fh:
            payload = json.load(fh)
    except urllib.error.HTTPError as exc:                      # prometheus explains itself in the body
        raise SystemExit(f"prometheus {path} -> HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach prometheus at {url}: {exc.reason}")
    if payload.get("status") != "success":
        raise SystemExit(f"prometheus {path} failed: {payload}")
    return payload["data"]


# Prometheus refuses a range query resolving to more than 11k points.
_MAX_POINTS = 9000


class NoSamples(Exception):
    """A SessionName exists on some metric, but carries no verl phase samples."""


def _bounds(url: str, query: str, start: float, end: float, step: float):
    data = _api(url, "query_range", query=query, start=start, end=end, step=step)
    if not data["result"]:
        return None
    values = data["result"][0]["values"]
    return float(values[0][0]), float(values[-1][0])


def session_window(url: str, session: str, lookback_days: int = 400) -> tuple[float, float]:
    """First and last sample of a session, in epoch seconds.

    Two passes: a coarse sweep over the whole lookback locates the run (the step has to scale
    with the range or Prometheus rejects the query), then a fine sweep around each edge pins it
    to the scrape. This needs no process-exit hook to record when the run stopped.
    """
    query = f'count(ray_verl_gpu_phase{{SessionName="{session}"}})'
    end = time.time()
    start = end - lookback_days * 86400
    coarse = max(60.0, (end - start) / _MAX_POINTS)

    rough = _bounds(url, query, start, end, coarse)
    if rough is None:
        raise NoSamples(
            f"no ray_verl_gpu_phase samples for SessionName={session!r}.\n"
            "  Either the session is outside Prometheus retention, or the label changed name\n"
            f"  in this Ray version. Check: curl -s '{url}/api/v1/label/SessionName/values'"
        )
    t0, t1 = rough
    lo = _bounds(url, query, max(t0 - coarse, start), min(t0 + coarse, end), 15) or (t0, t0)
    hi = _bounds(url, query, max(t1 - coarse, start), min(t1 + coarse, end), 15) or (t1, t1)
    return lo[0], hi[1]


def list_sessions(url: str) -> list[tuple[str, float, float]]:
    names = _api(url, "label/SessionName/values")
    out = []
    for name in names:
        try:
            out.append((name, *session_window(url, name)))
        except NoSamples:
            continue  # a Ray session with node metrics but no verl phases: not an experiment
    return sorted(out, key=lambda r: r[1])


def retention_seconds(url: str) -> float | None:
    try:
        flags = _api(url, "status/flags")
    except Exception:
        return None
    raw = flags.get("storage.tsdb.retention.time", "")
    m = re.fullmatch(r"(\d+)([smhdwy])", raw)
    if not m:
        return None
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}
    return int(m.group(1)) * mult[m.group(2)]


# ---------------------------------------------------------------------------
# PromQL selector injection
# ---------------------------------------------------------------------------
def inject_selector(expr: str, matcher: str) -> str:
    """Add ``matcher`` to every metric selector in ``expr``.

    A naive regex on ``{...}`` would corrupt ``label_replace(x, "gpu", "$1", "GpuIndex", "(.+)")``
    -- its arguments are string literals that happen to look like label syntax. So walk the
    expression tracking whether we are inside a double-quoted string, and only rewrite metric
    names found outside one. `$__interval`, `offset`, `and on (gpu, ip)`, `unless`, `or` and the
    `=~"$node"` template values contain no bare ``ray_`` token and are therefore untouched.

    Adding the same constant matcher to both sides of a vector match is safe: `on(...)` restricts
    matching to the labels it lists, and SessionName is equal on both sides regardless.
    """
    out = []
    i, n = 0, len(expr)
    in_string = False
    while i < n:
        ch = expr[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:      # keep escapes atomic: \" must not close the string
                out.append(expr[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        m = _METRIC_RE.match(expr, i)
        if not m:
            out.append(ch)
            i += 1
            continue

        name = m.group(0)
        out.append(name)
        i = m.end()
        j = i
        while j < n and expr[j].isspace():
            j += 1
        if j < n and expr[j] == "{":
            brace_end = expr.index("}", j)
            if "SessionName=" in expr[j:brace_end]:   # idempotent: already pinned
                continue
            out.append(expr[i:j + 1])                 # whitespace + '{'
            out.append(matcher + ", ")
            i = j + 1
        else:
            out.append("{" + matcher + "}")
    return "".join(out)


def pin_dashboard(dashboard: dict, session: str, name: str, t0: float, t1: float, live: bool = False) -> str:
    """Scope every query to one Ray session. Freeze the time range unless ``live``.

    The uid depends only on (name, session), so the live dashboard and the frozen one are the
    same URL: generate with --live while the run is going, re-generate without it afterwards and
    the same page stops following `now` and settles on the run's window.
    """
    matcher = f'SessionName="{session}"'

    for panel in dashboard.get("panels", []):
        for target in panel.get("targets", []):
            if "expr" in target:
                target["expr"] = inject_selector(target["expr"], matcher)

    # Template variables carry the query twice: `definition` (shown in the UI) and `query.query`.
    for var in dashboard.get("templating", {}).get("list", []):
        if var.get("type") != "query":
            continue
        if isinstance(var.get("definition"), str):
            var["definition"] = inject_selector(var["definition"], matcher)
        query = var.get("query")
        if isinstance(query, dict) and isinstance(query.get("query"), str):
            query["query"] = inject_selector(query["query"], matcher)
        elif isinstance(query, str):
            var["query"] = inject_selector(query, matcher)

    uid = f"vgp-{_slug(name)[:24]}-{hashlib.sha1(session.encode()).hexdigest()[:8]}"
    dashboard["uid"] = uid
    dashboard["title"] = f"verl · {name}" + (" (live)" if live else "")
    if not live:
        dashboard["time"] = {"from": f"{int(t0 * 1000)}", "to": f"{int(t1 * 1000)}"}
        dashboard["refresh"] = ""      # a finished run must not drift toward `now`
    dashboard["version"] = 1
    dashboard.pop("id", None)
    return uid


def _slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-") or "run"


def recolor(dashboard: dict) -> None:
    """Same palette as the live dashboard. Touches fieldConfig only, never `targets`."""
    panel = apply_phase_colors._phase_panel(dashboard)
    apply_phase_colors.apply(panel, apply_phase_colors.load_overrides(None))
    apply_phase_colors._sync_overrides(dashboard, apply_phase_colors.current_colors(panel))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest", action="store_true", help="use the most recently started Ray session")
    g.add_argument("--session", help="pin to this SessionName")
    g.add_argument("--list", action="store_true", help="list sessions Prometheus knows about, and exit")
    ap.add_argument("--name", help="human name for the experiment (used in title + uid)")
    ap.add_argument("--live", action="store_true",
                    help="scope to the session but keep the rolling time range and auto-refresh, "
                         "for watching a run in progress. Re-run without --live to freeze it.")
    ap.add_argument("--src", default=DEFAULT_SRC, help="base dashboard to clone")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="where Grafana's file provider looks")
    ap.add_argument("--prometheus", default=DEFAULT_PROM, help="Prometheus base URL")
    args = ap.parse_args()

    if args.list:
        rows = list_sessions(args.prometheus)
        if not rows:
            print("no sessions with ray_verl_gpu_phase samples in Prometheus")
            return 0
        print(f"  {'SessionName':<46} window")
        for name, t0, t1 in rows:
            fmt = "%Y-%m-%d %H:%M"
            print(f"  {name:<46} {time.strftime(fmt, time.localtime(t0))} -> "
                  f"{time.strftime('%H:%M', time.localtime(t1))}  ({(t1 - t0) / 3600:.1f}h)")
        return 0

    if args.latest:
        rows = list_sessions(args.prometheus)
        if not rows:
            raise SystemExit("no verl sessions found in Prometheus; is the run up and scraped?")
        session, t0, t1 = rows[-1]
    else:
        session = args.session
        try:
            t0, t1 = session_window(args.prometheus, session)
        except NoSamples as exc:
            raise SystemExit(str(exc))

    name = args.name or session
    keep = retention_seconds(args.prometheus)
    if not args.live and keep and t0 < time.time() - keep:
        print(f"WARNING: this run starts before Prometheus retention ({keep / 86400:.0f}d); "
              "the dashboard will be partly or wholly blank.", file=sys.stderr)

    with open(args.src) as fh:
        dashboard = json.load(fh)
    uid = pin_dashboard(dashboard, session, name, t0 - 60, t1 + 60, live=args.live)
    recolor(dashboard)

    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, f"{uid}.json")
    with open(dst, "w") as fh:
        json.dump(dashboard, fh, indent=2)
        fh.write("\n")

    if args.live:
        window = f"live ({dashboard['time']['from']} -> {dashboard['time']['to']}, refresh {dashboard['refresh']})"
    else:
        window = ("pinned " + time.strftime("%H:%M", time.localtime(t0))
                  + "->" + time.strftime("%H:%M", time.localtime(t1)))
    print(f"-> /d/{uid}   {dashboard['title']!r}   {window}   ({dst})")
    print("   Grafana rescans its dashboards dir every ~10s.")
    if args.live:
        print("   Re-run without --live once the experiment finishes to freeze it (same URL).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
