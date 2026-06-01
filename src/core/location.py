"""Temporal and spatial awareness for the chatbot.

Provides two pieces of context that are injected into the LLM prompt:

1. **Date / time** — always available from the system clock.
2. **Geographic location** — obtained via a free geo-IP lookup
   (ip-api.com) when the internet is reachable.  Falls back
   to the system timezone name when offline.

Internet detection
-------------------
A quick TCP connection to ``1.1.1.1:80`` (Cloudflare) is attempted
with a 3-second timeout.  The result is cached so subsequent calls
are instant.  Use ``force_refresh=True`` to re-check.

Usage
-----
    from src.core.location import get_context_string

    ctx = get_context_string()
    # "It is 2026-06-02 00:39 (CEST) in Europe/Madrid.  You are
    #  in Madrid, Community of Madrid, Spain."
"""

import socket
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------
_internet_available: bool | None = None
_last_internet_check: float = 0.0
_CACHE_TTL = 60.0  # seconds before re-checking internet

_geo_cache: dict | None = None
_GEO_CACHE_TTL = 3600.0  # 1 hour before re-fetching location
_last_geo_fetch: float = 0.0


# ---------------------------------------------------------------------------
# Internet reachability (lightweight TCP connect)
# ---------------------------------------------------------------------------

def _check_internet(host="1.1.1.1", port=80, timeout=3.0) -> bool:
    """Return ``True`` if a TCP connection to *host:port* succeeds.

    Uses ``1.1.1.1`` (Cloudflare DNS) because it is almost never
    blocked and avoids the overhead of a full HTTP request.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def has_internet(force_refresh=False) -> bool:
    """Return cached internet status, or re-check if stale / forced.

    The result is cached for ``_CACHE_TTL`` seconds to avoid
    hammering the network on every pipeline invocation.
    """
    global _internet_available, _last_internet_check
    now = time.time()
    if (
        not force_refresh
        and _internet_available is not None
        and (now - _last_internet_check) < _CACHE_TTL
    ):
        return _internet_available

    _internet_available = _check_internet()
    _last_internet_check = now
    return _internet_available


# ---------------------------------------------------------------------------
# Geographic location
# ---------------------------------------------------------------------------

def _get_local_timezone() -> str:
    """Read the IANA timezone name from ``/etc/timezone`` (Linux) or
    the ``/etc/localtime`` symlink.

    Falls back to ``time.tzname`` if neither is available.
    """
    etc_tz = Path("/etc/timezone")
    if etc_tz.exists():
        return etc_tz.read_text().strip()

    etc_link = Path("/etc/localtime")
    if etc_link.exists():
        try:
            resolved = etc_link.resolve()
            parts = resolved.parts
            # /usr/share/zoneinfo/Europe/Madrid → Europe/Madrid
            if "zoneinfo" in parts:
                idx = parts.index("zoneinfo")
                return "/".join(parts[idx + 1 :])
        except (OSError, ValueError):
            pass

    import time as _time
    return _time.tzname[0] or "Unknown"


def _fetch_location_online(timeout=5.0) -> dict | None:
    """Query ip-api.com for a free geo-IP location (no API key needed).

    Returns a dict with keys ``city``, ``region``, ``country``,
    ``countryCode``, ``timezone``, ``lat``, ``lon``, or ``None``
    if the request fails.
    """
    try:
        import requests
        resp = requests.get("http://ip-api.com/json/", timeout=timeout)
        data = resp.json()
        if data.get("status") == "success":
            return {
                "city": data.get("city", ""),
                "region": data.get("regionName", ""),
                "country": data.get("country", ""),
                "country_code": data.get("countryCode", ""),
                "timezone": data.get("timezone", ""),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
            }
    except Exception:
        pass
    return None


def get_location(force_refresh=False) -> dict:
    """Return a dict with location info, cached for ``_GEO_CACHE_TTL``.

    Online path
        Queries ip-api.com and returns structured city / region /
        country / timezone data.

    Offline path
        Returns only the system timezone name (read from
        ``/etc/timezone`` or ``/etc/localtime``).

    Returns
    -------
    dict
        Keys: ``"city"``, ``"region"``, ``"country"``,
        ``"country_code"``, ``"timezone"``, ``"source"``
        (``"geo-ip"`` or ``"system-clock"``).
    """
    global _geo_cache, _last_geo_fetch
    now = time.time()

    if (
        not force_refresh
        and _geo_cache is not None
        and (now - _last_geo_fetch) < _GEO_CACHE_TTL
    ):
        return _geo_cache

    if has_internet(force_refresh=force_refresh):
        remote = _fetch_location_online()
        if remote:
            remote["source"] = "geo-ip"
            _geo_cache = remote
            _last_geo_fetch = now
            return remote

    # Fallback: timezone-only, no geographic detail.
    tz = _get_local_timezone()
    _geo_cache = {
        "city": "",
        "region": "",
        "country": "",
        "country_code": "",
        "timezone": tz,
        "source": "system-clock",
    }
    _last_geo_fetch = now
    return _geo_cache


# ---------------------------------------------------------------------------
# Public convenience — build a human-readable context string
# ---------------------------------------------------------------------------

def get_context_string(lang="en") -> str:
    """Return a short, human-readable string describing the current
    time and the user's best-known location.

    Examples
    --------
    - Online:  *"It is 2026-06-02 00:39 (CEST) in Europe/Madrid.
                 You are in Madrid, Community of Madrid, Spain."*
    - Offline: *"It is 2026-06-02 00:39 (CEST)."*
    """
    loc = get_location()

    # ── date / time ──────────────────────────────────────────
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    tz_abbr = _tz_abbreviation()
    time_part = f"It is {date_str} {time_str} ({tz_abbr})"

    # ── timezone name (IANA) ─────────────────────────────────
    if loc.get("timezone"):
        time_part += f" in {loc['timezone']}"

    # ── location (online only) ───────────────────────────────
    parts = [time_part + "."]
    city = loc.get("city", "")
    region = loc.get("region", "")
    country = loc.get("country", "")

    if city and country:
        parts.append(f"You are in {city}, {region}, {country}." if region
                      else f"You are in {city}, {country}.")

    return "  ".join(parts)


def _tz_abbreviation() -> str:
    """Return the local timezone abbreviation (e.g. ``CEST``, ``EST``)."""
    import time as _time
    return _time.tzname[_time.daylight if _time.daylight else 0]


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Internet:", has_internet(force_refresh=True))
    print("Location:", get_location(force_refresh=True))
    print("Context:", get_context_string())
