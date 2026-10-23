"""Fetch real Bureau of Meteorology daily weather observations for Moorabbin
Airport (station 086077, the nearest official BoM station to Monash
Clayton) -- external validation data for the environmental sensor log, per
Keenan Granland's third suggestion on the EdStem forum (thread #81, "Confirming scope").

Important, honestly-scoped limitation: BoM's bulk historical-download
endpoint (Climate Data Online) actively blocks automated requests as
scraping (confirmed by testing against it -- it returns an explicit
"potential automated access" refusal). Their public monthly "Daily Weather
Observations" pages, however, are freely fetchable, but only for a rolling
window of roughly the last 15 months; anything older sits behind BoM's
paid/registered-access tier and is not fetched here. This means this
script can only ever cover recent months, not the full multi-year span of
the occupancy/environmental sensor data -- see doc/report/report.md for
which sensor this actually gives usable overlap with.

Usage:
    python tools/fetch_bom_weather.py
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

STATION_PRODUCT = "IDCJDW3052"  # Moorabbin Airport, station 086077
OUT_DIR = Path("data/raw/bom")

#: Months to attempt, as "YYYYMM" strings. BoM's free archive only holds a
#: rolling ~15 months, so months outside that window will 404 -- this is
#: expected and handled, not an error in this script.
MONTHS = [f"2025{m:02d}" for m in range(7, 13)] + [f"2026{m:02d}" for m in range(1, 10)]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_month(month: str) -> bool:
    """Download one month's CSV. Returns True if it existed and was saved."""
    url = f"https://www.bom.gov.au/climate/dwo/{month}/text/{STATION_PRODUCT}.{month}.csv"
    dest = OUT_DIR / f"{STATION_PRODUCT}.{month}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False  # outside BoM's free rolling window -- expected
        raise
    dest.write_bytes(data)
    return True


def main() -> None:
    """Download each month's BoM observations that are still available, and report any missing."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved, missing = [], []
    for month in MONTHS:
        if fetch_month(month):
            saved.append(month)
            print(f"  saved  {month}")
        else:
            missing.append(month)
            print(f"  404    {month}  (outside BoM's free rolling window)")
        time.sleep(0.5)  # be a polite, low-rate client

    print(f"\nSaved {len(saved)} month(s) to {OUT_DIR}/")
    if missing:
        print(f"Not available (outside free window): {missing}")


if __name__ == "__main__":
    main()
