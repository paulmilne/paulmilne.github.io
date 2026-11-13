#!/usr/bin/env python3
"""Download the public data behind the Economics series on wealth concentration.

Run once from the top of the blog repository:

    python3 src/economics/fetch_data.py            # everything except the large WID file
    python3 src/economics/fetch_data.py --wid      # also the World Inequality Database (large download)

Uses only the Python standard library. Files go to src/economics/data/, and
src/economics/data/manifest.json records what was fetched, from where and when,
and anything that failed. Nothing here needs an account or an API key.
"""
import csv, io, json, os, sys, time, zipfile, urllib.request, urllib.error
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
UA = {"User-Agent": "Mozilla/5.0 (research script; paulmilne.github.io)"}
manifest = {"run": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ok": [], "failed": []}

def get(url, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def save(name, url, check=None, binary=False):
    """Fetch url into data/name. check(bytes) -> bool guards against error pages."""
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        b = get(url)
        if check and not check(b):
            raise ValueError("unexpected content (%d bytes): %r" % (len(b), b[:80]))
        with open(path, "wb") as f:
            f.write(b)
        manifest["ok"].append({"file": name, "url": url, "bytes": len(b)})
        print("  ok     %-46s %9d bytes" % (name, len(b)))
        return b
    except Exception as e:
        manifest["failed"].append({"file": name, "url": url, "error": "%s: %s" % (type(e).__name__, e)})
        print("  FAILED %-46s %s: %s" % (name, type(e).__name__, str(e)[:90]))
        return None

# ---------------------------------------------------------------- FRED
# Distributional Financial Accounts (wealth by group) and the macro background.
FRED = {
    # shares of US household net worth, by wealth group
    "WFRBSTP1300": "share of net worth, top 0.1%",
    "WFRBST01134": "share of net worth, top 1%",
    "WFRBSN09161": "share of net worth, 90th-99th percentiles",
    "WFRBSN40188": "share of net worth, 50th-90th percentiles",
    "WFRBSB50215": "share of net worth, bottom 50%",
    # levels ($ millions)
    "WFRBLTP1246": "net worth, top 0.1%",
    "WFRBLT01026": "net worth, top 1%",
    "WFRBLN09053": "net worth, 90th-99th",
    "WFRBLN40080": "net worth, 50th-90th",
    "WFRBLB50107": "net worth, bottom 50%",
    # shares of asset classes held by the top 0.1%
    "WFRBSTP1291": "share of financial assets, top 0.1%",
    "WFRBSTP1281": "share of total assets, top 0.1%",
    "WFRBSTP1305": "share of real estate, top 0.1%",
    # macro background
    "FEDFUNDS": "federal funds rate",
    "DGS10": "10-year Treasury yield",
    "DFII10": "10-year real (TIPS) yield",
    "WALCL": "Federal Reserve total assets",
    "M2SL": "M2 money stock",
    "GDP": "US nominal GDP",
    "GDPC1": "US real GDP",
    "CPIAUCSL": "consumer prices",
    "TNWBSHNO": "household net worth",
    "NCBEILQ027S": "market value of corporate equities",
    "CP": "corporate profits after tax",
    "PRS85006173": "labour share, nonfarm business",
    "LABSHPUSA156NRUG": "labour share of GDP (Penn World Table)",
    "CSUSHPINSA": "Case-Shiller house price index",
    "SP500": "S&P 500 (last ten years)",
    "NASDAQCOM": "Nasdaq Composite",
    "FYFRGDA188S": "federal receipts, % of GDP",
    "GFDEGDQ188S": "federal debt, % of GDP",
    "TDSP": "household debt service ratio",
    "HDTGPDUSQ163N": "household debt, % of GDP",
}

def is_csv(b):
    head = b[:200].decode("utf-8", "replace").lower()
    return ("date" in head) and ("<html" not in head)

def fetch_fred():
    print("FRED")
    for sid, desc in FRED.items():
        save("fred/%s.csv" % sid, "https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s" % sid, check=is_csv)
        time.sleep(0.4)
    with open(os.path.join(OUT, "fred", "_series.json"), "w") as f:
        json.dump(FRED, f, indent=1)

# ---------------------------------------------------------------- Fed DFA
def fetch_dfa():
    print("Federal Reserve Distributional Financial Accounts (full tables)")
    b = save("dfa/dfa.zip", "https://www.federalreserve.gov/releases/z1/dataviz/download/zips/dfa.zip",
             check=lambda b: b[:2] == b"PK")
    if b:
        try:
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                z.extractall(os.path.join(OUT, "dfa"))
                print("         unpacked:", ", ".join(sorted(z.namelist()))[:300])
        except Exception as e:
            print("         could not unpack:", e)

# ---------------------------------------------------------------- World Bank
WB = {"NY.GDP.MKTP.CD": "gdp_current_usd", "NY.GDP.MKTP.KD.ZG": "gdp_growth", "SP.POP.TOTL": "population",
      "NE.GDI.TOTL.ZS": "investment_pct_gdp", "NY.GNS.ICTR.ZS": "saving_pct_gdp"}
def fetch_worldbank():
    print("World Bank")
    for ind, name in WB.items():
        url = ("https://api.worldbank.org/v2/country/WLD;USA;GBR;CHN;IND;DEU;FRA;JPN/indicator/%s"
               "?format=json&per_page=2000&date=1980:2026" % ind)
        save("worldbank/%s.json" % name, url, check=lambda b: b[:1] == b"[")

# ---------------------------------------------------------------- Forbes
def fetch_forbes():
    print("Forbes annual billionaire lists")
    for year in range(2010, datetime.now().year + 1):
        url = ("https://www.forbes.com/forbesapi/person/billionaires/%d/position/true.json"
               "?limit=5000&fields=rank,personName,finalWorth,source,countryOfCitizenship,industries,selfMade,birthDate,gender,year" % year)
        save("forbes/billionaires_%d.json" % year, url, check=lambda b: b[:1] == b"{" and b"personList" in b[:400])
        time.sleep(1.0)

# ---------------------------------------------------------------- Our World in Data (WID series, small files)
OWID = ["share-of-wealth-richest-1-percent", "wealth-share-richest-1-percent", "wealth-share-richest-10-percent",
        "income-share-top-1-before-tax-wid", "income-share-of-the-top-1-pip", "labor-share-of-gdp",
        "gdp-per-capita-maddison-project-database", "top-marginal-income-tax-rate"]
def fetch_owid():
    print("Our World in Data (some of these names may not exist; failures here are expected)")
    for slug in OWID:
        save("owid/%s.csv" % slug, "https://ourworldindata.org/grapher/%s.csv?v=1&csvType=full&useColumnShortNames=true" % slug,
             check=is_csv_owid)
        time.sleep(0.5)
def is_csv_owid(b):
    head = b[:200].decode("utf-8", "replace").lower()
    return ("entity" in head or "country" in head) and "<html" not in head

# ---------------------------------------------------------------- WID bulk (optional, large)
WID_COUNTRIES = ["US", "GB", "FR", "DE", "CN", "IN", "RU", "JP", "SE", "NO", "BR", "ZA", "WO"]
WID_KEEP = ("shweal", "sptinc", "sdiinc", "ahweal", "thweal", "wwealn", "mnninc", "anninc", "wwealp", "wwealg")
def fetch_wid():
    print("World Inequality Database bulk file (large: several hundred MB; kept countries: %s)" % ", ".join(WID_COUNTRIES))
    url = "https://wid.world/bulk_download/wid_all_data.zip"
    tmp = os.path.join(OUT, "wid", "wid_all_data.zip")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
            n = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk: break
                f.write(chunk); n += len(chunk)
                if n % (50 << 20) < (1 << 20): print("         %d MB" % (n >> 20), flush=True)
        kept = 0
        with zipfile.ZipFile(tmp) as z:
            names = z.namelist()
            for c in WID_COUNTRIES:
                for kind in ("data", "metadata"):
                    nm = "WID_%s_%s.csv" % (kind, c)
                    if nm not in names: continue
                    with z.open(nm) as src, open(os.path.join(OUT, "wid", nm), "w", newline="") as dst:
                        rd = csv.reader(io.TextIOWrapper(src, "utf-8"), delimiter=";")
                        wr = csv.writer(dst, delimiter=";")
                        header = next(rd); wr.writerow(header)
                        vi = header.index("variable")
                        for row in rd:
                            if row[vi].startswith(WID_KEEP):
                                wr.writerow(row); kept += 1
        os.remove(tmp)
        manifest["ok"].append({"file": "wid/WID_data_*.csv", "url": url, "rows": kept})
        print("  ok     wid: kept %d rows" % kept)
    except Exception as e:
        manifest["failed"].append({"file": "wid/wid_all_data.zip", "url": url, "error": "%s: %s" % (type(e).__name__, e)})
        print("  FAILED wid: %s: %s" % (type(e).__name__, str(e)[:120]))

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fetch_fred(); fetch_dfa(); fetch_worldbank(); fetch_forbes(); fetch_owid()
    if "--wid" in sys.argv: fetch_wid()
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("\n%d files fetched, %d failed. Details in src/economics/data/manifest.json" % (len(manifest["ok"]), len(manifest["failed"])))
