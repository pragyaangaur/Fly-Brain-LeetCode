"""Fill the page template with the real result files. Every number on the page comes from here.

    python scripts/build_site.py   ->   site/index.html
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
res = ROOT / "results"


def load(name):
    p = res / name
    return p.read_text() if p.exists() else "null"


html = (ROOT / "site" / "template.html").read_text()
html = html.replace("/*RESULTS*/null", json.dumps(json.loads(load("results.json")), separators=(",", ":")))
html = html.replace("/*ROBUST*/null", load("v1/robust.json"))
html = html.replace("/*TRACE*/null", load("trace.json"))
(ROOT / "site" / "index.html").write_text(html)
print("wrote site/index.html", len(html) // 1024, "KB")
