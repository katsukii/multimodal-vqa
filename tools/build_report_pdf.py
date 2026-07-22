"""Render report/report.md to a styled report/report.html (then Chrome prints it to PDF).

Usage:
    python tools/build_report_pdf.py        # -> report/report.html
    # then: Google Chrome --headless --print-to-pdf=report/report.pdf report/report.html
"""
from __future__ import annotations

import os

import markdown

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(REPO, "report", "report.md")
HTML = os.path.join(REPO, "report", "report.html")

CSS = """
@page { size: A4; margin: 14mm 15mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Helvetica Neue", Arial, "Hiragino Sans", sans-serif;
       font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 17pt; margin: 0 0 2pt; border-bottom: 2px solid #333; padding-bottom: 4pt; }
h2 { font-size: 12.5pt; margin: 12pt 0 4pt; color: #222; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 9pt 0 3pt; color: #333; }
p { margin: 4pt 0; }
ul { margin: 4pt 0; padding-left: 18pt; }
li { margin: 1.5pt 0; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt;
       background: #f2f2f2; padding: 1px 3px; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #e0e0e0; border-radius: 4px; padding: 7pt 9pt;
      white-space: pre-wrap; word-wrap: break-word; margin: 5pt 0; }
pre code { background: none; padding: 0; font-size: 8.6pt; line-height: 1.35; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; text-align: left; }
th { background: #eee; }
strong { color: #000; }
"""


def build() -> None:
    with open(MD, encoding="utf-8") as f:
        text = f.read()
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {HTML}")


if __name__ == "__main__":
    build()
