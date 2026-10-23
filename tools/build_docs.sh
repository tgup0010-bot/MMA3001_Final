#!/usr/bin/env bash
# Regenerate everything derived from the code and the report source, so the
# committed documentation always matches the code:
#   1. HTML API documentation (pdoc)      -> doc/api/
#   2. pytest report (HTML, JUnit XML, text) -> reports/test_report.*
#   3. PDF of the report (pandoc + Chromium) -> doc/report/MMA3001_Project_Report.pdf
#
# Usage (from the repository root, after `pip install -e ".[dev]"`):
#   bash tools/build_docs.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Every module in the package, so a new module can never be left out of the docs.
modules=$(python -c "import pkgutil, occupancy; print(' '.join(['occupancy'] + ['occupancy.' + m.name for m in pkgutil.iter_modules(occupancy.__path__)]))")
rm -rf doc/api
pdoc -o doc/api $modules
echo "API docs -> doc/api/"

# Strip the absolute path of this machine from the text report.
python -m pytest -v --junitxml=reports/test_report.xml --html=reports/test_report.html --self-contained-html \
    | sed "s#$(pwd)#.#g" > reports/test_report.txt
tail -n 1 reports/test_report.txt

# The PDF is the frozen final form of the report (Word files render
# differently between versions). MathML keeps the equations exact.
if command -v pandoc >/dev/null && node -e "require('playwright')" 2>/dev/null; then
    (cd doc/report && pandoc report.md -s --mathml --css report.css --embed-resources \
        --metadata pagetitle="MMA3001 Project Report" -o report.html)
    node tools/report_to_pdf.js
    rm -f doc/report/report.html
else
    echo "pandoc or Node Playwright not found: report PDF not rebuilt"
fi
