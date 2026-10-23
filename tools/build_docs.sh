#!/usr/bin/env bash
# Regenerate everything derived from the code and the report source, so the
# committed documentation always matches the code:
#   1. HTML API documentation (pdoc)      -> doc/api/
#   2. pytest report (HTML, JUnit XML, text) -> reports/test_report.*
#   3. Word version of the report (pandoc) -> doc/report/MMA3001_Project_Report.docx
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

if command -v pandoc >/dev/null; then
    # The existing Word file supplies the page layout and styles.
    (cd doc/report && pandoc report.md -o MMA3001_Project_Report.docx --reference-doc=MMA3001_Project_Report.docx)
    echo "Report -> doc/report/MMA3001_Project_Report.docx"
else
    echo "pandoc not found: Word report not rebuilt"
fi
