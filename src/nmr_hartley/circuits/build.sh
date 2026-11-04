#!/bin/sh
# Compile each circuitikz diagram to PDF (for PDF output) and SVG (for HTML output).
cd "$(dirname "$0")"
for f in hartley preamp system driver; do
  pdflatex -interaction=nonstopmode -halt-on-error "$f.tex" > /dev/null || { echo "FAILED: $f"; tail -20 "$f.log"; exit 1; }
  pdf2svg "$f.pdf" "$f.svg"
  rm -f "$f.aux" "$f.log"
done
echo "built: hartley preamp system driver (.pdf, .svg)"
