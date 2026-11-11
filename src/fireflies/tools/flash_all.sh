#!/usr/bin/env bash
# Flash every ESP32 on USB with the same firmware, one after another.
#   tools/flash_all.sh            # environment node_c3
#   tools/flash_all.sh node       # another environment from platformio.ini
# Run from the project folder (src/fireflies). Boards that are only powered (no data) are not seen.
set -euo pipefail
ENV="${1:-node_c3}"
cd "$(dirname "$0")/.."

shopt -s nullglob
PORTS=(/dev/cu.usbmodem* /dev/cu.usbserial* /dev/cu.wchusbserial* /dev/cu.SLAB_USBtoUART* /dev/ttyACM* /dev/ttyUSB*)
if [ ${#PORTS[@]} -eq 0 ]; then echo "No boards found on USB."; exit 1; fi
echo "Found ${#PORTS[@]} port(s): ${PORTS[*]}"

pio run -e "$ENV"                                  # build once
ok=0; failed=()
for p in "${PORTS[@]}"; do
  echo "=== $p"
  if pio run -e "$ENV" -t nobuild -t upload --upload-port "$p"; then ok=$((ok+1)); else failed+=("$p"); fi
done
echo "Flashed $ok of ${#PORTS[@]}."
if [ ${#failed[@]} -gt 0 ]; then echo "Failed: ${failed[*]}"; echo "Hold BOOT, tap RESET, release BOOT on those boards and rerun."; exit 1; fi
