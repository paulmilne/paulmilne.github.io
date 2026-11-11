# ESP32 fireflies

Firmware for the post *Fireflies on a Radio Link* (complexity/esp32-fireflies).
PlatformIO project; flash `node` (or `node_s3`) to every board, and optionally
`logger` to one board left on USB:

    pio run -e node -t upload
    pio run -e logger -t upload && pio device monitor -e logger > flashes.csv

Cheapest build: ESP32-C3 SuperMini boards, `pio run -e node_c3 -t upload`
(and `logger_c3` for the listener). `LED_PIN` is set per environment; change it for boards whose LED is elsewhere.
Boards with only an addressable RGB LED (for example M5Stack Atom) need the LED
lines replaced with the board's LED library.

The simulation and figure scripts are `ff.py`, `exps.py` and `figs5.py` in
`../kuramoto/`.

## Ten boards on a USB hub

With every board on a powered USB hub (3 A or more, USB-A ports, short USB-A to
USB-C cables), the Mac can flash and read all of them without moving anything:

    tools/flash_all.sh                      # build once, upload to every board on USB
    pip install pyserial matplotlib numpy
    python3 tools/fireflies_log.py          # live: nodes in each group flash, and their spread in ms
    python3 tools/fireflies_plot.py flashes_<time>.csv   # raster and order parameter, like the post

Each node prints `F,<node>,<seq>,<millis>` on every flash; the logger timestamps
lines as they arrive, so all boards share the Mac's clock (USB adds about 1 ms of
jitter). Start the logger before powering the hub to catch the approach to sync
from random phases, or press each board's RESET to restart it.
If an upload fails, hold BOOT, tap RESET, release BOOT and rerun.
