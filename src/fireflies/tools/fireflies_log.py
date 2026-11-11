"""Read every firefly on USB at once and log their flashes with one clock.

    python3 tools/fireflies_log.py                 # all ports, writes flashes_<time>.csv
    python3 tools/fireflies_log.py -o run1.csv --seconds 120
    python3 tools/fireflies_log.py --ports /dev/cu.usbmodem1101 /dev/cu.usbmodem1201

Each node prints "F,<node>,<seq>,<millis>" when it flashes. The boards' own clocks are
not synchronised, so every line is timestamped on arrival here (USB adds about a
millisecond of jitter). Prints a live line once a second: how many nodes flashed in
the last group flash and how spread out in time it was. Needs pyserial.
"""
import argparse, csv, glob, sys, threading, time, queue
import serial

PATTERNS = ["/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*", "/dev/ttyACM*", "/dev/ttyUSB*"]

def reader(port, q, stop):
    while not stop.is_set():
        try:
            with serial.Serial(port, 115200, timeout=0.2) as s:
                buf = b""
                while not stop.is_set():
                    chunk = s.read(256)
                    t = time.perf_counter()
                    if not chunk: continue
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        q.put((t, port, line.decode(errors="replace").strip()))
        except (serial.SerialException, OSError) as e:
            q.put((time.perf_counter(), port, f"#error {e}"))
            time.sleep(1.0)                                   # board unplugged or resetting: retry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="*")
    ap.add_argument("-o", "--out")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl-C)")
    ap.add_argument("--gap", type=float, default=0.15, help="flashes closer than this (s) count as one group flash")
    a = ap.parse_args()
    ports = a.ports or sorted(p for pat in PATTERNS for p in glob.glob(pat))
    if not ports: sys.exit("No ports found.")
    out = a.out or time.strftime("flashes_%Y%m%d_%H%M%S.csv")
    print(f"Logging {len(ports)} port(s) to {out}")
    q, stop = queue.Queue(), threading.Event()
    for p in ports: threading.Thread(target=reader, args=(p, q, stop), daemon=True).start()

    t0 = time.perf_counter(); group = []; last_report = t0
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_s", "port", "node", "seq", "board_ms"])
        try:
            while not a.seconds or time.perf_counter() - t0 < a.seconds:
                try: t, port, line = q.get(timeout=0.1)
                except queue.Empty: t, line = time.perf_counter(), ""
                if line.startswith("F,"):
                    try: _, node, seq, ms = line.split(",")
                    except ValueError: continue
                    w.writerow([f"{t - t0:.4f}", port, node, seq, ms])
                    if group and t - group[-1][0] > a.gap: group = []
                    group.append((t, node))
                elif line.startswith("I,"):
                    _, node, period = line.split(",")
                    print(f"  {port}: node {node}, period {period} ms")
                elif line.startswith("#error"):
                    print(f"  {port}: {line[1:]}")
                if time.perf_counter() - last_report >= 1.0 and group:
                    spread = (group[-1][0] - group[0][0]) * 1000
                    n = len({g[1] for g in group})
                    print(f"{time.perf_counter() - t0:7.1f} s  last group flash: {n:2d} node(s), spread {spread:5.1f} ms")
                    last_report = time.perf_counter(); f.flush()
        except KeyboardInterrupt:
            pass
    stop.set()
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
