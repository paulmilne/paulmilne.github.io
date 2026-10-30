"""Extract the Apollo 11 voice transcripts for a mission-time window.

    python3 transcripts.py SPACELOG_TRANSCRIPT_DIR OUT.json [get_start_s] [get_end_s]

Source: NASA's Apollo 11 Technical Air-to-Ground Voice Transcription (TEC)
and Onboard Voice Transcription (CM, from the command module's recorder),
both US government works, as cleaned and time-stamped by the Spacelog project
(github.com/Spacelog/spacelog, missions/a11/transcripts).

Spacelog's format: a "[dd:hh:mm:ss]" timestamp line, then one or more
"SPEAKER: text" lines, with metadata lines starting "_" (page, tape) and
inline markup such as "[glossary:P00]".
"""
import json
import re
import sys
from pathlib import Path

STAMP = re.compile(r"^\[(-?)(\d+):(\d+):(\d+):(\d+)\]\s*$")
LINE = re.compile(r"^([A-Z][A-Z0-9/ ()-]*?):\s?(.*)$")
MARKUP = re.compile(r"\[(?:glossary|time|link):([^\]|]+)(?:\|[^\]]*)?\]")

SPEAKERS = {
    "CDR": "Armstrong (CDR)",
    "CMP": "Collins (CMP)",
    "LMP": "Aldrin (LMP)",
    "CT": "Comms technician",
    "SC": "Spacecraft (unidentified)",
    "MS": "Mission Control",
}
# CapCom shifts on Apollo 11, read from Spacelog's mission notes (_meta) when available.
CAPCOM = [(0, "McCandless (CapCom)"), (6 * 3600 + 9 * 60 + 17, "Duke (CapCom)")]


def load_shifts(meta_path):
    """Replace CAPCOM with the shift list in Spacelog's _meta file."""
    global CAPCOM
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        shifts = meta["characters"]["CC"]["shifts"]
    except Exception:
        return
    out = []
    for item in shifts:
        who, stamp = item[0], item[1]
        d, h, mi, sec = (int(x) for x in stamp.split(":"))
        name = who.split("_")[-1].title()
        name = {"Mccandless": "McCandless"}.get(name, name)
        label = "CapCom (unidentified)" if name == "Cc" else f"{name} (CapCom)"
        out.append((((d * 24 + h) * 60 + mi) * 60 + sec, label))
    CAPCOM = sorted(out)


def capcom(t):
    name = CAPCOM[0][1]
    for start, who in CAPCOM:
        if t >= start:
            name = who
    return name


def parse(path, source):
    out, t = [], None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        m = STAMP.match(raw)
        if m:
            sign = -1 if m.group(1) else 1
            d, h, mi, s = (int(x) for x in m.groups()[1:])
            t = sign * (((d * 24 + h) * 60 + mi) * 60 + s)
            continue
        if t is None or not raw.strip() or raw.lstrip().startswith("_"):
            continue
        m = LINE.match(raw.strip())
        if m:
            code, text = m.group(1).strip(), m.group(2)
            name = capcom(t) if code == "CC" else SPEAKERS.get(code.split(" ")[0], code)
            if "(EAGLE)" in code:
                name += ", in Eagle"
            out.append([t, code, name, MARKUP.sub(r"\1", text).strip(), source])
        elif out and out[-1][0] == t:
            out[-1][3] += " " + MARKUP.sub(r"\1", raw.strip())
    return out


def main(src_dir, out_path, t0=-3600, t1=6 * 3600):
    src = Path(src_dir)
    load_shifts(src / "_meta")
    rows = parse(src / "TEC", "air-to-ground") + parse(src / "CM", "onboard")
    seen, unique = set(), []
    for r in sorted((r for r in rows if t0 <= r[0] <= t1), key=lambda r: (r[0], r[4])):
        key = (r[0], r[1], r[3], r[4])        # the TEC repeats some lines verbatim
        if key not in seen:
            seen.add(key)
            unique.append(r)
    rows = unique
    Path(out_path).write_text(json.dumps(dict(
        source="NASA Apollo 11 Technical Air-to-Ground and Onboard Voice Transcriptions, "
               "via Spacelog (spacelog.org)",
        columns=["get", "code", "speaker", "text", "channel"],
        lines=rows), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(len(rows), "lines ->", out_path)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(float(x) for x in a[2:4]))
