#!/usr/bin/env python3
"""
ER scoring within a manually drawn PanCK+ tumour region (QuPath, headless).

  python er_scoring.py import     # step 1: map CosMx cell boundaries onto each image
  (draw polygon, class = Tumour, save .qpdata in QuPath GUI)   # step 2
  python er_scoring.py measure    # measure ER for cells inside the polygon
  python er_scoring.py classify   # steps 3+4: histograms, thresholds, Neg/1+/2+/3+
  python er_scoring.py apply      # step 5: write the classes back onto the cells in the .qpdata

Requires: pip install numpy matplotlib
Add --force to `import` to overwrite existing .qpdata (this deletes drawn annotations).

THRESHOLDING (classify) works in three stages, stopping at the first that succeeds:
  1. Pronounced negative peak: a distinct low-intensity peak separated from the main
     population by a clear valley. t1 = the histogram elbow (valley / Triangle).
  2. Shoulder: no separate peak, but a low plateau/tail on the left flank of the main
     population. t1 = the foot of the main population's rise. Only accepted if it is
     stable across smoothing bandwidths and lands inside SHOULDER_BAND.
  3. Floor: neither stage is confident -> t1 = FLOOR_T1, flagged low_confidence.
t2 and t3 are then thirds of the positive range above t1 (all stages).
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# =====================================================================
# CONFIG
# =====================================================================
QUPATH_EXE = r"/Applications/QuPath-0.7.0-arm64.app/Contents/MacOS/QuPath-0.7.0-arm64"
IMAGE_DIR  = Path(r"{TIFs_file}")
CSV_DIR    = Path("{Cell_stats_file}")
OUTPUT_DIR = Path(r"{save_file}")

# How to match an image to its boundary CSV: grab the FOV token from the image
# name, then fill it into the template. e.g. image "..._F00002.ome.tif" -> CellBoundaries_F00002.csv
FOV_REGEX    = r"F\d+"
CSV_TEMPLATE = "CellBoundaries_{fov}.csv"

MARKER_CHANNEL = "ER"        # exact channel name preferred; falls back to a UNIQUE substring match
TUMOUR_CLASS   = "Tumour"    # class you give the polygon in QuPath

# --- Tumour (PanCK) detection ---
PANCK_CHANNEL    = "PanCK (AF488)"   # substring of channel name (must match exactly one channel)
PANCK_THRESHOLD  = 100
PANCK_DOWNSAMPLE = 1
PANCK_MIN_AREA   = 200.0     # um2 - used to be 500 was disincluding individual cells, but 200 is still enough to remove small specks of noise
PANCK_MIN_HOLE   = 2000.0    # um2 - changed from 4000 to 2000 to allow for small holes in tumour regions (e.g. blood vessels)

# --- Thresholding: general ---
THRESHOLD_MODE = "auto"      # "auto" (3-stage: negative peak -> shoulder -> floor) or "fixed"
FIXED_T1, FIXED_T2, FIXED_T3 = 5, 15, 25
AUTO_FLOOR_T1  = 5.0         # hard lower bound: t1 is never allowed below this (set 0 to disable)
STEP_2 = FIXED_T2 - FIXED_T1 # last-resort spacing only (positive range <= t1)
STEP_3 = FIXED_T3 - FIXED_T2
MIN_CELLS_FOR_AUTO = 100     # below this, stage 3 (floor)
HIST_BINS     = 60           # linear bins between min and 99.9th percentile (stage-1 elbow + thirds)
SMOOTH_BINS   = 3            # moving-average width used to find peak/valley in the linear histogram
MIN_BIN_FRAC  = 0.01         # bins with < this fraction of ALL cells are ignored

# --- Thresholding: stage 3 (floor) ---
FLOOR_T1 = 250.0             # lenient default 1+ cutoff when no confident negative peak / shoulder is found

# --- Thresholding: log-intensity density used by stages 1 and 2 ---
KDE_BW = 0.08                # Gaussian smoothing width in log10 units (~ +/-20% intensity)

# --- Thresholding: stage 1 (pronounced negative peak) ---
NEG_PEAK_MIN_REL     = 0.10  # left peak must be >= this fraction of the main peak's height
NEG_VALLEY_MAX_DEPTH = 0.50  # valley height / smaller of the two peaks must be <= this
PEAK_MIN_PROMINENCE  = 0.03  # peaks must stand out by >= this fraction of the tallest peak

# --- Thresholding: stage 2 (shoulder) ---
SHOULDER_BWS         = (0.06, 0.07, 0.08)  # the foot is recomputed at each bandwidth
SHOULDER_SLOPE_FRAC  = 0.30  # foot = walking left from steepest rise until slope < this fraction of it
SHOULDER_MAX_SPREAD  = 1.35  # max/min of the foot across bandwidths must be <= this, else unstable
SHOULDER_MIN_GAP_LOG = 0.15  # foot must be >= this many log10 units below the main peak
SHOULDER_BAND        = (100.0, 700.0)  # foot outside this range is rejected (sanity band, in ER units)
MIN_CELLS_SHOULDER   = 400   # fewer cells than this: shoulder is not attempted
# =====================================================================

IMPORT_GROOVY = r"""
import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.ROIs
import qupath.lib.io.PathIO
import qupath.lib.objects.classes.PathClass
import qupath.opencv.ml.pixel.PixelClassifiers

def csvFile    = new File('@@CSV@@')
def qpdataFile = new File('@@QPDATA@@')

setImageType('FLUORESCENCE')
clearAllObjects()

@@TUMOUR_BLOCK@@

def cellPointsMap = [:].withDefault { [] }
csvFile.withReader { reader ->
    String headerLine = reader.readLine()
    if (headerLine == null) return
    String delimiter = ","
    if (headerLine.contains("\t")) delimiter = "\t"
    else if (headerLine.contains(";")) delimiter = ";"
    def headers = headerLine.split(delimiter).collect { it.trim().toLowerCase() }
    int cellIdIdx = headers.indexOf("cellid")
    int xIdx = headers.indexOf("x_local")
    int yIdx = headers.indexOf("y_local")
    if (cellIdIdx == -1 || xIdx == -1 || yIdx == -1)
        throw new IllegalArgumentException("Missing cellID / x_local / y_local columns")
    String line
    while ((line = reader.readLine()) != null) {
        if (line.trim().isEmpty()) continue
        def tokens = line.split(delimiter)
        try {
            cellPointsMap[tokens[cellIdIdx].trim()] << [Double.parseDouble(tokens[xIdx].trim()),
                                                         Double.parseDouble(tokens[yIdx].trim())]
        } catch (Exception e) { }
    }
}

def plane = ImagePlane.getDefaultPlane()
def cells = []
cellPointsMap.each { cellID, points ->
    if (points.size() < 3) return
    double[] xs = points.collect { it[0] } as double[]
    double[] ys = points.collect { it[1] } as double[]
    // Detection objects (not annotations) so they behave as cells and can't be confused with your tumour polygon
    def det = PathObjects.createDetectionObject(ROIs.createPolygonROI(xs, ys, plane))
    det.setName(cellID)
    cells << det
}
addObjects(cells)
fireHierarchyUpdate()
println "Imported " + cells.size() + " cell boundaries"
cells.each {it.setColor(null)}
println "Cleared cell boundary colors"
PathIO.writeImageData(qpdataFile, getCurrentImageData())
"""

MEASURE_GROOVY = r"""
import qupath.lib.io.PathIO
import qupath.lib.analysis.features.ObjectMeasurements
import qupath.lib.analysis.features.ObjectMeasurements.Measurements
import qupath.lib.analysis.features.ObjectMeasurements.Compartments
import java.awt.image.BufferedImage

def qpdataFile  = new File('@@QPDATA@@')
def outCsv      = new File('@@CELLS_CSV@@')
def markerMatch = '@@MARKER@@'
def tumourClass = '@@TUMOUR_CLASS@@'

def server = getCurrentServer()
def data = PathIO.readImageData(qpdataFile.toPath(), null, server, BufferedImage)
def hierarchy = data.getHierarchy()

// Channel: exact name wins, else a substring match that must be unique (avoids ER vs HER2 mix-ups)
def channelNames = server.getMetadata().getChannels()*.getName()
def hits = channelNames.findAll { it == markerMatch }
if (hits.isEmpty()) hits = channelNames.findAll { it.contains(markerMatch) }
if (hits.size() != 1)
    throw new IllegalArgumentException("'${markerMatch}' matched ${hits} among ${channelNames}")
def markerName = hits[0]
println "Marker channel: ${markerName}"

def tumours = hierarchy.getAnnotationObjects().findAll { it.getPathClass()?.getName() == tumourClass }
if (tumours.isEmpty()) {
    println "NO_TUMOUR_ANNOTATION: nothing with class '${tumourClass}' found"
    return
}

// Cells whose centroid falls inside any tumour polygon
def cells = hierarchy.getDetectionObjects().findAll { c ->
    def r = c.getROI()
    tumours.any { it.getROI().contains(r.getCentroidX(), r.getCentroidY()) }
}
println "${cells.size()} cells inside ${tumours.size()} tumour annotation(s)"

cells.each {
    ObjectMeasurements.addIntensityMeasurements(server, it, 1.0, [Measurements.MEAN], [Compartments.CELL])
}

def keyFor = { c ->
    c.getMeasurementList().getMeasurementNames().find {
        it.contains(markerName) && it.toLowerCase().contains('mean')
    }
}
if (!cells.isEmpty())
    println "Measurement used: " + keyFor(cells[0]) + "  (all: " + cells[0].getMeasurementList().getMeasurementNames() + ")"

def cal = server.getPixelCalibration()
double pw = cal.hasPixelSizeMicrons() ? cal.getPixelWidthMicrons() : 1.0
double ph = cal.hasPixelSizeMicrons() ? cal.getPixelHeightMicrons() : 1.0

def sb = new StringBuilder("cell_id,x,y,area_um2,er_mean\n")
cells.each { c ->
    def k = keyFor(c)
    def v = k ? c.getMeasurementList().get(k) : Double.NaN
    def r = c.getROI()
    sb.append("${c.getName()},${r.getCentroidX()},${r.getCentroidY()},${r.getScaledArea(pw, ph)},${v}\n")
}
outCsv.parentFile.mkdirs()
outCsv.text = sb.toString()
println "Wrote " + outCsv
"""
APPLY_GROOVY = r"""
import qupath.lib.io.PathIO
import qupath.lib.objects.classes.PathClass
import java.awt.image.BufferedImage

def qpdataFile = new File('@@QPDATA@@')
def classCsv   = new File('@@CLASS_CSV@@')

def server = getCurrentServer()
def data = PathIO.readImageData(qpdataFile.toPath(), null, server, BufferedImage)
def hierarchy = data.getHierarchy()

def lookup = [:]
classCsv.eachLine { line, n ->
    if (n > 1) { def t = line.split(','); lookup[t[0]] = t[-1].trim() }
}
def colors = ['Negative': getColorRGB(0, 0, 255), '1+': getColorRGB(255, 220, 0),
              '2+': getColorRGB(255, 130, 0), '3+': getColorRGB(255, 0, 0)]

int k = 0
hierarchy.getDetectionObjects().each { c ->
    def cl = lookup[c.getName()]
    if (cl) { c.setPathClass(PathClass.fromString(cl, colors[cl])); k++ }
}
println "Classified ${k} cells"
PathIO.writeImageData(qpdataFile, data)
"""

def gs(value) -> str:
    """Safe inside a single-quoted Groovy string."""
    return str(value).replace("\\", "/").replace("'", "\\'")


def fill(template: str, **kw) -> str:
    for k, v in kw.items():
        template = template.replace(f"@@{k}@@", gs(v))
    return template


def find_images():
    return sorted(p for p in IMAGE_DIR.iterdir()
                  if p.is_file() and p.name.lower().endswith((".ome.tif", ".ome.tiff")))


def stem_of(image: Path) -> str:
    return image.name.split(".ome")[0]


def dirs():
    d = {n: OUTPUT_DIR / n for n in ("qpdata", "cells", "logs", "scripts", "results", "plots")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


import time

def run_qupath(image: Path, script_text: str, script_path: Path, log_path: Path, retries: int = 3) -> bool:
    script_path.write_text(script_text, encoding="utf-8")
    cmd = [QUPATH_EXE, "script", f"--image={image}", str(script_path)]
    for attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
            return proc.returncode == 0
        except OSError as e:
            print(f"(OSError {e}; retry {attempt}/{retries})", end="  ")
            time.sleep(3)
    log_path.write_text("run_qupath failed: repeated OSError launching QuPath\n", encoding="utf-8")
    return False

def tumour_detect() -> str:
    """Groovy block: PanCK threshold -> annotations classed as TUMOUR_CLASS."""
    block = r"""
// ---------- PanCK tumour detection ----------
def panckMatch = '@@PANCK_CHANNEL@@'
def channelNames = getCurrentServer().getMetadata().getChannels()*.getName()
def panckHits = channelNames.findAll { it == panckMatch }
if (panckHits.isEmpty()) panckHits = channelNames.findAll { it.contains(panckMatch) }
if (panckHits.size() != 1)
    throw new IllegalArgumentException("'${panckMatch}' matched ${panckHits} among ${channelNames}")

int channelIndex = channelNames.indexOf(panckHits[0])
println "PanCK channel ${channelIndex}: ${channelNames[channelIndex]}"

def server = getCurrentServer()
def resolution = server.getPixelCalibration().createScaledInstance(@@PANCK_DOWNSAMPLE@@, @@PANCK_DOWNSAMPLE@@)

def classifier = PixelClassifiers.createThresholdClassifier(
    resolution,
    channelIndex,
    (double) @@PANCK_THRESHOLD@@,
    PathClass.getInstance('Ignore*'),
    PathClass.getInstance('@@TUMOUR_CLASS@@')
)

def fullImage = createFullImageAnnotation(true)
createAnnotationsFromPixelClassifier(classifier, @@PANCK_MIN_AREA@@, @@PANCK_MIN_HOLE@@, "SPLIT", "DELETE_EXISTING")
removeObject(fullImage, true)

def red = getColorRGB(255, 0, 0)
getAnnotationObjects().each { it.setColor(red) }
fireHierarchyUpdate()
println "Detected ${getAnnotationObjects().size()} tumour region(s)"
if (getAnnotationObjects().isEmpty())
    println "NO_TUMOUR_REGIONS: nothing above PanCK threshold"
"""
    return fill(block, PANCK_CHANNEL=PANCK_CHANNEL, PANCK_THRESHOLD=PANCK_THRESHOLD,
                PANCK_DOWNSAMPLE=PANCK_DOWNSAMPLE, PANCK_MIN_AREA=PANCK_MIN_AREA,
                PANCK_MIN_HOLE=PANCK_MIN_HOLE, TUMOUR_CLASS=TUMOUR_CLASS)

# ---------------------------------------------------------------- stage 1/5
def stage_import(force: bool):
    d = dirs()
    images = find_images()
    print(f"{len(images)} image(s)")
    for i, image in enumerate(images, 1):
        stem = stem_of(image)
        qpdata = d["qpdata"] / f"{stem}.qpdata"
        print(f"[{i}/{len(images)}] {image.name}", end="  ")
        if qpdata.exists() and not force:
            print("skipped (qpdata exists; --force to overwrite)")
            continue
        m = re.search(FOV_REGEX, image.name)
        csv_path = CSV_DIR / CSV_TEMPLATE.format(fov=m.group(0)) if m else None
        if not csv_path or not csv_path.exists():
            print(f"NO MATCHING CSV ({csv_path})")
            continue
        script = IMPORT_GROOVY.replace("@@TUMOUR_BLOCK@@", tumour_detect())
        script = fill(script, CSV=csv_path, QPDATA=qpdata)
        ok = run_qupath(image, script, d["scripts"] / f"{stem}_import.groovy", d["logs"] / f"{stem}_import.log")
        print("done" if ok and qpdata.exists() else f"FAILED, see logs/{stem}_import.log")
    print(f"\nNext: (optional) open the .qpdata files in {d['qpdata']} to check the "
          f"'{TUMOUR_CLASS}' regions, then run: python er_scoring.py measure")


# ---------------------------------------------------------------- stage 2/5
def stage_measure():
    d = dirs()
    for i, image in enumerate(find_images(), 1):
        stem = stem_of(image)
        qpdata = d["qpdata"] / f"{stem}.qpdata"
        cells_csv = d["cells"] / f"{stem}_cells.csv"
        print(f"{image.name}", end="  ")
        if not qpdata.exists():
            print("no qpdata, run import first")
            continue
        cells_csv.unlink(missing_ok=True)
        script = fill(MEASURE_GROOVY, QPDATA=qpdata, CELLS_CSV=cells_csv,
                      MARKER=MARKER_CHANNEL, TUMOUR_CLASS=TUMOUR_CLASS)
        ok = run_qupath(image, script, d["scripts"] / f"{stem}_measure.groovy", d["logs"] / f"{stem}_measure.log")
        if ok and cells_csv.exists():
            print("done")
        else:
            print(f"no output (no '{TUMOUR_CLASS}' annotation, or error), see logs/{stem}_measure.log")


# ---------------------------------------------------------------- stage 3/5
# ---- stage-1 elbow on the linear histogram (unchanged from V2) ----
def find_elbow(counts, n):
    """Return (elbow_bin, method, peak_bin). method: 'valley', 'triangle' or 'none'."""
    k = SMOOTH_BINS
    sm = np.convolve(counts, np.ones(k) / k, mode="same")
    peak = int(sm.argmax())
    last = int(np.nonzero(counts)[0][-1])
    if last <= peak:
        return peak, "none", peak
    min_count = MIN_BIN_FRAC * n

    # 1) true valley: local minimum after the peak, followed by a real second bump
    for i in range(peak + 1, last):
        if sm[i] <= sm[i - 1] and sm[i] < sm[i + 1]:
            later = sm[i + 1:].max()
            if later - sm[i] >= min_count and later >= 1.25 * sm[i]:
                return i, "valley", peak

    # 2) no valley: Triangle elbow (point furthest from the peak->tail line)
    x = np.arange(peak, last + 1)
    y = sm[peak:last + 1]
    dx, dy = last - peak, sm[last] - sm[peak]
    dist = np.abs(dy * (x - peak) - dx * (y - sm[peak]))
    return peak + int(dist.argmax()), "triangle", peak


# ---- log-intensity density (stages 1 and 2) ----
def log_density(er, h):
    """Gaussian-smoothed density of log10(ER) on a fine 0.01 grid -> (grid, density), or None."""
    pos = er[er > 0]
    if pos.size < 10:
        return None
    L = np.log10(pos)
    lo = np.percentile(L, 0.5) - 0.3
    hi = np.percentile(L, 99.9) + 0.3
    grid = np.arange(lo, hi, 0.01)
    cnt, _ = np.histogram(L, bins=np.append(grid, grid[-1] + 0.01))
    kx = np.arange(-4 * h, 4 * h + 1e-9, 0.01)
    k = np.exp(-0.5 * (kx / h) ** 2)
    k /= k.sum()
    return grid + 0.005, np.convolve(cnt.astype(float), k, mode="same")


def local_peaks(d):
    """Indices of local maxima whose prominence is >= PEAK_MIN_PROMINENCE * tallest peak (numpy only)."""
    out = []
    top = d.max()
    for i in range(1, len(d) - 1):
        if d[i] > d[i - 1] and d[i] >= d[i + 1]:
            lo = i
            while lo > 0 and d[lo - 1] <= d[i]:
                lo -= 1
            hi = i
            while hi < len(d) - 1 and d[hi + 1] <= d[i]:
                hi += 1
            prom = d[i] - max(d[lo:i + 1].min(), d[i:hi + 1].min())
            if prom >= PEAK_MIN_PROMINENCE * top:
                out.append(i)
    return out


# ---- stage 1: pronounced negative peak ----
def negative_peak_depth(er):
    """Valley depth between a left-hand negative peak and the main peak, or None if there is no such peak.
    Depth = valley height / height of the smaller peak (0 = empty gap, 1 = no valley)."""
    dd = log_density(er, KDE_BW)
    if dd is None:
        return None
    g, d = dd
    main = int(d.argmax())
    left = [p for p in local_peaks(d) if p < main and d[p] >= NEG_PEAK_MIN_REL * d[main]]
    if not left:
        return None
    p0 = max(left, key=lambda p: d[p])
    vi = p0 + int(d[p0:main + 1].argmin())
    return float(d[vi] / min(d[p0], d[main]))


# ---- stage 2: shoulder ----
def shoulder_t1(er):
    """Foot of the main population's rise, recomputed at several bandwidths.
    Returns (t1 or None, spread or None, message)."""
    vals = []
    for h in SHOULDER_BWS:
        dd = log_density(er, h)
        if dd is None:
            return None, None, "too few positive values"
        g, d = dd
        pk = int(d.argmax())
        sl = np.gradient(d)[:pk]                 # slope on the left flank only
        if len(sl) < 5 or sl.max() <= 0:
            return None, None, "no rising left flank"
        j = int(sl.argmax())                     # steepest rise
        thr = SHOULDER_SLOPE_FRAC * sl[j]
        i = j
        while i > 0 and sl[i - 1] >= thr:        # walk left while still rising
            i -= 1
        if g[pk] - g[i] < SHOULDER_MIN_GAP_LOG:
            return None, None, "foot too close to main peak"
        vals.append(10 ** g[i])
    spread = max(vals) / min(vals)
    if spread > SHOULDER_MAX_SPREAD:
        return None, spread, f"unstable across bandwidths (x{spread:.2f})"
    return float(np.median(vals)), spread, f"stable across bandwidths (x{spread:.2f})"


def thirds_above(er, t1, notes):
    """Split the positive range above t1 into thirds -> (t1, t2, t3, upper, edges)."""
    n = er.size
    lo, hi = float(er.min()), float(np.percentile(er, 99.9))
    counts, edges = np.histogram(er, bins=HIST_BINS, range=(lo, hi))  # values > hi are ignored, not piled up
    b = int(np.clip(np.searchsorted(edges, t1, side="right") - 1, 0, HIST_BINS - 1))
    qualifying = [j for j in range(b + 1, HIST_BINS) if counts[j] >= MIN_BIN_FRAC * n]
    if qualifying:
        upper = float(edges[max(qualifying) + 1])
    else:
        upper = hi
        notes.append(f"no positive bin >={MIN_BIN_FRAC:.0%}: used full range")
    step = (upper - t1) / 3.0
    if step <= 0:
        t2, t3 = t1 + STEP_2, t1 + STEP_2 + STEP_3
        notes.append("positive range <= t1: used fixed steps")
    else:
        t2, t3 = t1 + step, t1 + 2 * step
    return t1, t2, t3, upper, edges


def auto_thresholds(er):
    """Three stages, first success wins: (1) negative peak, (2) shoulder, (3) floor."""
    n = er.size
    notes = []
    t1 = None
    stage, method = 3, "floor"
    elbow = ""
    elbow_method = ""
    depth = ""
    spread = ""

    if n < MIN_CELLS_FOR_AUTO:
        notes.append(f"<{MIN_CELLS_FOR_AUTO} cells")
    else:
        # ---- stage 1: pronounced negative peak (clean valley) -> existing histogram elbow ----
        dep = negative_peak_depth(er)
        if dep is not None:
            depth = dep
            if dep <= NEG_VALLEY_MAX_DEPTH:
                lo, hi = float(er.min()), float(np.percentile(er, 99.9))
                counts, edges = np.histogram(er, bins=HIST_BINS, range=(lo, hi))
                b, m, peak = find_elbow(counts, n)
                if m != "none":
                    t1 = 0.5 * (edges[b] + edges[b + 1])
                    elbow, elbow_method = t1, m
                    stage, method = 1, "negative_peak"
                    notes.append(f"clean negative peak (valley depth {dep:.2f}, elbow={m})")
                    if peak > HIST_BINS * 0.5:
                        notes.append("peak in upper half: mostly-positive image? check plot")

        # ---- stage 2: shoulder ----
        if t1 is None:
            if n < MIN_CELLS_SHOULDER:
                notes.append(f"<{MIN_CELLS_SHOULDER} cells: shoulder not attempted")
            else:
                t_sh, sp, msg = shoulder_t1(er)
                if sp is not None:
                    spread = sp
                if t_sh is None:
                    notes.append(f"no confident shoulder: {msg}")
                elif not (SHOULDER_BAND[0] <= t_sh <= SHOULDER_BAND[1]):
                    notes.append(f"shoulder foot {t_sh:.0f} outside band "
                                 f"{SHOULDER_BAND[0]:.0f}-{SHOULDER_BAND[1]:.0f}")
                else:
                    t1 = t_sh
                    stage, method = 2, "shoulder"
                    notes.append(f"shoulder foot, {msg}")

    # ---- stage 3: floor ----
    if t1 is None:
        t1 = FLOOR_T1
        stage, method = 3, "floor"
        notes.append("LOW CONFIDENCE: used floor")

    if t1 < AUTO_FLOOR_T1:
        t1 = AUTO_FLOOR_T1
        notes.append("hard lower bound applied")

    t1, t2, t3, upper, edges = thirds_above(er, t1, notes)
    return dict(t1=t1, t2=t2, t3=t3, stage=stage, method=method, elbow=elbow, elbow_method=elbow_method,
                valley_depth=depth, shoulder_spread=spread, upper=upper, edges=edges,
                low_confidence=(stage == 3), note="; ".join(notes))


def stage_classify():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = dirs()
    summary = []
    for cells_csv in sorted(d["cells"].glob("*_cells.csv")):
        stem = cells_csv.name.replace("_cells.csv", "")
        with open(cells_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        er = np.array([float(r["er_mean"]) for r in rows])
        keep = ~np.isnan(er)
        rows = [r for r, k in zip(rows, keep) if k]
        er = er[keep]
        if er.size == 0:
            print(f"{stem}: no valid cells")
            continue
        p = np.percentile(er, [5, 25, 50, 75, 95, 99])
        print(f"{stem}: percentiles 5/25/50/75/95/99: {p.round(0)}")

        # ---- thresholds ----
        info, note = None, ""
        if THRESHOLD_MODE == "fixed":
            t1, t2, t3 = FIXED_T1, FIXED_T2, FIXED_T3
        else:
            info = auto_thresholds(er)
            t1, t2, t3, note = info["t1"], info["t2"], info["t3"], info["note"]

        # ---- classify ----
        cls = np.where(er >= t3, "3+", np.where(er >= t2, "2+", np.where(er >= t1, "1+", "Negative")))
        n = {k: int((cls == k).sum()) for k in ("Negative", "1+", "2+", "3+")}
        total = er.size
        pos = n["1+"] + n["2+"] + n["3+"]
        pct = lambda v: 100.0 * v / total
        hscore = (n["1+"] + 2 * n["2+"] + 3 * n["3+"]) * 100.0 / total

        with open(d["results"] / f"{stem}_classified.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cell_id", "x", "y", "area_um2", "er_mean", "class"])
            for r, c in zip(rows, cls):
                w.writerow([r["cell_id"], r["x"], r["y"], r["area_um2"], r["er_mean"], c])

        # ---- histogram (same bins as the threshold calculation) ----
        fig, ax = plt.subplots(figsize=(7, 4))
        if info:
            ax.hist(er, bins=info["edges"], color="grey")
            ax.axhline(MIN_BIN_FRAC * total, color="blue", ls=":", lw=1)
            ax.axvline(info["upper"], color="blue", ls=":", lw=1)
        else:
            ax.hist(er, bins=int(np.sqrt(total)), range=(er.min(), np.percentile(er, 99.9)), color="grey")
        for t, lab in ((t1, "1+"), (t2, "2+"), (t3, "3+")):
            ax.axvline(t, color="red", ls="--", lw=1)
            ax.text(t, ax.get_ylim()[1] * 0.95, lab, color="red", ha="left", va="top")
        ax.set_xlabel("ER mean intensity (whole cell)   [blue: 1% bin height / top of positive range]")
        ax.set_ylabel("cells")
        stage_txt = f"  stage {info['stage']} ({info['method']})" if info else ""
        ax.set_title(f"{stem}  n={total}  %pos={pct(pos):.1f}  H={hscore:.0f}{stage_txt}", fontsize=9)
        fig.tight_layout()
        fig.savefig(d["plots"] / f"{stem}_ER_hist.png", dpi=150)
        plt.close(fig)

        summary.append(dict(image=stem, cells_in_tumour=total,
                            stage=info["stage"] if info else "",
                            method=info["method"] if info else "fixed",
                            low_confidence=info["low_confidence"] if info else "",
                            valley_depth=round(info["valley_depth"], 3) if info and info["valley_depth"] != "" else "",
                            shoulder_spread=round(info["shoulder_spread"], 3) if info and info["shoulder_spread"] != "" else "",
                            elbow_method=info["elbow_method"] if info else "",
                            elbow=round(info["elbow"], 1) if info and info["elbow"] != "" else "",
                            positive_range_top=round(info["upper"], 1) if info else "",
                            t1=round(t1, 1), t2=round(t2, 1), t3=round(t3, 1),
                            negative=n["Negative"], n_1plus=n["1+"], n_2plus=n["2+"], n_3plus=n["3+"],
                            pct_positive=round(pct(pos), 2), pct_1plus=round(pct(n["1+"]), 2),
                            pct_2plus=round(pct(n["2+"]), 2), pct_3plus=round(pct(n["3+"]), 2),
                            h_score=round(hscore, 1), note=note))
        label = f"stage {info['stage']}: {info['method']}" if info else "fixed"
        print(f"   t1/t2/t3 = {t1:.0f}/{t2:.0f}/{t3:.0f} ({label}) "
              f"%pos={pct(pos):.1f} H={hscore:.0f} {note}")

    if summary:
        out = OUTPUT_DIR / "summary_ER.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary[0].keys())
            w.writeheader()
            w.writerows(summary)
        print(f"\nSummary: {out}\nHistograms: {d['plots']}")

# ---------------------------------------------------------------- stage 5/5
def stage_apply():
    d = dirs()
    for image in find_images():
        stem = stem_of(image)
        qpdata = d["qpdata"] / f"{stem}.qpdata"
        class_csv = d["results"] / f"{stem}_classified.csv"
        print(image.name, end="  ")
        if not (qpdata.exists() and class_csv.exists()):
            print("skipped (no qpdata or classified csv)")
            continue
        script = fill(APPLY_GROOVY, QPDATA=qpdata, CLASS_CSV=class_csv)
        ok = run_qupath(image, script, d["scripts"] / f"{stem}_apply.groovy", d["logs"] / f"{stem}_apply.log")
        print("done" if ok else f"FAILED, see logs/{stem}_apply.log")

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("import", "measure", "classify", "apply"):
        sys.exit(__doc__)
    if sys.argv[1] in ("import", "measure", "apply"):
        if not Path(QUPATH_EXE).exists():
            sys.exit(f"QuPath not found: {QUPATH_EXE}")
        if not IMAGE_DIR.is_dir():
            sys.exit(f"Image folder not found: {IMAGE_DIR}")
    {"import": lambda: stage_import("--force" in sys.argv),
     "measure": stage_measure,
     "classify": stage_classify,
     "apply": stage_apply}[sys.argv[1]]()


if __name__ == "__main__":
    main()
