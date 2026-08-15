#!/usr/bin/env python3
"""
build_gallery.py

Scans Data/output/ for every image dataset produced by extract_cells.py
(any immediate subdirectory containing a manifest.csv) and generates a single,
self-contained, offline HTML gallery at Data/output/gallery.html that lets you
switch between datasets via a dropdown in the top band.

Usage:
    python3 build_gallery.py

Re-run this script any time extract_cells.py finishes processing a new image;
it fully regenerates Data/output/gallery.html from whatever dataset folders
currently exist under Data/output/. No manual HTML editing required.
"""

import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Data", "output")
GALLERY_PATH = os.path.join(OUTPUT_ROOT, "gallery.html")

MANIFEST_FIELDS = [
    "cell_id", "x1", "y1", "x2", "y2", "area", "dark_fraction",
    "flagged_possible_merge", "filename", "location_filename",
]


def discover_datasets(output_root):
    """Return sorted list of (dataset_name, manifest_path) for every immediate
    subdirectory of output_root that contains a manifest.csv."""
    datasets = []
    if not os.path.isdir(output_root):
        return datasets
    for entry in os.listdir(output_root):
        full_path = os.path.join(output_root, entry)
        if not os.path.isdir(full_path):
            continue
        manifest_path = os.path.join(full_path, "manifest.csv")
        if os.path.isfile(manifest_path):
            datasets.append((entry, manifest_path))
    datasets.sort(key=lambda d: d[0])
    return datasets


def load_manifest_rows(manifest_path):
    """Read a manifest.csv into the row shape the gallery JS expects
    (same shape as the original single-image gallery's CELLS array)."""
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "cell_id": int(row["cell_id"]),
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "area": int(row["area"]),
                "dark_fraction": float(row["dark_fraction"]),
                "flagged": row["flagged_possible_merge"].strip().lower() == "true",
                "filename": row["filename"],
                "location_filename": row["location_filename"],
            })
    rows.sort(key=lambda r: r["cell_id"])
    return rows


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Microglia Cell Gallery</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #f5f6f8;
    --panel-bg: #ffffff;
    --border: #d7dbe0;
    --text: #1f2430;
    --muted: #6b7280;
    --accent: #2563eb;
    --accent-bg: #eaf1ff;
    --flag-bg: #fff3cd;
    --flag-border: #e0a800;
    --flag-text: #7a5b00;
    --thumb-selected: #2563eb;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    height: 100%;
    margin: 0;
  }}
  body {{
    padding: 16px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    height: 100dvh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}

  /* ---------- Top band: title, toolbar, metadata ---------- */
  .top-band {{
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
    overflow: hidden;
  }}
  .titlebar {{
    display: flex;
    align-items: baseline;
    gap: 10px;
    flex-wrap: wrap;
  }}
  h1 {{
    font-size: 1.05rem;
    margin: 0;
    flex: none;
  }}
  .subtitle {{
    color: var(--muted);
    font-size: 0.78rem;
    margin: 0;
    flex: none;
  }}
  .toolbar {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    flex: none;
  }}
  button, select {{
    font-size: 0.85rem;
    padding: 4px 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--panel-bg);
    color: var(--text);
    cursor: pointer;
  }}
  button:hover, select:hover {{ background: var(--accent-bg); }}
  button:active {{ transform: translateY(1px); }}
  select#datasetSelect {{
    font-weight: 600;
    max-width: 320px;
  }}
  .dataset-picker {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .nav-group {{
    display: flex;
    gap: 4px;
  }}
  .dataset-picker label {{
    font-size: 0.78rem;
    color: var(--muted);
  }}
  #position {{
    font-weight: 600;
    font-size: 0.85rem;
    min-width: 100px;
    text-align: center;
  }}
  .flag-badge {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    background: var(--flag-bg);
    border: 1px solid var(--flag-border);
    color: var(--flag-text);
    font-weight: 600;
    font-size: 0.78rem;
  }}
  .flag-badge.hidden {{ display: none; }}

  .meta {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    padding: 6px 12px;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 0.8rem;
    flex: none;
  }}
  .meta div span.label {{
    color: var(--muted);
    margin-right: 4px;
  }}

  /* ---------- Middle band: side-by-side image viewer ---------- */
  .viewer {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
  }}
  .panel {{
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    flex: 1 1 300px;
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
  }}
  .panel h2 {{
    font-size: 0.95rem;
    margin: 0 0 8px 0;
    color: var(--muted);
    font-weight: 600;
    flex: none;
  }}
  .panel.cell-panel {{ flex: 0 0 260px; max-width: 260px; }}
  .panel.location-panel {{ flex: 2 1 640px; }}
  .cell-panel img {{
    display: block;
    flex: 1 1 auto;
    min-height: 0;
    width: auto;
    height: auto;
    max-width: 100%;
    max-height: 100%;
    margin: 0 auto;
    image-rendering: pixelated;
    object-fit: contain;
    background:
      linear-gradient(45deg, #eee 25%, transparent 25%),
      linear-gradient(-45deg, #eee 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #eee 75%),
      linear-gradient(-45deg, transparent 75%, #eee 75%);
    background-size: 16px 16px;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0px;
    border: 1px solid var(--border);
    border-radius: 4px;
  }}
  .location-panel img {{
    display: block;
    flex: 1 1 auto;
    min-height: 0;
    width: auto;
    height: auto;
    max-width: 100%;
    max-height: 100%;
    margin: 0 auto;
    object-fit: contain;
    border: 1px solid var(--border);
    border-radius: 4px;
  }}

  /* ---------- Bottom band: thumbnail strip ---------- */
  .thumbs-wrap {{
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
    overflow: hidden;
  }}
  .thumbs-header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .thumbs-wrap h2 {{
    font-size: 0.85rem;
    margin: 0;
    color: var(--muted);
    font-weight: 600;
    flex: none;
  }}
  .thumbs-hint {{
    color: var(--muted);
    font-size: 0.75rem;
    flex: none;
  }}
  .thumbs {{
    display: flex;
    gap: 6px;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 8px;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    height: 72px;
  }}
  .thumb {{
    flex: 0 0 auto;
    width: 56px;
    height: 56px;
    border: 2px solid transparent;
    border-radius: 6px;
    cursor: pointer;
    padding: 0;
    background: #ddd;
    position: relative;
  }}
  .thumb img {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    border-radius: 4px;
    background: #fff;
  }}
  .thumb.selected {{
    border-color: var(--thumb-selected);
  }}
  .thumb.flagged::after {{
    content: "";
    position: absolute;
    top: 2px;
    right: 2px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--flag-border);
    box-shadow: 0 0 0 1px #fff;
  }}

  @media (max-width: 700px) {{
    .panel.cell-panel {{ flex: 1 1 100%; max-width: 100%; }}
    .panel.location-panel {{ flex: 1 1 100%; }}
  }}

  /* Fallback for very short windows: let the page scroll rather than
     crushing content past the point of usefulness. */
  @media (max-height: 560px) {{
    html, body {{ height: auto; overflow: visible; }}
    body {{ display: block; }}
    .viewer {{ min-height: 320px; }}
  }}

  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #14161a;
      --panel-bg: #1d2025;
      --border: #333844;
      --text: #e5e7eb;
      --muted: #9aa0ab;
      --accent: #60a5fa;
      --accent-bg: #23324a;
      --flag-bg: #3a2f0d;
      --flag-border: #d9a520;
      --flag-text: #f2c94c;
    }}
  }}
</style>
</head>
<body>

<header class="top-band">
  <div class="titlebar">
    <h1>Microglia Cell Gallery</h1>
    <div class="subtitle" id="subtitle">&mdash; extracted cells with source-slide location</div>
  </div>

  <div class="toolbar">
    <div class="dataset-picker">
      <label for="datasetSelect">Image:</label>
      <select id="datasetSelect"></select>
    </div>
    <div class="nav-group">
      <button id="prevBtn">&larr; Prev</button>
      <button id="nextBtn">Next &rarr;</button>
    </div>
    <span id="position">Cell - / -</span>
    <span id="flagBadge" class="flag-badge hidden">FLAGGED: possible merged cluster</span>
  </div>

  <div class="meta">
    <div><span class="label">cell_id:</span><span id="metaId">-</span></div>
    <div><span class="label">bbox:</span><span id="metaBbox">-</span></div>
    <div><span class="label">area:</span><span id="metaArea">-</span> px&sup2;</div>
    <div><span class="label">dark_fraction:</span><span id="metaDark">-</span></div>
    <div><span class="label">flagged_possible_merge:</span><span id="metaFlag">-</span></div>
  </div>
</header>

<main class="viewer">
  <div class="panel cell-panel">
    <h2>Extracted cell crop</h2>
    <img id="cellImg" src="" alt="cell crop">
  </div>
  <div class="panel location-panel">
    <h2>Location in original slide (box = crop region)</h2>
    <img id="locationImg" src="" alt="location in slide">
  </div>
</main>

<footer class="thumbs-wrap">
  <div class="thumbs-header">
    <h2>All cells (click to jump)</h2>
    <span class="thumbs-hint">Amber dot = flagged_possible_merge. Use &larr;/&rarr; arrow keys or the buttons to step through cells.</span>
  </div>
  <div class="thumbs" id="thumbs"></div>
</footer>

<script>
// DATASETS maps dataset folder name -> array of cell rows for that dataset
// (same row shape the original single-image gallery used). Everything is
// embedded inline so this file works fully offline via file://, with no
// fetch()/XHR of any kind.
const DATASETS = {datasets_json};

const DATASET_NAMES = Object.keys(DATASETS);

let currentDataset = DATASET_NAMES[0];
let currentIndex = 0;

const cellImg = document.getElementById('cellImg');
const locationImg = document.getElementById('locationImg');
const positionEl = document.getElementById('position');
const flagBadge = document.getElementById('flagBadge');
const metaId = document.getElementById('metaId');
const metaBbox = document.getElementById('metaBbox');
const metaArea = document.getElementById('metaArea');
const metaDark = document.getElementById('metaDark');
const metaFlag = document.getElementById('metaFlag');
const thumbsEl = document.getElementById('thumbs');
const subtitleEl = document.getElementById('subtitle');
const datasetSelect = document.getElementById('datasetSelect');

function currentCells() {{
  return DATASETS[currentDataset];
}}

function render() {{
  const cells = currentCells();
  const c = cells[currentIndex];
  const prefix = currentDataset + '/';

  cellImg.src = prefix + 'cells/' + c.filename;
  cellImg.alt = 'cell ' + c.cell_id;
  locationImg.src = prefix + 'locations/' + c.location_filename;
  locationImg.alt = 'location of cell ' + c.cell_id;

  positionEl.textContent = 'Cell ' + (currentIndex + 1) + ' / ' + cells.length;

  metaId.textContent = c.cell_id;
  metaBbox.textContent = '(' + c.x1 + ', ' + c.y1 + ') - (' + c.x2 + ', ' + c.y2 + ')';
  metaArea.textContent = c.area;
  metaDark.textContent = c.dark_fraction;
  metaFlag.textContent = c.flagged ? 'True' : 'False';

  if (c.flagged) {{
    flagBadge.classList.remove('hidden');
  }} else {{
    flagBadge.classList.add('hidden');
  }}

  // update thumbnail selection
  const thumbs = thumbsEl.querySelectorAll('.thumb');
  thumbs.forEach((t, i) => {{
    if (i === currentIndex) {{
      t.classList.add('selected');
      t.scrollIntoView({{ behavior: 'smooth', inline: 'center', block: 'nearest' }});
    }} else {{
      t.classList.remove('selected');
    }}
  }});
}}

function goTo(index) {{
  const cells = currentCells();
  if (index < 0) index = 0;
  if (index > cells.length - 1) index = cells.length - 1;
  currentIndex = index;
  render();
}}

function buildThumbs() {{
  thumbsEl.innerHTML = '';
  const prefix = currentDataset + '/';
  const frag = document.createDocumentFragment();
  currentCells().forEach((c, i) => {{
    const btn = document.createElement('button');
    btn.className = 'thumb' + (c.flagged ? ' flagged' : '');
    btn.title = 'Cell ' + c.cell_id + (c.flagged ? ' (flagged)' : '');
    btn.addEventListener('click', () => goTo(i));

    const img = document.createElement('img');
    img.src = prefix + 'cells/' + c.filename;
    img.alt = 'cell ' + c.cell_id + ' thumbnail';
    img.loading = 'lazy';

    btn.appendChild(img);
    frag.appendChild(btn);
  }});
  thumbsEl.appendChild(frag);
}}

function selectDataset(name) {{
  if (!DATASETS.hasOwnProperty(name)) return;
  currentDataset = name;
  currentIndex = 0;
  datasetSelect.value = name;
  subtitleEl.textContent = currentDataset + ' \\u2014 extracted cells with source-slide location';
  buildThumbs();
  render();
}}

function buildDatasetSelect() {{
  const frag = document.createDocumentFragment();
  DATASET_NAMES.forEach((name) => {{
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    frag.appendChild(opt);
  }});
  datasetSelect.appendChild(frag);
  datasetSelect.value = currentDataset;
}}

datasetSelect.addEventListener('change', (e) => selectDataset(e.target.value));

document.getElementById('prevBtn').addEventListener('click', () => goTo(currentIndex - 1));
document.getElementById('nextBtn').addEventListener('click', () => goTo(currentIndex + 1));

document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') {{ goTo(currentIndex - 1); }}
  else if (e.key === 'ArrowRight') {{ goTo(currentIndex + 1); }}
}});

buildDatasetSelect();
subtitleEl.textContent = currentDataset + ' \\u2014 extracted cells with source-slide location';
buildThumbs();
render();
</script>

</body>
</html>
"""


def build_gallery():
    datasets = discover_datasets(OUTPUT_ROOT)
    if not datasets:
        raise SystemExit(
            "No dataset folders with manifest.csv found under " + OUTPUT_ROOT
        )

    all_data = {}
    for name, manifest_path in datasets:
        all_data[name] = load_manifest_rows(manifest_path)

    datasets_json = json.dumps(all_data, separators=(",", ":"))
    html = HTML_TEMPLATE.format(datasets_json=datasets_json)

    with open(GALLERY_PATH, "w") as f:
        f.write(html)

    print("Wrote {} with {} dataset(s):".format(GALLERY_PATH, len(all_data)))
    for name in all_data:
        print("  - {}: {} cells".format(name, len(all_data[name])))


if __name__ == "__main__":
    build_gallery()
