"""Extract cyan-outlined cells from a microscopy image onto white-background crops.

Cyan detection and morphology parameters are derived from the image itself
(Otsu thresholding, measured outline stroke width, contour-area statistics)
rather than hardcoded constants, so the pipeline isn't tuned to one image.
"""

import argparse
import csv
import os

import cv2
import numpy as np


class DebugSession:
    """Pipeline-wide debug state, shared by every cell being traced.

    Debugging a ~100-cell field at full frame is useless -- you cannot see
    which pixels drove a decision. So every step is written cropped to a
    cell's own neighbourhood (plus context, to show the walls it shares with
    neighbours) and upscaled, since a cell is often only ~40px across.

    The global stages (cyan mask, closing, region labelling) produce one
    frame-wide image each, identical for every cell. They are held here once
    and cropped per cell at write time; buffering them per cell would
    multiply memory by the number of cells traced.

    Files are named `NN_<what>__<function>.png` so each artifact points at the
    function in this file that produced it. PNG, not JPG: these are masks, and
    JPEG ringing would invent pixels that were never there. The one exception
    is the whole-slide orientation shot, which is a photograph and would cost
    ~1MB per cell as PNG.
    """

    def __init__(self, output_dir, wanted, context, scale):
        self.output_dir = output_dir
        self.wanted = wanted        # None => trace every accepted cell
        self.context = context
        self.scale = scale
        self.global_steps = {}
        self.global_notes = []
        self.written = []

    def wants(self, cell_id):
        return self.wanted is None or cell_id in self.wanted

    # Same add()/note() surface a CellRecorder exposes, so the global stages
    # can take a session wherever a per-cell recorder would go.
    def add(self, name, image, full_frame=True, scale=True):
        self.global_steps[name] = (image, full_frame, scale)

    def note(self, key, value):
        self.global_notes.append((key, value))

    def recorder(self, cell_id):
        return CellRecorder(self, cell_id)


class CellRecorder:
    """Per-cell steps for one cell, plus the writer for its trace folder.

    A recorder is created per candidate region and only flushed once the
    region is accepted, so a region that is rejected and whose id passes to a
    later region simply leaves nothing behind.
    """

    def __init__(self, session, cell_id):
        self.session = session
        self.cell_id = cell_id
        self.steps = {}
        self.notes = []
        self.crop_box = None
        self.dir = None

    def add(self, name, image, full_frame=True, scale=True):
        """full_frame=False for images already in crop coordinates; scale=False
        for whole-slide views, meant to be seen wide rather than magnified."""
        self.steps[name] = (image, full_frame, scale)

    def note(self, key, value):
        self.notes.append((key, value))

    def set_target(self, bbox, img_shape):
        x, y, w, h = bbox
        img_h, img_w = img_shape[:2]
        pad = self.session.context
        self.crop_box = (
            max(0, x - pad),
            max(0, y - pad),
            min(img_w, x + w + pad),
            min(img_h, y + h + pad),
        )
        self.dir = os.path.join(self.session.output_dir, "debug",
                                f"cell_{self.cell_id:03d}")

    def _prepare(self, image, full_frame, allow_scale):
        if full_frame:
            x1, y1, x2, y2 = self.crop_box
            image = image[y1:y2, x1:x2]
        if not allow_scale:
            return image
        scale = self.session.scale
        if scale <= 0:  # auto: make the longest side at least ~420px
            longest = max(image.shape[0], image.shape[1], 1)
            scale = max(1, int(round(420 / longest)))
        if scale != 1:
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)
        return image

    def flush(self):
        os.makedirs(self.dir, exist_ok=True)
        all_steps = dict(self.session.global_steps)
        all_steps.update(self.steps)

        for name, (image, full_frame, scale) in all_steps.items():
            params = [cv2.IMWRITE_JPEG_QUALITY, 82] if name.lower().endswith(".jpg") else []
            cv2.imwrite(os.path.join(self.dir, name),
                        self._prepare(image, full_frame, scale), params)

        x1, y1, x2, y2 = self.crop_box
        with open(os.path.join(self.dir, "summary.txt"), "w") as f:
            f.write(f"Debug trace for cell {self.cell_id}\n")
            f.write(f"debug crop window (x1,y1,x2,y2): ({x1}, {y1}, {x2}, {y2})\n")
            f.write(f"context padding: {self.session.context}px\n\n")
            f.write("Values computed per step:\n")
            for key, value in self.session.global_notes + self.notes:
                f.write(f"  {key}: {value}\n")
            f.write("\nSteps (file -> function in extract_cells.py):\n")
            for name in sorted(all_steps):
                stem = os.path.splitext(name)[0]
                func = stem.rsplit("__", 1)[-1] if "__" in stem else "-"
                f.write(f"  {name} -> {func}()\n")

        self.session.written.append(self.cell_id)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def build_cyan_mask(img_bgr, hue_low, hue_high, min_component_area, debug=None):
    """Isolate cyan outline pixels via an Otsu threshold on saturation*value,
    computed only over hue-band pixels so it adapts to this image's own
    brightness/contrast rather than assuming a fixed cutoff."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hue_band = cv2.inRange(h, hue_low, hue_high)

    sv_score = (s.astype(np.float32) * v.astype(np.float32)) / 255.0
    sv_score_u8 = np.clip(sv_score, 0, 255).astype(np.uint8)

    band_pixels = sv_score_u8[hue_band > 0]
    if band_pixels.size == 0:
        return np.zeros_like(hue_band)

    otsu_thresh, _ = cv2.threshold(
        band_pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = np.zeros_like(hue_band)
    mask[(hue_band > 0) & (sv_score_u8 >= otsu_thresh)] = 255

    if debug:
        debug.add("02_hue_band__build_cyan_mask.png", hue_band)
        debug.add("03_sv_score__build_cyan_mask.png", sv_score_u8)
        debug.add("04_cyan_raw__build_cyan_mask.png", mask)
        debug.note("hue band", f"{hue_low}-{hue_high}")
        debug.note("otsu threshold on S*V (within hue band)", f"{otsu_thresh:.1f}")
        debug.note("cyan pixels before denoise", int(np.count_nonzero(mask)))

    cleaned = remove_small_components(mask, min_component_area)

    if debug:
        debug.add("05_cyan_denoised__remove_small_components.png", cleaned)
        debug.note("cyan pixels after denoise", int(np.count_nonzero(cleaned)))
        debug.note("min component area", min_component_area)

    return cleaned


def remove_small_components(mask, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def estimate_stroke_half_width(mask, debug=None):
    """Median distance-to-background over foreground pixels approximates
    half the drawn outline's stroke thickness."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    nonzero = dist[mask > 0]
    if debug:
        vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        debug.add("06_distance_transform__estimate_stroke_half_width.png", vis)
    if nonzero.size == 0:
        return 1.0
    return max(1.0, float(np.median(nonzero)))


def close_mask_gaps(mask, stroke_half_width, debug=None):
    kernel_size = int(round(stroke_half_width * 2)) * 2 + 1
    kernel_size = max(3, min(kernel_size, 15))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if debug:
        debug.add("07_closed__close_mask_gaps.png", closed)
        # Which pixels the closing invented -- the step most likely to fuse
        # two neighbouring cells into one region.
        debug.add("08_closing_added__close_mask_gaps.png",
                  cv2.subtract(closed, mask))
        debug.note("stroke half-width (median dist transform)", f"{stroke_half_width:.2f}px")
        debug.note("closing kernel", f"{kernel_size}x{kernel_size} ellipse, 2 iterations")
    return closed, kernel_size


def find_enclosed_regions(closed_mask, absolute_min_area):
    """Each cell's outline is a closed loop, but adjacent cells' loops often
    touch or share a wall -- so `findContours(RETR_EXTERNAL)` sees a whole
    touching cluster as a single outer boundary and treats the individual
    cells inside it as ignored 'holes'. Instead, flood-fill the *background*
    around the cyan lines: every area fully walled off by cyan pixels is its
    own connected component regardless of how many neighbors it touches,
    which is what actually separates one cell from the next in this image.

    4-connectivity is required here (not 8) so a diagonal run of cyan pixels
    still blocks the fill from leaking corner-to-corner between two cells
    that only touch at a point.
    """
    non_cyan = cv2.bitwise_not(closed_mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(non_cyan, connectivity=4)

    img_h, img_w = closed_mask.shape[:2]
    enclosed_ids = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < absolute_min_area:
            continue
        touches_border = x <= 0 or y <= 0 or x + w >= img_w or y + h >= img_h
        if touches_border:
            continue
        enclosed_ids.append(i)
    return labels, stats, enclosed_ids


def region_dark_fraction(gray, region_pixel_mask, global_dark_thresh, debug=None):
    """A real cell interior contains dark phase-contrast structure; a small
    background pocket trapped between several touching outlines does not.
    Compare against a threshold computed once for the whole image (Otsu),
    rather than per-region, since many enclosed regions are too small/uniform
    for their own Otsu split to be meaningful."""
    values = gray[region_pixel_mask]
    if values.size == 0:
        return 0.0
    if debug:
        dark = np.zeros_like(gray)
        dark[region_pixel_mask & (gray <= global_dark_thresh)] = 255
        debug.add("11_dark_pixels__region_dark_fraction.png", dark)
        debug.note("global dark threshold (Otsu on grayscale)", int(global_dark_thresh))
    return float(np.count_nonzero(values <= global_dark_thresh)) / values.size


def build_cell_mask(region_pixel_mask, cyan_mask, debug=None):
    """Interior region plus the drawn outline pixels immediately bordering
    it, so the crop includes the cyan boundary rather than stopping just
    inside it."""
    region_u8 = (region_pixel_mask.astype(np.uint8)) * 255
    dilated = cv2.dilate(region_u8, np.ones((3, 3), np.uint8), iterations=2)
    bordering_outline = cv2.bitwise_and(dilated, cyan_mask)
    cell_mask = cv2.bitwise_or(region_u8, bordering_outline)
    if debug:
        debug.add("12_bordering_outline__build_cell_mask.png", bordering_outline)
        debug.add("13_cell_mask__build_cell_mask.png", cell_mask)
        debug.note("cell mask area (px)", int(np.count_nonzero(cell_mask)))
    return cell_mask


def count_dark_bodies(gray, cell_mask, min_body_area=150, debug=None):
    """Advisory check for regions that likely still contain more than one
    cell (e.g. two loops joined by an over-eager gap-close): count distinct,
    reasonably-sized dark blobs inside the final mask via a local Otsu split."""
    region_pixels = gray[cell_mask > 0]
    if region_pixels.size == 0:
        return 0

    otsu_thresh, _ = cv2.threshold(
        region_pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    dark = np.zeros_like(gray)
    dark[(cell_mask > 0) & (gray <= otsu_thresh)] = 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    bodies = sum(1 for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_body_area)

    if debug:
        vis = np.zeros_like(gray)
        for i in range(1, num):
            # Blobs big enough to count as a cell body are white; the rejected
            # speckle is dim, so an "almost 2 cells" call is visible.
            vis[labels == i] = 255 if stats[i, cv2.CC_STAT_AREA] >= min_body_area else 80
        debug.add("15_dark_bodies__count_dark_bodies.png", vis)
        debug.note("min body area", min_body_area)
        debug.note("dark bodies counted (>1 => flagged as merge)", max(bodies, 1))

    return max(bodies, 1)


def crop_and_composite(img, cell_mask, bbox, padding):
    img_h, img_w = img.shape[:2]
    x, y, w, h = bbox
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    crop_img = img[y1:y2, x1:x2]
    crop_mask = cell_mask[y1:y2, x1:x2]
    mask_3ch = cv2.merge([crop_mask, crop_mask, crop_mask])

    white_canvas = np.full_like(crop_img, 255)
    composite = np.where(mask_3ch > 0, crop_img, white_canvas)
    return composite, (x1, y1, x2, y2)


def draw_location_image(img, bbox, color=(0, 200, 0)):
    x1, y1, x2, y2 = bbox
    marked = img.copy()
    cv2.rectangle(marked, (x1, y1), (x2, y2), color, 3)
    return marked


def draw_debug_overview(img, accepted_bboxes, rejected_bboxes, flagged_ids):
    debug = img.copy()
    legend = "green=accepted  amber=flagged possible merge  red=rejected (enclosed, not enough cell content)"
    cv2.putText(debug, legend, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(debug, legend, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for idx, (x1, y1, x2, y2) in enumerate(accepted_bboxes):
        color = (0, 165, 255) if idx in flagged_ids else (0, 200, 0)
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        cv2.putText(debug, str(idx), (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    for x1, y1, x2, y2 in rejected_bboxes:
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return debug


def main():
    parser = argparse.ArgumentParser(
        description="Extract cyan-outlined cells from a microscopy image onto white-background crops"
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("-o", "--output", default=None,
                         help="Output directory (default: Data/output/<image_basename>)")
    parser.add_argument("--hue-low", type=int, default=70,
                         help="Lower bound of cyan hue band, OpenCV 0-179 scale (default: 70)")
    parser.add_argument("--hue-high", type=int, default=110,
                         help="Upper bound of cyan hue band (default: 110)")
    parser.add_argument("--min-component-area", type=int, default=15,
                         help="Drop raw mask components smaller than this before closing (default: 15)")
    parser.add_argument("--absolute-min-area", type=int, default=80,
                         help="Minimum enclosed-region area in px to consider as a candidate cell (default: 80)")
    parser.add_argument("--min-dark-fraction", type=float, default=0.05,
                         help="Minimum fraction of a region's pixels that must be 'dark' "
                              "(phase-contrast cell structure) to count as a cell, not an "
                              "empty gap between touching outlines (default: 0.05)")
    parser.add_argument("--padding", type=int, default=10,
                         help="Padding in px around each cropped cell (default: 10)")
    parser.add_argument("--debug-cell", type=int, nargs="+", default=None, metavar="ID",
                         help="Dump every pipeline step for these cell ids into "
                              "<output>/debug/cell_<ID>/. Normal output is unaffected.")
    parser.add_argument("--debug-all", action="store_true",
                         help="Write a debug trace for every extracted cell. Slower and "
                              "produces ~18 images per cell.")
    parser.add_argument("--debug-context", type=int, default=40,
                         help="Px of surrounding context in debug crops, so shared walls "
                              "with neighbouring cells are visible (default: 40)")
    parser.add_argument("--debug-scale", type=int, default=0,
                         help="Upscale factor for debug images; 0 = auto-fit to ~420px "
                              "(default: 0). Nearest-neighbour, so pixels stay crisp.")
    args = parser.parse_args()

    debug_session = None
    if args.debug_all or args.debug_cell:
        wanted = None if args.debug_all else set(args.debug_cell)
        debug_session = DebugSession(None, wanted, args.debug_context, args.debug_scale)

    img = load_image(args.image)
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    output_dir = args.output or os.path.join("Data", "output", base_name)
    cells_dir = os.path.join(output_dir, "cells")
    locations_dir = os.path.join(output_dir, "locations")
    os.makedirs(cells_dir, exist_ok=True)
    os.makedirs(locations_dir, exist_ok=True)

    if debug_session:
        debug_session.output_dir = output_dir
        debug_session.add("01_input__load_image.png", img)

    cyan_mask = build_cyan_mask(img, args.hue_low, args.hue_high,
                                args.min_component_area, debug=debug_session)
    cv2.imwrite(os.path.join(output_dir, "cyan_mask_debug.jpg"), cyan_mask)

    stroke_half_width = estimate_stroke_half_width(cyan_mask, debug=debug_session)
    closed_mask, kernel_size = close_mask_gaps(cyan_mask, stroke_half_width,
                                               debug=debug_session)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global_dark_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    labels, stats, candidate_ids = find_enclosed_regions(closed_mask, args.absolute_min_area)

    # The "all candidate regions" backdrop is the same for every cell, so it is
    # painted once here; each traced cell only overlays its own region on a
    # copy. Rebuilding it per cell would be ~100 full-frame passes.
    regions_base = None
    if debug_session:
        regions_base = np.full(img.shape[:2], 40, dtype=np.uint8)
        for region_id in candidate_ids:
            regions_base[labels == region_id] = 130
        regions_base[closed_mask > 0] = 0

    manifest_rows = []
    accepted_bboxes = []
    rejected_gap_bboxes = []
    flagged_ids = set()
    idx = 0

    for region_id in candidate_ids:
        region_pixel_mask = labels == region_id
        x, y, w, h, area = stats[region_id]

        # `idx` is the id this region will receive if it passes the filter
        # below, so a recorder can be opened before any per-cell work runs.
        # It is only flushed on acceptance, so a rejected region leaves
        # nothing behind even though it briefly held the id.
        recorder = None
        if debug_session is not None and debug_session.wants(idx):
            recorder = debug_session.recorder(idx)
            recorder.set_target((x, y, w, h), img.shape)

            # Orientation shots, before any processing is shown. Every later
            # step is a tight crop of a busy neighbourhood, which is useless
            # without first establishing which box in it is the subject.
            dx1, dy1, dx2, dy2 = recorder.crop_box
            slide = img.copy()
            cv2.rectangle(slide, (dx1, dy1), (dx2, dy2), (0, 200, 255), 3)
            cv2.rectangle(slide, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(slide, f"cell {idx}", (dx1, max(16, dy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
            # Downscaled: this only has to answer "whereabouts on the slide",
            # and at full resolution it would dominate the trace's size --
            # with --debug-all that is one full-frame image per cell.
            slide_scale = min(1.0, 900 / max(slide.shape[0], slide.shape[1]))
            if slide_scale < 1.0:
                slide = cv2.resize(slide, None, fx=slide_scale, fy=slide_scale,
                                   interpolation=cv2.INTER_AREA)
            recorder.add("00a_target_on_slide__set_target.jpg", slide,
                         full_frame=False, scale=False)

            in_context = img.copy()
            cv2.rectangle(in_context, (x, y), (x + w, y + h), (255, 0, 255), 1)
            recorder.add("00b_target_in_context__set_target.png", in_context)

            region_vis = regions_base.copy()
            region_vis[region_pixel_mask] = 255
            recorder.add("09_regions__find_enclosed_regions.png", region_vis)
            recorder.add("10_target_region__find_enclosed_regions.png",
                         region_pixel_mask.astype(np.uint8) * 255)
            recorder.note("region bbox (x, y, w, h)", f"({x}, {y}, {w}, {h})")
            recorder.note("region area (px)", int(area))

        dark_fraction = region_dark_fraction(gray, region_pixel_mask,
                                             global_dark_thresh, debug=recorder)
        if dark_fraction < args.min_dark_fraction:
            rejected_gap_bboxes.append((x, y, x + w, y + h))
            continue

        cell_mask = build_cell_mask(region_pixel_mask, cyan_mask, debug=recorder)
        composite, (x1, y1, x2, y2) = crop_and_composite(img, cell_mask, (x, y, w, h), args.padding)

        num_bodies = count_dark_bodies(gray, cell_mask, debug=recorder)
        flagged = num_bodies > 1
        if flagged:
            flagged_ids.add(idx)

        if recorder:
            outline = img.copy()
            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(outline, contours, -1, (255, 0, 255), 1)
            recorder.add("14_mask_outline_on_input__build_cell_mask.png", outline)
            recorder.add("16_composite__crop_and_composite.png", composite, full_frame=False)
            recorder.note("dark fraction", f"{dark_fraction:.3f}")
            recorder.note("crop bbox with padding (x1, y1, x2, y2)", f"({x1}, {y1}, {x2}, {y2})")
            recorder.note("flagged as possible merge", flagged)
            recorder.flush()

        filename = f"cell_{idx:03d}.jpg"
        cv2.imwrite(os.path.join(cells_dir, filename), composite)

        location_filename = f"cell_{idx:03d}_location.jpg"
        location_color = (0, 165, 255) if flagged else (0, 200, 0)
        location_img = draw_location_image(img, (x1, y1, x2, y2), color=location_color)
        cv2.imwrite(os.path.join(locations_dir, location_filename), location_img)

        accepted_bboxes.append((x1, y1, x2, y2))
        manifest_rows.append({
            "cell_id": idx,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "area": int(area),
            "dark_fraction": round(dark_fraction, 3),
            "flagged_possible_merge": flagged,
            "filename": filename,
            "location_filename": location_filename,
        })
        idx += 1

    debug_img = draw_debug_overview(img, accepted_bboxes, rejected_gap_bboxes, flagged_ids)
    cv2.imwrite(os.path.join(output_dir, "debug_overview.jpg"), debug_img)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["cell_id", "x1", "y1", "x2", "y2", "area", "dark_fraction",
                           "flagged_possible_merge", "filename", "location_filename"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Estimated outline stroke half-width: {stroke_half_width:.1f}px "
          f"(closing kernel: {kernel_size}x{kernel_size})")
    print(f"Accepted: {idx} cells ({len(flagged_ids)} flagged as possible merges)")
    print(f"Rejected (enclosed but not enough dark/cell content): {len(rejected_gap_bboxes)}")
    print(f"Output written to: {output_dir}")

    if debug_session:
        written = debug_session.written
        debug_dir = os.path.join(output_dir, "debug")
        if written:
            print(f"Debug traces written for {len(written)} cell(s) -> {debug_dir}")
        if debug_session.wanted is not None:
            missing = sorted(debug_session.wanted - set(written))
            if missing:
                print(f"Debug: no such cell id(s) {missing} "
                      f"({idx} cells extracted, ids 0-{max(idx - 1, 0)}).")


if __name__ == "__main__":
    main()
