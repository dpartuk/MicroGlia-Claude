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


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def build_cyan_mask(img_bgr, hue_low, hue_high, min_component_area):
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

    return remove_small_components(mask, min_component_area)


def remove_small_components(mask, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def estimate_stroke_half_width(mask):
    """Median distance-to-background over foreground pixels approximates
    half the drawn outline's stroke thickness."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    nonzero = dist[mask > 0]
    if nonzero.size == 0:
        return 1.0
    return max(1.0, float(np.median(nonzero)))


def close_mask_gaps(mask, stroke_half_width):
    kernel_size = int(round(stroke_half_width * 2)) * 2 + 1
    kernel_size = max(3, min(kernel_size, 15))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
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


def region_dark_fraction(gray, region_pixel_mask, global_dark_thresh):
    """A real cell interior contains dark phase-contrast structure; a small
    background pocket trapped between several touching outlines does not.
    Compare against a threshold computed once for the whole image (Otsu),
    rather than per-region, since many enclosed regions are too small/uniform
    for their own Otsu split to be meaningful."""
    values = gray[region_pixel_mask]
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(values <= global_dark_thresh)) / values.size


def build_cell_mask(region_pixel_mask, cyan_mask):
    """Interior region plus the drawn outline pixels immediately bordering
    it, so the crop includes the cyan boundary rather than stopping just
    inside it."""
    region_u8 = (region_pixel_mask.astype(np.uint8)) * 255
    dilated = cv2.dilate(region_u8, np.ones((3, 3), np.uint8), iterations=2)
    bordering_outline = cv2.bitwise_and(dilated, cyan_mask)
    return cv2.bitwise_or(region_u8, bordering_outline)


def count_dark_bodies(gray, cell_mask, min_body_area=150):
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

    num, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    bodies = sum(1 for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_body_area)
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
    args = parser.parse_args()

    img = load_image(args.image)
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    output_dir = args.output or os.path.join("Data", "output", base_name)
    cells_dir = os.path.join(output_dir, "cells")
    locations_dir = os.path.join(output_dir, "locations")
    os.makedirs(cells_dir, exist_ok=True)
    os.makedirs(locations_dir, exist_ok=True)

    cyan_mask = build_cyan_mask(img, args.hue_low, args.hue_high, args.min_component_area)
    cv2.imwrite(os.path.join(output_dir, "cyan_mask_debug.jpg"), cyan_mask)

    stroke_half_width = estimate_stroke_half_width(cyan_mask)
    closed_mask, kernel_size = close_mask_gaps(cyan_mask, stroke_half_width)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global_dark_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    labels, stats, candidate_ids = find_enclosed_regions(closed_mask, args.absolute_min_area)

    manifest_rows = []
    accepted_bboxes = []
    rejected_gap_bboxes = []
    flagged_ids = set()
    idx = 0

    for region_id in candidate_ids:
        region_pixel_mask = labels == region_id
        x, y, w, h, area = stats[region_id]

        dark_fraction = region_dark_fraction(gray, region_pixel_mask, global_dark_thresh)
        if dark_fraction < args.min_dark_fraction:
            rejected_gap_bboxes.append((x, y, x + w, y + h))
            continue

        cell_mask = build_cell_mask(region_pixel_mask, cyan_mask)
        composite, (x1, y1, x2, y2) = crop_and_composite(img, cell_mask, (x, y, w, h), args.padding)

        num_bodies = count_dark_bodies(gray, cell_mask)
        flagged = num_bodies > 1
        if flagged:
            flagged_ids.add(idx)

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


if __name__ == "__main__":
    main()
