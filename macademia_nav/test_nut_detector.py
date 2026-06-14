import cv2
import numpy as np
import argparse
import sys
from dataclasses import dataclass
from typing import List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Config — tweak these without touching detection logic
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # HSV colour ranges — covers real macadamia nut shades (tan → dark brown)
    hsv_ranges: List[Tuple[np.ndarray, np.ndarray]] = None

    # Shape filters
    min_area:         int   = 300
    max_area:         int   = 60_000
    min_circularity:  float = 0.40
    min_aspect_ratio: float = 0.55
    max_aspect_ratio: float = 1.80
    min_solidity:     float = 0.75   # filled-ness; rejects crescents/shadows

    # Confidence thresholds
    high_conf_circularity: float = 0.65
    high_conf_solidity:    float = 0.85

    def __post_init__(self):
        if self.hsv_ranges is None:
            # Range 1 — light tan / raw macadamia shell
            # Range 2 — medium brown
            # Range 3 — dark roasted brown
            self.hsv_ranges = [
                (np.array([8,  40,  120]), np.array([22, 180, 255])),
                (np.array([5,  60,  60]),  np.array([20, 200, 180])),
                (np.array([3,  80,  30]),  np.array([15, 255, 120])),
            ]


# ──────────────────────────────────────────────────────────────────────────────
# Detection helpers
# ──────────────────────────────────────────────────────────────────────────────
def build_mask(hsv: np.ndarray, cfg: Config) -> np.ndarray:
    """Combine all HSV ranges into one cleaned mask."""
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in cfg.hsv_ranges:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))

    # Morphological cleanup
    k_open  = np.ones((3, 3), np.uint8)
    k_close = np.ones((7, 7), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open,  iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=2)
    return combined


def shape_metrics(contour) -> Tuple[float, float, float]:
    """Return (circularity, aspect_ratio, solidity) for a contour."""
    area      = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circ      = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0

    x, y, w, h  = cv2.boundingRect(contour)
    aspect      = (w / h) if h > 0 else 0.0

    hull         = cv2.convexHull(contour)
    hull_area    = cv2.contourArea(hull)
    solidity     = (area / hull_area) if hull_area > 0 else 0.0

    return circ, aspect, solidity


def confidence_score(area: float, circ: float, solidity: float,
                     cfg: Config) -> float:
    """
    Simple 0-100 confidence score combining three independent signals.
    Each contributes up to 33 points.
    """
    # Area score — peaks at midpoint of allowed range
    area_mid   = (cfg.min_area + cfg.max_area) / 2
    area_score = max(0.0, 1 - abs(area - area_mid) / area_mid) * 33

    # Circularity score
    circ_score = min(circ / 1.0, 1.0) * 34

    # Solidity score
    sol_score  = min(solidity / 1.0, 1.0) * 33

    return area_score + circ_score + sol_score


def multi_scale_detect(image: np.ndarray, cfg: Config):
    """
    Run detection at 3 scales and merge results.
    Helps catch both small and large nuts in the same frame.
    """
    all_boxes    = []
    all_scores   = []
    all_contours = []

    scales = [0.5, 1.0, 1.5]

    for scale in scales:
        if scale != 1.0:
            h, w   = image.shape[:2]
            resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        else:
            resized = image

        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask    = build_mask(hsv, cfg)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt) / (scale ** 2)   # normalise to original scale
            if not (cfg.min_area < area < cfg.max_area):
                continue

            circ, aspect, solidity = shape_metrics(cnt)
            if circ     < cfg.min_circularity:  continue
            if solidity < cfg.min_solidity:     continue
            if not (cfg.min_aspect_ratio < aspect < cfg.max_aspect_ratio): continue

            # Scale bounding box back to original image coords
            x, y, w, h = cv2.boundingRect(cnt)
            box = (
                int(x / scale), int(y / scale),
                int(w / scale), int(h / scale)
            )

            score = confidence_score(area, circ, solidity, cfg)
            all_boxes.append(box)
            all_scores.append(score)
            all_contours.append((cnt, scale))

    return all_boxes, all_scores


def non_max_suppression(boxes, scores, iou_thresh=0.4):
    """Remove duplicate detections from multi-scale overlaps."""
    if not boxes:
        return []

    boxes_arr  = np.array([[x, y, x+w, y+h] for x, y, w, h in boxes], dtype=float)
    scores_arr = np.array(scores, dtype=float)

    x1, y1, x2, y2 = boxes_arr[:,0], boxes_arr[:,1], boxes_arr[:,2], boxes_arr[:,3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores_arr.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w_inter = np.maximum(0, xx2 - xx1)
        h_inter = np.maximum(0, yy2 - yy1)
        inter   = w_inter * h_inter
        iou     = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        order = order[1:][iou < iou_thresh]

    return keep


# ──────────────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────────────
def draw_detections(image: np.ndarray, boxes, scores, cfg: Config) -> np.ndarray:
    output = image.copy()

    for (x, y, w, h), score in zip(boxes, scores):
        cx, cy = x + w // 2, y + h // 2

        # Colour by confidence
        if score >= 75:
            colour, label = (0, 255, 0),   "HIGH"
        elif score >= 50:
            colour, label = (0, 200, 255), "MED"
        else:
            colour, label = (0, 100, 255), "LOW"

        # Draw bounding box + crosshair
        cv2.rectangle(output, (x, y), (x + w, y + h), colour, 2)
        cv2.drawMarker(output, (cx, cy), colour,
                       cv2.MARKER_CROSS, markerSize=12, thickness=1)

        # Label
        tag = f"Nut {label} {score:.0f}%"
        (tw, th), _ = cv2.getTextSize(
            tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(output,
                      (x, y - th - 8), (x + tw + 4, y), colour, -1)
        cv2.putText(output, tag,
                    (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 2)

    # Summary banner
    n = len(boxes)
    banner = f"Nuts detected: {n}"
    cv2.rectangle(output, (0, 0), (300, 40), (30, 30, 30), -1)
    cv2.putText(output, banner,
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (255, 255, 255), 2)

    return output


def build_debug_panel(image: np.ndarray, cfg: Config) -> np.ndarray:
    """Return a side-by-side debug panel: original | mask | edges."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = build_mask(hsv, cfg)

    mask_bgr  = cv2.cvtColor(mask,  cv2.COLOR_GRAY2BGR)
    edges     = cv2.Canny(mask, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    h = image.shape[0]
    panel = np.hstack([
        cv2.resize(image,     (int(image.shape[1] * 300 / h), 300)),
        cv2.resize(mask_bgr,  (int(mask_bgr.shape[1]  * 300 / h), 300)),
        cv2.resize(edges_bgr, (int(edges_bgr.shape[1] * 300 / h), 300)),
    ])

    for i, label in enumerate(["Original", "HSV Mask", "Edges"]):
        cv2.putText(panel, label,
                    (i * (panel.shape[1] // 3) + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return panel


# ──────────────────────────────────────────────────────────────────────────────
# HSV calibration helper (run with --calibrate)
# ──────────────────────────────────────────────────────────────────────────────
def calibrate(image_path: str) -> None:
    """Click pixels on the nut to read their HSV values for tuning."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot open {image_path}")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h, s, v = hsv[y, x]
            print(f"  Pixel ({x},{y})  HSV = ({h}, {s}, {v})")
            print(f"  Suggested range: "
                  f"lower=({max(0,h-15)}, {max(0,s-40)}, {max(0,v-40)})  "
                  f"upper=({min(180,h+15)}, 255, 255)")

    cv2.imshow("Calibrate — click on nuts", image)
    cv2.setMouseCallback("Calibrate — click on nuts", on_click)
    print("Click on nut pixels. Press Q to quit.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Macadamia Nut Detector v3")
    parser.add_argument("image",       nargs="?", default="test_image.jpg")
    parser.add_argument("--calibrate", action="store_true",
                        help="Click-to-sample HSV calibration mode")
    parser.add_argument("--debug",     action="store_true",
                        help="Show mask + edge debug panel")
    parser.add_argument("--save",      metavar="PATH",
                        help="Save output image to file")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: cannot open '{args.image}'")
        sys.exit(1)

    if args.calibrate:
        calibrate(args.image)
        return

    cfg = Config()

    # ── Detect ──────────────────────────────────────────────────────────────
    boxes, scores = multi_scale_detect(image, cfg)
    keep          = non_max_suppression(boxes, scores, iou_thresh=0.4)

    final_boxes  = [boxes[i]  for i in keep]
    final_scores = [scores[i] for i in keep]

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Nuts detected: {len(final_boxes)}")
    for i, (box, sc) in enumerate(zip(final_boxes, final_scores)):
        x, y, w, h = box
        print(f"  Nut {i+1}: centre=({x+w//2}, {y+h//2})  "
              f"size={w}×{h}px  confidence={sc:.1f}%")
    print(f"{'─'*50}\n")

    # ── Draw & show ─────────────────────────────────────────────────────────
    output = draw_detections(image, final_boxes, final_scores, cfg)

    cv2.imshow("Nut Detection v3", output)

    if args.debug:
        debug = build_debug_panel(image, cfg)
        cv2.imshow("Debug Panel", debug)

    if args.save:
        cv2.imwrite(args.save, output)
        print(f"Saved to {args.save}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()