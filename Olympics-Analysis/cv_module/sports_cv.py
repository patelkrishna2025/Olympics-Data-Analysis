"""
=============================================================
 Olympics Intelligence System
 MODULE: Computer Vision — Sports Image Analyser
 Operations: Color analysis, motion blur detection,
             crowd detection, dominant colours, filter gallery,
             Olympic-ring colour scoring, sport environment hints
=============================================================
"""
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SportsImageResult:
    width: int
    height: int
    brightness: float
    contrast: float
    dominant_colors: list        # top-3 hex
    color_mood: str              # e.g. "Indoor Arena / Outdoor Field"
    edge_density: float          # detail / action richness
    motion_blur_score: float     # 0=sharp, 1=blurry (laplacian variance)
    crowd_density: str           # Low / Medium / High estimate
    sport_env_hint: str          # Aquatics / Track / Stadium / Gym …
    olympic_ring_match: str      # dominant ring colour
    annotated_frame: Optional[np.ndarray] = field(default=None, repr=False)


def _rgb_to_hex(r, g, b) -> str:
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def _dominant_colors(img_rgb: np.ndarray, k: int = 3) -> list[str]:
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 6000:
        idx = np.random.choice(len(pixels), 6000, replace=False)
        pixels = pixels[idx]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    counts = np.bincount(labels.flatten())
    order  = np.argsort(-counts)
    return [_rgb_to_hex(*centers[i]) for i in order]


def _color_mood(img_rgb: np.ndarray) -> str:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    avg_v = hsv[:, :, 2].mean()
    avg_s = hsv[:, :, 1].mean()
    avg_h = hsv[:, :, 0].mean()
    if avg_v > 180 and 85 <= avg_h <= 140:
        return "🏊 Aquatics / Pool"
    if avg_v > 160 and 30 <= avg_h <= 85 and avg_s > 60:
        return "🏃 Outdoor Track / Field"
    if avg_v < 70:
        return "🏟️ Indoor Arena / Night Event"
    if avg_v > 180 and avg_s < 40:
        return "🤸 Gymnasium / Indoor"
    return "🏅 General Sports Venue"


def _motion_blur_score(gray: np.ndarray) -> float:
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalise: lower lap_var = more blur
    score = max(0.0, min(1.0, 1.0 - lap_var / 2000.0))
    return round(score, 3)


def _crowd_density(img_rgb: np.ndarray) -> str:
    """Estimate crowd density via texture complexity in upper region."""
    h, w = img_rgb.shape[:2]
    upper = img_rgb[:h//3, :, :]
    gray_up = cv2.cvtColor(upper, cv2.COLOR_RGB2GRAY)
    edges   = cv2.Canny(gray_up, 50, 150)
    density = edges.mean()
    if density > 30:
        return "🔴 High (crowded venue)"
    elif density > 12:
        return "🟡 Medium"
    return "🟢 Low (quiet / close-up)"


def _olympic_ring_match(img_rgb: np.ndarray) -> str:
    """Match image dominant hue to one of the 5 Olympic ring colours."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    avg_h = hsv[:, :, 0].mean()
    avg_s = hsv[:, :, 1].mean()
    avg_v = hsv[:, :, 2].mean()
    if avg_s < 40:
        return "⚪ White / Neutral"
    if avg_h < 15 or avg_h > 165:
        return "🔴 Red"
    if 15 <= avg_h < 35:
        return "🟡 Yellow / Gold"
    if 35 <= avg_h < 85:
        return "🟢 Green"
    if 85 <= avg_h < 130:
        return "🔵 Blue"
    return "⚫ Black / Dark"


def _sport_env_hint(mood: str, edge_density: float, blur_score: float) -> str:
    if "Aquatics" in mood or "Pool" in mood:
        return "🏊 Likely Aquatics / Swimming Event"
    if "Track" in mood or "Field" in mood:
        return "🏃 Likely Athletics / Track & Field"
    if "Indoor" in mood and edge_density > 0.15:
        return "🤸 Likely Gymnastics / Indoor Sport"
    if blur_score > 0.5:
        return "💨 Fast-action Sport (high motion blur)"
    if edge_density > 0.18:
        return "🥊 Combat or High-Detail Sport"
    return "🏅 General / Ceremonial"


class OlympicsCVAnalyser:
    """
    Analyse any uploaded sports image with full CV pipeline.
    No deep-learning — pure OpenCV classical methods.
    """

    def analyse(self, frame_bgr: np.ndarray) -> SportsImageResult:
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w    = gray.shape

        brightness    = round(float(gray.mean()), 1)
        contrast      = round(float(gray.std()), 1)
        dom_colors    = _dominant_colors(img_rgb)
        mood          = _color_mood(img_rgb)
        edges         = cv2.Canny(gray, 80, 180)
        edge_density  = round(float(edges.mean()) / 255.0, 4)
        blur_score    = _motion_blur_score(gray)
        crowd         = _crowd_density(img_rgb)
        ring_match    = _olympic_ring_match(img_rgb)
        sport_hint    = _sport_env_hint(mood, edge_density, blur_score)

        # Annotate — edge overlay in gold
        annotated = frame_bgr.copy()
        edge_col  = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        gold_mask = np.zeros_like(annotated)
        gold_mask[edges > 0] = [0, 215, 255]   # gold BGR
        annotated = cv2.addWeighted(annotated, 0.85, gold_mask, 0.35, 0)

        return SportsImageResult(
            width=w, height=h,
            brightness=brightness, contrast=contrast,
            dominant_colors=dom_colors,
            color_mood=mood,
            edge_density=edge_density,
            motion_blur_score=blur_score,
            crowd_density=crowd,
            sport_env_hint=sport_hint,
            olympic_ring_match=ring_match,
            annotated_frame=annotated,
        )

    @staticmethod
    def apply_filters(img_rgb: np.ndarray) -> dict[str, np.ndarray]:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        out  = {}

        out["Original"]          = img_rgb
        out["Grayscale"]         = gray
        out["Edge Detection"]    = cv2.Canny(gray, 80, 180)
        out["Gaussian Blur"]     = cv2.GaussianBlur(img_rgb, (13, 13), 0)
        out["Sharpen"]           = cv2.filter2D(
                                        img_rgb, -1,
                                        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                                   )
        out["Emboss"]            = cv2.filter2D(
                                        gray, -1,
                                        np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
                                   )
        out["Invert"]            = cv2.bitwise_not(img_rgb)

        # Sepia
        k = np.array([[0.272,0.534,0.131],
                      [0.349,0.686,0.168],
                      [0.393,0.769,0.189]])
        out["Sepia"]             = np.clip(img_rgb @ k.T, 0, 255).astype(np.uint8)

        # Olympic gold tint
        gold = img_rgb.copy().astype(np.float32)
        gold[:, :, 0] = np.clip(gold[:, :, 0] * 1.3, 0, 255)
        gold[:, :, 1] = np.clip(gold[:, :, 1] * 1.1, 0, 255)
        gold[:, :, 2] = np.clip(gold[:, :, 2] * 0.5, 0, 255)
        out["Olympic Gold Tint"] = gold.astype(np.uint8)

        # High contrast
        out["High Contrast"]     = cv2.convertScaleAbs(img_rgb, alpha=1.8, beta=-50)

        # Binary threshold
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        out["Binary Threshold"]  = thresh

        return out

    @staticmethod
    def pixel_stats(img_rgb: np.ndarray) -> list[dict]:
        return [
            {
                "Channel": ch,
                "Mean":    round(img_rgb[:, :, i].mean(), 2),
                "Std Dev": round(img_rgb[:, :, i].std(),  2),
                "Min":     int(img_rgb[:, :, i].min()),
                "Max":     int(img_rgb[:, :, i].max()),
            }
            for i, ch in enumerate(["Red", "Green", "Blue"])
        ]
