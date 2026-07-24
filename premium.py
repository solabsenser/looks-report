from pathlib import Path

import cv2
import numpy as np

JPEG_QUALITY = 75
MAX_REPORT_SIDE = 1200


def _jpeg_path(image_path, suffix):
    path = Path(image_path)
    return str(path.with_name(f"{path.stem}{suffix}.jpg"))


def _write_jpeg(path, image):
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return path


def _resize_to_limit(image, max_side=MAX_REPORT_SIDE):
    h, w = image.shape[:2]
    longest_side = max(h, w)

    if longest_side <= max_side:
        return image

    scale = max_side / longest_side
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def generate_mesh_overlay(image_path, landmarks):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image = _resize_to_limit(image)
    h, w = image.shape[:2]

    for lm in landmarks:

        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(
            image,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )

    output = _jpeg_path(image_path, "_mesh")
    return _write_jpeg(output, image)


def generate_heatmap(
    image_path,
    landmarks,
    scores
):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image = _resize_to_limit(image)
    overlay = image.copy()

    h, w = image.shape[:2]

    eye_score = scores["eye_score"]
    nose_score = scores["nose_score"]
    jaw_score = scores["jaw_score"]

    def score_color(score):

        if score >= 7.5:
            return (0, 255, 0)

        elif score >= 6:
            return (0, 255, 255)

        return (0, 0, 255)

    left_eye = (
        int(landmarks[33].x * w),
        int(landmarks[33].y * h)
    )

    right_eye = (
        int(landmarks[263].x * w),
        int(landmarks[263].y * h)
    )

    nose = (
        int(landmarks[1].x * w),
        int(landmarks[1].y * h)
    )

    chin = (
        int(landmarks[152].x * w),
        int(landmarks[152].y * h)
    )

    cv2.circle(overlay, left_eye, 35, score_color(eye_score), -1)
    cv2.circle(overlay, right_eye, 35, score_color(eye_score), -1)
    cv2.circle(overlay, nose, 40, score_color(nose_score), -1)
    cv2.circle(overlay, chin, 60, score_color(jaw_score), -1)

    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)

    output = _jpeg_path(image_path, "_heatmap")
    return _write_jpeg(output, image)


def generate_debug_overlay(image_path, landmarks):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image = _resize_to_limit(image)
    h, w = image.shape[:2]

    forehead = (int(landmarks[10].x * w), int(landmarks[10].y * h))
    chin = (int(landmarks[152].x * w), int(landmarks[152].y * h))
    left_face = (int(landmarks[234].x * w), int(landmarks[234].y * h))
    right_face = (int(landmarks[454].x * w), int(landmarks[454].y * h))

    cv2.line(image, forehead, chin, (255, 0, 0), 2)
    cv2.line(image, left_face, right_face, (0, 255, 0), 2)
    cv2.putText(image, "FACE HEIGHT", forehead, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(image, "FACE WIDTH", left_face, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output = _jpeg_path(image_path, "_debug")
    return _write_jpeg(output, image)


def generate_premium_report(original_path, mesh_path, heatmap_path, debug_path):

    if not all([original_path, mesh_path, heatmap_path, debug_path]):
        return None

    original = cv2.imread(original_path)
    mesh = cv2.imread(mesh_path)
    heatmap = cv2.imread(heatmap_path)
    debug = cv2.imread(debug_path)

    if original is None or mesh is None or heatmap is None or debug is None:
        return None

    original = _resize_to_limit(original)
    h, w = original.shape[:2]

    mesh = cv2.resize(mesh, (w, h), interpolation=cv2.INTER_AREA)
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_AREA)
    debug = cv2.resize(debug, (w, h), interpolation=cv2.INTER_AREA)

    cv2.putText(mesh, "FaceMesh", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(heatmap, "Heatmap", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(debug, "Debug", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(original, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    final = np.vstack([np.hstack([mesh, heatmap]), np.hstack([debug, original])])

    output = _jpeg_path(original_path, "_premium")
    return _write_jpeg(output, final)
