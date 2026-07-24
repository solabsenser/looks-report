import cv2
import numpy as np


def generate_mesh_overlay(image_path, landmarks):

    image = cv2.imread(image_path)

    if image is None:
        return None

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

    output = image_path.replace(
        ".jpg",
        "_mesh.jpg"
    )

    cv2.imwrite(output, image)

    return output


def generate_heatmap(
    image_path,
    landmarks,
    scores
):

    image = cv2.imread(image_path)

    if image is None:
        return None

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

    cv2.circle(
        overlay,
        left_eye,
        35,
        score_color(eye_score),
        -1
    )

    cv2.circle(
        overlay,
        right_eye,
        35,
        score_color(eye_score),
        -1
    )

    cv2.circle(
        overlay,
        nose,
        40,
        score_color(nose_score),
        -1
    )

    cv2.circle(
        overlay,
        chin,
        60,
        score_color(jaw_score),
        -1
    )

    image = cv2.addWeighted(
        overlay,
        0.35,
        image,
        0.65,
        0
    )

    output = image_path.replace(
        ".jpg",
        "_heatmap.jpg"
    )

    cv2.imwrite(output, image)

    return output


def generate_debug_overlay(
    image_path,
    landmarks
):

    image = cv2.imread(image_path)

    if image is None:
        return None

    h, w = image.shape[:2]

    forehead = (
        int(landmarks[10].x * w),
        int(landmarks[10].y * h)
    )

    chin = (
        int(landmarks[152].x * w),
        int(landmarks[152].y * h)
    )

    left_face = (
        int(landmarks[234].x * w),
        int(landmarks[234].y * h)
    )

    right_face = (
        int(landmarks[454].x * w),
        int(landmarks[454].y * h)
    )

    cv2.line(
        image,
        forehead,
        chin,
        (255, 0, 0),
        2
    )

    cv2.line(
        image,
        left_face,
        right_face,
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        "FACE HEIGHT",
        forehead,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    cv2.putText(
        image,
        "FACE WIDTH",
        left_face,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    output = image_path.replace(
        ".jpg",
        "_debug.jpg"
    )

    cv2.imwrite(output, image)

    return output

def generate_premium_report(
    original_path,
    mesh_path,
    heatmap_path,
    debug_path
):

    import cv2
    import numpy as np

    original = cv2.imread(original_path)
    mesh = cv2.imread(mesh_path)
    heatmap = cv2.imread(heatmap_path)
    debug = cv2.imread(debug_path)

    if (
        original is None
        or mesh is None
        or heatmap is None
        or debug is None
    ):
        return None

    h, w = original.shape[:2]

    mesh = cv2.resize(mesh, (w, h))
    heatmap = cv2.resize(heatmap, (w, h))
    debug = cv2.resize(debug, (w, h))

    cv2.putText(
        mesh,
        "FaceMesh",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        heatmap,
        "Heatmap",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        debug,
        "Debug",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        original,
        "Original",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    top = np.hstack([
        mesh,
        heatmap
    ])

    bottom = np.hstack([
        debug,
        original
    ])

    final = np.vstack([
        top,
        bottom
    ])

    output = original_path.replace(
        ".jpg",
        "_premium.jpg"
    )

    cv2.imwrite(
        output,
        final
    )

    return output
