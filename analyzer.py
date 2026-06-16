import os
import re
import traceback
from pathlib import Path
from threading import Lock
from urllib.request import urlretrieve
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Меняем импорт с Google на Groq
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = Path(
    os.getenv("FACE_LANDMARKER_MODEL_PATH", "models/face_landmarker.task")
)
MODEL_DOWNLOAD_LOCK = Lock()
ANALYZER_BACKEND = "mediapipe-tasks-face-landmarker-v2 + Groq Llama3"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# Инициализируем клиент Groqа
groq_client = Groq(api_key=GROQ_API_KEY)


def ensure_face_landmarker_model():
    if FACE_LANDMARKER_MODEL_PATH.exists():
        return FACE_LANDMARKER_MODEL_PATH

    with MODEL_DOWNLOAD_LOCK:
        if FACE_LANDMARKER_MODEL_PATH.exists():
            return FACE_LANDMARKER_MODEL_PATH

        FACE_LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = FACE_LANDMARKER_MODEL_PATH.with_suffix(".tmp")
        urlretrieve(FACE_LANDMARKER_MODEL_URL, temp_path)
        temp_path.replace(FACE_LANDMARKER_MODEL_PATH)

    return FACE_LANDMARKER_MODEL_PATH


def detect_face_landmarks(rgb_image):
    model_path = ensure_face_landmarker_model()

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        results = landmarker.detect(mp_image)

    if not results.face_landmarks:
        return None

    return results.face_landmarks[0]

def landmark_distance(lm1, lm2, w, h):
    dx = (lm1.x - lm2.x) * w
    dy = (lm1.y - lm2.y) * h
    return math.sqrt(dx * dx + dy * dy)

def calculate_head_pose(landmarks, w, h):

    left_eye = landmarks[33]
    right_eye = landmarks[263]

    nose = landmarks[1]

    left_eye_x = left_eye.x * w
    left_eye_y = left_eye.y * h

    right_eye_x = right_eye.x * w
    right_eye_y = right_eye.y * h

    nose_x = nose.x * w

    # Roll (наклон головы)

    roll = math.degrees(
        math.atan2(
            right_eye_y - left_eye_y,
            right_eye_x - left_eye_x
        )
    )

    # Yaw (поворот влево-вправо)

    eye_center_x = (
        left_eye_x +
        right_eye_x
    ) / 2

    yaw = (
        (nose_x - eye_center_x)
        /
        abs(right_eye_x - left_eye_x)
    ) * 100

    return {
        "roll": round(roll, 2),
        "yaw": round(yaw, 2)
    }

def calculate_confidence(metrics):

    confidence = 100

    # Качество фото

    sharpness = metrics["sharpness"]

    if sharpness < 20:
        confidence -= 20

    elif sharpness < 50:
        confidence -= 10

    # Яркость

    brightness = metrics["brightness"]

    if brightness < 50:
        confidence -= 15

    elif brightness < 80:
        confidence -= 5

    # Наклон головы

    roll = abs(metrics["head_roll"])

    if roll > 10:
        confidence -= 15

    elif roll > 5:
        confidence -= 5

    # Поворот головы

    yaw = abs(metrics["head_yaw"])

    if yaw > 15:
        confidence -= 15

    elif yaw > 8:
        confidence -= 5

    return max(0, min(100, round(confidence)))
    
def calculate_face_metrics(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = detect_face_landmarks(rgb)

    if landmarks is None:
        return None
        
    head_pose = calculate_head_pose(
        landmarks,
        w,
        h
    )
    
    # Перевод ключевых точек в пиксели
    left_eye_x, left_eye_y = landmarks[33].x * w, landmarks[33].y * h
    right_eye_x, right_eye_y = landmarks[263].x * w, landmarks[263].y * h
    eye_width_reference = max(
        abs(right_eye_x - left_eye_x),
        1.0
    )

    face_width_reference = landmark_distance(
        landmarks[234],
        landmarks[454],
        w,
        h
    )

    face_width_reference = max(
        face_width_reference,
        1.0
    )
    nose_x, nose_y = landmarks[1].x * w, landmarks[1].y * h
    chin_y = landmarks[152].y * h
    forehead_y = landmarks[10].y * h

    face_height = abs(chin_y - forehead_y)
    face_ratio = face_height / face_width_reference

    pairs = [
        (33, 263), (133, 362), (70, 300), (107, 336),
        (61, 291), (323, 93), (172, 397), (78, 308)
    ]

    # Коррекция поворота
    dy = right_eye_y - left_eye_y
    dx = right_eye_x - left_eye_x
    angle = math.atan2(dy, dx)
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)

    total_deviation = 0.0
    cx, cy = (left_eye_x + right_eye_x) / 2, (left_eye_y + right_eye_y) / 2

    # Расчет симметрии
    for p1, p2 in pairs:
        pt1_x, pt1_y = landmarks[p1].x * w, landmarks[p1].y * h
        pt2_x, pt2_y = landmarks[p2].x * w, landmarks[p2].y * h
        
        x1_opt = cos_a * (pt1_x - cx) - sin_a * (pt1_y - cy)
        y1_opt = sin_a * (pt1_x - cx) + cos_a * (pt1_y - cy)
        
        x2_opt = cos_a * (pt2_x - cx) - sin_a * (pt2_y - cy)
        y2_opt = sin_a * (pt2_x - cx) + cos_a * (pt2_y - cy)
        
        total_deviation += abs(abs(x1_opt) - abs(x2_opt))
        total_deviation += abs(y1_opt - y2_opt)

    nose_x_opt = cos_a * (nose_x - cx) - sin_a * (nose_y - cy)
    total_deviation += abs(nose_x_opt) * 2

    # АДАПТИВНАЯ НОРМАЛИЗАЦИЯ И «АНТИ-ИНФЛЯЦИЯ»
    raw_error = total_deviation / eye_width_reference

    scale_multiplier = (
        0.35 if eye_width_reference > 200
        else 0.25
    )
    
    # Считаем сырой балл
    raw_score = max(0.0, (1.0 - (raw_error * scale_multiplier)) * 10)
    
    # Жесткое ограничение (анти-инфляция): 
    # выше 7.0 балл растет медленнее, чтобы не плодить десятки
    if raw_score > 7.0:
        symmetry = 7.0 + ((raw_score - 7.0) * 0.6)
    else:
        symmetry = raw_score
        
    symmetry = round(max(0.0, min(10.0, symmetry)), 1)

    # Яркость
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = round(gray.mean(), 1)
    
# Резкость фото
    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()
    
    if sharpness > 200:
        image_quality = 10.0
    elif sharpness > 100:
        image_quality = 8.0
    elif sharpness > 50:
        image_quality = 6.0
    elif sharpness > 20:
        image_quality = 4.0
    else:
        image_quality = 2.0
        
# Canthal Tilt
    canthal_tilt = math.degrees(
        math.atan2(
            -(right_eye_y - left_eye_y),
            (right_eye_x - left_eye_x)
        )
    )

# Ширина носа
    nose_width = landmark_distance(
        landmarks[129],
        landmarks[358],
        w,
        h
    )

    nose_length = landmark_distance(
        landmarks[168],
        landmarks[2],
        w, h
    )

# Ширина рта
    mouth_width = landmark_distance(
        landmarks[61],
        landmarks[291],
        w,
        h
    )

    # ===== ЧЕЛЮСТЬ =======
    jaw_width = landmark_distance(
        landmarks[172],
        landmarks[397],
        w,
        h
    )

    chin_height = landmark_distance(
        landmarks[18],
        landmarks[152],
        w, h
    )
    
    #==== СКУЛЫ =====
    cheekbone_width = landmark_distance(
        landmarks[234],
        landmarks[454],
        w, h
    )
    
    # ======= ГУБЫ ========
    upper_lip = landmark_distance(
        landmarks[13],
        landmarks[0],
        w, h
    )

    lower_lip = landmark_distance(
        landmarks[14],
        landmarks[17],
        w, h
    )

    # ====== ГЛАЗА =======
    eye_spacing = landmark_distance(
        landmarks[133],
        landmarks[362],
        w,
        h
    )
    
    left_eye_width = landmark_distance(
        landmarks[33],
        landmarks[133],
        w, h
    )

    right_eye_width = landmark_distance(
        landmarks[362],
        landmarks[263],
        w, h
    )

    eye_width_ratio = (
        (left_eye_width + right_eye_width) / 2
    ) / face_width_reference

    left_eye_height = landmark_distance(
        landmarks[159],
        landmarks[145],
        w, h
    )

    right_eye_height = landmark_distance(
        landmarks[386],
        landmarks[374],
        w, h
    )
    
    # ====== ПЕРЕМЕННЫЕ ======
    eye_width = (
        left_eye_width +
        right_eye_width
    ) / 2

    eye_height = (
        left_eye_height +
        right_eye_height
    ) / 2

    eye_width_ratio = (
        eye_width /
        face_width_reference
    )

    eye_height_ratio = (
        eye_height /
        face_width_reference
    )

    nose_ratio = nose_width / face_width_reference
    mouth_ratio = mouth_width / face_width_reference
    jaw_ratio = jaw_width / face_width_reference
    eye_spacing_ratio = eye_spacing / eye_width_reference
    eye_aspect_ratio = eye_width / eye_height
    nose_length_ratio = nose_length / face_height
    chin_ratio = chin_height / face_height
    jaw_to_cheek_ratio = jaw_width / cheekbone_width
    
# Face Shape (V2)

    if face_ratio < 1.25:

        if jaw_ratio >= 0.80:
            face_shape = "Square"
        else:
            face_shape = "Round"

    elif face_ratio < 1.45:

        if jaw_ratio >= 0.78:
            face_shape = "Square-Oval"
        else:
            face_shape = "Oval"

    else:

        if jaw_ratio >= 0.80:
            face_shape = "Rectangle"
        else:
            face_shape = "Oblong"

    # Facial Thirds
    forehead_point = landmarks[10]
    nose_base_point = landmarks[2]
    chin_point = landmarks[152]

    upper_third = abs(
        nose_base_point.y - forehead_point.y
    )

    lower_third = abs(
        chin_point.y - nose_base_point.y
    )

    thirds_ratio = upper_third / max(lower_third, 0.001)
    print({
        "face_ratio": round(face_ratio, 2),
        "symmetry": symmetry,
        "brightness": brightness,
        "sharpness": round(sharpness, 1),
        "canthal_tilt": round(canthal_tilt, 2),
        "nose_ratio": round(nose_ratio, 3),
        "mouth_ratio": round(mouth_ratio, 3),
        "jaw_ratio": round(jaw_ratio, 3),
        "eye_spacing_ratio": round(eye_spacing_ratio, 3),
        "face_shape": face_shape,
        "thirds_ratio": round(thirds_ratio, 2),
        "head_roll": head_pose["roll"],
        "head_yaw": head_pose["yaw"],
        "face_shape_reason":
            f"ratio={round(face_ratio,2)}, jaw={round(jaw_ratio,2)}",
        "eye_width_ratio": round(eye_width_ratio, 3),
        "eye_height_ratio": round(eye_height_ratio, 3),
        "eye_aspect_ratio": round(eye_aspect_ratio, 3),
        "nose_length_ratio": round(nose_length_ratio, 3),
        "jaw_to_cheek_ratio": round(jaw_to_cheek_ratio, 3),
    })

    return {
        "face_ratio": round(face_ratio, 2),
        "symmetry": symmetry,
        "brightness": brightness,

        "sharpness": round(sharpness, 1),
        "image_quality": image_quality,

        "canthal_tilt": round(canthal_tilt, 2),

        "nose_ratio": round(nose_ratio, 3),
        "mouth_ratio": round(mouth_ratio, 3),
        "jaw_ratio": round(jaw_ratio, 3),
        "eye_spacing_ratio": round(eye_spacing_ratio, 3),
        "face_shape": face_shape,
        "thirds_ratio": round(thirds_ratio, 2),
        "head_roll": head_pose["roll"],
        "head_yaw": head_pose["yaw"],
        "eye_width_ratio": round(eye_width_ratio, 3),
        "eye_height_ratio": round(eye_height_ratio, 3),
        "eye_aspect_ratio": round(eye_aspect_ratio, 3),
        "nose_length_ratio": round(nose_length_ratio, 3),
        "jaw_to_cheek_ratio": round(jaw_to_cheek_ratio, 3)
    }

def clamp(value, min_value=0.0, max_value=10.0):
    return max(min_value, min(max_value, value))


def calculate_scores(metrics):

    # =====================
    # СИММЕТРИЯ
    # =====================

    symmetry_score = metrics["symmetry"]

    # =====================
    # ОБЩИЕ ПРОПОРЦИИ
    # =====================

    face_ratio = metrics["face_ratio"]

    proportion_score = (
        10 -
        abs(face_ratio - 1.50) * 10
    )

    proportion_score = clamp(proportion_score)

    # =====================
    # ВЕРТИКАЛЬНЫЕ ТРЕТИ
    # =====================

    thirds_ratio = metrics["thirds_ratio"]

    thirds_score = (
        10 -
        abs(thirds_ratio - 1.0) * 8
    )

    thirds_score = clamp(thirds_score)

    # =====================
    # ГЛАЗА
    # =====================

    eye_spacing = metrics["eye_spacing_ratio"]
    eye_width = metrics["eye_width_ratio"]
    eye_height = metrics["eye_height_ratio"]
    canthal_tilt = metrics["canthal_tilt"]

    spacing_score = (
        8 -
        abs(eye_spacing - 0.38) * 50
    )

    spacing_score = clamp(spacing_score)

    width_score = (
        8 -
        abs(eye_width - 0.18) * 60
    )

    width_score = clamp(width_score)

    height_score = (
        8 -
        abs(eye_height - 0.055) * 140
    )

    height_score = clamp(height_score)

    tilt_score = (
        8 -
        abs(canthal_tilt - 4.0) * 0.8
    )

    tilt_score = clamp(tilt_score)

    eye_score = (
        spacing_score * 0.35 +
        width_score * 0.25 +
        height_score * 0.20 +
        tilt_score * 0.20
    )

    eye_score = clamp(eye_score)

    # =====================
    # НОС
    # =====================

    nose_width = metrics["nose_ratio"]
    nose_length = metrics["nose_length_ratio"]

    nose_width_score = (
        8 -
        abs(nose_width - 0.28) * 50
    )

    nose_width_score = clamp(nose_width_score)

    nose_length_score = (
        8 -
        abs(nose_length - 0.30) * 50
    )

    nose_length_score = clamp(nose_length_score)

    nose_score = (
        nose_width_score * 0.5 +
        nose_length_score * 0.5
    )

    nose_score = clamp(nose_score)

    # =====================
    # ЧЕЛЮСТЬ
    # =====================

    jaw_ratio = metrics["jaw_ratio"]
    jaw_to_cheek = metrics["jaw_to_cheek_ratio"]

    jaw_width_score = (
        8 -
        abs(jaw_ratio - 0.82) * 35
    )

    jaw_width_score = clamp(jaw_width_score)

    jaw_structure_score = (
        8 -
        abs(jaw_to_cheek - 0.85) * 40
    )

    jaw_structure_score = clamp(jaw_structure_score)

    jaw_score = (
        jaw_width_score * 0.5 +
        jaw_structure_score * 0.5
    )

    jaw_score = clamp(jaw_score)

    # =====================
    # КАЧЕСТВО ФОТО
    # =====================

    image_score = metrics["image_quality"]

    confidence = calculate_confidence(metrics)

    # =====================
    # ОБЩИЙ БАЛЛ
    # =====================

    overall_score = (
        symmetry_score * 0.25 +
        proportion_score * 0.20 +
        thirds_score * 0.15 +
        eye_score * 0.15 +
        jaw_score * 0.15 +
        nose_score * 0.05 +
        image_score * 0.05
    )

    overall_score = clamp(overall_score)

    return {
        "symmetry_score": round(symmetry_score, 1),
        "proportion_score": round(proportion_score, 1),
        "thirds_score": round(thirds_score, 1),
        "eye_score": round(eye_score, 1),
        "nose_score": round(nose_score, 1),
        "jaw_score": round(jaw_score, 1),
        "confidence": confidence,
        "overall_score": round(overall_score, 1)
    }

def get_tier(score):

    if score >= 9.5:
        return "TRUE ADAM"

    elif score >= 8.5:
        return "CHAD"

    elif score >= 7.5:
        return "CHADLITE"

    elif score >= 6.8:
        return "HTN"

    elif score >= 5.5:
        return "MTN"

    elif score >= 4.5:
        return "LTN"

    elif score >= 3.5:
        return "SUB 5"

    elif score >= 2.5:
        return "SUB 3"

    return "SUBHUMAN"
    
def build_ai_feedback_prompt(metrics, scores, tier):
    return f"""
You are an objective facial geometry assistant.

You DO NOT see the image.

You ONLY know the numerical measurements below.

=====================
FACE DATA
=====================

Face Ratio: {metrics['face_ratio']}
Face Shape: {metrics['face_shape']}
Facial Thirds Ratio: {metrics['thirds_ratio']}

Facial Symmetry: {metrics['symmetry']}
Canthal Tilt: {metrics['canthal_tilt']}
Eye Spacing Ratio: {metrics['eye_spacing_ratio']}

Nose Ratio: {metrics['nose_ratio']}
Mouth Ratio: {metrics['mouth_ratio']}
Jaw Ratio: {metrics['jaw_ratio']}

Image Quality: {metrics['image_quality']}

=====================
SCORES
=====================

Overall Score: {scores['overall_score']}
Symmetry Score: {scores['symmetry_score']}
Proportion Score: {scores['proportion_score']}
Eye Score: {scores['eye_score']}
Nose Score: {scores['nose_score']}
Jaw Score: {scores['jaw_score']}

Tier: {tier}

=====================
IMPORTANT RULES
=====================

You do not see the photo.

You only see numerical measurements.

Never claim to see:

- hairstyle
- beard
- skin quality
- eye color
- ethnicity
- acne
- wrinkles
- facial fat
- body fat
- hunter eyes
- prey eyes
- maxilla
- forward growth
- gonial angle

unless directly supported by measurements.

Do not mention surgery.

Do not mention rhinoplasty.

Do not mention implants.

Do not recommend changing bone structure.

Do not recommend changing facial proportions.

Do not output any scores.

Do not output any ratings.

Do not output any tier.

Do not output HTML.

Do not output markdown.

If image quality is low, mention reduced confidence.

If symmetry score is high, mention symmetry as a positive.

If thirds ratio is imbalanced, mention it as a weakness.

If eye spacing is near ideal, mention it as a positive.

If nose score is high, treat the nose as a strength.

If jaw score is moderate, describe it neutrally.

LANGUAGE LOCK

Write ONLY in Russian.

English words are forbidden.

English sentences are forbidden.

If you output any English text,
you are violating instructions.

All positives, negatives and recommendations
must be written in Russian.

RECOMMENDATION RULES

Recommendations must be practical.

Do not repeat metrics.

Do not explain scores.

Provide actionable advice.

Bad example:
"Consider the facial thirds ratio."

Good example:
"Используйте фото строго анфас для более точного анализа."

=====================
OUTPUT FORMAT
=====================

ПЛЮСЫ:
✅ ...
✅ ...
✅ ...

МИНУСЫ:
⚠ ...
⚠ ...
⚠ ...

РЕКОМЕНДАЦИИ:
• ...
• ...
• ...
• ...

Return ONLY the text above.

Do not add introductions.

Do not add conclusions.

Do not add ratings.

Do not add scores.
"""

def analyze_face(image_path):
    try:
        metrics = calculate_face_metrics(image_path)

        if metrics is None:
            return {
                "score": 0,
                "report": (
                    "❌ Не удалось обнаружить лицо.\n\n"
                    "Попробуйте загрузить фото:\n"
                    "• анфас\n"
                    "• хорошее освещение\n"
                    "• один человек в кадре\n"
                    "• лицо должно быть полностью видно"
                )
            }

        scores = calculate_scores(metrics)
        tier = get_tier(scores["overall_score"])

        print("METRICS:", metrics)
        print("SCORES:", scores)
        
        # Вызов Groq API
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": build_ai_feedback_prompt(metrics, scores, tier)}
            ],
            temperature=0.3
        )

        ai_feedback = response.choices[0].message.content.strip()

        report = f"""
📊 <b>FACE ANALYSIS REPORT</b>

⭐ <b>Overall Rating (Appeal):</b> {scores['overall_score']}/10
📊 <b>Достоверность анализа:</b> {scores['confidence']}%

👁 <b>Симметрия лица (PSL):</b> {scores['symmetry_score']}/10
📏 <b>Пропорции лица:</b> {scores['proportion_score']}/10
📐 <b>Вертикальные пропорции:</b> {scores['thirds_score']}/10
🦴 <b>Выраженность челюсти:</b> {scores['jaw_score']}/10
👃 <b>Нос:</b> {scores['nose_score']}/10
👀 <b>Область глаз:</b> {scores['eye_score']}/10

🧔 <b>Потенциал внешности:</b> {tier}

{ai_feedback}
""".strip()

        return {
            "score": scores["overall_score"],
            "report": report
        }

    except Exception as e:
        print(traceback.format_exc())

        return {
            "score": 0,
            "report": f"❌ Ошибка анализа изображения.\n\nДетали: {e}"
        }


if __name__ == "__main__":
    result = analyze_face("photo.jpg")
    print("SCORE:", result["score"])
    print("\nREPORT:\n", result["report"])
