import os
import re
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

# Инициализируем клиент Groq
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

def calculate_face_metrics(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = detect_face_landmarks(rgb)

    if landmarks is None:
        return None

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

    # Размеры
    face_width = landmark_distance(
        landmarks[234],
        landmarks[454],
        w,
        h
    )

    face_width = max(face_width, 1.0)
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
    raw_error = total_deviation / face_width
    scale_multiplier = 0.35 if face_width > 200 else 0.25 
    
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

# Ширина рта
    mouth_width = landmark_distance(
        landmarks[61],
        landmarks[291],
        w,
        h
    )

# Ширина челюсти
    jaw_width = landmark_distance(
        landmarks[234],
        landmarks[454],
        w,
        h
    )

# Расстояние между глазами
    eye_spacing = landmark_distance(
        landmarks[133],
        landmarks[362],
        w,
        h
    )

    nose_ratio = nose_width / face_width_reference
    mouth_ratio = mouth_width / face_width_reference
    jaw_ratio = jaw_width / face_width_reference
    eye_spacing_ratio = eye_spacing / eye_width_reference
    
    # Face Shape
    if face_ratio >= 1.55:
        face_shape = "Rectangle"
    elif face_ratio >= 1.40:
        face_shape = "Oval"
    elif face_ratio >= 1.25:
        face_shape = "Round"
    else:
        face_shape = "Square"

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
    }

def clamp(value, min_value=0.0, max_value=10.0):
    return max(min_value, min(max_value, value))


def calculate_scores(metrics):

    # Симметрия уже готова
    symmetry_score = metrics["symmetry"]

    # Пропорции лица
    ratio = metrics["face_ratio"]

    ideal_ratio = 1.55

    proportion_score = 10 - abs(ratio - ideal_ratio) * 12
    proportion_score = clamp(proportion_score)

    # Глаза
    eye_spacing = metrics["eye_spacing_ratio"]

    eye_score = 10 - abs(eye_spacing - 0.38) * 40
    eye_score = clamp(eye_score)

    # Нос
    nose_ratio = metrics["nose_ratio"]

    nose_score = 10 - abs(nose_ratio - 0.25) * 25
    nose_score = clamp(nose_score)

    # Челюсть
    jaw_ratio = metrics["jaw_ratio"]

    jaw_score = 10 - abs(jaw_ratio - 0.75) * 15
    jaw_score = clamp(jaw_score)

    # Качество фото
    image_score = metrics["image_quality"]

    overall_score = (
        symmetry_score * 0.30 +
        proportion_score * 0.20 +
        eye_score * 0.15 +
        jaw_score * 0.20 +
        nose_score * 0.10 +
        image_score * 0.05
    )

    return {
        "symmetry_score": round(symmetry_score, 1),
        "proportion_score": round(proportion_score, 1),
        "eye_score": round(eye_score, 1),
        "nose_score": round(nose_score, 1),
        "jaw_score": round(jaw_score, 1),
        "overall_score": round(overall_score, 1)
    }
    
def build_prompt(metrics, scores):
    return f"""
You are a facial geometry analyst.

Your task is NOT to invent ratings.

Your task is to explain the objective measurements already calculated by the system.

========================
OBJECTIVE FACE METRICS
========================

Face Ratio: {metrics['face_ratio']}
Facial Symmetry: {metrics['symmetry']}/10
Face Shape: {metrics['face_shape']}
Facial Thirds Ratio: {metrics['thirds_ratio']}

Canthal Tilt: {metrics['canthal_tilt']}
Eye Spacing Ratio: {metrics['eye_spacing_ratio']}

Nose Width Ratio: {metrics['nose_ratio']}
Mouth Width Ratio: {metrics['mouth_ratio']}
Jaw Width Ratio: {metrics['jaw_ratio']}

Brightness: {metrics['brightness']}
Sharpness: {metrics['sharpness']}
Image Quality: {metrics['image_quality']}/10

========================
OBJECTIVE SCORES
========================

Overall Score: {scores['overall_score']}/10

Symmetry Score: {scores['symmetry_score']}/10
Proportion Score: {scores['proportion_score']}/10
Eye Score: {scores['eye_score']}/10
Nose Score: {scores['nose_score']}/10
Jaw Score: {scores['jaw_score']}/10

========================
IMPORTANT RULES
========================

1. DO NOT invent scores.

2. DO NOT significantly change scores.

3. Overall Rating MUST stay within ±0.3 of:
{scores['overall_score']}

4. Use the supplied objective metrics.

5. If image quality is low, mention that confidence is reduced.

6. If facial symmetry is high, mention it as a positive.

7. If facial thirds ratio is imbalanced, mention it as a weakness.

8. If eye spacing is close to ideal, mention it as a positive.

9. If nose ratio deviates from average proportions, mention it.

10. Avoid exaggerated praise or insults.

11. Be realistic and balanced.

12. Write ONLY in Russian.

========================
TIER SCALE
========================

9.5+ = TRUE ADAM

8.5-9.4 = CHAD

7.5-8.4 = CHADLITE

6.8-7.4 = HTN

5.5-6.7 = MTN

4.5-5.4 = LTN

3.5-4.4 = SUB 5

2.5-3.4 = SUB 3

0-2.4 = SUBHUMAN

========================
OUTPUT FORMAT
========================

OUTPUT FORMAT RULES:
- Use standard Telegram HTML formatting ONLY (<b> and </b> for bold text).
- NEVER use asterisks (*) or markdown.
- Do NOT include example text inside the template brackets below. Generate your own honest Russian analysis from scratch.
- Follow the template EXACTLY:

📊 <b>FACE ANALYSIS REPORT</b>
⭐ <b>Overall Rating (Appeal):</b> [Calculate real unique score]/10

👁 <b>Симметрия лица (PSL):</b> {metrics['symmetry']}/10
📏 <b>Пропорции лица:</b> [Calculate real score based on ratio {metrics['face_ratio']}]/10
🦴 <b>Выраженность челюсти:</b> [Calculate real score considering jaw sharpness or bloat]/10
✨ <b>Дефиниция костной структуры (Fat/Bloat):</b> [Calculate real score based on tissue quality]/10
👃 <b>Нос:</b> [Calculate real score]/10
👄 <b>Губы:</b> [Calculate real score]/10
👀 <b>Область глаз:</b> [Calculate real score]/10

🧔 <b>Потенциал внешности: [ENTER ONLY THE SELECTED TYPOLOGY IN UPPERCASE FROM THE TIER SCALE, MATCHING THE OVERALL RATING]</b>

<b>Плюсы (Геометрия лица):</b>
✅ [Write a genuine specific advantage of this face in Russian, based strictly on metrics]
✅ [Write another genuine advantage of this face in Russian]

<b>Минусы (Диспропорции, асимметрии и Bloat):</b>
⚠ [Write a genuine geometric flaw, facial bloat, or asymmetry in Russian, avoid contradictions]
⚠ [Write another genuine flaw or asymmetry in Russian]

<b>План по Looksmaxxing:</b>
• [Practical advice based on the flaws: Softmaxxing / Leanmaxxing / Mewing contextually]
• [Practical advice tailored to the specific face traits]
• [Practical advice for grooming, hair, or styling based on their proportions]
• [Optional Hardmaxxing advice ONLY if tier is SUB 5, SUB 3 or SUBHUMAN, otherwise write a 4th Softmaxxing point instead]

The entire report MUST be written strictly in professional Russian language. Avoid mixing English words into the Russian text except for specific accepted culture terms in brackets or quotes. Maintain a realistic, balanced, and fair tone.
"""


def extract_score(report):
    # Ищем любую конструкцию вокруг "Overall Rating" или "Rating", игнорируя HTML-теги и текст в скобках
    match = re.search(
        r"Overall\s+Rating.*?(?:Appeal)?.*?:?\s*.*?(\d+(?:\.\d+)?)", 
        report, 
        re.IGNORECASE
    )
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return 0.0
    return 0.0


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

        print("METRICS:", metrics)
        print("SCORES:", scores)

        # Вызов Groq API
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": build_prompt(metrics, scores)}
            ],
            temperature=0.3
        )

        report = response.choices[0].message.content.strip()
        score = extract_score(report)

        return {
            "score": score,
            "report": report
        }

    except Exception as e:
        return {
            "score": 0,
            "report": (
                "❌ Ошибка анализа изображения.\n\n"
                f"Детали: {str(e)}"
            )
        }


if __name__ == "__main__":
    result = analyze_face("photo.jpg")
    print("SCORE:", result["score"])
    print("\nREPORT:\n", result["report"])
