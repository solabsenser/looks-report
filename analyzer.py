import os
import re
from pathlib import Path
from threading import Lock
from urllib.request import urlretrieve

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


def calculate_face_metrics(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = detect_face_landmarks(rgb)

    if landmarks is None:
        return None

    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose = landmarks[1]
    chin = landmarks[152]
    forehead = landmarks[10]

    # Расчет пропорций лица
    face_width = max(abs(right_eye.x - left_eye.x), 0.001)
    face_height = abs(chin.y - forehead.y)
    face_ratio = face_height / face_width

    # УЛЬТРА-СТРОГАЯ ПРОВЕРКА СИММЕТРИИ ПО ВСЕМУ ЛИЦУ (16 точек)
    pairs = [
        (33, 263),   # Внешние углы глаз
        (133, 362),  # Внутренние углы глаз
        (70, 300),   # Внешние края бровей
        (107, 336),  # Внутренние края бровей
        (61, 291),   # Углы губ
        (323, 93),   # Края скул
        (172, 397),  # Углы челюсти
        (78, 308)    # Внешний контур губ
    ]

    total_deviation = 0.0
    eye_center_x = (left_eye.x + right_eye.x) / 2

    for p1, p2 in pairs:
        pt1 = landmarks[p1]
        pt2 = landmarks[p2]
        
        dist_to_center_l = abs(pt1.x - eye_center_x)
        dist_to_center_r = abs(pt2.x - eye_center_x)
        total_deviation += abs(dist_to_center_l - dist_to_center_r)
        total_deviation += abs(pt1.y - pt2.y)

    total_deviation += abs(nose.x - eye_center_x) * 2

    # Сбалансированный коэффициент 1.2
    error_factor = (total_deviation / face_width) * 1.2
    symmetry = max(0.0, min(10.0, (1.0 - error_factor) * 10))

    # Расчет яркости
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    return {
        "face_ratio": round(face_ratio, 2),
        "symmetry": round(symmetry, 1),
        "brightness": round(brightness, 1)
    }


def build_prompt(metrics):
    return f"""
You are an advanced, uncompromising AI Facial Aesthetics and Looksmaxxing Analyst. Your objective is to evaluate facial symmetry, proportions, and feature harmony with forensic precision, using deep lookism and looksmaxxing culture terminology based on standard community definitions.

GLOSSARY & TERMINOLOGY TO USE IN ANALYSES:
- Canthal Tilt (Кантальный тилт): The angle between the inner and outer corners of the eyes. Positive (положительный) is ideal, negative (отрицательный) is a flaw.
- Hunter Eyes (Охотничий взгляд): Deep-set, horizontally elongated eyes with a positive canthal tilt and minimal eyelid exposure. The opposite is Bug eyes (пучеглазие).
- Forward Growth (Вперед-направленный рост челюсти): Well-developed maxilla and mandible creating sharp facial definition and strong profile projection.
- Mewing (Мьюинг): Correct tongue posture against the palate to improve jawline and midface structure over time.
- Softmaxxing: Maximizing natural features via fat loss, skin care, gym, chewing hard gum, hair styling, and posture.
- Hardmaxxing: Surgical and invasive interventions (genioplasty, jaw implants, rhinoplasty, orbital rim implants).

THE LOOKISM TIER SCALE (HOLISTIC APPEARANCE TIER):
Classify the subject based on their overall feature harmony, NOT as a direct mathematical average:
- [TRUE ADAM] (Tier 1): Unobtainable perfection, flawless symmetry, absolute dominance. Score: 9.5+
- [CHAD] (Tier 2): Peak human dimorphism, highly dominant, sharp forward-facing features. Score: 8.5 - 9.4
- [HTN] (High-Tier Normie - Tier 3): Clearly above average. Very good symmetry, harmonious features. Score: 7.5 - 8.4
- [MTN] (Mid-Tier Normie - Tier 4): Average human standard. Moderate definition, acceptable proportions. Score: 6.0 - 7.4
- [LTN] (Low-Tier Normie - Tier 5): Below average. Lacks facial mass definition or clear symmetry. Score: 4.5 - 5.9
- [SUB 5] (Tier 6): Significantly below average. Visible asymmetries, harmony issues. Score: 3.0 - 4.4
- [SUB 3] (Tier 7): Heavily flawed. Severe distortions or lack of definition. Score: 2.9 or less

CRITICAL DYNAMIC SCORING RULES:
- The actual measured facial symmetry score is {metrics['symmetry']}/10. Use this exact number for "Симметрия лица".
- Do NOT hardcode the overall rating to any static placeholder! Dynamically calculate a fair, realistic "Overall Rating" based on the face ratio ({metrics['face_ratio']}) and feature alignment. 

OUTPUT FORMAT RULES:
- Use standard Telegram HTML formatting ONLY (<b> and </b> for bold text).
- NEVER use asterisks (*) or markdown.
- Follow the template EXACTLY, replacing X.X with dynamic, unique calculated scores:

📊 <b>FACE ANALYSIS REPORT</b>
⭐ <b>Overall Rating:</b> X.X/10

👁 <b>Симметрия лица:</b> {metrics['symmetry']}/10
📏 <b>Пропорции лица:</b> X.X/10
🦴 <b>Выраженность челюсти:</b> X.X/10
👃 <b>Нос:</b> X.X/10
👄 <b>Губы:</b> X.X/10
👀 <b>Область глаз:</b> X.X/10

🧔 <b>Потенциал внешности: [ENTER ONLY THE SELECTED TYPOLOGY IN UPPERCASE FROM THE SCALE]</b>

<b>Плюсы (Геометрия лица):</b>
✅ [Objective advantage in Russian, e.g., выраженный вперед-направленный рост челюсти (forward growth)]
✅ [Objective advantage in Russian, e.g., положительный кантальный тилт и потенциал hunter eyes]

<b>Минусы (Диспропорции и асимметрии):</b>
⚠ [Geometric flaw in Russian, e.g., недостаточная проекция подбородка и слабая линия челюсти]
⚠ [Geometric flaw in Russian, e.g., отрицательный кантальный тилт или асимметрия крыльев носа]

<b>План по Looksmaxxing:</b>
• [Practical advice, e.g., Softmaxxing: регулярный мьюинг и жевание жесткой резинки для дефиниции челюсти]
• [Practical advice, e.g., Softmaxxing: снижение процента подкожного жира для проявления костной структуры лица]
• [Practical advice, e.g., Softmaxxing: уход за кожей и подбор прически под вертикальные пропорции лица]
• [Optional Hardmaxxing advice ONLY if tier is SUB 5 or SUB 3, e.g., Hardmaxxing: рассмотрение гениопластики при сильной рецессии подбородка. If tier is higher, write another Softmaxxing point instead]

The entire report MUST be written strictly in professional Russian language ONLY. Avoid mixing English words into the Russian text except for specific accepted culture terms in brackets. Maintain a realistic, balanced, and fair tone.
"""


def extract_score(report):
    match = re.search(
        r"Overall Rating:\s*(\d+(?:\.\d+)?)",
        report
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

        # Вызов Groq API вместо Gemini
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": build_prompt(metrics)}
            ],
            temperature=0.3  # Чуть ниже температуру, чтобы строго следовала шаблону
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
