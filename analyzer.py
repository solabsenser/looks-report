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
You are an advanced, uncompromising AI Facial Aesthetics and Lookism Analyst. Your primary function is to strictly evaluate facial symmetry, proportions, and feature harmony with forensic precision.

CRITICAL INSTRUCTIONS:
1. STRICT IMPARTIALITY: Evaluate each facial feature independently. Disregard overall perception when scoring specific traits (Jawline, Eyes, Nose, Lips). A person with sub-par jawline can still have high-tier eye symmetry.
2. SUB-SCORE GUIDELINES: Do not automatically drop sub-scores below 4.0 or inflate above 7.5. Look for specific flaws or harmonious details.
3. PROPORTION RATIONALITY: Evaluate "Пропорции лица" based on the provided Face Ratio ({metrics['face_ratio']}). Realistic ratios within 1.3 - 1.8 are normal and common, typically scoring between 5.0 - 6.5, not lower unless the distortion is heavy. Golden ratio (1.618) is the ideal benchmark.

THE LOOKISM TIER SCALE (APPEARANCE TIER):
You MUST classify the subject into one of the categories below based on your overall evaluation of their facial metrics. This classification is NOT a direct average of scores, but a holistic determination of their resemblance to the specific types shown in the standard "Lookism Scales".
- [true adam] (Tier 1): Unobtainable perfection, often defined by stylized (e.g., specific long dark hair aesthetic), flawless symmetry, and absolute feature dominance. Overall score ~9.5+
- [chad] (Tier 2): Peak human dimorphism, highly dominant, forward-facing sharp features, extreme symmetry. Overall score ~8.5 - 9.4
- [htn] (High-Tier Normie - Tier 3): Clearly above average. Very good symmetry, harmonious and defined features. Stylized, clean hair. Overall score ~7.5 - 8.4
- [mtn] (Mid-Tier Normie - Tier 4): Average human standard. Moderate defining features, some symmetry, acceptable proportions. Overall score ~6.0 - 7.4
- [ltn] (Low-Tier Normie - Tier 5): Below average. Definable features but lacks definition or symmetry, often with less stylized/more simple hair. Overall score ~4.5 - 5.9
- [sub 5] (Tier 6): Significantly below average. Moderate asymmetries and facial feature harmony issues. Distinct lack of facial definition. Overall score ~3.0 - 4.4
- [sub 3] (Tier 7): Heavily flawed. Severe asymmetry, feature distortion, or lack of facial mass definition. Overall score ~2.9 or less

OUTPUT FORMAT RULES:
- Use standard Telegram Markdown formatting. 
- Use **double asterisks** for bold text to highlight key areas as shown in the template.
- KEEP ALL EMOJIS in the template below; they must appear exactly as written.
- Follow the template exactly:

📊 **FACE ANALYSIS REPORT**
⭐ **Overall Rating:** X.X/10

👁 **Симметрия лица:** {metrics['symmetry']}/10
📏 **Пропорции лица:** X.X/10
🦴 **Выраженность челюсти:** X.X/10
👃 **Нос:** X.X/10
👄 **Губы:** X.X/10
👀 **Область глаз:** X.X/10

🧔 **Потенциал внешности:** **[Enter one classification from the SCALE: true adam, chad, htn, mtn, ltn, sub 5, or sub 3 in uppercase or brackets]**

**Плюсы:**
✅ [State an objective geometric advantage, e.g., flawless eye symmetry or defined jawline structure]
✅ [State another objective geometric advantage, e.g., balanced lip proportions]

**Минусы:**
⚠ [Identify a specific asymmetry or suboptimal proportion in Russian, e.g., небольшая асимметрия носа]
⚠ [Identify a minor feature harmony issue, e.g., челюсть могла бы быть более определенной]

**Рекомендации:**
• [Constructive grooming advice tailored to the weaker area, e.g., hair styling to balance a certain feature]
• [Style or grooming advice focusing on improvement, e.g., макияж для коррекции]
• [Style/maintenance advice, e.g., regular skincare to improve feature presentation]

The entire report MUST be written strictly in professional Russian language ONLY. Avoid mixing English words into the Russian text. Maintain a realistic, balanced, and fair tone.
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
