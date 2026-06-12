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
You are an advanced, uncompromising, and brutally realistic AI Facial Aesthetics and Looksmaxxing Analyst. Your objective is to evaluate facial symmetry, proportions, feature harmony, and tissue quality with forensic precision, avoiding any rating inflation, cope, or false praise.

LOOKSMAXXING TERMINOLOGY TO INTEGRATION (IN RUSSIAN):
- PSL Score (геометрия костей), Appeal (общая миловидность), Canthal Tilt (кантальный тилт), Hunter Eyes (охотничий взгляд), Forward Growth (рост челюсти), Mewing (мьюинг), Fraud (фродинг), Cope (коуп).
- Facial Bloat / Bloating (Одутловатость/Лишний жир): Подкожный жир на лице и наличие второго подбородка, которые скрывают истинную костную структуру и углы челюсти. В луксмаксинге это главный враг дефиниции (definition).

CRITICAL CALIBRATION RULES (ANTI-INFLATION & BODY FAT):
1. Evaluate weight and tissue quality: Pay strict attention to facial bloat, puffiness, or a double chin. If the face looks round, heavy, or lacks submandibular definition due to excess weight, you MUST significantly drop the scores for "Выраженность челюсти", "Пропорции лица", and "Overall Rating".
2. Do NOT give high tiers like CHAD or CHADLITE to faces with visible facial bloat, soft jawlines, or double chins. Excess facial fat automatically drops the subject into MTN, LTN, or SUB 5, depending on the severity.
3. The actual measured facial symmetry score is exactly {metrics['symmetry']}/10. The measured face ratio is {metrics['face_ratio']}.
4. Avoid contradictions: Do not praise a jawline for "forward growth" if it is buried under a double chin or hidden by facial bloat.

THE LOOKISM TIER SCALE:
- [TRUE ADAM] (Tier 1): Пик человеческой внешности, абсолютное доминирование черт. Score: 9.5+
- [CHAD] (Tier 2): Высшая оценка внешности, идеальные факторы привлекательности, резкие маскулинные костные углы. Score: 8.5 - 9.4
- [CHADLITE] (Tier 3): Красивый, явно выше среднего, отличная костная структура без грамма лишнего жира. Score: 7.5 - 8.4
- [HTN] (High Tier Normie - Tier 4): Чуть красивее среднего, хорошая гармония, приятный Appeal. Score: 6.8 - 7.4
- [MTN] (Middle Tier Normie - Tier 5): Средняя внешность, типичный стандарт, умеренные пропорции. Score: 5.5 - 6.7
- [LTN] (Low Tier Normie - Tier 6): Чуть ниже среднего, не хватает дефиниции костей, лицо может быть слегка заплывшим (bloated). Score: 4.5 - 5.4
- [SUB 5] (Tier 7): Плохо, заметные диспропорции, асимметрии или выраженная одутловатость/лишний вес. Score: 3.5 - 4.4
- [SUB 3] (Tier 8): Выше минимума, тяжелые эстетические дефекты или сильное ожирение лица. Score: 2.5 - 3.4
- [SUBHUMAN] (Tier 9): Низшая оценка внешности, полное отсутствие гармонии. Score: 2.4 or less

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
✨ <b>Дефиниция костной структуры (Fat/Bloat):</b> [Calculate score: 10 means zero fat/ultra sharp, lower means heavy bloat or double chin]/10
👃 <b>Нос:</b> [Calculate real score]/10
👄 <b>Губы:</b> [Calculate real score]/10
👀 <b>Область глаз:</b> [Calculate real score]/10

🧔 <b>Потенциал внешности: [ENTER ONLY THE SELECTED TYPOLOGY IN UPPERCASE FROM THE SCALE]</b>

<b>Плюсы (Геометрия лица):</b>
✅ [Write a genuine specific advantage of this face in Russian]
✅ [Write another genuine advantage of this face in Russian]

<b>Минусы (Диспропорции, асимметрии и Bloat):</b>
⚠ [Write a genuine geometric flaw, facial bloat, or double chin issue in Russian, avoid contradictions]
⚠ [Write another genuine flaw or asymmetry in Russian]

<b>План по Looksmaxxing:</b>
• [Practical advice, e.g., Softmaxxing: если есть лишний жир/bloat, жестко рекомендуй дефицит калорий и кардио (leanmaxxing) для проявления костей]
• [Practical advice, e.g., Softmaxxing: конкретное действие для улучшения линии челюсти (мьюинг и т.д.)]
• [Practical advice, e.g., Доступный фродинг (fraud): как визуально скрыть минусы прической или ракурсом]
• [Optional Hardmaxxing advice ONLY if tier is SUB 5, SUB 3 or SUBHUMAN, otherwise write a 4th Softmaxxing point instead]

The entire report MUST be written strictly in professional Russian language. Avoid mixing English words into the Russian text except for specific accepted culture terms in brackets or quotes. Maintain a realistic, balanced, and fair tone.
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
