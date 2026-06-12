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


def calculate_face_metrics(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Получаем реальный размер картинки в пикселях
    h, w, _ = image.shape

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = detect_face_landmarks(rgb)

    if landmarks is None:
        return None

    # Переводим ключевые точки в реальные пиксели экрана
    left_eye_x, left_eye_y = landmarks[33].x * w, landmarks[33].y * h
    right_eye_x, right_eye_y = landmarks[263].x * w, landmarks[263].y * h
    nose_x, nose_y = landmarks[1].x * w, landmarks[1].y * h
    chin_y = landmarks[152].y * h
    forehead_y = landmarks[10].y * h

    # Расчет пропорций лица (в пикселях)
    face_width = max(abs(right_eye_x - left_eye_x), 1.0)
    face_height = abs(chin_y - forehead_y)
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

    # Высчитываем точный угол наклона головы в пиксельной сетке
    dy = right_eye_y - left_eye_y
    dx = right_eye_x - left_eye_x
    angle = math.atan2(dy, dx)
    
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)

    total_deviation = 0.0
    
    # Центр вращения в пикселях
    cx = (left_eye_x + right_eye_x) / 2
    cy = (left_eye_y + right_eye_y) / 2

    # Проходим по парам точек с переводом в пиксели и выравниванием
    for p1, p2 in pairs:
        pt1_x, pt1_y = landmarks[p1].x * w, landmarks[p1].y * h
        pt2_x, pt2_y = landmarks[p2].x * w, landmarks[p2].y * h
        
        # Поворачиваем левую точку
        x1_opt = cos_a * (pt1_x - cx) - sin_a * (pt1_y - cy)
        y1_opt = sin_a * (pt1_x - cx) + cos_a * (pt1_y - cy)
        
        # Поворачиваем правую точку
        x2_opt = cos_a * (pt2_x - cx) - sin_a * (pt2.y - cy)
        y2_opt = sin_a * (pt2_x - cx) + cos_a * (pt2.y - cy)
        
        # Сравниваем пиксельные отклонения
        total_deviation += abs(abs(x1_opt) - abs(x2_opt))
        total_deviation += abs(y1_opt - y2_opt)

    # Проверяем положение кончика носа
    nose_x_opt = cos_a * (nose_x - cx) - sin_a * (nose_y - cy)
    total_deviation += abs(nose_x_opt) * 2

    # Деление пиксельной ошибки на пиксельную ширину лица дает идеальную точность
    error_factor = total_deviation / face_width
    
    # Корректируем жесткость: уменьшаем влияние микро-ошибок (множитель 0.35)
    symmetry = max(0.0, min(10.0, (1.0 - (error_factor * 0.35)) * 10))

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
You are an advanced, independent, and brutally realistic AI Facial Aesthetics and Looksmaxxing Analyst. Your objective is to evaluate facial symmetry, proportions, feature harmony, and tissue quality with forensic precision, blending the objective data from MediaPipe with core looksmaxxing community concepts without any rating inflation or cope.

EXTENDED LOOKSMAXXING GLOSSARY (USE CONTEXTUALLY, NOT BLINDLY):
- Gonial Angle (Гониальный угол): Угол нижней челюсти. Идеал для мужчин ~110-120°. Большой угол делает лицо круглым, слишком острый — диспропорциональным.
- Eyelid Exposure (Уровни верхнего века): Видимость кожи над верхним веком. Высокое веко делает взгляд уставшим ("bug eyes"), минимальное или отсутствующее веко создает "hunter eyes".
- Midface Ratio (Компактность лица): Соотношение высоты средней трети лица к ширине. Компактное среднее лицо делает череп более маскулинным и привлекательным.
- Leanmaxxing: Снижение процента жира в организме ради проявления костных углов.
- Bloat / Bloating (Одутловатость): Лишний подкожный жир на лице и наличие второго подбородка, скрывающие истинную костную структуру (PSL).
- Canthal Tilt: Наклон глаз (положительный/отрицательный).
- Forward Growth: Вперед-направленный рост челюсти и максиллы.
- Appeal: Общая гармония и миловидность лица вопреки строгим костным правилам.
- Cope: Самообман, отрицание или нежелание признавать реальные минусы геометрии.

THE LOOKISM TIER SCALE:
- [TRUE ADAM] (Tier 1): Пик человеческой внешности, абсолютное доминирование черт. Score: 9.5+
- [CHAD] (Tier 2): Высшая оценка внешности, идеальные факторы привлекательности, резкие маскулинные костные углы. Score: 8.5 - 9.4
- [CHADLITE] (Tier 3): Красивый, явно выше среднего, отличная костная структура. Score: 7.5 - 8.4
- [HTN] (High Tier Normie - Tier 4): Чуть красивее среднего, хорошая гармония, приятный Appeal. Score: 6.8 - 7.4
- [MTN] (Middle Tier Normie - Tier 5): Средняя внешность, типичный стандарт, умеренные пропорции. Score: 5.5 - 6.7
- [LTN] (Low Tier Normie - Tier 6): Чуть ниже среднего, не хватает дефиниции костей. Score: 4.5 - 5.4
- [SUB 5] (Tier 7): Плохо, заметные диспропорции, асимметрии или выраженный bloat. Score: 3.5 - 4.4
- [SUB 3] (Tier 8): Выше минимума, тяжелые эстетические дефекты или сильное ожирение лица. Score: 2.5 - 3.4
- [SUBHUMAN] (Tier 9): Низшая оценка внешности, полное отсутствие гармонии. Score: 2.4 or less

CRITICAL LOGIC & MEDIAPIPE INTEGRATION RULES:
1. FAITHFUL TO MEDIAPIPE: You MUST anchor your analysis on the provided input: Face Height-to-Width Ratio is {metrics['face_ratio']} and Facial Symmetry is {metrics['symmetry']}/10. If symmetry is low (below 5.5), the overall score CANNOT be high, and you must note the asymmetry in minuses.
2. STRICT MATH TIER MATCHING: Calculate the "Overall Rating (Appeal)" first. Then, you MUST select the text tier from THE LOOKISM TIER SCALE that exactly matches that calculated score. 
   - If the score is between 5.5 and 6.7, the tier MUST be strictly [MTN]. 
   - If the score is between 6.8 and 7.4, the tier MUST be strictly [HTN].
   - Do NOT mix them up like giving a 6.2 score and calling it HTN. That is a critical error!
3. OBJECTIVE TISSUE EVALUATION: Do NOT assume every face has "facial bloat" or a "double chin"! Evaluate strictly based on the image. If the face is lean, thin, or has clearly visible jaw contours, award a HIGH score (8.0 - 9.5) for "Дефиниция костной структуры" and praise the leanness. ONLY drop scores for bloat if there are clear, visible signs of excess fat or a double chin.
4. CONTEXTUAL TERM USAGE: Use glossary terms only when they apply. Do not repeat the word "Mewing" in every single bullet point. Diversify your advice (e.g., chewing gum, leanmaxxing, skincare, hair style, posture).
5. NO CONTRADICTIONS: Ensure your dynamic sub-scores match your text description perfectly. Do not praise a jawline for "forward growth" if it is simultaneously called "weak" in minuses.

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

🧔 <b>Потенциал внешности: [ENTER ONLY THE SELECTED TYPOLOGY IN UPPERCASE FROM THE SCALE, MATCHING THE OVERALL RATING]</b>

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
