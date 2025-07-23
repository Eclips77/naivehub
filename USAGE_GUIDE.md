# NaiveHub - מדריך שימוש מקיף
## 🎯 איך הפרויקט עובד - הסבר מלא

### 📁 ארכיטקטורת הפרויקט

```
NaiveHub/
├── servers/                    # שרתי FastAPI
│   ├── trainer_server.py      # שרת אימון (פורט 8001)
│   └── classifier_server.py   # שרת סיווג (פורט 8000)
├── services/                  # ליבת האלגוריתמים
│   ├── trainer.py            # אלגוריתם אימון Naive Bayes
│   ├── classifier.py         # אלגוריתם חיזוי
│   └── evaluator.py          # הערכת מודלים
├── managers/                  # ניהול נתונים
│   ├── data_manager.py       # טעינה וניהול נתונים
│   ├── classifier_manager.py # ניהול מודלי סיווג
│   └── trainer_manager.py    # ניהול תהליך האימון
├── utils/                     # כלי עזר
│   ├── data_loader.py        # טעינת נתונים
│   ├── data_cleaner.py       # ניקוי נתונים
│   ├── data_splitter.py      # חלוקת נתונים
│   └── model_loader.py       # טעינת מודלים
├── models/                    # תיקיית מודלים (נוצרת אוטומטית)
├── Data/                      # תיקיית נתונים (אופציונלי)
└── main.py                    # נקודת כניסה עיקרית
```

---

## 🔄 זרימת העבודה המלאה

### שלב 1: הכנת הנתונים 📊

**איפה שמים את קובץ הנתונים:**
```
# אפשרות 1: בתיקיית הפרויקט הראשית
my_data.csv

# אפשרות 2: בתיקיית Data
Data/my_data.csv

# אפשרות 3: נתיב מלא
C:/path/to/my_data.csv
```

**פורמט הנתונים הנדרש:**
```csv
feature1,feature2,feature3,target_column
sunny,hot,high,no
rainy,mild,high,no
sunny,hot,normal,yes
cloudy,mild,high,yes
```

**דרישות:**
- הקובץ חייב להיות בפורמט CSV
- שורה ראשונה עם שמות עמודות
- עמודת המטרה (target) יכולה להיות בכל מקום
- נתונים קטגוריאליים (טקסט) או מספריים

---

### שלב 2: הפעלת המערכת 🚀

#### דרך 1: עם Docker (מומלץ)
```bash
# הפעלת המערכת המלאה
docker-compose up -d

# בדיקת סטטוס
docker ps

# עצירת המערכת
docker-compose down
```

#### דרך 2: הפעלה ידנית
```bash
# Terminal 1 - שרת אימון
cd servers
python -m uvicorn trainer_server:app --host 0.0.0.0 --port 8001

# Terminal 2 - שרת סיווג
cd servers
python -m uvicorn classifier_server:app --host 0.0.0.0 --port 8000
```

---

### שלב 3: אימון מודל 🧠

#### דרך 1: דרך API (מומלץ)
```bash
# הכנת בקשת אימון
curl -X POST "http://localhost:8001/train" \
-H "Content-Type: application/json" \
-d '{
  "file_path": "my_data.csv",
  "target_column": "play_tennis",
  "model_name": "tennis_model"
}'
```

#### דרך 2: דרך Python
```python
import requests

# נתוני אימון
train_data = {
    "file_path": "my_data.csv",
    "target_column": "play_tennis", 
    "model_name": "tennis_model"
}

# שליחת בקשה לשרת אימון
response = requests.post("http://localhost:8001/train", json=train_data)
print(response.json())
```

#### דרך 3: שימוש ישיר בקוד
```python
from managers.data_manager import DataManager
from services.trainer import NaiveBayesTrainer

# טעינת נתונים
data_manager = DataManager("my_data.csv")
train_df, test_df = data_manager.prepare_data()

# אימון מודל
trainer = NaiveBayesTrainer()
model = trainer.fit(train_df, "play_tennis")
```

---

### שלב 4: טעינת מודל לשרת סיווג 📥

```python
# טעינת מודל מהשרת אימון לשרת סיווג
load_request = {
    "model_name": "tennis_model"
}

response = requests.post("http://localhost:8000/load_model", json=load_request)
print(response.json())
```

**מה קורה פנימית:**
1. שרת הסיווג פונה לשרת האימון
2. מקבל את המודל
3. שומר אותו כקובץ JSON ב-`models/tennis_model.json`
4. טוען אותו לזיכרון לחיזויים מהירים

---

### שלב 5: ביצוע חיזויים 🎯

```python
# חיזוי יחיד
predict_data = {
    "model_name": "tennis_model",
    "record": {
        "weather": "sunny",
        "temperature": "hot", 
        "humidity": "high"
    }
}

response = requests.post("http://localhost:8000/predict", json=predict_data)
prediction = response.json()["prediction"]
print(f"Prediction: {prediction}")
```

---

### שלב 6: הערכת מודל 📈

```python
# הערכת דיוק המודל
eval_data = {
    "model_name": "tennis_model",
    "test_data": [
        {"weather": "sunny", "temperature": "hot", "humidity": "high", "play_tennis": "no"},
        {"weather": "cloudy", "temperature": "mild", "humidity": "normal", "play_tennis": "yes"}
    ]
}

response = requests.post("http://localhost:8000/evaluate", json=eval_data)
accuracy = response.json()["accuracy_percentage"]
print(f"Model accuracy: {accuracy}")
```

---

## 📂 מה אמור להיות בתיקיית models?

### לפני אימון ראשון:
```
models/
└── (ריקה)
```

### אחרי אימון וטעינה:
```
models/
├── tennis_model.json      # מודל שמור בפורמט JSON
├── iris_model.json        # מודל נוסף
└── customer_model.json    # מודל נוסף
```

### תוכן קובץ מודל לדוגמה:
```json
{
  "classes": ["no", "yes"],
  "priors": {
    "no": 0.35714285714285715,
    "yes": 0.6428571428571429
  },
  "likelihoods": {
    "weather": {
      "no": {
        "sunny": 0.6,
        "rainy": 0.4,
        "cloudy": 0.0
      },
      "yes": {
        "sunny": 0.2222222222222222,
        "rainy": 0.3333333333333333,
        "cloudy": 0.4444444444444444
      }
    }
  },
  "feature_columns": ["weather", "temperature", "humidity"],
  "target_column": "play_tennis"
}
```

---

## 🛠️ דוגמאות שימוש מלאות

### דוגמה 1: פרויקט חיזוי מזג אויר
```python
# 1. הכנת נתונים
weather_data = {
    'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy'],
    'temperature': ['hot', 'hot', 'hot', 'mild', 'cool'], 
    'humidity': ['high', 'high', 'high', 'high', 'normal'],
    'windy': ['false', 'true', 'false', 'false', 'false'],
    'play': ['no', 'no', 'yes', 'yes', 'yes']
}

df = pd.DataFrame(weather_data)
df.to_csv('weather.csv', index=False)

# 2. אימון מודל
train_request = {
    "file_path": "weather.csv",
    "target_column": "play",
    "model_name": "weather_predictor"
}

# 3. חיזוי
predict_request = {
    "model_name": "weather_predictor", 
    "record": {
        "outlook": "sunny",
        "temperature": "cool",
        "humidity": "high", 
        "windy": "true"
    }
}
```

### דוגמה 2: סיווג לקוחות
```python
# נתוני לקוחות
customer_data = {
    'age': ['young', 'young', 'middle_aged', 'senior', 'senior'],
    'income': ['high', 'high', 'high', 'medium', 'low'],
    'student': ['no', 'no', 'no', 'no', 'yes'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes']
}

# אימון וחיזוי
train_request = {
    "file_path": "customers.csv",
    "target_column": "buys_computer", 
    "model_name": "customer_model"
}
```

---

## 🐳 עבודה עם Docker

### הפעלה:
```bash
# בניית ההפעלה הראשונית
docker-compose build

# הפעלת המערכת
docker-compose up -d

# צפייה בלוגים
docker-compose logs -f

# עצירה
docker-compose down
```

### בדיקת וליומים:
```bash
# רשימת מודלים שמורים
docker exec naivehub-classifier ls -la /app/models/

# בדיקת תוכן מודל
docker exec naivehub-classifier cat /app/models/tennis_model.json
```

---

## 🔧 פתרון בעיות נפוצות

### בעיה 1: "File not found"
```python
# פתרון: ודא שהקובץ קיים
import os
if os.path.exists("my_data.csv"):
    print("File exists")
else:
    print("File not found - check path")
```

### בעיה 2: "Model not loaded"
```python
# פתרון: טען את המודל קודם
load_request = {"model_name": "my_model"}
requests.post("http://localhost:8000/load_model", json=load_request)
```

### בעיה 3: "Connection refused"
```bash
# פתרון: ודא שהשרתים רצים
curl http://localhost:8001/health
curl http://localhost:8000/health
```

---

## 📋 רשימת נקודות קצה (API)

### שרת אימון (8001):
- `POST /train` - אימון מודל חדש
- `GET /models` - רשימת מודלים זמינים  
- `GET /model/{name}` - קבלת מודל ספציפי
- `GET /health` - בדיקת בריאות

### שרת סיווג (8000):
- `POST /load_model` - טעינת מודל מהשרת אימון
- `POST /predict` - ביצוע חיזוי
- `POST /evaluate` - הערכת דיוק מודל
- `GET /models` - רשימת מודלים זמינים בשרת אימון
- `GET /health` - בדיקת בריאות

---

## ✅ טסטים ובדיקות

```bash
# הרצת טסטים מקיפים
python test_naivehub.py

# או עם הסקריפט המהיר
quick-test.bat     # Windows
./quick-test.sh    # Linux/Mac
```

זה הסבר מקיף על כל הפרויקט! האם יש משהו ספציפי שתרצה שאני אפרט עליו יותר?
