# NaiveHub - מדריך Postman מעודכן
## 🚀 המערכת החדשה - שרת אימון עם הערכה

### 📋 שינויים במערכת:

**🧠 שרת אימון (פורט 8001):**
- ✅ אימון מודלים (`POST /train`)
- ✅ קבלת מודלים (`GET /model/{name}`)
- ✅ רשימת מודלים (`GET /models`)
- ✅ **הערכת מודלים** (`POST /evaluate`) ← **חדש!**
- ✅ בדיקת בריאות (`GET /health`)

**🔮 שרת סיווג (פורט 8000):**
- ✅ טעינת מודלים (`POST /load_model`)
- ✅ חיזויים (`POST /predict`)
- ✅ רשימת מודלים זמינים (`GET /models`)
- ✅ בדיקת בריאות (`GET /health`)
- ❌ **הערכה הוסרה** (עברה לשרת אימון)

---

## 🎯 תהליך העבודה החדש:

### 1️⃣ **אימון מודל** (שרת אימון)
```
POST http://localhost:8001/train

Body:
{
    "file_path": "tennis_data.csv",
    "target_column": "play_tennis",
    "model_name": "tennis_model"
}
```

### 2️⃣ **הערכת מודל** (שרת אימון) ← **פשוט יותר!**
```
POST http://localhost:8001/evaluate

Body:
{
    "model_name": "tennis_model"
}
```

**זהו! רק שם המודל!** 🎉

### 3️⃣ **טעינת מודל לחיזויים** (שרת סיווג)
```
POST http://localhost:8000/load_model

Body:
{
    "model_name": "tennis_model"
}
```

### 4️⃣ **ביצוע חיזויים** (שרת סיווג)
```
POST http://localhost:8000/predict

Body:
{
    "model_name": "tennis_model",
    "record": {
        "weather": "sunny",
        "temperature": "hot",
        "humidity": "high",
        "windy": "false"
    }
}
```

---

## 📊 **הערכת מודל - הדרך החדשה**

### **בקשה פשוטה:**
```json
{
    "model_name": "tennis_model"
}
```

### **תגובה מפורטת:**
```json
{
    "model_name": "tennis_model",
    "accuracy": 0.8,
    "accuracy_percentage": "80.00%",
    "test_samples": 5,
    "classes": ["no", "yes"],
    "features": ["weather", "temperature", "humidity", "windy"]
}
```

---

## 🏗️ **בניית הקונטיינרים מחדש:**

### עצור את המערכת הקיימת:
```bash
docker-compose down
```

### בנה מחדש:
```bash
docker-compose build --no-cache
```

### הפעל:
```bash
docker-compose up -d
```

### בדוק שהכל רץ:
```bash
docker ps
```

---

## 🎮 **תרחיש מלא לבדיקה:**

### 1. ודא שהשרתים רצים:
```bash
curl http://localhost:8001/health
curl http://localhost:8000/health
```

### 2. צור קובץ נתונים ושלח לקונטיינר:
```bash
docker cp tennis_data.csv naivehub-trainer:/app/
```

### 3. אמן מודל:
```
POST http://localhost:8001/train
{
    "file_path": "tennis_data.csv",
    "target_column": "play_tennis",
    "model_name": "tennis_model"
}
```

### 4. הערך מודל (חדש!):
```
POST http://localhost:8001/evaluate
{
    "model_name": "tennis_model"
}
```

### 5. טען מודל לחיזויים:
```
POST http://localhost:8000/load_model
{
    "model_name": "tennis_model"
}
```

### 6. בצע חיזוי:
```
POST http://localhost:8000/predict
{
    "model_name": "tennis_model",
    "record": {
        "weather": "sunny",
        "temperature": "hot",
        "humidity": "high",
        "windy": "false"
    }
}
```

---

## ✨ **יתרונות המערכת החדשה:**

1. **פשטות**: הערכה רק עם שם מודל
2. **ביצועים**: נתוני בדיקה נשמרים באימון
3. **אמינות**: השרת שאימן יודע לבדוק
4. **הפרדה**: שרת סיווג מתמחה רק בחיזויים

---

## 🔧 **פקודות בדיקה:**

### בדוק קבצים בקונטיינר אימון:
```bash
docker exec naivehub-trainer ls -la /app/
```

### בדוק לוגים:
```bash
docker logs naivehub-trainer
docker logs naivehub-classifier
```

### בדוק בריאות שרתים:
```bash
curl http://localhost:8001/health
curl http://localhost:8000/health
```

---

## 🎯 **דוגמה מלאה ב-Postman:**

### Collection: "NaiveHub v2"

```
📁 Training Server (8001)
├── POST Train Model
├── POST Evaluate Model ← חדש!
├── GET List Models
├── GET Get Model
└── GET Health

📁 Classification Server (8000)  
├── POST Load Model
├── POST Predict
├── GET List Models
└── GET Health
```

המערכת עכשיו פשוטה וחכמה יותר! 🚀
