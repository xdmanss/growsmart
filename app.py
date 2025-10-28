import os
import uuid
import sqlite3
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    send_from_directory,
    flash,
    abort,
)
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    logout_user,
    current_user,
    UserMixin,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename



########################################
# APP / CONFIG
########################################

app = Flask(__name__)
app.secret_key = "growsmart-super-secret-key"  # change in prod

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "growsmart.db")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}

########################################
# LOGIN MANAGER
########################################

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

########################################
# DB HELPERS
########################################

def get_db(retries=3, delay=1):
    import time
    for i in range(retries):
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.OperationalError:
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise


def init_db():
    conn = get_db()
    cur = conn.cursor()

    # users table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            name TEXT,
            password_hash TEXT
        );
        """
    )

    # scans table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            advice TEXT,
            timestamp TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """
    )
    conn.commit()
    conn.close()

init_db()

########################################
# USER CLASS FOR FLASK-LOGIN
########################################

class User(UserMixin):
    def __init__(self, id, email, name, password_hash):
        self.id = id
        self.email = email
        self.name = name
        self.password_hash = password_hash

def load_user_from_db(user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, email, name, password_hash FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return User(row["id"], row["email"], row["name"], row["password_hash"])
    return None

@login_manager.user_loader
def load_user(user_id):
    return load_user_from_db(user_id)

########################################
# STATIC / COPY / CONTENT
########################################

LANG_CONTENT = {
    "tips_best_accuracy": {
        "en": "Tips for best accuracy",
        "ar": "نصائح لأفضل دقة",
    },
    "tips_items": [
        {"en": "Good lighting (daylight)", "ar": "إضاءة جيدة (ضوء النهار)"},
        {"en": "Leaf fills most of the frame", "ar": "الورقة تغطي أغلب الصورة"},
        {"en": "No background clutter", "ar": "خلفية نظيفة بدون فوضى"},
    ],
    # >>> UPDATED PART STARTS HERE <<<
    "about_sections": [
        {
            "title": "AI Plant Health",
            "content": "GrowSmart uses AI to detect plant diseases and guide treatment."
        },
        {
            "title": "Sustainability",
            "content": "We promote water-saving, pesticide reduction, and food security."
        },
        {
            "title": "Innovation",
            "content": "Combining technology and agriculture for UAE’s green future."
        },
        {
            "title": "Team",
            "content": "Developed by Team GrowSmart — passionate about smart farming."
        }
    ],
    # >>> UPDATED PART ENDS HERE <<<
}

CARE_LIBRARY = [
    {
        "slug": "tomato-leaf-mold",
        "title": "Tomato Leaf Mold",
        "symptoms": "Yellow spots on top, olive/brown fuzzy mold underneath the leaf.",
        "treatment": "Remove infected leaves, improve airflow, avoid wet leaves overnight.",
    },
    {
        "slug": "apple-scab",
        "title": "Apple Scab",
        "symptoms": "Olive spots on leaves/fruit that darken and crack.",
        "treatment": "Prune for airflow, clear fallen leaves, consider early-season fungicide.",
    },
    {
        "slug": "healthy",
        "title": "Healthy Leaf",
        "symptoms": "Uniform color, no lesions, no fuzzy growth.",
        "treatment": "Maintain stable watering and monitor weekly.",
    },
]

########################################
# METRICS HELPERS
########################################

def compute_metrics(user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM scans WHERE user_id = ?", (user_id,))
    total_scans = cur.fetchone()["c"]

    cur.execute(
        "SELECT COUNT(*) AS c FROM scans WHERE user_id = ? AND LOWER(prediction) != 'healthy leaf';",
        (user_id,),
    )
    suspected = cur.fetchone()["c"]

    cur.execute(
        "SELECT COUNT(*) AS c FROM scans WHERE user_id = ? AND LOWER(prediction) = 'healthy leaf';",
        (user_id,),
    )
    healthy = cur.fetchone()["c"]

    cur.execute(
        "SELECT timestamp FROM scans WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1;",
        (user_id,),
    )
    row = cur.fetchone()
    last_time = row["timestamp"] if row else None

    # percentage for cards
    disease_pct = 0
    healthy_pct = 0
    if total_scans > 0:
        disease_pct = round((suspected / total_scans) * 100, 1)
        healthy_pct = round((healthy / total_scans) * 100, 1)

    return {
        "total_scans": total_scans,
        "suspected_disease": suspected,
        "healthy_leaves": healthy,
        "disease_pct": disease_pct,
        "healthy_pct": healthy_pct,
        "last_scan_time": last_time,
    }

########################################
# AUTH HELPERS
########################################

def create_user(email, name, password):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cur.fetchone():
        conn.close()
        return None, "Email already registered."
    pw_hash = generate_password_hash(password)
    cur.execute(
        "INSERT INTO users (email, name, password_hash) VALUES (?, ?, ?)",
        (email, name, pw_hash),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return load_user_from_db(new_id), None

def authenticate_user(email, password):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, email, name, password_hash FROM users WHERE email = ?",
        (email,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    if not check_password_hash(row["password_hash"], password):
        return None
    return User(row["id"], row["email"], row["name"], row["password_hash"])

########################################
# AI MODEL FOR REAL DISEASE PREDICTION
########################################

def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

import torch
from torchvision import transforms
from PIL import Image

MODEL_PATH = os.path.join(BASE_DIR, "growsmart_mnv2.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    print("✅ GrowSmart model loaded successfully.")
except Exception as e:
    model = None
    print("❌ Error loading model:", e)

# Transformations same as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Classes from your dataset
CLASS_NAMES = [
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Apple___scab",
    "Bell_Pepper___Bacterial_spot",
    "Bell_Pepper___healthy",
    "Cherry___healthy",
    "Cherry___Powdery_mildew",
    "Grape___Black_rot",
    "Grape___Esca_Black_measles",
    "Grape___healthy",
    "Grape___Leaf_blight",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Strawberry___Healthy",
    "Strawberry___Leaf_Scorch",
    "Tomato___Bacterial_Spot",
    "Tomato___Early_Blight",
    "Tomato___Healthy",
    "Tomato___Late_Blight"
]

def run_inference_on_image(image_path: str):
    if model is None:
        return {"label": "Model not loaded", "confidence": 0, "advice": "Model failed to load."}

    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, idx = torch.max(probs, 1)

        label = CLASS_NAMES[idx.item()]
        conf = round(confidence.item() * 100, 2)

        # Generate simple advice
        if "healthy" in label.lower():
            advice = "Looks healthy. Maintain watering and monitor regularly."
        else:
            advice = f"Detected signs of {label.replace('___', ' ')}. Check care guide for recommendations."

        print(f"🧠 Prediction: {label} ({conf}%)")  # For debugging

        return {"label": label, "confidence": conf, "advice": advice}

    except Exception as e:
        print("❌ Inference error:", e)
        return {"label": "Error", "confidence": 0, "advice": str(e)}

########################################
# DB OPERATIONS FOR SCANS
########################################

def save_scan(user_id, filename, inf_res):
    conn = get_db()
    cur = conn.cursor()
    scan_id = uuid.uuid4().hex
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    cur.execute(
        """
        INSERT INTO scans (id, user_id, filename, prediction, confidence, advice, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            scan_id,
            user_id,
            filename,
            inf_res["label"],
            float(inf_res["confidence"]),
            inf_res["advice"],
            ts,
        ),
    )
    conn.commit()
    conn.close()
    return scan_id, ts

def get_recent_scans(user_id, limit=None):
    conn = get_db()
    cur = conn.cursor()
    q = """
        SELECT id, filename, prediction, confidence, advice, timestamp
        FROM scans
        WHERE user_id = ?
        ORDER BY timestamp DESC
    """
    if limit:
        q += " LIMIT ?"
        cur.execute(q, (user_id, limit))
    else:
        cur.execute(q, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def get_scan_by_id(user_id, scan_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, filename, prediction, confidence, advice, timestamp
        FROM scans
        WHERE user_id = ? AND id = ?
        """,
        (user_id, scan_id),
    )
    row = cur.fetchone()
    conn.close()
    return row

def delete_scan(user_id, scan_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM scans WHERE user_id = ? AND id = ?",
        (user_id, scan_id),
    )
    conn.commit()
    changes = cur.rowcount
    conn.close()
    return changes

########################################
# HELPERS FOR TEMPLATE CONTEXT
########################################

def public_scan_dict(row):
    return {
        "id": row["id"],
        "filename": row["filename"],
        "image_url": url_for("uploaded_file", filename=row["filename"]),
        "prediction": row["prediction"],
        "confidence": row["confidence"],
        "timestamp": row["timestamp"],
    }

########################################
# ROUTES: AUTH
########################################

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    error_msg = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = authenticate_user(email, password)
        if user:
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            error_msg = "Invalid email or password."

    return render_template("login.html", error_msg=error_msg)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    error_msg = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user, err = create_user(email, name, password)
        if err:
            error_msg = err
        else:
            login_user(user)
            return redirect(url_for("dashboard"))

    return render_template("signup.html", error_msg=error_msg)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

########################################
# ROUTES: CORE PAGES
########################################

@app.route("/")
@login_required
def dashboard():
    metrics = compute_metrics(current_user.id)
    scans = get_recent_scans(current_user.id, limit=5)

    # This data feeds dashboard cards and mini-chart
    scans_for_chart = [
        {"t": r["timestamp"], "label": r["prediction"], "conf": r["confidence"]}
        for r in scans
    ]

    tips_carousel = [
        "Use natural light, not flash, for clearer diagnosis.",
        "Separate sick leaves immediately to avoid spreading fungus.",
        "Overwatering can mimic disease. Check soil moisture first.",
        "Ventilation reduces mold risk in greenhouse crops.",
    ]

    return render_template(
        "dashboard.html",
        metrics=metrics,
        recent_scans=[public_scan_dict(r) for r in scans],
        scans_for_chart=scans_for_chart,
        tips_carousel=tips_carousel,
    )

@app.route("/analysis")
@login_required
def analysis():
    metrics = compute_metrics(current_user.id)
    all_scans = get_recent_scans(current_user.id)

    # top disease frequency
    freq = {}
    for s in all_scans:
        freq[s["prediction"]] = freq.get(s["prediction"], 0) + 1

    # chart data
    timeline = [
        {"t": s["timestamp"], "conf": s["confidence"], "label": s["prediction"]}
        for s in all_scans
    ]

    return render_template(
        "analysis.html",
        metrics=metrics,
        freq=freq,
        timeline=timeline,
    )

@app.route("/diagnose", methods=["GET", "POST"])
@login_required
def diagnose():
    just_uploaded = False
    result_public = None
    advice_text = None

    if request.method == "POST":
        # check file
        if "plant_image" not in request.files:
            flash("No file submitted.")
            return redirect(request.url)

        file = request.files["plant_image"]
        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload an image.")
            return redirect(request.url)

        # save upload
        safe_name = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)

        # inference
        inf = run_inference_on_image(save_path)

        # save to db
        scan_id, ts = save_scan(current_user.id, unique_name, inf)

        just_uploaded = True
        result_public = {
            "id": scan_id,
            "filename": unique_name,
            "image_url": url_for("uploaded_file", filename=unique_name),
            "prediction": inf["label"],
            "confidence": inf["confidence"],
            "timestamp": ts,
        }
        advice_text = inf["advice"]

    # ✅ No longer load old image automatically on GET
    return render_template(
        "diagnose.html",
        just_uploaded=just_uploaded,
        last_result=result_public,
        advice=advice_text,
        lang_content=LANG_CONTENT,
    )


@app.route("/history")
@login_required
def history():
    scans = get_recent_scans(current_user.id)
    return render_template(
        "history.html",
        history_items=[public_scan_dict(s) for s in scans],
    )

@app.route("/care")
@login_required
def care():
    # Example diseases for each fruit
    CARE_LIBRARY = {
        "Tomato": {
            "Bacterial Spot": {
                "symptoms": "Small dark spots with yellow halo.",
                "treatment": "Remove infected leaves, avoid overhead watering."
            },
            "Early Blight": {
                "symptoms": "Brown target-like spots on lower leaves.",
                "treatment": "Rotate crops, use copper fungicide."
            },
            "Late Blight": {
                "symptoms": "Dark, water-soaked lesions spreading fast.",
                "treatment": "Remove infected plants, improve airflow."
            },
            "Healthy": {
                "symptoms": "Uniform color, no lesions.",
                "treatment": "Monitor weekly and keep soil balanced."
            }
        },
        "Apple": {
            "Black Rot": {
                "symptoms": "Circular black spots on leaves or fruit.",
                "treatment": "Prune infected branches and dispose properly."
            },
            "Cedar Apple Rust": {
                "symptoms": "Orange spots with spore horns on leaves.",
                "treatment": "Remove cedar trees nearby, apply fungicide."
            },
            "Scab": {
                "symptoms": "Olive green spots on leaves, cracking fruit.",
                "treatment": "Prune trees and remove fallen leaves."
            },
            "Healthy": {
                "symptoms": "Shiny green leaves, no deformities.",
                "treatment": "Regular pruning and watering."
            }
        },
        "Grape": {
            "Black Rot": {
                "symptoms": "Brown circular lesions on leaves and fruit shrivels.",
                "treatment": "Remove infected fruit, prune for better airflow."
            },
            "Leaf Blight": {
                "symptoms": "Yellowing leaves with necrotic spots.",
                "treatment": "Apply fungicide, ensure good drainage."
            },
            "Healthy": {
                "symptoms": "Green intact leaves, strong stems.",
                "treatment": "Maintain balanced watering and sunlight."
            }
        },
        "Potato": {
            "Early Blight": {
                "symptoms": "Concentric brown rings on older leaves.",
                "treatment": "Use copper fungicide, rotate crops yearly."
            },
            "Late Blight": {
                "symptoms": "Dark patches and white mold underside.",
                "treatment": "Destroy infected plants, avoid overhead watering."
            },
            "Healthy": {
                "symptoms": "Uniform green leaves, no discoloration.",
                "treatment": "Keep soil well-drained and nutrient rich."
            }
        },
        "Strawberry": {
            "Leaf Scorch": {
                "symptoms": "Purplish spots that dry out leaves.",
                "treatment": "Remove infected leaves, water at roots only."
            },
            "Healthy": {
                "symptoms": "Bright green leaves and clean fruit.",
                "treatment": "Keep rows ventilated, water in morning."
            }
        }
    }

    fruits = list(CARE_LIBRARY.keys())
    selected_fruit = request.args.get("fruit")
    selected_disease = request.args.get("disease")

    diseases, selected_details = [], None
    if selected_fruit:
        diseases = [{"name": k} for k in CARE_LIBRARY[selected_fruit].keys()]
        if selected_disease in CARE_LIBRARY[selected_fruit]:
            selected_details = CARE_LIBRARY[selected_fruit][selected_disease]

    return render_template(
        "care.html",
        fruits=fruits,
        diseases=diseases,
        selected_fruit=selected_fruit,
        selected_disease=selected_disease,
        selected_details=selected_details,
    )


@app.route("/care/<slug>")
@login_required
def care_detail(slug):
    c = next((x for x in CARE_LIBRARY if x["slug"] == slug), None)
    if not c:
        abort(404)
    return render_template("care_detail.html", item=c)

@app.route("/about")
def about():
    # about is allowed public so judges can see mission without logging in
    return render_template("about.html", content=LANG_CONTENT["about_sections"])

@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html")

@app.route("/admin")
@login_required
def admin_panel():
    metrics = compute_metrics(current_user.id)
    scans_preview = get_recent_scans(current_user.id, limit=5)
    return render_template(
        "admin.html",
        metrics=metrics,
        history_preview=[public_scan_dict(s) for s in scans_preview],
    )

########################################
# API ROUTES
########################################

@app.route("/api/history")
@login_required
def api_history():
    scans = get_recent_scans(current_user.id)
    pubs = [public_scan_dict(s) for s in scans]
    return jsonify(
        {"ok": True, "count": len(pubs), "items": pubs}
    )

@app.route("/api/delete/<scan_id>", methods=["POST"])
@login_required
def api_delete(scan_id):
    removed = delete_scan(current_user.id, scan_id)
    return jsonify({"ok": True, "removed": removed, "scan_id": scan_id})

@app.route("/api/download/<scan_id>")
@login_required
def api_download(scan_id):
    row = get_scan_by_id(current_user.id, scan_id)
    if not row:
        return abort(404)
    filename = row["filename"]
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
        as_attachment=True,
        download_name=filename,
    )

@app.route("/api/export_scans")
@login_required
def api_export_scans():
    scans = get_recent_scans(current_user.id)
    pubs = [public_scan_dict(s) for s in scans]
    return jsonify({"ok": True, "items": pubs})

########################################
# STATIC UPLOADS
########################################

@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

########################################
# ERROR HANDLERS
########################################

@app.errorhandler(404)
def not_found(e):
    return (
        render_template(
            "error.html",
            code=404,
            message_en="Page not found.",
            message_ar="الصفحة غير موجودة.",
        ),
        404,
    )

@app.errorhandler(500)
def server_error(e):
    return (
        render_template(
            "error.html",
            code=500,
            message_en="Internal server error.",
            message_ar="خطأ داخلي في الخادم.",
        ),
        500,
    )

########################################
# MAIN
########################################

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
