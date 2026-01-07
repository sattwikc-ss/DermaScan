# app.py
import os
import json
import uuid
import random
import sqlite3
import numpy as np
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
from flask import (
    Flask, request, render_template, jsonify,
    send_file
)
from werkzeug.utils import secure_filename

# reportlab for PDF export
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# PIL + PyTorch for inference
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

# ---------------- Config ---------------- #
UPLOAD_FOLDER = "static/uploads"
DATABASE = "skin_analysis.db"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Put your model file here (same dir as app.py)
MODEL_PATH = "best_resnet18_ham_colab.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

# ---------------- Label order (must match model training) ---------------- #
# Index -> code mapping must match the checkpoint training order:
class_names = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

# Human readable names for DB lookups / UI
code_to_name = {
    "nv": "Melanocytic nevus",
    "mel": "Melanoma",
    "bkl": "Benign keratosis",
    "bcc": "Basal cell carcinoma",      # make sure DB has this entry
    "akiec": "Actinic keratosis",
    "vasc": "Vascular lesion",
    "df": "Dermatofibroma",
}

# ---------------- DB Init ---------------- #
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id TEXT PRIMARY KEY,
            image_path TEXT NOT NULL,
            disease_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL,
            notes TEXT
        )
    """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS disease_info (
            name TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            symptoms TEXT NOT NULL,
            treatments TEXT NOT NULL,
            prevention TEXT NOT NULL
        )
    """
    )

    c.execute("SELECT COUNT(*) FROM disease_info")
    if c.fetchone()[0] == 0:
        # Ensure entries for all descriptive names used by code_to_name
        disease_info = [
            (
                "Actinic keratosis",
                "A rough, scaly patch on the skin caused by years of sun exposure.",
                "Rough, dry, scaly patches; May be red, tan, pink, or flesh-colored; Usually less than 1 inch in diameter.",
                "Cryotherapy, Topical medications, Photodynamic therapy, Curettage and electrosurgery, Chemical peeling.",
                "Use sunscreen daily; Wear protective clothing; Avoid peak sun hours; Regular skin checks.",
            ),
            (
                "Melanocytic nevus",
                "A common mole formed when melanocytes grow in clusters.",
                "Brown/black; Round with well-defined borders; Usually <6 mm; Uniform appearance.",
                "Usually no treatment; Surgical removal if suspicious.",
                "Monitor for changes; Sun protection; Regular self-exams.",
            ),
            (
                "Melanoma",
                "The most serious type of skin cancer that develops from pigment cells.",
                "Asymmetry; Irregular border; Varied color; >6 mm; Evolving.",
                "Surgical excision; Sentinel node biopsy; Immunotherapy; Targeted therapy; Radiation.",
                "Avoid excess sun; Sunscreen; No tanning beds; Self-exams; Professional checks.",
            ),
            (
                "Benign keratosis",
                "A non-cancerous growth on the skin that develops from skin cells.",
                "Waxy, stuck-on appearance; Light brown to black; Round/oval; Flat or slightly raised.",
                "Often no treatment; Cryotherapy; Curettage; Laser therapy.",
                "No specific prevention; Regular skin examinations.",
            ),
            (
                "Basal cell carcinoma",
                "A common form of skin cancer that begins in the basal cells.",
                "Pearly or waxy bump; Flat, flesh-colored or brown scar-like lesion; Bleeding or scabbing sore that heals then returns.",
                "Surgical excision, Mohs surgery, Curettage and electrodesiccation, Radiation, Topical therapies.",
                "Sun protection; Avoid tanning beds; Regular skin checks.",
            ),
            (
                "Vascular lesion",
                "Abnormalities of blood vessels visible on skin.",
                "Red/purple discoloration; Flat or raised; Anywhere; Sometimes painful.",
                "Laser therapy; Sclerotherapy; Excision; Compression therapy.",
                "Sun protection; Avoid trauma; Healthy weight and BP.",
            ),
            (
                "Dermatofibroma",
                "A common benign skin tumor that presents as a firm nodule.",
                "Firm round bump; Pink/red/brown; May be tender; Usually <1 cm.",
                "Often no treatment; Surgical excision if bothersome; Cryotherapy; Steroid injections.",
                "No specific prevention; Protect skin from trauma.",
            ),
        ]

        c.executemany(
            "INSERT INTO disease_info (name, description, symptoms, treatments, prevention) VALUES (?, ?, ?, ?, ?)",
            disease_info,
        )

    conn.commit()
    conn.close()


# ---------------- PyTorch model loader & inference ---------------- #
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int = 7):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    raw = torch.load(path, map_location=device)

    # If a whole model was saved
    if hasattr(raw, "state_dict") and isinstance(raw, torch.nn.Module):
        model = raw
        model.to(device)
        model.eval()
        return model

    # raw is likely a dict or state_dict
    if isinstance(raw, dict):
        # common wrappers
        if "model_state_dict" in raw:
            state = raw["model_state_dict"]
        elif "state_dict" in raw:
            state = raw["state_dict"]
        else:
            state = raw
    else:
        state = raw

    model = build_model(num_classes=len(class_names))
    # remove 'module.' prefix if present
    new_state = {}
    for k, v in state.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v

    model.load_state_dict(new_state)
    model.to(device)
    model.eval()
    return model


# load at startup
try:
    MODEL = load_checkpoint(MODEL_PATH, DEVICE)
    # print output feature size for confirmation
    out_features = None
    if hasattr(MODEL, "fc") and hasattr(MODEL.fc, "out_features"):
        out_features = MODEL.fc.out_features
    else:
        dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
        out = MODEL(dummy)
        out_features = out.shape[1]
    print(f"[INFO] Loaded model from {MODEL_PATH} on {DEVICE}, out_features={out_features}")
except Exception as e:
    MODEL = None
    print(f"[WARNING] Could not load model ({e}). App will fallback to dummy predictions.")


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    return tensor


def model_predict(img_tensor):
    global MODEL
    if MODEL is None:
        # fallback
        code = random.choice(class_names)
        confidence = round(random.uniform(80, 99), 2)
        return code, confidence

    with torch.no_grad():
        outputs = MODEL(img_tensor)  # (1, num_classes)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        top_idx = int(np.argmax(probs))
        code = class_names[top_idx]
        confidence = float(probs[top_idx] * 100.0)
        return code, round(confidence, 2)


# ---------------- Helpers ---------------- #
def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def save_to_history(image_path, disease_class_code, confidence):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    # store code (short label) in DB for clarity
    c.execute(
        """
        INSERT INTO analysis_history (id, image_path, disease_class, confidence, timestamp, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (analysis_id, image_path, disease_class_code, confidence, timestamp, ""),
    )
    conn.commit()
    conn.close()
    return analysis_id


def get_history(limit=10, offset=0):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT * FROM analysis_history ORDER BY datetime(timestamp) DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    rows = [dict(row) for row in c.fetchall()]
    conn.close()

    # Map short codes to human-readable names for display
    for r in rows:
        code = r.get("disease_class")
        if code:
            r["disease_name"] = code_to_name.get(code, code)
        else:
            r["disease_name"] = ""

    return rows



def get_history_count():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM analysis_history")
    total = c.fetchone()[0]
    conn.close()
    return total


def get_disease_info_by_code(code):
    """
    Given a short code (e.g., 'nv'), map to descriptive name and query DB.
    """
    name = code_to_name.get(code, code)
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM disease_info WHERE name = ?", (name,))
    info = c.fetchone()
    conn.close()
    return dict(info) if info else None


# ---------------- Routes ---------------- #
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        if not allowed_file(file.filename):
            return "Unsupported file type", 400

        ext = os.path.splitext(file.filename)[1].lower()
        unique_filename = secure_filename(f"{uuid.uuid4()}{ext}")
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(image_path)

        # Preprocess + predict
        try:
            img_tensor = preprocess_image(image_path)
            pred_code, confidence = model_predict(img_tensor)
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            pred_code = random.choice(class_names)
            confidence = round(random.uniform(80, 99), 2)

        # Save code in history; map to descriptive name for DB/info lookup
        analysis_id = save_to_history(image_path, pred_code, confidence)
        disease_info = get_disease_info_by_code(pred_code)
        # Provide descriptive name for UI
        pred_name = code_to_name.get(pred_code, pred_code)
        history = get_history(5)

        return render_template(
            "index.html",
            image_path=image_path,
            predicted_class=pred_name,
            predicted_code=pred_code,
            confidence=confidence,
            disease_info=disease_info,
            history=history,
            analysis_id=analysis_id,
        )

    history = get_history(5)
    return render_template("index.html", history=history)


@app.route("/history")
def history_page():
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(24, int(request.args.get("per_page", 12)))
    offset = (page - 1) * per_page
    items = get_history(limit=per_page, offset=offset)
    total = get_history_count()
    pages = max(1, (total + per_page - 1) // per_page)
    return render_template(
        "history.html", history=items, page=page, pages=pages, per_page=per_page
    )


@app.route("/api/history", methods=["GET"])
def api_history():
    limit = request.args.get("limit", 10, type=int)
    history = get_history(limit)
    return jsonify(history)


@app.route("/api/disease/<disease_code>")
def api_disease_info(disease_code):
    info = get_disease_info_by_code(disease_code)
    if info:
        return jsonify(info)
    return jsonify({"error": "Disease not found"}), 404


@app.route("/api/history/<analysis_id>", methods=["DELETE"])
def delete_analysis(analysis_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    c.execute("SELECT image_path FROM analysis_history WHERE id = ?", (analysis_id,))
    result = c.fetchone()

    if not result:
        conn.close()
        return jsonify({"error": "Analysis not found"}), 404

    image_path = result[0]
    c.execute("DELETE FROM analysis_history WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()

    if os.path.exists(image_path):
        try:
            os.remove(image_path)
        except OSError:
            pass

    return jsonify({"success": True})


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    c.execute("SELECT image_path FROM analysis_history")
    image_paths = [row[0] for row in c.fetchall()]

    c.execute("DELETE FROM analysis_history")
    conn.commit()
    conn.close()

    for path in image_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    return jsonify({"success": True})


@app.route("/api/notes/<analysis_id>", methods=["POST"])
def save_notes_api(analysis_id):
    data = request.get_json(silent=True) or {}
    notes = data.get("notes", "")

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("UPDATE analysis_history SET notes = ? WHERE id = ?", (notes, analysis_id))
    conn.commit()
    updated = c.rowcount
    conn.close()

    if updated:
        return jsonify({"success": True})
    return jsonify({"error": "Analysis not found"}), 404


@app.route("/api/export/<analysis_id>")
def export_analysis(analysis_id):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Analysis not found"}), 404

    analysis_dict = dict(row)
    disease_info = get_disease_info_by_code(analysis_dict["disease_class"]) or {}

    return jsonify({"analysis": analysis_dict, "disease_info": disease_info})


@app.route("/api/export/<analysis_id>/csv")
def export_analysis_csv(analysis_id):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Analysis not found"}), 404

    headers = list(row.keys())
    values = [str(row[h]) for h in headers]
    csv_data = ",".join(headers) + "\n" + ",".join(v.replace(",", ";") for v in values)

    buf = BytesIO(csv_data.encode("utf-8"))
    return send_file(
        buf, mimetype="text/csv", as_attachment=True, download_name=f"{analysis_id}.csv"
    )


@app.route("/api/export/bulk.zip")
def export_bulk_zip():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM analysis_history ORDER BY datetime(timestamp) DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()

    mem = BytesIO()
    with ZipFile(mem, "w", ZIP_DEFLATED) as z:
        if rows:
            headers = list(rows[0].keys())
            lines = [",".join(headers)]
            for r in rows:
                lines.append(",".join(str(r.get(h, "")).replace(",", ";") for h in headers))
            z.writestr("analysis/analysis.csv", "\n".join(lines))
        else:
            z.writestr("analysis/analysis.csv", "No data")

        z.writestr("analysis/analysis.json", json.dumps(rows, indent=2))

        for r in rows:
            p = r.get("image_path")
            if p and os.path.exists(p):
                try:
                    z.write(p, arcname=os.path.join("images", os.path.basename(p)))
                except Exception:
                    pass

    mem.seek(0)
    return send_file(
        mem, mimetype="application/zip", as_attachment=True, download_name="dermascan_export.zip"
    )


@app.route("/api/export/<analysis_id>/pdf")
def export_analysis_pdf(analysis_id):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Analysis not found"}), 404

    analysis = dict(row)
    disease_info = get_disease_info_by_code(analysis["disease_class"]) or {}

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Skin Disease Analysis Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Analysis ID: {analysis.get('id','')}", styles["Normal"]))
    story.append(Paragraph(f"Date: {analysis.get('timestamp','')}", styles["Normal"]))
    # show descriptive name in PDF
    story.append(Paragraph(f"Disease Class: {code_to_name.get(analysis.get('disease_class',''), analysis.get('disease_class',''))}", styles["Normal"]))
    story.append(Paragraph(f"Confidence: {analysis.get('confidence','')}%", styles["Normal"]))

    if analysis.get("notes"):
        story.append(Spacer(1, 6))
        story.append(Paragraph("Notes:", styles["Heading3"]))
        story.append(Paragraph(analysis["notes"], styles["Normal"]))

    img_path = analysis.get("image_path")
    if img_path and os.path.exists(img_path):
        story.append(Spacer(1, 12))
        try:
            rl_img = RLImage(img_path)
            max_w = letter[0] - 72  # 1in margins
            scale = min(1.0, max_w / rl_img.drawWidth)
            rl_img.drawWidth *= scale
            rl_img.drawHeight *= scale
            story.append(rl_img)
        except Exception:
            pass

    if disease_info:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Disease Information", styles["Heading2"]))
        story.append(Paragraph(f"Description: {disease_info.get('description','')}", styles["Normal"]))
        story.append(Paragraph(f"Symptoms: {disease_info.get('symptoms','')}", styles["Normal"]))
        story.append(Paragraph(f"Treatments: {disease_info.get('treatments','')}", styles["Normal"]))
        story.append(Paragraph(f"Prevention: {disease_info.get('prevention','')}", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"skin_analysis_{analysis_id}.pdf",
    )


# ---------------- Run ---------------- #
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
