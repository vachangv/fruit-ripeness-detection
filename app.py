from flask import Flask, render_template, request
import os
from image_processing import analyze_image
from fruit_detector import detect_fruit


app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def welcome():
    """Professional landing page with project details"""
    return render_template("index.html")

@app.route("/detector")
def index():
    """Main fruit detection page"""
    return render_template("form.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]
    if file.filename == "":
        return "No selected file"

    # Save image
    filename = "upload.jpg"
    image_fs_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(image_fs_path)

    # Absolute path for YOLO & OpenCV
    image_fs_path = os.path.abspath(image_fs_path)

    # URL path for browser
    image_url = "/static/uploads/upload.jpg"

    # Detect fruit using YOLO
    detected_fruit, detect_conf = detect_fruit(image_fs_path)

    if detected_fruit is None:
        return render_template(
        "result.html",
        image_path=image_url,
        fruit_type="Not Supported",
        color="N/A",
        ripeness="Not supported",
        quality="N/A",
        confidence=0,
        recommendation=(
            "This fruit is not supported in the current version. "
            "Supported fruits: Apple, Banana, Orange, Tomato."
        )
    )


    # Ripeness analysis
    color, ripeness, quality, confidence, recommendation = analyze_image(
        image_fs_path, detected_fruit
    )

    return render_template(
        "result.html",
        image_path=image_url,   # 👈 URL, not filesystem path
        fruit_type=detected_fruit.capitalize(),
        color=color,
        ripeness=ripeness,
        quality=quality,
        confidence=confidence,
        recommendation=recommendation
    )



if __name__ == "__main__":
    app.run(debug=True)
