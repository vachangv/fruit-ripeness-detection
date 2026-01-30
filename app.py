from flask import Flask, render_template, request
import os
from image_processing import analyze_image


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
        return "No file part"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    # Get selected fruit type
    fruit_type = request.form.get("fruit_type", "banana")

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(image_path)

    # CALL IMAGE PROCESSING WITH FRUIT TYPE
    color, ripeness, quality, confidence, recommendation = analyze_image(image_path, fruit_type)

    return render_template(
        "result.html",
        image_path=image_path,
        color=color,
        ripeness=ripeness,
        quality=quality,
        confidence=confidence,
        recommendation=recommendation,
        fruit_type=fruit_type.capitalize()
    )


if __name__ == "__main__":
    app.run(debug=True)