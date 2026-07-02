import os
from flask import Flask, render_template, request
import qrcode
import time

app = Flask(__name__)

# Define the path for saving generated QR codes
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
GENERATED_DIR = os.path.join(STATIC_DIR, "generated")

# Ensure the 'generated' directory exists
os.makedirs(GENERATED_DIR, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    qr_image_url = None
    input_data = ""

    if request.method == "POST":
        input_data = request.form.get("data")
        if input_data:
            # Generate a unique filename to avoid browser caching issues
            filename = f"qr_{int(time.time())}.png"
            filepath = os.path.join(GENERATED_DIR, filename)

            # Create QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(input_data)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")
            img.save(filepath)

            qr_image_url = f"/static/generated/{filename}"

    return render_template("index.html", qr_image_url=qr_image_url, input_data=input_data)

if __name__ == "__main__":
    app.run(debug=True)