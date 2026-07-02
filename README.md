# QR Code Generator

A simple web application built with Python and Flask to generate and download QR codes from text or URLs.

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS
- **QR Code Library**: `qrcode`
- **Image Processing**: `Pillow`

## Features

- Enter any text or URL into an input field.
- Instantly generate a QR code.
- Display the generated QR code on the page.
- Download the QR code as a PNG image.

## Project Structure

```
qr-code-generator/
├── app.py
├── requirements.txt
├── static/
│   ├── css/
│   │   └── style.css
│   └── generated/
└── templates/
    └── index.html
```

## Setup and Installation

1.  **Clone the repository or create the project files.**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**
    ```bash
    python app.py
    ```

5.  Open your web browser and navigate to `http://127.0.0.1:5000`.