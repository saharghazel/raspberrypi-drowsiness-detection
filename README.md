**Driver Drowsiness Detection | Raspberry Pi 4 & Edge AI**

Real-time eye-state monitoring prototype designed for Raspberry Pi 4. The system combines computer vision with an on-device TensorFlow Lite CNN to classify eye images and display drowsiness alerts through a Flask dashboard. Optional GPIO outputs support a physical LED and buzzer.
![Dashboard interface preview](images/dashboard_preview.png)
Dashboard preview with illustrative detection data.

**Features**

- Live video capture from a USB webcam
- Face and eye localization using OpenCV Haar cascades
- Eye-state inference with a TensorFlow Lite model
- Web dashboard with live video, predictions, scores, statistics and alert history
- Visual and browser-audio alerts; optional LED and buzzer via GPIO
  
**Tech Stack**
  
Hardware: Raspberry Pi 4, USB webcam; optional LED and active buzzer  
Software: Python, OpenCV, NumPy, TensorFlow Lite, Flask, GPIO Zero  
Model input: 64 × 64 grayscale eye images

**Processing Pipeline**

`USB camera → Face/eye detection → Grayscale + resize (64 × 64) → Normalization → TFLite inference → Alert logic → Web dashboard / GPIO`
The inference score is evaluated against a configurable threshold in `app.py`. The current backend raises a persistent alert after 10 consecutive sleepy classifications.

**Repository Structure**

```text
.
├── app.py                 # Video capture, inference, Flask API and optional GPIO
├── model_yeux.tflite      # Trained eye-state classification model
├── templates/
│   └── dashboard.html     # Web interface
├── images/
│   └── dashboard_preview.png
├── docs/
│   └── project_report.pdf # Technical report (when included)
├── requirements.txt
└── .gitignore
```
**Run on Raspberry Pi 4**

Requirements: Raspberry Pi OS, Python 3, a USB webcam accessible through Video4Linux, and a TensorFlow Lite interpreter compatible with your OS and Python version.
```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-flask python3-gpiozero python3-venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```
Install a compatible TensorFlow Lite interpreter inside the environment (for example, `tflite-runtime` where a wheel is available). Then start the server:
```bash
python3 app.py
```
Open http://127.0.0.1:5000 on the Raspberry Pi. To access the dashboard from another device on a trusted local network:
```bash
HOST=0.0.0.0 python3 app.py
```
To select a different USB camera: `CAMERA_INDEX=1 python3 app.py`.
Optional hardware alerts
Connect an LED (with a current-limiting resistor) to BCM GPIO17 and a GPIO-compatible active buzzer to BCM GPIO27, using an appropriate driver circuit when necessary. Enable GPIO alerts with:
```bash
GPIO_ALERTS=1 python3 app.py
```
**Project Status**

This project was developed and successfully tested on a Raspberry Pi 4 as part of an academic engineering project at the National Engineering School of Sousse (ENISo), Tunisia.

The system integrates embedded computer vision, TensorFlow Lite inference, and real-time drowsiness detection.

Academic Evaluation: 17/20

The project demonstrates the practical implementation of Edge AI on embedded hardware for driver drowsiness monitoring.Author

**Sahar Ghazel  
Electronics Engineering Student — National Engineering School of Sousse (ENISo), Tunisia**
