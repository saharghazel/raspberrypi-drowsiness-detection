Driver Drowsiness Detection — Raspberry Pi 4 Deployment Adaptation
Status: Raspberry Pi 4–oriented adaptation of an existing PC prototype. Not yet verified on Raspberry Pi 4 hardware. Do not describe this as a completed Pi deployment until you run and test it on the board.
Overview
USB camera → OpenCV Haar face/eye detection → grayscale 64×64 normalization → provided TFLite eye-classification model → Flask web dashboard → optional LED and buzzer.
The provided model is kept unchanged. `templates/dashboard.html` is also kept unchanged. Its page text still refers to PC simulation; that description comes from the original dashboard and must not be mistaken for confirmation of Pi deployment.
Files
`app.py`: Python Flask inference server with Raspberry Pi camera and optional GPIO configuration.
`model\_yeux.tflite`: original supplied model (unchanged).
`templates/dashboard.html`: original supplied HTML dashboard (unchanged).
`docs/`: add your reviewed project report here as `project\_report.pdf`.
`images/`: add genuine screenshots/photos from your experiments.
Raspberry Pi prerequisites
Raspberry Pi 4, supported Raspberry Pi OS installation, internet access for setup.
USB UVC camera (`/dev/video0` or another `/dev/video\*` index); the provided code uses Video4Linux and is not configured for a CSI/libcamera camera.
Python 3, Flask, OpenCV, NumPy, and a compatible ARM TensorFlow Lite interpreter (prefer `tflite-runtime` where available, alternatively `ai-edge-litert` or TensorFlow).
Optional: LED with series resistor and suitable active buzzer wired using a safe GPIO-compatible circuit.
Initial software setup (Raspberry Pi OS)
```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-flask python3-gpiozero python3-venv
```
Install a Pi/OS/Python-version-compatible TensorFlow Lite interpreter using the appropriate package distribution. Availability depends on architecture and Raspberry Pi OS release. For example, in an environment where a compatible wheel exists:
```bash
python3 -m pip install tflite-runtime
```
On externally managed Raspberry Pi OS Python, prefer an isolated virtual environment (`python3 -m venv --system-site-packages .venv` and use `.venv/bin/pip`) instead of modifying system Python. If no compatible interpreter package is provided, consult the framework's current ARM installation instructions. Do not assume this command works on every Pi image.
Run
```bash
# In the project directory (and with interpreter installed):
python3 app.py
```
Open `http://127.0.0.1:5000` on the Pi itself. For access from another computer on your trusted LAN:
```bash
HOST=0.0.0.0 python3 app.py
```
Then browse to `http://<PI\_IP\_ADDRESS>:5000`. The Flask development server is for local demonstrations, not internet exposure.
If the USB webcam is not at index zero:
```bash
CAMERA\_INDEX=1 python3 app.py
```
Report
 in `docs/project\_report.pdf`
