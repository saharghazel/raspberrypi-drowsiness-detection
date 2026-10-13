# Driver Drowsiness Detection Using Raspberry Pi 4

## Overview

An embedded computer vision system designed to detect signs of driver drowsiness in real time using a Raspberry Pi 4, a camera, and machine learning.

The system analyzes visual information and activates an alert when drowsiness is detected.

## Technologies Used

- **Hardware:** Raspberry Pi 4, USB Camera, Buzzer, LED
- **Programming:** Python
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** TensorFlow Lite
- **Operating System:** Raspberry Pi OS

## System Architecture

Camera → Image Acquisition → Preprocessing → TFLite Model → Drowsiness Prediction → Alert System

## Features

- Real-time camera acquisition
- Eye-state analysis
- Lightweight TFLite inference
- Drowsiness warning using a buzzer and LED

## AI Model

The project uses a TensorFlow Lite model for eye-state classification.

- Input dimensions: 64 × 64 × 1
- Output: Binary classification
- Deployment platform: Raspberry Pi 4

## Evaluation

During an initial test with 20 samples (10 awake and 10 sleepy), the model classified all samples correctly. This is a small preliminary test, not a comprehensive accuracy benchmark.

## Installation

List the required libraries, hardware setup, and installation commands here.

## Demonstration

Add screenshots or a link to a short demonstration video.

## Author

Sahar Ghazel

Electronics Engineering Student — ENISo, Tunisia

Interests: Embedded Systems, Edge AI, Computer Vision, and Microcontrollers.
