# SignLink – Real-Time ASL Translation System

## Overview

SignLink is a real-time American Sign Language (ASL) to text translation system designed to enable seamless communication between Deaf and hearing individuals. The system uses computer vision and AI to recognize sign language gestures from live video and convert them into readable text in real time.

SignLink can be used in two ways:

* As a virtual camera integrated with video platforms (e.g., Zoom, FaceTime)
* As a standalone application for in-person communication

## Problem

Deaf and Hard-of-Hearing individuals face significant communication barriers, especially in time-sensitive environments like healthcare. Access to interpreters is often limited, delayed, or unavailable in emergencies.

## Solution

SignLink provides real-time ASL-to-text translation, enabling direct and independent communication without relying on interpreters. This improves accessibility in critical scenarios such as:

* Emergency rooms
* Doctor appointments
* Video calls
* Everyday conversations

## How It Works

* Video input is captured from a webcam
* MediaPipe extracts hand and body landmarks from each frame
* A Temporal Vision Transformer processes sequences of movements
* A phrase assembly component converts recognized signs into continuous subtitles

## Project Structure

```
data_pose_personal/   # Custom dataset
src/                  # Core logic
features_hands.py     # Feature extraction
model.py              # Model
```


