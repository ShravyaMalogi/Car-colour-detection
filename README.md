# 🚦 Car Colour Detection & People Counting at Traffic Signals  

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)](https://opencv.org/)  
[![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-orange)](https://pjreddie.com/darknet/yolo/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

> Detects blue cars, counts vehicles, and counts people at traffic signals — all in **real-time**.  

---

## 📌 Overview  
This project uses **YOLO object detection** combined with a **colour classification model** to detect cars at a traffic signal and determine their colour.  
- **Blue cars** → Highlighted with **red bounding boxes**  
- **Other cars** → Highlighted with **blue bounding boxes**  
- **People** → Counted and displayed on-screen  

It’s useful for **traffic monitoring, enforcement, and analytics**.  

---

## 🚀 Features  
- 🚗 **Colour-specific car detection** (blue vs other colours)  
- 🟥 Red bounding box for blue cars, 🟦 blue bounding box for others  
- 👥 People counting at traffic signals  
- 📊 Real-time display of **car count + people count**  
- ⚡ Works with live camera or video files  

---

## 🛠 Tech Stack  
- **Python 3.x**  
- **OpenCV** for image & video processing  
- **YOLO** for car & person detection  
- **NumPy** for data handling  
- **Pillow** for image preprocessing  

---

## 📜 How It Works  
1. **Detection** → YOLO detects cars & people in each frame  
2. **Classification** → Car colours are identified (blue or not)  
3. **Bounding Boxes** →  
   - Blue cars → Red boxes  
   - Other cars → Blue boxes  
4. **Counting** → Number of cars & people displayed live  

---

## 📈 Example Output  
