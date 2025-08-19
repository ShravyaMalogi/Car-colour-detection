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

## 📂 Repository Structure

```
Car-colour-detection/
├── app.py
│ 
├── models/
│ └── blue_car_classifier.keras
│
├── yolo/
│ └── yolov8s.pt
│
├── notebooks/
│ └── Car_Colour_Detection.ipynb
│
├── data/
│ ├── input/
│ │ └── sample_input.mp4
│ └── output/
│   └── sample_output.mp4
│
├── results/
│ ├── evaluation_metrics.png
│ ├── confusion_matrix.png
│ └── classification_report.png
│
├── requirements.txt
└── README.md

``` 

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
<img src="data/previews/preview.jpeg" alt="Sample Output" width="400"/>

---

## 🏃 How to Run  
```bash
# Clone the repository
git clone https://github.com/ShravyaMalogi/Car-colour-detection.git
cd Car-colour-detection

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
