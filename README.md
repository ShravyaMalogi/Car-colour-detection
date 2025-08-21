# 🚙 Car Colour Detection along with People Count 

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)](https://opencv.org/)  
[![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-orange)](https://pjreddie.com/darknet/yolo/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

> Detects blue cars, counts vehicles, and counts people at traffic signals.  

---

## 📌 Overview  
This project uses YOLO object detection with a color classification model to **detect cars** at traffic signals, **identify their colors**, and **count people**.

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
├── style.css
│
├── notebooks/
│ └── Car_Colour_Detection.ipynb
│
├── data/
│ ├── input/
│ │ └── sample_inputs
│ └── output/
│   └── sample_outputs
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
- 🚗 **Colour-specific car detection**   
- 🔲 **Bounding Box** for cars 
- 📊 Display of **car colour & count + people count**  
- ⚡ Works with images and video files
- 🎯 Achieved **~96% Accuracy**  

---

## 📂 Dataset  
- **Source:** [Car Colours Dataset](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset)  
- **Format Used:** Images sorted into color-labeled folders
  
---

## 📜 How It Works  
1. **Detection** → YOLO detects cars & people in each frame  
2. **Classification** → Car colours are identified  
3. **Bounding Boxes** →  
   - Blue cars → Red boxes  
   - Other cars → Blue boxes  
4. **Counting** → Number of cars & people displayed on top 

---

## 📈 Example Output  

![visuals](/data/previews/preview1.jpg)
![output](/data/output/sample_output3.jpg)

---

## ▶️ How to Run  

In GitHub Codespaces
```bash
# Update the package list and then install the libgl1 graphics library
sudo apt-get update && sudo apt-get install libgl1

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

On Local System
```bash
# Clone the repository
git clone https://github.com/ShravyaMalogi/Car-colour-detection.git
cd Car-colour-detection

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
