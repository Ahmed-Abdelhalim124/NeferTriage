***NeferAI Triage***

This project implements an **AI-powered health emergency detection system** that can identify **falls** and **chest pain indicators** in real-time from video input. It combines **deep learning models**, **pose estimation**, and **YOLO-based person detection** to provide reliable alerts in critical situations.  

---

## 🚀 Features  
- ✅ Fall Detection using a custom-trained PyTorch model with temporal smoothing  
- ✅ Chest Pain Detection with EfficientNet and hand–chest interaction analysis  
- ✅ YOLOv8 Person Detection to isolate individuals for focused health analysis  
- ✅ MediaPipe Pose & Hand Landmarks for advanced body and hand positioning checks  
- ✅ Unified Video Processing with annotated video output  
- ✅ Emergency Alerts (fall, chest pain, or combined critical events)  

---

## 🛠️ Tech Stack  
- Python 3.10+  
- PyTorch & Torchvision  
- Ultralytics YOLOv8  
- MediaPipe  
- OpenCV  
- Gradio  
- NumPy & PIL  

---

## 📂 Project Structure  

├── app.py # Main implementation
├── best_chestpain_model2
├── fall_detector
├── README.md
└── requirements.txt 

📊 Output

Annotated video with labels and alerts:

✅ NO FALL / SITTING

🚨 FALL DETECTED

🫀 CHEST PAIN ALERT

🔴 CRITICAL: CHEST PAIN + FALL
