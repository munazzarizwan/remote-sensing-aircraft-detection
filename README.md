# remote-sensing-aircraft-detection
YOLOv8-based aircraft detection in satellite images with a complete offline processing pipeline.

This project implements an object detection system to detect aircraft in remote sensing satellite images using YOLOv8.

🔍 Features
Detects airplanes in satellite images
Fully offline working system
Input → Processing → Output pipeline
Outputs images with bounding boxes
🛠️ Tech Stack
Python
YOLOv8 (Ultralytics)
OpenCV
📂 Project Structure
input_image/ → Place input images here
output_image/ → Output results saved here
best.pt → Trained model
predict_script.py → Detection script
run.bat → Run file
▶️ How to Run
Install dependencies:
Place image in input_image/
Run bat file
Check results in output_image/
📌 Note
This project works completely offline and processes satellite images to detect aircraft using deep learning.
