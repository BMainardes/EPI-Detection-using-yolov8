Overview
This project implements a real-time detection system for identifying personal protective equipment (PPE) using the YOLOv8 (You Only Look Once) object detection model. The system is capable of detecting various types of PPE such as helmets, safety vests, gloves, and more, ensuring that individuals in workplaces are complying with safety regulations.

Features
Real-time detection: Leverages YOLOv8 to detect PPE in real-time via connected cameras.
High accuracy: Uses the advanced capabilities of YOLOv8 for object detection, ensuring precise identification of equipment.
Scalability: Can be deployed in various industrial and construction settings.
Customizable: Easily adaptable to detect new types of PPE by retraining the model with custom datasets.
Technologies Used
Python: Main programming language.
YOLOv8: Object detection model.
OpenCV: For image and video processing.
Roboflow: For dataset preparation and augmentation.
Pytorch: Machine learning framework for YOLOv8.
Setup Instructions
Prerequisites
Python 3.8 or higher
pip for managing Python packages
A machine with GPU support is recommended for faster inference.



# Cloning the Repository

```
git clone https://github.com/BMainardes/EPI-Detection-using-yolov8.git
```
```
cd EPI-Detection-using-yolov8
```
# Install the required dependencies:
```
pip install -r requirements.txt
```
Download the pre-trained YOLOv8 weights or train your own model (instructions below).


Prepare your dataset using Roboflow or manually, ensuring proper labeling of PPE categories.
Modify the data.yaml file to include the correct paths to your dataset.
Train the model using the following command:

```
python train.py --data data.yaml --cfg yolov8.yaml --weights yolov8n.pt --epochs 50
```
Output
The detection results will be saved in the runs/detect/ directory, with bounding boxes drawn around detected PPE items.

Dataset
The dataset for this project should contain labeled images of individuals wearing various types of PPE (e.g., helmets, safety vests, etc.). You can use pre-labeled datasets or create your own using image labeling tools such as LabelImg or Roboflow.
