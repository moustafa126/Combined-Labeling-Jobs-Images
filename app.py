import streamlit as st
import os
# Install libGL.so.1 if it's missing
if not os.path.isfile('/usr/lib/x86_64-linux-gnu/libGL.so.1'):
    os.system('sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx')
    
from ultralytics import YOLO
from PIL import Image

# Function to load the YOLO model
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to perform inference and draw bounding boxes
def detect_objects(image_path, model, conf_threshold):
    # Set the confidence threshold based on user input
    results = model(image_path, conf=conf_threshold)
    im_array = results[0].plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # Convert to RGB PIL image
    return im

# Streamlit app
st.title("Bryan & Lys Folder Images Only:n Sink Detection YOLO Models")

# Dropdown menu to select the model
model_options_pt = [
    'combined_classes_yolov8m.pt',
    'combined_classes_yolov8s.pt',
    'combined_classes_yolov9c.pt',
    'only_deficiency_yolov8m.pt',
    'only_deficiency_yolov8s.pt',
    'only_deficiency_yolov9c.pt',
    'two_classes_yolov8m.pt',
    'two_classes_yolov8s.pt',
    'two_classes_yolov9c.pt'
]

model_options_onnx = [
    'combined_classes_yolov8m.onnx',
    'combined_classes_yolov8s.onnx',
    'combined_classes_yolov9c.onnx',
    'only_deficiency_yolov8m.onnx',
    'only_deficiency_yolov8s.onnx',
    'only_deficiency_yolov9c.onnx',
    'two_classes_yolov8m.onnx',
    'two_classes_yolov8s.onnx',
    'two_classes_yolov9c.onnx'
]

model_options = model_options_pt
model_choice = st.selectbox("Choose a model", model_options)

# Confidence threshold input slider
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# File uploader to upload multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Submit") and uploaded_files:
    # Load the model
    model_path = os.path.join('models', model_choice)
    model = load_model(model_path)

    # Process each uploaded image
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Perform object detection
        image_with_boxes = detect_objects("temp_image.jpg", model, conf_threshold)

        # Display the image with bounding boxes
        st.image(image_with_boxes, caption=f'Detected Dirt in Sinks {uploaded_file.name}', use_column_width=True)