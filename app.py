import os
from flask import Flask, request, jsonify, send_file
import base64
import numpy as np
from model import load_image, preprocess_image, segment_image,detect_objects,classify_image,caption_image
from flask_cors import CORS
from heapq import nlargest
app = Flask(__name__)
CORS(app)
with open("ImageNetLabels.txt", "r") as f:
    class_labels = f.read().splitlines()[1:]




@app.route("/image_processing", methods=["POST"])
def image_processing():
    if 'image' in request.files:
        file = request.files['image']
    elif 'image_url' in request.json:
        file = request.json['image_url']
    else:
        return jsonify({"error": "No image file or URL provided"}), 400

    image, _ = load_image(file)

    # Classification
    top_5_predictions = classify_image(image)
    
    # Segmentation
    segmented_image_buffer = segment_image(image)
    # Image captioning
    captioned_image = caption_image(image)

    # Encode the segmented image buffer to base64 for sending in JSON
    segmented_image_base64 = base64.b64encode(segmented_image_buffer.getvalue()).decode("utf-8")

    # Object detection
    detected_objects_buffer = detect_objects(image)
    detected_objects_base64 = base64.b64encode(detected_objects_buffer.getvalue()).decode("utf-8")


    return jsonify({"top_5_predictions": top_5_predictions, "segmented_image": segmented_image_base64, "detected_objects": detected_objects_base64,"captioned_image":captioned_image})



@app.route('/list_files')
def list_files():
    files = os.listdir('.')
    return {'files': files}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010)
