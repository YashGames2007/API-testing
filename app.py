from flask import Flask, request, jsonify
import base64
import os
from datetime import datetime
import cv2
import requests
import threading
from PIL import Image
import numpy as np

# Replace with your RoboFlow API details
ROBOFLOW_API_KEY = "CnvKCPst0TvPtUwtzTf4"
ROBOFLOW_MODEL = "toy-cars-hbeml/1"
ROBOFLOW_VERSION = "1"

# RoboFlow API URL
API_URL = f"https://detect.roboflow.com/toy-cars-hbeml/1?api_key=CnvKCPst0TvPtUwtzTf4&overlap=70&confidence=50"

app = Flask(__name__)

UPLOAD_FOLDER = "new"
split_folder = "splits"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

cam1_image = "cam1_image.jpg"
cam2_image = "cam2_image.jpg"

cam1_count = 5
cam2_count = 0
cid = 0
cam_image_folder = "./cams"
signal = 2
time = 15
s1_count = 0
s2_count = 0
s3_count = 0
s4_count = 0
is_ambulance = False

@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid request format"}), 400

        # Extract image data
        image_data = data["image"]
        cid = data["cid"]

        # Get current timestamp
        # timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        # image_filename = f"Image_{timestamp}.jpg"
        # image_path = os.path.join(UPLOAD_FOLDER, image_filename)

        if cid == 1:
            image_filename = cam1_image
            cam_image_path = os.path.join(cam_image_folder, image_filename)
            # cam1_count = cid  # Now it updates the global variable
            # cam1_count = int(datetime.now().strftime("%M"))
            cam2_count = 99
        else:
            image_filename = cam2_image
            cam_image_path = os.path.join(cam_image_folder, image_filename)
            cam2_count = cid  # Now it updates the global variable
            # cam1_count = 99
            cam2_count = int(datetime.now().strftime("%M"))

        # cam1_count = int(datetime.now().strftime("%H"))
        # cam2_count = int(datetime.now().strftime("%M"))
        # Decode Base64 image
        image_bytes = base64.b64decode(image_data)

        # Save the image
        # with open(image_path, "wb") as img_file:
        #     img_file.write(image_bytes)
        with open(cam_image_path, "wb") as img_file:
            img_file.write(image_bytes)

        thread = threading.Thread(target=detect_objects, args=())
        thread.start()

        return jsonify({
            # "i1": cam1_count,
            # "i2": cam2_count,
            "cid": cid
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route('/ppost', methods=['POST'])
def ppost():
    global s1_count, s2_count, s3_count, s4_count, signal, time, is_ambulance  # Add this line

    try:
        data = request.get_json()
        if not data or "i1" not in data:
            return jsonify({"error": "Invalid request format"}), 400

        # Extract image data
        i1 = data["i1"]
        i2 = data["i2"]
        i3 = data["i3"]
        i4 = data["i4"]

        # Write Code to set time of signals if any cars are detected
        if is_ambulance:
            is_ambulance = False
            return jsonify({
            "signal": signal,
            "time": 17,
            "s1": i1,
            "s2": i2,
            "s3": i3,
            "s4": i4,
            })
        else:
            previous = [i1, i2, i3, i4]
            
            if max(previous) > 60:
                signal = previous.index(max(previous)) + 1
                time = previous[signal - 1]

                return jsonify({
                    "signal": signal,
                    "time": time * 15,
                    "s1": i1,
                    "s2": i2,
                    "s3": i3,
                    "s4": i4,
                })

            counts = [s1_count, s2_count, s3_count, s4_count]
            # Find the largest count
            largest = max(counts)

            # Find the index of the largest number
            signal = counts.index(largest) + 1
            time = counts[signal - 1]

        # print("\n\n\n\n" + i1, i2, i3, i4)
        return jsonify({
            "signal": signal,
            "time": time * 40,
            "s1": i1,
            "s2": i2,
            "s3": i3,
            "s4": i4,
            # "s1": s1_count,
            # "s2": s2_count,
            # "s3": s3_count,
            # "s4": s4_count,
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


def detect_objects():
    global s1_count, s2_count, s3_count, s4_count, is_ambulance, signal  # Add this line
    # global cam1_count, cam2_count, s1, s2, s3, s4  # Add this line

    split_images()

    images = [
        "cam2_left_half.jpg",   #1
        "cam2_right_half.jpg",  #2
        "cam1_left_half.jpg",   #3
        "cam1_right_half.jpg",  #4
    ]

    counts = [0, 0, 0, 0]
    
    for i, image in enumerate(images):
        # Open image using Pillow
        image_pil = Image.open(os.path.join(split_folder, image))

        # Convert Pillow Image to NumPy array
        image_np = np.array(image_pil)

        # Convert RGB to BGR (OpenCV uses BGR color order)
        image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Encode image as JPEG
        _, img_encoded = cv2.imencode('.jpg', image_np_bgr)

        # Send image to RoboFlow API
        response = requests.post(API_URL, files={"file": img_encoded.tobytes()})
        response_json = response.json()

        # Process detections
        detections = response_json.get("predictions", [])
        object_count = len(detections)
        counts[i] = object_count

        # Detecting Ambulance
        for detection in detections:
            if dict(detection)["class"] == "Ambulance":
                is_ambulance = True
                signal = i + 1  #Starts with 1


        # Draw bounding boxes
        for obj in detections:
            x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
            label = obj["class"]
            confidence = obj["confidence"]

            # Draw rectangle
            cv2.rectangle(image_np_bgr, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
            cv2.putText(image_np_bgr, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"Detected {object_count} objects.")
        
        # # Update global count variables
        # if i == 0:
        #     cam1_count = object_count
        # elif i == 1:
        #     cam1_count = object_count
        # elif i == 2:
        #     cam2_count = object_count
        # elif i == 3:
        #     cam2_count = object_count

        # Save the processed image with bounding boxes
        output_path = f"./output//{image}"
        cv2.imwrite(output_path, image_np_bgr)
        print(f"Saved output as {output_path}")

    # Assign final counts to s1, s2, s3, s4
    s1_count, s2_count, s3_count, s4_count = counts

    return counts



def split_images():
    for i in [1, 2]:
        # Open the image
        image_path = f"cam{i}_image.jpg"  # Replace with your image file
        image = Image.open(os.path.join(cam_image_folder, image_path))

        # Get image dimensions
        width, height = image.size
        midpoint = width // 2  # Middle of the image

        # Split the image into two halves
        left_half = image.crop((0, 0, midpoint, height))  # Left half
        right_half = image.crop((midpoint, 0, width, height))  # Right half
        # Get current timestamp
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        # Save the images
        left_half.save(os.path.join(split_folder, f"cam{i}_left_half.jpg"))
        # left_half.save(os.path.join(UPLOAD_FOLDER, f"cam{i}_left_half {timestamp}.jpg"))
        right_half.save(os.path.join(split_folder, f"cam{i}_right_half.jpg"))
        # left_half.save(os.path.join(UPLOAD_FOLDER, f"cam{i}_right_half {timestamp}.jpg"))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
