from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from processing import *
import base64

app = Flask(__name__)

def encode_image_to_base64(image):
    """Encode a CV2 image to base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    algorithm = request.args.get('algorithm')
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    np_image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if algorithm == 'coloralg':
        saturation_factor = float(request.form.get('saturation', 1.0))
        color_results = coloralg(image, saturation_factor=saturation_factor)
        results = {
            'binary_image': encode_image_to_base64(color_results['binary_image']),
            'contours_image': encode_image_to_base64(color_results['contours_image']),
            'ellipses_image': encode_image_to_base64(color_results['ellipses_image']),
            'histogram_image': encode_image_to_base64(color_results['histogram_image']),
            'average_length': color_results['average_length'],
            'grain_count': color_results['grain_count']
        }
    elif algorithm == 'contouralg':
        contour_results = contouralg(image)
        results = {
            'binary_image': encode_image_to_base64(contour_results['binary_image']),
            'contours_image': encode_image_to_base64(contour_results['contours_image']),
            'ellipses_image': encode_image_to_base64(contour_results['ellipses_image']),
            'histogram_image': encode_image_to_base64(contour_results['histogram_image']),
            'average_length': contour_results['average_length'],
            'grain_count': contour_results['grain_count']
        }
    else:
        return jsonify({'error': 'Invalid algorithm'}), 400

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
