'''
Copyright (c) [2024], MeqdadDev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# Required packages
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2 as cv
from werkzeug.utils import secure_filename

# Local modules
import image_processor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def index():
    return render_template('index.html')

def apply_image_processing(image, filename):
    resized = image_processor.resize_image_512(image, filename)
    rgb = image_processor.bgr2rgb(resized, filename)
    gray = image_processor.rgb2gray(rgb, filename)
    rgb_split_channels = image_processor.split_rgb_channels(rgb, filename)
    average_blur = image_processor.average_blur(rgb, filename)
    gaussian_blur = image_processor.gaussian_blur(rgb, filename)
    dejtect_edges = image_processor.detect_edges(rgb, filename)
    canny_edges = image_processor.canny_edge_detection(gray, filename)
    hsv_model = image_processor.hsv_model(rgb, filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        apply_image_processing(image, filename)
        return redirect(url_for('result', filename=filename))
    return render_template('upload.html')

@app.route('/result/<filename>')
def result(filename):
    return render_template("result.html", filename=filename)

@app.route('/view/rgb/<filename>')
def view_image(filename):
    filename = "rgb_" + filename
    return '<div align="center"><h3>RGB Image:</h3> \n<img src="' + \
        url_for('static', filename='uploads/' + filename) + '"></div>'

@app.route('/view/gray_scale/<filename>')
def view_gray(filename):
    filename = "gray_" + filename
    return '<div align="center"><h3>Gray Scale Image:</h3> \n<img src="' + \
        url_for('static', filename='uploads/' + filename) + '"></div>'

@app.route("/view/split_rgb_channels/<filename>")
def view_rgb_split_channels(filename):
    filename = "split_rgb_channels_" + filename
    return '<div align="center"><h3>RGB Channels Image [Split]:</h3> \n<img src="' + \
        url_for('static', filename='uploads/' + filename) + '"></div>'

@app.route("/view/average_blur/<filename>")
def average_blur(filename):
    filename = "average_blur_" + filename
    return '<div align="center"><h3>Average Blur Filters [Box Mask]:</h3> \n<img src="' + \
        url_for('static', filename='uploads/' + filename) + '"></div>'

@app.route("/view/gaussian_blur/<filename>")
def gaussian_blur(filename):
    filename = "gaussian_blur_" + filename
    return '<div align="center"><h3>Gaussian Blur Filters:</h3> \n<img src="' + \
        url_for('static', filename='uploads/' + filename) + '"></div>'

@app.route("/view/detect_edges/<filename>")
def detect_edges(filename):
    filename = "detect_edges_" + filename
    return '<div align="center"><h3>Detect Edges:</h3> \n<img src="' + \
        url_for('static', filename='uploads/' + filename) + '"></div>'

@app.route("/view/canny_edges/<filename>")
def canny_edges(filename):
    filename = "canny_edges_" + filename
    return '<div align="center"><h3>Canny Edge Detection (70, 150):</h3> \n<img src="' + \
        url_for('static', filename='uploads/' + filename) + '"></div>'

@app.route("/view/hsv_model/<filename>")
def hsv_model(filename):
    filename = "hsv_model_" + filename
    return '<div align="center"><h3>HSV Color Model:</h3> \n<img src="' + \
        url_for('static', filename='uploads/' + filename) + '"></div>'

if __name__ == '__main__':
    app.run(debug=True)

