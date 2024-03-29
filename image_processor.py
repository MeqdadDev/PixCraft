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

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

PROJECT_UPLOADS = 'static/uploads'

def add_signature(img):
    font = cv.FONT_HERSHEY_SIMPLEX
    position = (20, img.shape[0] - 30)
    color = (255, 255, 255)
    thickness = 1
    cv.putText(img, 'By PixCraft', position, font, 1, color, thickness, cv.LINE_AA)

def resize_image_512(img, filename):
    resized = cv.resize(img, (512, 512))
    add_signature(resized)
    cv.imwrite(PROJECT_UPLOADS + "/resized_512_" + filename, resized)
    return resized

def bgr2rgb(img, filename):
    rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    add_signature(rgb_image)
    cv.imwrite(PROJECT_UPLOADS + "/rgb_" + filename, img)
    return rgb_image

def rgb2gray(img, filename=None):
    gray_image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    add_signature(gray_image)
    if filename != None:
        cv.imwrite(PROJECT_UPLOADS + "/gray_" + filename, gray_image)
    return gray_image

def split_rgb_channels(img, filename):
    r, g, b = cv.split(img)
    # show the channels in one plot
    plt.figure(figsize=(15, 5))
    # Row=1, Cols=3, Fig.No.=1
    plt.subplot(1, 3, 1)
    plt.imshow(r, cmap='Reds_r')
    plt.title('Red Channel')
    # plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap='Greens_r')
    plt.title('Green Channel')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(b, cmap='Blues_r')
    plt.title('Blue Channel')
    
    plt.savefig(PROJECT_UPLOADS + "/split_rgb_channels_" + filename)
        
    channels_fig = cv.imread(PROJECT_UPLOADS + "/split_rgb_channels_" + filename)
    add_signature(channels_fig)
    return channels_fig

def average_blur(img, filename):
    prefix = "/average_blur_"
    sizes = [3, 9, 17, 25]
    output_images = []

    for size in sizes:
        blur_filter = np.ones((size, size)) / size**2
        output_images.append(cv.filter2D(img, -1, blur_filter))

    titles = ["Blur 3x3 Filter", "Blur 9x9 Filter", "Blur 17x17 Filter", "Blur 25x25 Filter"]
    save_images_figure(output_images, titles, filename, prefix)

    blur_fig = cv.imread(PROJECT_UPLOADS + prefix + filename)
    add_signature(blur_fig)
    return blur_fig

def gaussian_blur(img, filename):
    prefix = "/gaussian_blur_"
    g_scales = [3, 9, 17, 25]
    g_sizes = [19, 55, 101, 151]  # X6 times of scale
    g_output_images = []

    for i in range(len(g_scales)):
        g_output_images.append(cv.GaussianBlur(img, (g_sizes[i], g_sizes[i]), g_scales[i]))

    titles = ["(si-19, sc-3) Gaussian Filter", "(si-55, sc-9) Gaussian Filter", \
              "(si-101, sc-17) Gaussian Filter", "(si-151, sc-25) Gaussian Filter"]
    
    save_images_figure(g_output_images, titles, filename, prefix)

    g_blur_fig = cv.imread(PROJECT_UPLOADS + prefix + filename)
    add_signature(g_blur_fig)
    return g_blur_fig

def detect_edges(img, filename):
    prefix = "/detect_edges_"
    gray_image = rgb2gray(img)

    h_filter = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    catch_v_edges = cv.filter2D(gray_image, -1, h_filter)

    # convole the image with vertical mask to catch horizontal edges
    v_filter = np.transpose(h_filter)
    catch_h_edges = cv.filter2D(gray_image, -1, v_filter)

    # convole the image with a Laplacian mask to catch all edges
    laplacian_filter = np.array([[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, -1]])
    catch_all_edges = cv.filter2D(gray_image, -1, laplacian_filter)

    images = [catch_v_edges, catch_h_edges, catch_all_edges]
    titles = ["All Edges (Laplacian-Filter)", "H-Edges by V-Filter", "V-Edges by H-Filter"]

    save_images_figure(images, titles, filename, prefix)

    # plt.savefig(PROJECT_UPLOADS + "/detect_edges_" + filename)
    detect_edges_fig = cv.imread(PROJECT_UPLOADS + prefix + filename)
    add_signature(detect_edges_fig)
    return detect_edges_fig

def canny_edge_detection(img, filename):
    canny_edges = cv.Canny(img, 70, 150)
    cv.imwrite(PROJECT_UPLOADS + "/canny_edges_" + filename, canny_edges)
    return canny_edges

def hsv_model(img, filename):
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    cv.imwrite(PROJECT_UPLOADS + "/hsv_model_" + filename, hsv)
    return hsv


def save_images_figure(images, titles, filename, prefix):
    """
    Display multiple images in a single figure with corresponding titles.

    Args:
    - images: List of images to be displayed.
    - titles: List of titles for each image.
    - filename: Name of processed image file.
    - prefix: Prefix to added before filename.
    Returns:
    - None
    """
    if len(images) != len(titles):
        raise ValueError("The number of images does not match the number of titles.")
    num_images = len(images)
    plt.figure(figsize=(num_images * 5, 5))
    for i, (image, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, num_images, i)
        plt.title(title)
        plt.imshow(image, cmap="gray")
        # plt.axis('off')
    # plt.show()
    plt.savefig(PROJECT_UPLOADS + prefix + filename)
