## For a single image

# !pip install kneed
import matplotlib.image as img
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.vq import whiten, kmeans
import seaborn as sns
import numpy as np
from kneed import KneeLocator # Now you can import it
import zipfile
import os
from PIL import Image
from webcolors import hex_to_rgb, rgb_to_name
import cv2
from google.colab.patches import cv2_imshow
from skimage import color
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the model
data_cat = ['Multi_color', 'Single_color']
model = load_model('/content/drive/MyDrive/Colab Notebooks/cnn1.h5')

def preprocess_and_predict(img_path):
    image = load_img(img_path, target_size=(180, 180))
    img_arr = img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)

    # Make prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    predicted_class = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100

    print(f'Color: {predicted_class} with {confidence:.2f}% confidence')

    # Display the image and prediction
    # plt.imshow(image)
    # plt.title(f'Color: {predicted_class} with {confidence:.2f}% confidence')
    # plt.axis('off')
    # plt.show()

    return predicted_class

# CSS color names and their corresponding hex values
css_colors = {
    '#F0F8FF': 'AliceBlue',
    '#FAEBD7': 'AntiqueWhite',
    '#00FFFF': 'Aqua',
    '#7FFFD4': 'Aquamarine',
    '#F0FFFF': 'Azure',
    '#F5F5DC': 'Beige',
    '#FFE4C4': 'Bisque',
    '#000000': 'Black',
    '#FFEBCD': 'BlanchedAlmond',
    '#0000FF': 'Blue',
    '#8A2BE2': 'BlueViolet',
    '#A52A2A': 'Brown',
    '#DEB887': 'BurlyWood',
    '#5F9EA0': 'CadetBlue',
    '#7FFF00': 'Chartreuse',
    '#D2691E': 'Chocolate',
    '#FF7F50': 'Coral',
    '#6495ED': 'CornflowerBlue',
    '#FFF8DC': 'Cornsilk',
    '#DC143C': 'Crimson',
    '#00FFFF': 'Cyan',
    '#00008B': 'DarkBlue',
    '#008B8B': 'DarkCyan',
    '#B8860B': 'DarkGoldenRod',
    '#A9A9A9': 'DarkGray',
    '#006400': 'DarkGreen',
    '#BDB76B': 'DarkKhaki',
    '#8B008B': 'DarkMagenta',
    '#556B2F': 'DarkOliveGreen',
    '#FF8C00': 'DarkOrange',
    '#9932CC': 'DarkOrchid',
    '#8B0000': 'DarkRed',
    '#E9967A': 'DarkSalmon',
    '#8FBC8F': 'DarkSeaGreen',
    '#483D8B': 'DarkSlateBlue',
    '#2F4F4F': 'DarkSlateGray',
    '#00CED1': 'DarkTurquoise',
    '#9400D3': 'DarkViolet',
    '#FF1493': 'DeepPink',
    '#00BFFF': 'DeepSkyBlue',
    '#696969': 'DimGray',
    '#1E90FF': 'DodgerBlue',
    '#B22222': 'FireBrick',
    '#FFFAF0': 'FloralWhite',
    '#228B22': 'ForestGreen',
    '#FF00FF': 'Fuchsia',
    '#DCDCDC': 'Gainsboro',
    '#F8F8FF': 'GhostWhite',
    '#FFD700': 'Gold',
    '#DAA520': 'GoldenRod',
    '#808080': 'Gray',
    '#008000': 'Green',
    '#ADFF2F': 'GreenYellow',
    '#F0FFF0': 'HoneyDew',
    '#FF69B4': 'HotPink',
    '#CD5C5C': 'IndianRed',
    '#4B0082': 'Indigo',
    '#FFFFF0': 'Ivory',
    '#F0E68C': 'Khaki',
    '#E6E6FA': 'Lavender',
    '#FFF0F5': 'LavenderBlush',
    '#7CFC00': 'LawnGreen',
    '#FFFACD': 'LemonChiffon',
    '#ADD8E6': 'LightBlue',
    '#F08080': 'LightCoral',
    '#E0FFFF': 'LightCyan',
    '#FAFAD2': 'LightGoldenRodYellow',
    '#D3D3D3': 'LightGray',
    '#90EE90': 'LightGreen',
    '#FFB6C1': 'LightPink',
    '#FFA07A': 'LightSalmon',
    '#20B2AA': 'LightSeaGreen',
    '#87CEFA': 'LightSkyBlue',
    '#778899': 'LightSlateGray',
    '#B0C4DE': 'LightSteelBlue',
    '#FFFFE0': 'LightYellow',
    '#00FF00': 'Lime',
    '#32CD32': 'LimeGreen',
    '#FAF0E6': 'Linen',
    '#FF00FF': 'Magenta',
    '#800000': 'Maroon',
    '#66CDAA': 'MediumAquaMarine',
    '#0000CD': 'MediumBlue',
    '#BA55D3': 'MediumOrchid',
    '#9370DB': 'MediumPurple',
    '#3CB371': 'MediumSeaGreen',
    '#7B68EE': 'MediumSlateBlue',
    '#00FA9A': 'MediumSpringGreen',
    '#48D1CC': 'MediumTurquoise',
    '#C71585': 'MediumVioletRed',
    '#191970': 'MidnightBlue',
    '#F5FFFA': 'MintCream',
    '#FFE4E1': 'MistyRose',
    '#FFE4B5': 'Moccasin',
    '#FFDEAD': 'NavajoWhite',
    '#000080': 'Navy',
    '#FDF5E6': 'OldLace',
    '#808000': 'Olive',
    '#6B8E23': 'OliveDrab',
    '#FFA500': 'Orange',
    '#FF4500': 'OrangeRed',
    '#DA70D6': 'Orchid',
    '#EEE8AA': 'PaleGoldenRod',
    '#98FB98': 'PaleGreen',
    '#AFEEEE': 'PaleTurquoise',
    '#DB7093': 'PaleVioletRed',
    '#FFEFD5': 'PapayaWhip',
    '#FFDAB9': 'PeachPuff',
    '#CD853F': 'Peru',
    '#FFC0CB': 'Pink',
    '#DDA0DD': 'Plum',
    '#B0E0E6': 'PowderBlue',
    '#800080': 'Purple',
    '#663399': 'RebeccaPurple',
    '#FF0000': 'Red',
    '#BC8F8F': 'RosyBrown',
    '#4169E1': 'RoyalBlue',
    '#8B4513': 'SaddleBrown',
    '#FA8072': 'Salmon',
    '#F4A460': 'SandyBrown',
    '#2E8B57': 'SeaGreen',
    '#FFF5EE': 'SeaShell',
    '#A0522D': 'Sienna',
    '#C0C0C0': 'Silver',
    '#87CEEB': 'SkyBlue',
    '#6A5ACD': 'SlateBlue',
    '#708090': 'SlateGray',
    '#FFFAFA': 'Snow',
    '#00FF7F': 'SpringGreen',
    '#4682B4': 'SteelBlue',
    '#D2B48C': 'Tan',
    '#008080': 'Teal',
    '#D8BFD8': 'Thistle',
    '#FF6347': 'Tomato',
    '#40E0D0': 'Turquoise',
    '#EE82EE': 'Violet',
    '#F5DEB3': 'Wheat',
    '#FFFFFF': 'White',
    '#F5F5F5': 'WhiteSmoke',
    '#FFFF00': 'Yellow',
    '#9ACD32': 'YellowGreen'
}

# Function to load image and resize
def load_and_resize_image(image_path, max_pixels=50000):
    # Open image using PIL
    image_pil = Image.open(image_path)

    original_width, original_height = image_pil.size
    # Calculate aspect ratio
    aspect_ratio = original_width / original_height
    # Calculate new dimensions while maintaining aspect ratio
    new_height = int(np.sqrt(max_pixels / aspect_ratio))
    new_width = int(new_height * aspect_ratio)
    # Resize image
    image_resized = image_pil.resize((new_width, new_height))
    # Convert image to NumPy array
    # Removed the normalization that caused the floating point issue
    return np.array(image_resized)

# Convert RGB to Hex
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

# Find closest CSS color name and hex value
def closest_css_color(rgb):
    min_diff = float('inf')
    closest_color_name = None
    closest_color_hex = None
    for hex_value, name in css_colors.items():
        css_rgb = hex_to_rgb(hex_value)
        diff = np.sqrt(np.sum((np.array(css_rgb) - np.array(rgb)) ** 2))
        if diff < min_diff:
            min_diff = diff
            closest_color_name = name
            closest_color_hex = hex_value
    return closest_color_name, closest_color_hex

# Function to convert hex to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Function to convert RGB to HSV
def rgb_to_hsv(rgb_color):
    color = np.uint8([[list(rgb_color)]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)[0][0]
    return hsv_color

# Function to find and draw contours for the given color
def find_and_draw_contours(image, hex_color, tolerance=30):
    # Convert hex color to RGB
    target_color_rgb = hex_to_rgb(hex_color)

    # Convert target color to HSV
    target_color_hsv = rgb_to_hsv(target_color_rgb)
    # print(f"Target HSV Color: {target_color_hsv}")

    # Define lower and upper bounds for the color
    lower_bound = np.array([max(target_color_hsv[0] - tolerance, 0), max(target_color_hsv[1] - tolerance, 0), max(target_color_hsv[2] - tolerance, 0)])
    upper_bound = np.array([min(target_color_hsv[0] + tolerance, 179), min(target_color_hsv[1] + tolerance, 255), min(target_color_hsv[2] + tolerance, 255)])
    # print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

    # Convert image to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Create a mask for the color
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    print("Mask:")
    cv2_imshow(mask)

    # The black region in the mask has the value of 0,
    # so when multiplied with the original image removes all non-target regions
    # result = cv2.bitwise_and(image, image, mask=mask)
    # print("Result:")
    # cv2_imshow(result)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the image
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    return image_with_contours

# Define function to process each image
def process_image(image_path):
    prediction = preprocess_and_predict(image_path)
    # Load and resize image
    image = load_and_resize_image(image_path)
    print("\n")
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    print("Resized Image Dimensions:", image.shape)

    # Remove alpha channel if present
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Convert RGB image to CIELAB
    image_lab = color.rgb2lab(image)

    # Flatten the image array and extract LAB values
    l, a, b = [], [], []
    for row in image_lab:
        for pixel in row:
            l.append(pixel[0])
            a.append(pixel[1])
            b.append(pixel[2])

    # Create DataFrame
    color_df = pd.DataFrame({'L': l, 'A': a, 'B': b})

    # Scale the LAB values
    color_df['scaled_L'] = whiten(color_df['L'])
    color_df['scaled_A'] = whiten(color_df['A'])
    color_df['scaled_B'] = whiten(color_df['B'])

    if prediction == 'Single_color':
        optimal_clusters = 1
    else:
        # Preparing data for elbow plot
        distortions = []
        num_clusters = range(1, 11)  # Evaluate 1 to 7 clusters


        for i in num_clusters:
            _, distortion = kmeans(
                color_df[['scaled_L', 'scaled_A', 'scaled_B']], i
            )
            distortions.append(distortion)

        # Use kneed to find the elbow
        kneedle = KneeLocator(num_clusters, distortions, curve='convex', direction='decreasing')
        optimal_clusters = kneedle.elbow

    print(f'Optimal number of clusters: {optimal_clusters}')

    # Plot elbow plot
    # elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

    # sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
    # plt.xticks(num_clusters)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Distortion')
    # plt.axvline(optimal_clusters, linestyle='--', color='red', label=f'Optimal Clusters: {optimal_clusters}')
    # plt.legend()
    # plt.show()

    # Reshape image for clustering
    # image_reshaped = image.reshape(-1, 3)

    # # Ensure pixel values are in range [0, 255]
    # if image.max() <= 1.0:
    #     image_reshaped *= 255

    # # Convert to DataFrame
    # image_df = pd.DataFrame(image_reshaped, columns=['L', 'A', 'B'])

    # Perform k-means clustering to find dominant colors
    # Convert pixel values to float for k-means
    cluster_centers, _ = kmeans(color_df[['L', 'A', 'B']].astype(float), optimal_clusters)

    # Convert LAB cluster centers to RGB
    dominant_colors_lab = cluster_centers.reshape(-1, 1, 3)
    dominant_colors_rgb = color.lab2rgb(dominant_colors_lab).reshape(-1, 3)

    # print("Dominant colors (in LAB space):", cluster_centers)
    # print("Dominant colors (in RGB space):", dominant_colors_rgb)

    # Convert dominant colors to Hex and find closest CSS color names
    dominant_hex = [rgb_to_hex(color * 255) for color in dominant_colors_rgb]
    closest_css_colors = [closest_css_color(color * 255) for color in dominant_colors_rgb]

    # Print Hex codes and corresponding CSS color names
    for i, hex_code in enumerate(dominant_hex):
        print(f"Color {i + 1}: {hex_code} - Closest CSS color: {closest_css_colors[i]}")

    # Print Hex codes and corresponding CSS color names
    # for i, (hex_code, (color_name, _)) in enumerate(zip(dominant_hex, closest_css_colors)):
    #     print(f"Color {i + 1}: {hex_code} - Closest CSS color: {color_name}")

    # Create reference colors palette from closest CSS colors
    reference_colors = [np.array(hex_to_rgb(hex_code)) / 255.0 for _, hex_code in closest_css_colors]


    # Visualize dominant colors
    plt.imshow([dominant_colors_rgb])
    plt.axis('off')
    plt.title('Dominant colors')
    plt.show()

    # Visulaize Reference colors
    plt.imshow([reference_colors])
    plt.axis('off')
    plt.title('Reference colors')
    plt.show()

    # Draw contours for each dominant color
    # for hex_code in dominant_hex:
    #     image_with_contours = find_and_draw_contours(image, hex_code)
    #     plt.imshow(image_with_contours)
    #     plt.axis('off')
    #     plt.title(f'Contours for color: {hex_code}')
    #     plt.show()

# process_image('/content/WhatsApp Image 2025-04-05 at 7.30.54 PM.jpeg')
# process_image('/content/WhatsApp Image 2025-04-05 at 7.35.47 PM.jpeg')