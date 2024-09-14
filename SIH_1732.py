import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from matplotlib import pyplot as plt

# Function to load the grayscale image
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Error: Could not load the image. Please check the path and try again.")
    return img

# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(image)
    return cl

# Function to apply Gamma Correction using TensorFlow
def adjust_gamma(image, gamma=1.2):
    image = img_to_array(image)
    image = tf.image.adjust_gamma(image, gamma)
    image = array_to_img(image)
    return np.array(image)

# Adaptive filter based on Laplacian noise estimation
def adaptive_filter(image, h):
    rows, cols = image.shape
    denoised_image = np.zeros_like(image)
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            local_window = image[i-1:i+2, j-1:j+2]
            local_mean = np.mean(local_window)
            local_variance = np.var(local_window)
            
            if local_variance == 0:
                denoised_image[i, j] = image[i, j]
            else:
                local_noise_variance = max(0, local_variance - h)
                denoised_image[i, j] = (image[i, j] - local_mean) * (local_noise_variance / local_variance) + local_mean
                
    return denoised_image

# Function to tile the large image with overlap
def tile_image(image, tile_size, overlap):
    h, w = image.shape
    tiles = []
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append((x, y, tile))
    return tiles, h, w

# Function to blend overlapping tiles
def blend_tiles(result_image, x, y, tile, overlap):
    h, w = tile.shape
    blend_region = result_image[y:y + h, x:x + w]
    blend_weight = np.ones(tile.shape, dtype=np.float32)
    
    if x > 0:
        blend_weight[:, :overlap] *= np.linspace(0, 1, overlap)[np.newaxis, :]
    if y > 0:
        blend_weight[:overlap, :] *= np.linspace(0, 1, overlap)[:, np.newaxis]
    
    result_image[y:y + h, x:x + w] = (result_image[y:y + h, x:x + w] * (1 - blend_weight) + tile * blend_weight).astype(np.uint8)

# Function to apply Retinex enhancement (placeholder, implement as needed)
def retinex_enhancement(tile):
    return tile  # Implement Retinex if needed

# Mean Absolute Error
def mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

# Mean Squared Error
def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

# Peak Signal-to-Noise Ratio
def psnr(img1, img2):
    mse_value = mse(img1, img2)
    if mse_value == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel) - 10 * np.log10(mse_value)

# Signal-to-Noise Ratio
def snr(img1, img2):
    noise = img1 - img2
    variance = np.var(noise)
    if variance == 0:
        return float('inf')
    return 10 * np.log10(1 / variance)

# Main function to enhance large grayscale images with overlapping tiles and blending
def enhance_low_light_image(image_path, h, tile_size=1000, overlap=50):
    image = load_image(image_path)
    tiles, img_h, img_w = tile_image(image, tile_size, overlap)
    result_image = np.zeros((img_h, img_w), dtype=np.uint8)

    for (x, y, tile) in tiles:
        retinex_image = retinex_enhancement(tile)
        contrast_enhanced_image = apply_clahe(retinex_image)
        gamma_corrected_image = adjust_gamma(contrast_enhanced_image, gamma=1.2)
        adaptive_filtered_image = adaptive_filter(gamma_corrected_image, h)
        blend_tiles(result_image, x, y, adaptive_filtered_image, overlap)

    # Calculate and print image quality metrics
    print(f"MAE: {mae(image, result_image):.2f}")
    print(f"MSE: {mse(image, result_image):.2f}")
    print(f"PSNR: {psnr(image, result_image):.2f} dB")
    print(f"SNR: {snr(image, result_image):.2f} dB")

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result_image, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return result_image

# Example usage
image_path = '/Users/rishabhrai/Coding/python/low light images/slice_x0_y49000.png'
h = 0.5  # Noise variance estimation
enhanced_image = enhance_low_light_image(image_path, h, tile_size=1000, overlap=100)

# Save the final enhanced grayscale image
cv2.imwrite('enhanced_lunar_image_combined.png', enhanced_image)
