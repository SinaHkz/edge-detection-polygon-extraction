import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Define Sobel filters for horizontal and vertical edge detection
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Horizontal edges
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Vertical edges

# Function to apply Sobel edge detection to an image
def sobel_edge_detection(image):
    # Convert to grayscale if the image is in RGB
    if len(image.shape) == 3:  # Check if the image has 3 channels (RGB)
        image = image.mean(axis=2).astype(np.uint8)  # Convert to grayscale

    # Apply Gaussian blur to reduce noise
    image = gaussian_filter(image, sigma=1.5)  # Sigma controls the amount of blur

    # Get the size of the image
    height, width = image.shape

    # Create empty arrays for horizontal and vertical gradients
    grad_x = np.zeros_like(image, dtype=np.float32)
    grad_y = np.zeros_like(image, dtype=np.float32)

    # Apply Sobel operator using nested loops
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx, gy = 0, 0  # Initialize gradients
            # Iterate over the Sobel kernel
            for ki in range(-1, 2):  # Kernel height
                for kj in range(-1, 2):  # Kernel width
                    pixel = image[i + ki, j + kj]  # Get pixel value
                    gx += pixel * SOBEL_X[ki + 1, kj + 1]  # Apply horizontal kernel
                    gy += pixel * SOBEL_Y[ki + 1, kj + 1]  # Apply vertical kernel
            grad_x[i, j] = gx  # Store horizontal gradient
            grad_y[i, j] = gy  # Store vertical gradient

    # Compute the edge magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Apply threshold to create a binary edge map
    edges = (magnitude > 50).astype(np.uint8) * 255  # Threshold can be adjusted

    return edges

# Main program
if __name__ == "__main__":
    # Load the image
    image_path = 'table.jpg'  # Replace with your image path
    image = np.array(Image.open(image_path))

    # Detect edges using Sobel operator with noise reduction
    edges = sobel_edge_detection(image)

    # Save the result as a binary image
    output_path = 'edges_with_noise_reduction.jpg'
    Image.fromarray(edges).convert("L").save(output_path)
