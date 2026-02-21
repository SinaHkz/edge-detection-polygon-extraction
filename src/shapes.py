import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from skimage import measure  # For detecting contours

# Define Sobel filters for horizontal and vertical edge detection
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Horizontal edges
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Vertical edges

# Function to apply Sobel edge detection to an image using loops
def sobel_edge_detection(image):
    if len(image.shape) == 3:  # Convert to grayscale if the image is RGB
        image = image.mean(axis=2).astype(np.uint8)
    image = gaussian_filter(image, sigma=1)  # Reduce noise with Gaussian blur

    # Initialize gradient arrays
    height, width = image.shape
    grad_x = np.zeros_like(image, dtype=np.float32)
    grad_y = np.zeros_like(image, dtype=np.float32)

    # Apply Sobel operator using loops
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

    # Compute magnitude and apply a threshold
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edges = (magnitude > 50).astype(np.uint8) * 255
    return edges

# Function to find contours and create a polygonal graph
def create_polygon_from_edges(edge_map):
    # Find contours using skimage.measure
    contours = measure.find_contours(edge_map, level=128)  # Level threshold for edges

    # Select the longest contour (assumes it's the main object)
    if len(contours) > 0:
        largest_contour = max(contours, key=len)  # Pick the largest contour
        return largest_contour
    return None

# Main program
if __name__ == "__main__":
    # Load and process the image
    image_path = '4.jpg'  # Replace with your image path
    image = np.array(Image.open(image_path))
    edges = sobel_edge_detection(image)

    # Find the polygon from the edges
    polygon = create_polygon_from_edges(edges)

    if polygon is not None:
        # Create a blank image for the polygon
        height, width = edges.shape
        output_image = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(output_image)

        # Draw the polygon onto the blank image
        polygon_points = [(int(x[1]), int(x[0])) for x in polygon]
        draw.line(polygon_points + [polygon_points[0]], fill="red", width=2)

        # Save the resulting image
        output_path = "polygon_edges.jpg"
        output_image.save(output_path)
        print(f"Polygonal edges saved to {output_path}")
    else:
        print("No polygon found in the edges.")
