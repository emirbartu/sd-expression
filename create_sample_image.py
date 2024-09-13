from PIL import Image, ImageDraw

def create_sample_image(filename='sample_image.jpg', size=(512, 512), color='white'):
    # Create a new image with the given size and color
    image = Image.new('RGB', size, color)
    
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    
    # Draw a simple face (circle for head, dots for eyes, curve for smile)
    center_x, center_y = size[0] // 2, size[1] // 2
    radius = min(size) // 3
    
    # Draw head
    draw.ellipse([center_x - radius, center_y - radius, 
                  center_x + radius, center_y + radius], 
                 outline='black', width=3)
    
    # Draw eyes
    eye_radius = radius // 8
    left_eye_center = (center_x - radius // 2, center_y - radius // 4)
    right_eye_center = (center_x + radius // 2, center_y - radius // 4)
    draw.ellipse([left_eye_center[0] - eye_radius, left_eye_center[1] - eye_radius,
                  left_eye_center[0] + eye_radius, left_eye_center[1] + eye_radius], 
                 fill='black')
    draw.ellipse([right_eye_center[0] - eye_radius, right_eye_center[1] - eye_radius,
                  right_eye_center[0] + eye_radius, right_eye_center[1] + eye_radius], 
                 fill='black')
    
    # Draw smile
    smile_start = (center_x - radius // 2, center_y + radius // 4)
    smile_end = (center_x + radius // 2, center_y + radius // 4)
    draw.arc([smile_start[0], smile_start[1], 
              smile_end[0], smile_end[1] + radius // 4], 
             start=0, end=180, fill='black', width=3)
    
    # Save the image
    image.save(filename)
    print(f"Sample image created: {filename}")

if __name__ == "__main__":
    create_sample_image()
