import gradio as gr
import os
from PIL import Image, ImageDraw

# Default image path
print(os.path.dirname(__file__))
image_path = os.path.join(os.path.dirname(__file__), 'img_18.jpg')

# Gradio function to detect faces
def rotate_image(image):
    return image.rotate(90)

# Define the Gradio app
iface = gr.Interface(
    fn=rotate_image,
    inputs = gr.Image(type="pil", value=
                      image_path),
    outputs="image",
    live=False,
)

if __name__ == "__main__":
    # Run the Gradio app
    iface.launch(share=False, server_name="0.0.0.0")