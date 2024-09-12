from PIL import Image
import streamlit as st
from models.Flow_MS import FlowMS, mask_to_gaussian
from utils.util import parse_args
import torch
import numpy as np
import cv2
import streamlit_drawable_canvas as sdc

@st.cache_data
def load_image(image_file, _n_steps):
    img = Image.open(image_file).convert('RGB')
    noise_seg, color_seg = generate_segmentation(img, model, _n_steps)
    return img, noise_seg, color_seg

def generate_segmentation(_img, _model, _n_steps):
    image = np.array(_img)
    image = cv2.resize(image, (_model.args.size, _model.args.size))
    # turn image into torch tensor
    image = image.astype(np.float32)
    image = (image / 127.5) - 1
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).to(_model.device)
    noise_seg, color_seg = _model.segment_edit(image, _n_steps)
    color_seg = color_seg.squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    return noise_seg, color_seg

def sample_image(_noise, _model, _n_steps):
    image = _model.sample_edit(_noise, _n_steps)
    image = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    # clip the image to be between 0 and 1
    image = np.clip(image, 0, 1)
    image = (image * 255.0).astype(np.uint8)
    return image

@st.cache_resource
def load_model(_args):
    model = FlowMS(_args)
    model.load_model(_args.checkpoint)
    return model

# allow the user to upload an image
args = parse_args()
n_steps = st.slider("Number of steps to take:", 1, 100, 20)
model = load_model(args)
st.title("Image Segmentation")
st.write("Upload an image to segment:")
uploaded_image = st.file_uploader("Choose an image...", type="png")
if uploaded_image is not None:
    image, noise_seg, color_seg = load_image(uploaded_image, n_steps)
    recon_image = sample_image(noise_seg, model, n_steps)
    st.image(image, caption="Uploaded Image", width=300)
    st.image(color_seg, caption="Noise Segmentation", width=300)
    # create a canvas to allow the user to edit the color_seg
    st.write("Edit the segmentation:")
    # pick the class to edit
    #0: 'background'	1: 'skin'	2: 'nose'   3: 'eye_g'	4: 'l_eye'	5: 'r_eye' 6: 'l_brow'	7: 'r_brow'	8: 'l_ear' 9: 'r_ear'	10: 'mouth'	11: 'u_lip' 12: 'l_lip'	13: 'hair'	14: 'hat' 15: 'ear_r'	16: 'neck_l'	17: 'neck' 18: 'cloth'
    classes = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
    # add the name of the class in the select box
    class_pick = st.selectbox("Select the class to edit:", list(enumerate(classes)))
    class_to_edit = class_pick[0]
    # stroke color depends on the class
    stroke_color = f"rgb({model.colors[class_to_edit][0]}, {model.colors[class_to_edit][1]}, {model.colors[class_to_edit][2]})"
    canvas_result = sdc.st_canvas(
        fill_color="rgb(255, 255, 255)",  # Fixed fill color with some opacity
        stroke_width=10,
        stroke_color=stroke_color,
        update_streamlit=True,
        width=300,
        height=300,
        background_image=Image.fromarray(color_seg),
    )

    # create a button to generate the new image
    if st.button("Generate Image"):
        #convert canvas to numpy array
        edited_color_seg = np.array(canvas_result.image_data)
        # resize to 64x64
        edited_color_seg = cv2.resize(edited_color_seg, (model.args.size, model.args.size), interpolation=cv2.INTER_NEAREST)
        # make it gray scale
        edited_color_seg = cv2.cvtColor(edited_color_seg, cv2.COLOR_RGB2GRAY)
        # make it binary
        edited_color_seg[edited_color_seg <= 1] = 0
        edited_color_seg[edited_color_seg > 1] = 1
        new_noise = mask_to_gaussian(torch.from_numpy(edited_color_seg).unsqueeze(0), model.mean[class_to_edit], model.var, model.args.dist, noise_seg.shape).to(model.device)
        noise_seg = new_noise + noise_seg*(1-torch.from_numpy(edited_color_seg).to(model.device))
        recon_image = sample_image(noise_seg, model, n_steps)
        st.image(recon_image, caption="Generated Image", width=300)
        
