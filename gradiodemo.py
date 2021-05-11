import os
import random
from data import get_image_list
from model import create_model
from data import read_img_path, tensor_to_img, save_image
import gradio as gr
import torchtext
from PIL import Image

torch.hub.download_url_to_file('https://en.wikipedia.org/wiki/The_Great_Wave_off_Kanagawa#/media/File:Tsunami_by_hokusai_19th_century.jpg', 'wave.jpg')
torch.hub.download_url_to_file('https://cdn.pixabay.com/photo/2020/10/02/13/49/bridge-5621201_1280.jpg', 'building.jpg')

torchtext.utils.download_from_url("https://drive.google.com/uc?id=1RILKwUdjjBBngB17JHwhZNBEaW4Mr-Ml", root="./weights/")

def sketch2anime(img, load_size=512, gpu_ids=[]):
    model = create_model(gpu_ids)
    img, aus_resize = read_img_path(img.name, load_size)
    aus_tensor = model(img)
    aus_img = tensor_to_img(aus_tensor)
    image_pil = Image.fromarray(aus_img)
    image_pil = image_pil.resize(aus_resize, Image.BICUBIC)
    return image_pil

  
title = "Anime2Sketch"
description = "demo for Anime2Sketch. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.05703'>Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis</a> | <a href='https://github.com/Mukosame/Anime2Sketch'>Github Repo</a></p>"

gr.Interface(
    sketch2anime, 
    [gr.inputs.Image(type="file", label="Input")], 
    gr.outputs.Image(type="pil", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[
        ["test_samples/madoka.jpg"],
        ["building.jpg"],
        ["wave.jpg"]
    ]).launch(debug=True)