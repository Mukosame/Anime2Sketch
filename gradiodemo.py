import os
import argparse
import shutil
import sys
from subprocess import call
from types import SimpleNamespace
from random import randint
import gradio as gr
import torchtext

torchtext.utils.download_from_url("https://drive.google.com/uc?id=1RILKwUdjjBBngB17JHwhZNBEaW4Mr-Ml", root="./weights/")

def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)

main_environment = os.getcwd()
def sketch2anime(img):
    _id = randint(1, 10000)
    INPUT_DIR = "/tmp/input_image" + str(_id) + "/"
    OUTPUT_DIR = "/tmp/output_image" + str(_id) + "/"
    run_cmd("rm -rf " + INPUT_DIR)
    run_cmd("rm -rf " + OUTPUT_DIR)
    run_cmd("mkdir " + INPUT_DIR)
    run_cmd("mkdir " + OUTPUT_DIR)
    img.save(INPUT_DIR + "1.jpg", "JPEG")
    opts = SimpleNamespace(dataroot=INPUT_DIR, output_dir=OUTPUT_DIR)

    # resolve relative paths before changing directory
    opts.input_folder = os.path.abspath(opts.dataroot)
    opts.output_folder = os.path.abspath(opts.output_dir)

    stage_1_input_dir = opts.input_folder
    stage_1_output_dir = os.path.join(opts.output_folder, "stage_1_restore_output")
    if not os.path.exists(stage_1_output_dir):
        os.makedirs(stage_1_output_dir)
        stage_1_command = (
            "sudo python3 test.py --dataroot "
            + stage_1_input_dir
            + " --load_size 512 "
            + " --output_dir "
            + stage_1_output_dir
        )
        run_cmd(stage_1_command)

    return os.path.join(OUTPUT_DIR, "stage_1_restore_output", "1.jpg")
  
title = "Anime2Sketch"
description = "demo for Anime2Sketch. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.05703'>Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis</a> | <a href='https://github.com/Mukosame/Anime2Sketch'>Github Repo</a></p>"

gr.Interface(
    sketch2anime, 
    [gr.inputs.Image(type="pil", label="Input")], 
    "image",
    title=title,
    description=description,
    article=article,
    examples=[
        ["test_samples/madoka.jpg"],
    ]).launch(debug=True)