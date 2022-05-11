"""Test script for anime-to-sketch translation
Example:
    python3 test.py --dataroot /your_path/dir --load_size 512
    python3 test.py --dataroot /your_path/img.jpg --load_size 512
"""

import os
import torch
from data import get_image_list
from model import create_model
from data import read_img_path, tensor_to_img, save_image
import argparse
from tqdm.auto import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anime-to-sketch test options.')
    parser.add_argument('--dataroot','-i', default='test_samples/', type=str)
    parser.add_argument('--load_size','-s', default=512, type=int)
    parser.add_argument('--output_dir','-o', default='results/', type=str)
    parser.add_argument('--gpu_ids', '-g', default=[], help="gpu ids: e.g. 0 0,1,2 0,2.")
    opt = parser.parse_args()

    # create model
    gpu_list = ','.join(str(x) for x in opt.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    device = torch.device('cuda' if len(opt.gpu_ids)>0 else 'cpu')
    model = create_model().to(device)      # create a model given opt.model and other options
    model.eval()
    # get input data
    if os.path.isdir(opt.dataroot):
        test_list = get_image_list(opt.dataroot)
    elif os.path.isfile(opt.dataroot):
        test_list = [opt.dataroot]
    else:
        raise Exception("{} is not a valid directory or image file.".format(opt.dataroot))
    # save outputs
    save_dir = opt.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    for test_path in tqdm(test_list):
        basename = os.path.basename(test_path)
        aus_path = os.path.join(save_dir, basename)
        img,  aus_resize = read_img_path(test_path, opt.load_size)
        aus_tensor = model(img.to(device))
        aus_img = tensor_to_img(aus_tensor)
        save_image(aus_img, aus_path, aus_resize)
