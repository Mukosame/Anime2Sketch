import os 
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch 

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    """if a given filename is a valid image
    Parameters:
        filename (str) -- image filename
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_list(path):
    """read the paths of valid images from the given directory path
    Parameters:
        path (str)    -- input directory path
    """
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def get_transform(load_size=0, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if load_size > 0:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def read_img_path(path, load_size):
    """read tensors from a given image path
    Parameters:
        path (str)     -- input image path
        load_size(int) -- the input size. If <= 0, don't resize
    """
    img = Image.open(path).convert('RGB')
    aus_resize = None
    if load_size > 0:
        aus_resize = img.size
    transform = get_transform(load_size=load_size)
    image = transform(img)
    return image.unsqueeze(0), aus_resize 

def tensor_to_img(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, output_resize=None):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array)    -- input numpy array
        image_path (str)             -- the path of the image
        output_resize(None or tuple) -- the output size. If None, don't resize
    """

    image_pil = Image.fromarray(image_numpy)
    if output_resize:
        image_pil = image_pil.resize(output_resize, Image.BICUBIC)
    image_pil.save(image_path)