# Anime2Sketch
*Anime2Sketch: A sketch extractor for illustration, anime art, manga*

By [Xiaoyu Xiang](https://engineering.purdue.edu/people/xiaoyu.xiang.1)

![teaser demo](demos/vinland_saga.gif)

## Gradio Web Demo
- [Web Demo](https://gradio.app/g/AK391/Anime2Sketch) by [**AK391**](https://github.com/AK391)
![gradio_web_demo](figures/gradiodemo.png)


## Updates
- 2021.5.2: Upload more example results of anime video.
- 2021.4.30: Upload the test scripts. Now our repo is ready to run!
- 2021.4.11: Upload the pretrained weights, and more test results.
- 2021.4.8: Create the repo.

## Introduction
The repository contains the testing codes and pretrained weights for Anime2Sketch.

Anime2Sketch is a sketch extractor that works well on illustration, anime art, and manga. It is an application based on the paper ["Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis"](https://arxiv.org/abs/2104.05703).

## Prerequisites
- Linux or macOS
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- CPU or NVIDIA GPU + CUDA CuDNN
- [Pillow](https://pillow.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/)


## Get Started
### Installation 
Install the required packages: ```pip install -r requirements.txt```

### Download Pretrained Weights
Please download the weights from [GoogleDrive](https://drive.google.com/drive/folders/1Srf-WYUixK0wiUddc9y3pNKHHno5PN6R?usp=sharing), and put it into the [weights/](weights/) folder.

### Test
```Shell
python3 test.py --dataroot /your_input/dir --load_size 512 --output_dir /your_output/dir
```
The above command includes three arguments:
- dataroot: your test file or directory
- load_size: due to the memory limit, we need to resize the input image before processing. By default, we resize it to `512x512`.
- output_dir: path of the output directory

Run our example:
```Shell
python3 test.py --dataroot test_samples/madoka.jpg --load_size 512 --output_dir results/
```

### Train
This project is a sub-branch of [AODA](https://github.com/Mukosame/AODA). Please check it for the training instructions.

## More Results
Our model works well on illustration arts:
![madoka demo](demos/madoka_in_out.png)
![demo1](demos/demo1_in_out.png)
Turn handrawn photos to clean linearts:
![demo2](demos/demo2_in_out.png)
Simplify freehand sketches:
![demo3](demos/demo3_in_out.png)
And more anime results:
![demo4](demos/vinland_3.gif)
![demo5](demos/vinland_1.gif)

## Contact
[Xiaoyu Xiang](https://engineering.purdue.edu/people/xiaoyu.xiang.1).

You can also leave your questions as issues in the repository. I will be glad to answer them!

## License
This project is released under the [MIT License](LICENSE).

## Citations
```BibTex
@misc{Anime2Sketch,
  author = {Xiaoyu Xiang, Ding Liu, Xiao Yang, Yiheng Zhu, Xiaohui Shen},
  title = {Anime2Sketch: A Sketch Extractor for Anime Arts with Deep Networks},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Mukosame/Anime2Sketch}}
}

@misc{xiang2021adversarial,
      title={Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis}, 
      author={Xiang, Xiaoyu and Liu, Ding and Yang, Xiao and Zhu, Yiheng and Shen, Xiaohui and Allebach, Jan P},
      year={2021},
      eprint={2104.05703},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
