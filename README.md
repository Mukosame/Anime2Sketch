# Anime2Sketch
*Anime2Sketch: A sketch extractor for illustration, anime art, manga (under construction)*

By [Xiaoyu Xiang](https://engineering.purdue.edu/people/xiaoyu.xiang.1)

![madoka demo](demos/madoka_in_out.png)

## Updates
- 2021.4.11: Upload the pretrained weights, and more test results.
- 2021.4.8: Create the repo

## Introduction
The repository contains the testing codes and pretrained weights for Anime2Sketch.

Anime2Sketch is a sketch extractor that works well on illustration, anime art, and manga. 

## Prerequisites
- Linux or macOS
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- CPU or NVIDIA GPU + CUDA CuDNN


## Get Started
### Installation 
Install the required packages: ```pip install -r requirements.txt```

### Download Pretrained Weights
Please download the weights from [GoogleDrive](https://drive.google.com/drive/folders/1Srf-WYUixK0wiUddc9y3pNKHHno5PN6R?usp=sharing), and put it into the [weights/](weights/) folder.

### Test
```Shell
python3 test.py --dataroot /your_path/dir --load_size 512
```
The above command includes two arguments:
- dataroot: your test file or directory
- load_size: due to the memory limit, we need to resize the input image before processing. By default, we resize it to `512x512`.

## More Results
Our model works well on anime arts:
![demo1](demos/demo1_in_out.png)
Turn handrawn photos to clean linearts:
![demo2](demos/demo2_in_out.png)
Simplify freehand sketches:
![demo3](demos/demo3_in_out.png)

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
```
