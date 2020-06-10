---
layout: page
title: Technology Watch
permalink: /watch/
tcc: true
---

# Table of Content

1. TOC
{:toc}


# Machine/Deep Learning tools
## Technics
### Transformers
- [The illustrated Transformers](http://jalammar.github.io/illustrated-transformer/)
- [Transformers explained](https://arxiv.org/abs/2001.04451)
- [Paper](https://arxiv.org/abs/1706.03762)

```
@misc{vaswani2017attention,
    title={Attention Is All You Need},
    author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year={2017},
    eprint={1706.03762},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### BERT
### BART
### XLNet
### GPT
### GPT2
### GPT3

### Reformers
[Colab](https://arxiv.org/abs/2001.04451)

## Useful Libs
### Wrappers
#### Vision
#### Text


| Tool                                             | Binary Classification | Multi-Label Classification | Question Answering | Tokenization | Generation | Named Entity Recognition |
|--------------------------------------------------|-----------------------|----------------------------|--------------------|--------------|------------|--------------------------|
| [Blurr](https://ohmeow.github.io/blurr/#Imports) |                       |                            |                    |              |            |                          |
|                                                  |                       |                            |                    |              |            |                          |
|                                                  |                       |                            |                    |              |            |                          |
#### Structured Data
### AutoMl
#### AutoKeras 

[AutoKeras](https://autokeras.com)

```python
from tensorflow.keras.datasets import mnist

import autokeras as ak

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)

# Initialize the ImageClassifier.
clf = ak.ImageClassifier(max_trials=3)
# Search for the best model.
clf.fit(x_train, y_train, epochs=10)
# Evaluate on the testing data.
print('Accuracy: {accuracy}'.format(
    accuracy=clf.evaluate(x_test, y_test)))
```

```
@inproceedings{jin2019auto,
  title={Auto-Keras: An Efficient Neural Architecture Search System},
  author={Jin, Haifeng and Song, Qingquan and Hu, Xia},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1946--1956},
  year={2019},
  organization={ACM}
}
```
# Deep Learning use cases

## Nothing to Image
### Generative
#### Face
##### Disentangled Image Generation Through Structured Noise Injection
![https://github.com/yalharbi/StructuredNoiseInjection/raw/master/example_fakes_alllocal.png](https://github.com/yalharbi/StructuredNoiseInjection/raw/master/example_fakes_alllocal.png)
- [Code](https://github.com/yalharbi/StructuredNoiseInjection)
- [Paper](https://arxiv.org/abs/2004.12411)

```
@misc{alharbi2020disentangled,
    title={Disentangled Image Generation Through Structured Noise Injection},
    author={Yazeed Alharbi and Peter Wonka},
    year={2020},
    eprint={2004.12411},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<iframe width="100%" src="https://www.youtube.com/embed/7h-7wso9E0k" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Image to Anything

### Image to Image
#### Inpainting
##### High-Resolution Image Inpainting with Iterative Confidence Feedback and Guided Upsampling
![https://s1.ax1x.com/2020/03/18/8wQG5T.jpg](https://s1.ax1x.com/2020/03/18/8wQG5T.jpg)
- [Project Page](https://zengxianyu.github.io/iic/)
- [APP](http://47.57.135.203:2333/)
- [Paper](https://arxiv.org/abs/2005.11742)

##### EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning
![https://user-images.githubusercontent.com/1743048/50673917-aac15080-0faf-11e9-9100-ef10864087c8.png](https://user-images.githubusercontent.com/1743048/50673917-aac15080-0faf-11e9-9100-ef10864087c8.png)
- [Code](https://github.com/zengxianyu/edge-connect)
- [Paper](https://arxiv.org/abs/1901.00212)

```
@inproceedings{nazeri2019edgeconnect,
  title={EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning},
  author={Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  journal={arXiv preprint},
  year={2019},
}
```

##### Progressive Image Inpainting with Full-Resolution Residual Network
![Before](https://github.com/zengxianyu/Inpainting_FRRN/blob/master/examples/ex_damaged2.png?raw=true)
![After](https://github.com/zengxianyu/Inpainting_FRRN/blob/master/examples/ex_final2.png?raw=true)
- [Code](https://github.com/zengxianyu/Inpainting_FRRN)
- [Paper](https://arxiv.org/abs/1907.10478)

```
@misc{guo2019progressive,
    title={Progressive Image Inpainting with Full-Resolution Residual Network},
    author={Zongyu Guo and Zhibo Chen and Tao Yu and Jiale Chen and Sen Liu},
    year={2019},
    eprint={1907.10478},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

#### Super resolution
##### PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models
![http://pulse.cs.duke.edu/assets/094.jpeg](http://pulse.cs.duke.edu/assets/094.jpeg)
- [Project Page](PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models)
- [Colab](https://colab.research.google.com/drive/1-cyGV0FoSrHcQSVq3gKOymGTMt0g63Xc?usp=sharing#sandboxMode=true)
- [Code](https://github.com/adamian98/pulse)
- [Paper](https://arxiv.org/abs/2003.03808)

```
@InProceedings{PULSE_CVPR_2020, 
author = {Menon, Sachit and Damian, Alex and Hu, McCourt and Ravi, Nikhil and Rudin, Cynthia}, 
title = {PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models}, 
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
month = {June}, 
year = {2020} 
}
```

##### Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining
![https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention/raw/master/Figs/Visual_3.png](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention/raw/master/Figs/Visual_3.png)
- [Code](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention)
- [Paper](https://arxiv.org/abs/2006.01424)

```
@inproceedings{Mei2020image,
  title={Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining},
  author={Mei, Yiqun and Fan, Yuchen and Zhou, Yuqian and Huang, Lichao and Huang, Thomas S and Shi, Humphrey},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}
```

#### Image-to-Image translation
##### DeepFaceDrawing: Deep Generation of Face Images from Sketches
![http://geometrylearning.com/DeepFaceDrawing/imgs/teaser.jpg](http://geometrylearning.com/DeepFaceDrawing/imgs/teaser.jpg)
- [Paper](http://geometrylearning.com/DeepFaceDrawing/)

<iframe width="100%" src="https://www.youtube.com/embed/HSunooUTwKs" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

##### UGATIT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation (ICLR 2020)
![https://github.com/taki0112/UGATIT/blob/master/assets/teaser.png?raw=true](https://github.com/taki0112/UGATIT/blob/master/assets/teaser.png?raw=true)
- [Code](https://github.com/taki0112/UGATIT#paper--official-pytorch-code)
- [Paper](https://arxiv.org/abs/1907.10830)

```
@inproceedings{
Kim2020U-GAT-IT:,
title={U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation},
author={Junho Kim and Minjae Kim and Hyeonwoo Kang and Kwang Hee Lee},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJlZ5ySKPH}
}
```

- [Author's Site](https://www.notion.so/Make-everyone-s-life-more-fun-using-AI-b15459d868bb490184e256cd95f26107€)

##### Selfie to Anime
![https://github.com/jqueguiner/databuzzword/blob/master/images/A578852A-9A4D-4D90-88E0-A4D81C7D41B3.jpeg](https://github.com/jqueguiner/databuzzword/blob/master/images/A578852A-9A4D-4D90-88E0-A4D81C7D41B3.jpeg?raw=true)
- [Project Page](https://selfie2anime.com/)
- [Code](https://github.com/t04glovern/selfie2anime)
- [API](https://market-place.ai.ovh.net/#!/apis/59a0426c-c148-4cff-a042-6cc148fcffa5/pages/06641de1-1b1c-4bd2-a41d-e11b1c3bd230)
- [Paper](https://github.com/t04glovern/selfie2anime/blob/master/assets/Deploying-Models-to-the-Masses.pdf)

```
@misc{kim2019ugatit,
    title={U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation},
    author={Junho Kim and Minjae Kim and Hyeonwoo Kang and Kwanghee Lee},
    year={2019},
    eprint={1907.10830},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
- [Author's Site](https://www.notion.so/Make-everyone-s-life-more-fun-using-AI-b15459d868bb490184e256cd95f26107)

#### Segmentation
##### Poly-YOLO: higher speed, more precise detection and instance segmentation for YOLOv3
![https://gitlab.com/irafm-ai/poly-yolo/-/raw/master/poly-yolo-titlepage-image.jpg?inline=false](https://gitlab.com/irafm-ai/poly-yolo/-/raw/master/poly-yolo-titlepage-image.jpg?inline=false)
- [Code](https://gitlab.com/irafm-ai/poly-yolo)
- [Paper](https://arxiv.org/abs/2005.13243)

```
@misc{hurtik2020polyyolo,
    title={Poly-YOLO: higher speed, more precise detection and instance segmentation for YOLOv3},
    author={Petr Hurtik and Vojtech Molek and Jan Hula and Marek Vajgl and Pavel Vlasanek and Tomas Nejezchleba},
    year={2020},
    eprint={2005.13243},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<iframe width="100%" src="https://www.youtube.com/embed/2KxNnEV-Zes" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

##### Attention-Guided Hierarchical Structure Aggregation for Image Matting
![https://wukaoliu.github.io/HAttMatting/figures/visualization.png](https://wukaoliu.github.io/HAttMatting/figures/visualization.png)
- [Project Page](https://wukaoliu.github.io/HAttMatting/)
- [Code](https://github.com/wukaoliu/CVPR2020-HAttMatting)
- [Paper](https://wukaoliu.github.io/HAttMatting/)

```
@InProceedings{Qiao_2020_CVPR,
    author = {Qiao, Yu and Liu, Yuhao and Yang, Xin and Zhou, Dongsheng and Xu, Mingliang and Zhang, Qiang and Wei, Xiaopeng},
    title = {Attention-Guided Hierarchical Structure Aggregation for Image Matting},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

##### Foreground-aware Semantic Representations for Image Harmonization
![https://github.com/saic-vul/image_harmonization/raw/master/images/ih_teaser.jpg](https://github.com/saic-vul/image_harmonization/raw/master/images/ih_teaser.jpg)
- [Code](https://github.com/saic-vul/image_harmonization)
- [Paper](https://arxiv.org/abs/2006.00809)

```
@article{sofiiuk2020harmonization,
  title={Foreground-aware Semantic Representations for Image Harmonization},
  author={Konstantin Sofiiuk, Polina Popenova, Anton Konushin},
  journal={arXiv preprint arXiv:2006.00809},
  year={2020}
}
```

### Image to Text
##### Image Captioning
![https://raw.githubusercontent.com/jayeshsaita/image_captioning_pytorch/master/data/sample_output/output_6774537791.jpg](https://raw.githubusercontent.com/jayeshsaita/image_captioning_pytorch/master/data/sample_output/output_6774537791.jpg)
- [Code](https://github.com/jayeshsaita/image_captioning_pytorch)


### Image to Sound/Speech

### Image to Video


## Text to Anything

### Text to Image

### Text to Text
#### Text Generation
##### Next Word Prediction
![UI](https://raw.githubusercontent.com/renatoviolin/next_word_prediction/master/word_prediction.gif)
- [Code](https://github.com/renatoviolin/next_word_prediction)

#### Code to Code
##### Unsupervised Translation of Programming Languages
- [Paper](https://arxiv.org/abs/2006.03511)

```
@misc{lachaux2020unsupervised,
    title={Unsupervised Translation of Programming Languages},
    author={Marie-Anne Lachaux and Baptiste Roziere and Lowik Chanussot and Guillaume Lample},
    year={2020},
    eprint={2006.03511},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<iframe width="100%" src="https://www.youtube.com/embed/xTzFJIknh7E" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
### Text to Sound/Speech

### Text to Video


## Sound/Speech to Anything

### Sound/Speech to Image
##### Audio to Image Conversion
- [Code](https://www.kaggle.com/timolee/audio-data-conversion-to-images-eda)

### Sound/Speech to Text
##### Speech Command Reognitiob
- [Code](https://github.com/jayeshsaita/Speech-Commands-Recognition)
- [Paper](https://www.linkedin.com/in/jayeshsaita)

### Sound/Speech to Sound/Speech

### Sound/Speech to Video



## Video to Anything

### Video to Video
#### Segmentation
##### MSeg : A Composite Dataset for Multi-domain Semantic Segmentation
![https://user-images.githubusercontent.com/62491525/83893958-abb75e00-a71e-11ea-978c-ab4080b4e718.gif](https://user-images.githubusercontent.com/62491525/83893958-abb75e00-a71e-11ea-978c-ab4080b4e718.gif)
- [Code](https://github.com/mseg-dataset)
- [Paper](https://vladlen.info/papers/MSeg.pdf)

```
@InProceedings{MSeg_2020_CVPR,
author = {Lambert, John and Zhuang, Liu and Sener, Ozan and Hays, James and Koltun, Vladlen},
title = {MSeg A Composite Dataset for Multi-domain Semantic Segmentation},
booktitle = {Computer Vision and Pattern Recognition (CVPR)},
year = {2020}
}
```

<iframe width="100%" src="https://www.youtube.com/embed/PzBK6K5gyyo" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

##### Motion Supervised co-part Segmentation
![https://github.com/AliaksandrSiarohin/motion-cosegmentation/blob/master/sup-mat/beard-line.gif?raw=true](https://github.com/AliaksandrSiarohin/motion-cosegmentation/blob/master/sup-mat/beard-line.gif?raw=true)
- [Code](https://github.com/AliaksandrSiarohin/motion-cosegmentation)
- [Paper](http://arxiv.org/abs/2004.03234)

```
@article{Siarohin_2020_motion,
  title={Motion Supervised co-part Segmentation},
  author={Siarohin, Aliaksandr and Roy, Subhankar and Lathuilière, Stéphane and Tulyakov, Sergey and Ricci, Elisa and Sebe, Nicu},
  journal={arXiv preprint},
  year={2020}
}
```

<iframe width="100%" src="https://www.youtube.com/embed/RJ4Nj1wV5iA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Video to Image

### Video to Text

### Video toSound/Speech

### Video to Video


# Inference
## Fastai
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to announce <a href="https://twitter.com/hashtag/fastinference?src=hash&amp;ref_src=twsrc%5Etfw">#fastinference</a>! The bad news: fastshap and ClassConfusion are gone. The good news: they moved to a new home! In this module we have all of the above plus some speed-up and QOL integrations into <a href="https://twitter.com/fastdotai?ref_src=twsrc%5Etfw">@fastdotai</a>&#39;s inference methods, see here: 1/<a href="https://t.co/SLgJahtSr5">https://t.co/SLgJahtSr5</a> <a href="https://t.co/1oFkMe4SsP">pic.twitter.com/1oFkMe4SsP</a></p>&mdash; Zach #masks4all Mueller (@TheZachMueller) <a href="https://twitter.com/TheZachMueller/status/1269818072577331200?ref_src=twsrc%5Etfw">June 8, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>


[How to deploy Fastai on Ubuntu](https://jianjye.com/p/deploy-fastai-digitalocean-ubuntu-flask-supervisor/)
## HuggingFace
[Deploying HuggingFace to Production](https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18)