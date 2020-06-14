---
layout: page
title: Technology Watch
permalink: /watch/
toc: true
---
1. TOC
{:toc}
# Machine/Deep Learning tools
## Common issues
### Unbalanced Dataset
#### Solution : Oversampling
- [Oversampling with FastAI](https://www.kaggle.com/tanlikesmath/oversampling-mnist-with-fastai)

## Technics

### Clustering
- [Understanding K-Means, K-Means++ and, K-Medoids Clustering Algorithms](https://towardsdatascience.com/understanding-k-means-k-means-and-k-medoids-clustering-algorithms-ad9c9fbf47ca)
- [Centroid Initialization Methods for k-means Clustering](https://www.kdnuggets.com/2020/06/centroid-initialization-k-means-clustering.html)

### Data Labeling
- [Reducing your labeled data requirements (2–5x) for Deep Learning: Deep Mind’s new "Contrastive Predictive Coding 2.0"](https://medium.com/@lessw/reducing-your-labeled-data-requirements-2-5x-for-deep-learning-google-brains-new-contrastive-2ac0da0367ef)
- [Annotation tools for computer vision and NLP](https://machinelearningtokyo.com/2020/06/04/annotation-tools-for-computer-vision-and-nlp/amp/?__twitter_impression=true)

### Reinforcement Learning 
#### TensorTrade
- [Code](https://github.com/tensortrade-org/tensortrade/blob/master/README.md)

### Text few shot learning
- [Zero shot learning](https://joeddav.github.io/blog/2020/05/29/ZSL.html)

### Text summarization
Great tutorial serie here
- [Part 1](https://medium.com/hackernoon/text-summarizer-using-deep-learning-made-easy-490880df6cd)
- [Part 2](https://medium.com/hackernoon/abstractive-text-summarization-tutorial-2-text-representation-made-very-easy-ef4511a1a46)
- [Part 3](https://medium.com/hackernoon/tutorial-3-what-is-seq2seq-for-text-summarization-and-why-68ebaa644db0)
- [Part 4](https://medium.com/hackernoon/multilayer-bidirectional-lstm-gru-for-text-summarization-made-easy-tutorial-4-a63db108b44f)
- [Part 5](https://medium.com/hackernoon/beam-search-attention-for-text-summarization-made-easy-tutorial-5-3b7186df7086)
- [Part 6](https://medium.com/hackernoon/build-an-abstractive-text-summarizer-in-94-lines-of-tensorflow-tutorial-6-f0e1b4d88b55)
- [Part 7](https://medium.com/hackernoon/combination-of-abstractive-extractive-methods-for-text-summarization-tutorial-7-8a4fb85d67e2)
- [Part 8](https://medium.com/hackernoon/teach-seq2seq-models-to-learn-from-their-mistakes-using-deep-curriculum-learning-tutorial-8-a730a387754)
- [Part 9](https://medium.com/analytics-vidhya/deep-reinforcement-learning-deeprl-for-abstractive-text-summarization-made-easy-tutorial-9-c6914999c76c)
- [Part 10](https://medium.com/analytics-vidhya/deep-reinforcement-learning-deeprl-for-abstractive-text-summarization-made-easy-tutorial-9-c6914999c76c)
- [Code](https://github.com/theamrzaki/text_summurization_abstractive_methods)

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
- [HuggingFace/nlp - Create fastai Dataloaders, show batch, and create dataset for LM, MLM](https://forums.fast.ai/t/huggingface-nlp-create-fastai-dataloaders-show-batch-and-create-dataset-for-lm-mlm/73179)

### BERT
#### Blogs
- [Blog Text Classification using Transformers](https://app.wandb.ai/cayush/bert-finetuning/reports/Sentence-classification-with-Huggingface-BERT-and-W%26B--Vmlldzo4MDMwNA)
- [Community Notebooks](https://github.com/huggingface/transformers/tree/master/notebooks#community-notebooks)

#### BERTweet: A pre-trained language model for English Tweets
- [Code](https://github.com/VinAIResearch/BERTweet/blob/master/README.md#models2)

#### COVID-Twitter-BERT
- [Code](https://github.com/digitalepidemiologylab/covid-twitter-bert)
- [Paper](https://arxiv.org/abs/2005.07503)

```
@article{muller2020covid,
  title={COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter},
  author={M{\"u}ller, Martin and Salath{\'e}, Marcel and Kummervold, Per E},
  journal={arXiv preprint arXiv:2005.07503},
  year={2020}
}
```

### GPT2

#### aitextgen -- Train a GPT-2 Text-Generating Model w/ GPU
- [Code](https://github.com/minimaxir/aitextgen)
- [Doc](https://docs.aitextgen.io/)
- [Sample](https://colab.research.google.com/drive/15qBZx5y9rdaQSyWpsreMDnTiZ5IlN0zD?usp=sharing)
- [Author](https://twitter.com/minimaxir)

#### Relates Blogs
- [Tune GPT2 to generate positive reviews](https://lvwerra.github.io/trl/04-gpt2-sentiment-ppo-training/)

### Question Answering
- [Deep Learning has (almost) all the answers: Yes/No Question Answering with Transformers](https://medium.com/illuin/deep-learning-has-almost-all-the-answers-yes-no-question-answering-with-transformers-223bebb70189)
- [Building a simple Open Domain Question Answering pipeline in French](https://medium.com/illuin/building-an-open-domain-question-answering-pipeline-in-french-97304e63c369)

### Reformers
- [Colab](https://arxiv.org/abs/2001.04451)


## Benchmaks
### Text
#### XGLUE: Expanding cross-lingual understanding and generation with tasks from real-world scenarios
![](https://www.microsoft.com/en-us/research/uploads/prod/2020/06/XGLUE-homepage-feat-image-800x550.png)
- [Blog](https://www.microsoft.com/en-us/research/blog/xglue-expanding-cross-lingual-understanding-and-generation-with-tasks-from-real-world-scenarios/)
- [Code](https://github.com/microsoft/Unicoder)
- [Dataset](https://github.com/microsoft/XGLUE)
- [Paper](https://www.microsoft.com/en-us/research/publication/xglue-a-new-benchmark-dataset-for-cross-lingual-pre-training-understanding-and-generation/)

## Useful Libs
### Wrapper
#### Vision
#### Text
##### fast.ai Code-First Intro to Natural Language Processing
- [Project Page](https://www.fast.ai/2019/07/08/fastai-nlp/)
- [Code](https://github.com/fastai/course-nlp)

here is the associated tutorial serie:

<iframe width="50%" min-height="75%" src="https://www.youtube.com/embed/videoseries?list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


- [FastAI + HuggingFace = FastHugs](https://www.ntentional.com/nlp/training%20technique/classification/2020/04/17/fasthugs_seq_classification.html)

### Text
#### NLTK
- [Project Page](https://www.nltk.org)
- [Code](https://github.com/nltk/nltk)
- [Doc](https://www.nltk.org/index.html)
- [Cheat Sheet](https://cheatography.com/murenei/cheat-sheets/natural-language-processing-with-python-and-nltk/)

I consider [@SentDex](https://twitter.com/Sentdex) founder [pythonprogramming.net](https://pythonprogramming.net) and [https://www.youtube.com/channel/sentdex](https://www.youtube.com/channel/sentdex) as the best tutorial for NLTK

<iframe width="50%" min-height="75%" src="https://www.youtube.com/embed/videoseries?list=PLI142kNg_e0Q57BmOF9H4UnXiWNSVZZ-O" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


#### SpaCy
- [Project Page](https://spacy.io/)
- [Code](https://github.com/explosion/spaCy)
- [Doc](https://spacy.io/api/doc)
- [Cheat Sheet](https://www.datacamp.com/community/blog/spacy-cheatsheet)
- [Spacy on Twitter @spacy_io](https://twitter.com/spacy_io)
- [Spacy on Linkedin](https://www.linkedin.com/company/explosion-ai/)
- [Spacy on Youtube](https://www.youtube.com/c/ExplosionAI)

A good Spacy tutorial Youtube serie here :

<iframe width="50%" min-height="75%" src="https://www.youtube.com/embed/videoseries?list=PLJ39kWiJXSiz1LK8d_fyxb7FTn4mBYOsD" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Spacy channel :

<iframe width="50%" min-height="75%" src="https://www.youtube.com/embed/videoseries?list=PLBmcuObd5An559HbDr_alBnwVsGq-7uTF" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

#### Transformers (HuggingFace)
- [Project Page](https://huggingface.co/)
- [Code](https://github.com/huggingface/transformers)
- [Colab](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb)
- [Doc](https://huggingface.co/transformers/)
- [HuggingFace on Twitter @HuggingFace](https://twitter.com/huggingface)
- [HuggingFace on Linkedin](https://www.linkedin.com/company/huggingface/)

##### Related blogs
- [A Step by Step Guide to Tracking Hugging Face Model Performance](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-Hugging-Face-Model-Performance--VmlldzoxMDE2MTU)
- [Does model size matter? A comparison of BERT and DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-A-comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU)
- [Open-Dialog Chatbots for Learning New Languages](https://nathancooper.io/i-am-a-nerd/chatbot/deep-learning/gpt2/2020/05/12/chatbot-part-1.html)

#### Simple Transformers (based on HuggingFace)
- [Project Page](https://simpletransformers.ai)
- [Code](https://github.com/ThilinaRajapakse/simpletransformers)
- [Doc](https://simpletransformers.ai/docs/installation/)
- [Blog Post](https://towardsdatascience.com/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3)


Simple Transformers is a wrapper on top of HuggingFace's Transformer Library take makes it easy to setup and use, here is an example of binary classification :

```python
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0]]
train_df = pd.DataFrame(train_data)

eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0]]
eval_df = pd.DataFrame(eval_data)

# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base') # You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

```

#### cdQA: Unsupervised QA
- [Project Page](https://cdqa-suite.github.io/cdQA-website/)
- [Code](https://github.com/cdqa-suite/cdQA)

#### Facebook UnsupervisedQA
- [Code](https://github.com/facebookresearch/UnsupervisedQA)
- [Blog](https://medium.com/illuin/unsupervised-question-answering-4758e5f2be9b)

### Others
#### Facebook MMF
A modular framework for vision & language multimodal research from Facebook AI Research (FAIR)
![](https://camo.githubusercontent.com/fa3fcc9fb23c9d5e4ba8a7a822c15d53dc892ef7/68747470733a2f2f692e696d6775722e636f6d2f42503873596e6b2e6a7067)
- [Code](https://github.com/facebookresearch/mmf)
- [Doc](https://mmf.readthedocs.io/en/latest/)
- [Paper](http://learningsys.org/nips18/assets/papers/35CameraReadySubmissionPythia___A_platform_for_vision_language_multi_modal_research.pdf) 

```
@inproceedings{singh2018pythia,
  title={Pythia-a platform for vision \& language research},
  author={Singh, Amanpreet and Goswami, Vedanuj and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  booktitle={SysML Workshop, NeurIPS},
  volume={2018},
  year={2018}
}
```

## Hands-on
### NLP
- [Modern NLP in Python](https://github.com/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb)


| Tool | Binary Classification | Multi-Label Classification | Question Answering | Tokenization | Generation | Named Entity Recognition |
|-|-|-|-|-|-|-|

#### Structured Data
### AutoML
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

#### OVHcloud autoML
![Demo](https://labs.ovh.com/sites/default/files/inline-images/upload_source_1.png| width=250)
![Demo2](https://labs.ovh.com/sites/default/files/inline-images/optimization_result_2.png | width=250)
![Demo3](https://labs.ovh.com/sites/default/files/inline-images/model_kpi_2.png | width=250)

- [Site](https://labs.ovh.com/machine-learning-platform)
- [Code](https://github.com/ovh/prescience-client)
- [Forum](https://gitter.im/ovh/ai)

# Deep Learning use cases

## Nothing to Image
### Generative
#### Face
##### Disentangled Image Generation Through Structured Noise Injection
![https://github.com/yalharbi/StructuredNoiseInjection/raw/master/example_fakes_alllocal.png](https://github.com/yalharbi/StructuredNoiseInjection/raw/master/example_fakes_alllocal.png?raw=true | width=250)
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

<iframe width="50%" src="https://www.youtube.com/embed/7h-7wso9E0k" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Image to Anything

### Image to Image
#### Avatarify : Deepdake for zoom
- [Code](https://github.com/alievk/avatarify)
- [Blog](https://github.com/DashBarkHuss/100-days-of-code/blob/master/post-log.md#avatarify-1)

#### Inpainting
##### High-Resolution Image Inpainting with Iterative Confidence Feedback and Guided Upsampling
![https://s1.ax1x.com/2020/03/18/8wQG5T.jpg](https://s1.ax1x.com/2020/03/18/8wQG5T.jpg | width=250)
- [Project Page](https://zengxianyu.github.io/iic/)
- [APP](http://47.57.135.203:2333/)
- [Paper](https://arxiv.org/abs/2005.11742)

##### EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning
![https://user-images.githubusercontent.com/1743048/50673917-aac15080-0faf-11e9-9100-ef10864087c8.png](https://user-images.githubusercontent.com/1743048/50673917-aac15080-0faf-11e9-9100-ef10864087c8.png | width=250)
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
![Before](https://github.com/zengxianyu/Inpainting_FRRN/blob/master/examples/ex_damaged2.png?raw=true | width=250)
![After](https://github.com/zengxianyu/Inpainting_FRRN/blob/master/examples/ex_final2.png?raw=true | width=250)
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
![http://pulse.cs.duke.edu/assets/094.jpeg](http://pulse.cs.duke.edu/assets/094.jpeg | width=250)
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
![https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention/raw/master/Figs/Visual_3.png](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention/raw/master/Figs/Visual_3.png | width=250)
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
![http://geometrylearning.com/DeepFaceDrawing/imgs/teaser.jpg](http://geometrylearning.com/DeepFaceDrawing/imgs/teaser.jpg | width=250)
- [Paper](http://geometrylearning.com/DeepFaceDrawing/)

<iframe width="50%" src="https://www.youtube.com/embed/HSunooUTwKs" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

##### UGATIT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation (ICLR 2020)
![https://github.com/taki0112/UGATIT/blob/master/assets/teaser.png?raw=true](https://github.com/taki0112/UGATIT/blob/master/assets/teaser.png?raw=true | width=250)
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

##### Selfie to Anime
![https://github.com/jqueguiner/databuzzword/blob/master/images/A578852A-9A4D-4D90-88E0-A4D81C7D41B3.jpeg](https://github.com/jqueguiner/databuzzword/blob/master/images/A578852A-9A4D-4D90-88E0-A4D81C7D41B3.jpeg?raw=true | width=250)
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
![https://gitlab.com/irafm-ai/poly-yolo/-/raw/master/poly-yolo-titlepage-image.jpg?inline=false](https://gitlab.com/irafm-ai/poly-yolo/-/raw/master/poly-yolo-titlepage-image.jpg?inline=false | width=250)
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

<iframe width="50%" src="https://www.youtube.com/embed/2KxNnEV-Zes" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

##### Attention-Guided Hierarchical Structure Aggregation for Image Matting
![https://wukaoliu.github.io/HAttMatting/figures/visualization.png](https://wukaoliu.github.io/HAttMatting/figures/visualization.png?raw=true | width=250)
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
![https://github.com/saic-vul/image_harmonization/raw/master/images/ih_teaser.jpg](https://github.com/saic-vul/image_harmonization/raw/master/images/ih_teaser.jpg?raw=true | width=250)
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

##### Single-Stage Semantic Segmentation from Image Labels (CVPR 2020)
![https://github.com/visinf/1-stage-wseg/blob/master/figures/results.gif?raw=true](https://github.com/visinf/1-stage-wseg/blob/master/figures/results.gif?raw=true | width=250)
- [Code](https://github.com/visinf/1-stage-wseg)
- [Paper](https://arxiv.org/abs/2005.08104)

```
@inproceedings{Araslanov:2020:WSEG,
  title     = {Single-Stage Semantic Segmentation from Image Labels},
  author    = {Araslanov, Nikita and and Roth, Stefan},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```

### Others
#### Background Matting: The World is Your Green Screen
![](https://camo.githubusercontent.com/89ad795b21ae7c739811372739e53985b1e7feab/68747470733a2f2f686f6d65732e63732e77617368696e67746f6e2e6564752f7e736f756d796139312f70617065725f7468756d626e61696c732f6d617474696e672e706e67 | width=250)
- [Project Page](http://grail.cs.washington.edu/projects/background-matting/)
- [Code](https://github.com/senguptaumd/Background-Matting)
- [Paper](https://arxiv.org/abs/2004.00626)

```
@InProceedings{BMSengupta20,
  title={Background Matting: The World is Your Green Screen},
  author = {Soumyadip Sengupta and Vivek Jayaram and Brian Curless and Steve Seitz and Ira Kemelmacher-Shlizerman},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2020}
}
```

- [Blog Post](https://link.medium.com/suCEIzEed7)

#### 3D Photography using Context-aware Layered Depth Inpainting (CVPR 2020)
![](https://camo.githubusercontent.com/8dd5b529c99cdfcedd043c8239b68c4d7a23a148/68747470733a2f2f66696c65626f782e6563652e76742e6564752f7e6a626875616e672f70726f6a6563742f334450686f746f2f334450686f746f5f7465617365722e6a7067 | width=250)
- [Project Page](https://shihmengli.github.io/3D-Photo-Inpainting/)
- [Code](https://github.com/vt-vl-lab/3d-photo-inpainting/blob/master/README.md)
- [Colab](https://colab.research.google.com/drive/1706ToQrkIZshRSJSHvZ1RuCiM__YX3Bz)
- [Paper](https://arxiv.org/abs/2004.04727)

```
@inproceedings{Shih3DP20,
  author = {Shih, Meng-Li and Su, Shih-Yang and Kopf, Johannes and Huang, Jia-Bin},
  title = {3D Photography using Context-aware Layered Depth Inpainting},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```

#### Project an image centroid to another image using OpenCV
![https://github.com/cyrildiagne/screenpoint/blob/master/example/match_debug.png?raw=true](https://github.com/cyrildiagne/screenpoint/blob/master/example/match_debug.png?raw=true)
- [Code](https://github.com/cyrildiagne/screenpoint/blob/master/README.md)

### Image to Text

##### CompGuessWhat?!: a Multi-Task Evaluation Framework for Grounded Language Learning
![](https://compguesswhat.github.io/img/attribute_prediction.png)
- [Code](https://github.com/CompGuessWhat)
- [Paper](https://arxiv.org/abs/2006.02174)

```
@inproceedings{suglia2020compguesswhat,
  title={CompGuessWhat?!: a Multi-task Evaluation Framework for Grounded Language Learning},
  author={Suglia, Alessandro, Konstas, Ioannis, Vanzo, Andrea, Bastianelli, Emanuele, Desmond Elliott, Stella Frank and Oliver Lemon},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```

https://compguesswhat.github.io/paper/
##### YOLOv4: Optimal Speed and Accuracy of Object Detection
![](https://i.ibb.co/mz376Rd/Image-PNG.png)
- [Code](https://github.com/AlexeyAB/darknet)
- [Paper](https://arxiv.org/abs/2004.10934)

```
@misc{bochkovskiy2020yolov4,
    title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
    author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
    year={2020},
    eprint={2004.10934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

##### Image Captioning with PyTorch
![https://raw.githubusercontent.com/jayeshsaita/image_captioning_pytorch/master/data/sample_output/output_6774537791.jpg](https://raw.githubusercontent.com/jayeshsaita/image_captioning_pytorch/master/data/sample_output/output_6774537791.jpg | width=250)
- [Code](https://github.com/jayeshsaita/image_captioning_pytorch)

##### ResNeSt: Split-Attention Networks
![](https://raw.githubusercontent.com/zhanghang1989/ResNeSt/master/miscs/abstract.jpg)
- [Code](https://github.com/zhanghang1989/ResNeSt)
- [Paper](https://arxiv.org/abs/2004.08955)

```
@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}
```

##### Hands-on guide to sign language classification using CNN
[Hands-on guide to sign language classification using CNN](https://analyticsindiamag.com/hands-on-guide-to-sign-language-classification-using-cnn/)

### Image to Sound/Speech

### Image to Video


## Text to Anything

### Text to Image
#### Network Fusion for Content Creation with Conditional INNs (CVPR 2020)
- [Project Page](https://compvis.github.io/network-fusion/)
- [Paper](https://arxiv.org/abs/2005.13580)

### Text to Text

#### Bilingual Translation
- [Example](https://gist.github.com/sshleifer/a5498e4d829a016b5875516d659ed50f)
- [Code](https://github.com/sshleifer/marian)
- [Author](https://github.com/sshleifer/marian)

#### T5 finetuning 
- [Paper](https://arxiv.org/abs/1910.10683)

#### Training Electra
- [Pre-train ELECTRA from Scratch for Spanish] (https://chriskhanhtran.github.io/posts/electra-spanish/)

#### Text Translation
- [Blog](https://towardsdatascience.com/build-your-own-machine-translation-service-with-transformers-d0709df0791b)

#### Text Generation
##### Lyrics Generation
- [Colab](https://colab.research.google.com/drive/12g07FS2WkNctNy_bYb7a5ZNFAsJcN0dz?usp=sharing)
- [Author](https://eilab.gatech.edu/mark-riedl)

##### Next Word Prediction
![UI](https://raw.githubusercontent.com/renatoviolin/next_word_prediction/master/word_prediction.gif =250x)
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

<iframe width="50%" src="https://www.youtube.com/embed/xTzFJIknh7E" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Text to Sound/Speech
#### Pitchtron: Towards audiobook generation from ordinary people’s voices 
- [Project Page](https://sunghee.kaist.ac.kr/entry/pitchtron)
- [Paper](https://arxiv.org/abs/2005.10456)
- [Code](https://github.com/hash2430/pitchtron)
- [Samples](https://sunghee.kaist.ac.kr/entry/pitchtron)

<audio controls="controls">
    <source src="https://www.dropbox.com/s/g9ch0aw2wbgmwek/hard_emotive_1.wav?dl=1" type="audio/wav"></source>
    <p>Your browser does not support the audio element.</p>
</audio>

#### Transformers TTS
- [Project Page](https://as-ideas.github.io/TransformerTTS/)
- [Code](https://github.com/as-ideas/TransformerTTS)
- [Samples](https://as-ideas.github.io/TransformerTTS/)

<audio controls="controls">
    <source src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformertts/Trump.wav?raw=true" type="audio/wav"></source>
    <p>Your browser does not support the audio element.</p>
</audio>

### Text to Video


## Sound/Speech to Anything

### Sound/Speech to Image
##### Audio to Image Conversion
- [Code](https://www.kaggle.com/timolee/audio-data-conversion-to-images-eda)

### Sound/Speech to Text
##### Speech Command Recognition
- [Code](https://github.com/jayeshsaita/Speech-Commands-Recognition)
- [Author](https://www.linkedin.com/in/jayeshsaita)


### Sound/Speech to Sound/Speech
#### Speaker-independent-emotional-voice-conversion-based-on-conditional-VAW-GAN-and-CWT
- [Project Page](https://kunzhou9646.github.io/speaker-independent-emotional-vc/)
- [Code](https://github.com/KunZhou9646/Speaker-independent-emotional-voice-conversion-based-on-conditional-VAW-GAN-and-CWT)
- [Paper](https://www.researchgate.net/publication/341388058_Converting_Anyone's_Emotion_Towards_Speaker-Independent_Emotional_Voice_Conversion)
- [Sample](https://kunzhou9646.github.io/speaker-independent-emotional-vc/)

```
@unknown{unknown,
author = {Zhou, Kun and Sisman, Berrak and Zhang, Mingyang and Li, Haizhou},
year = {2020},
month = {05},
pages = {},
title = {Converting Anyone's Emotion: Towards Speaker-Independent Emotional Voice Conversion},
doi = {10.13140/RG.2.2.20921.60006}
}
```

### Sound/Speech to Video



## Video to Anything

### Video to Video
#### Segmentation
##### MSeg : A Composite Dataset for Multi-domain Semantic Segmentation
![https://user-images.githubusercontent.com/62491525/83893958-abb75e00-a71e-11ea-978c-ab4080b4e718.gif](https://user-images.githubusercontent.com/62491525/83893958-abb75e00-a71e-11ea-978c-ab4080b4e718.gif | width=250)
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

<iframe width="50%" src="https://www.youtube.com/embed/PzBK6K5gyyo" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

##### Motion Supervised co-part Segmentation
![https://github.com/AliaksandrSiarohin/motion-cosegmentation/blob/master/sup-mat/beard-line.gif?raw=true](https://github.com/AliaksandrSiarohin/motion-cosegmentation/blob/master/sup-mat/beard-line.gif?raw=true | width=250)
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

<iframe width="50%" src="https://www.youtube.com/embed/RJ4Nj1wV5iA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Video to Image

### Video to Text

### Video toSound/Speech

### Video to Video


# Inference
## Python serving
- [Why we switched from Flask to FastAPI for production machine learning
](https://towardsdatascience.com/why-we-switched-from-flask-to-fastapi-for-production-machine-learning-765aab9b3679)

## Fastai
{% twitter https://twitter.com/TheZachMueller/status/1269818072577331200?ref_src=twsrc%5Etfw %}

[How to deploy Fastai on Ubuntu](https://jianjye.com/p/deploy-fastai-digitalocean-ubuntu-flask-supervisor/)

## HuggingFace
- [Deploying HuggingFace to Production](https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18)
- [Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://github.com/huggingface/transformers/blob/master/examples/movement-pruning/README.md)

```
@article{sanh2020movement,
    title={Movement Pruning: Adaptive Sparsity by Fine-Tuning},
    author={Victor Sanh and Thomas Wolf and Alexander M. Rush},
    year={2020},
    eprint={2005.07683},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Hummingbird 
python library that compiles trained ML models into tensor computation for faster inference. Supported models include sklearn decision trees, random forest, lightgbm, xgboost.   

- [Code](https://github.com/microsoft/hummingbird)
- [Examples](https://github.com/microsoft/hummingbird/tree/master/notebooks)


# Tools
## Terminal
### Rich 
Rich is a Python library for rich text and beautiful formatting in the terminal
![https://github.com/willmcgugan/rich/raw/master/imgs/features.png?raw=true](https://github.com/willmcgugan/rich/raw/master/imgs/features.png?raw=true)
- [Code](https://github.com/willmcgugan/rich)
- [Doc](https://rich.readthedocs.io/en/latest/)

## Python
### PyAudio FFT
![https://raw.githubusercontent.com/tr1pzz/Realtime_PyAudio_FFT/master/assets/teaser.gif](https://raw.githubusercontent.com/tr1pzz/Realtime_PyAudio_FFT/master/assets/teaser.gif)
- [https://github.com/tr1pzz/Realtime_PyAudio_FFT](https://github.com/tr1pzz/Realtime_PyAudio_FFT)

### Process Mining : alpha-miner
- [Project Page](http://pm4py.pads.rwth-aachen.de/documentation/process-discovery/alpha-miner/)


### Image Feature extractor
- [Code](https://github.com/aleSuglia/image-feature-extractors/blob/master/README.md)

# Cool projects
## Web based Training   
- [Teachable Machine](https://teachablemachine.withgoogle.com/)

## How to evaluate Longformer on TriviaQA using NLP 
- [Colab](https://colab.research.google.com/drive/1m7eTGlPmLRgoPkkA7rkhQdZ9ydpmsdLE?usp=sharing#scrollTo=lbNZdYkugq7-)

## Data Visualization
- [Choosing Fonts for Your Data Visualization](https://medium.com/nightingale/choosing-a-font-for-your-data-visualization-2ed37afea637)


# Hardware
## GPU
### Nvidia
#### Ampere
- [NVIDIA Ampere Architecture In-Depth](https://devblogs.nvidia.com/nvidia-ampere-architecture-in-depth/)



# MOOC
## Fast.ai
- [Image](https://www.fast.ai/)
- [NLP](https://www.fast.ai/2019/07/08/fastai-nlp/)

## Benchmark
### NLP
![](https://airev.us/wp-content/uploads/2020/05/tabela-duz%CC%87a_pop-1024x1536.png)
- [Full Blog Post](https://airev.us/ultimate-guide-to-natural-language-processing-courses)
- [Missing in Blog Post](https://www.fast.ai/2019/07/08/fastai-nlp/)


