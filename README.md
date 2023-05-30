[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11110514&assignment_repo_type=AssignmentRepo)

# "PixTales: The Story Behind the Picture"

Welcome to PIXTALES - The story behind the picture, a deep learning approach to image captioning. The magic behind this repository is all about connecting visuals to language - generating a narrative for every image. Our project is driven by the power of neural networks and deep learning, aiming to create meaningful and accurate descriptions for any image. 

## What is PIXTALES?

PIXTALES is a project that uses a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to create a model capable of describing the content of images. We utilize the CNN as an "encoder" to transform an input image into a complex feature representation, and the RNN acts as a "decoder", turning those features into a rich, human-readable text. 

## Goals

Our Objective is to implement different models with different configurations and structures that are able to train and provide some results, and highlight the best results. Another goal is to try different ways to test our model, using different metrics and testing it with different datasets such as COCO dataset or Flickr30k dataset, but in principle we are going to use the Flickr8k dataset.

## Data

You can get all the data you will need in the following folder from google drive so you don't need to search for it.

[Image Captioning Data](https://drive.google.com/drive/folders/1skoIZFClsh_Ol-wiwG_Foo53BQF8KOMW?usp=sharing)

## Getting Started

First, you need to set clone this repository in your local (or virtual) machine

```
git@github.com:DCC-UAB/dlnn-project_ia-group_05.git
```

You will need to install the needed libraries and set up a good environment with pytorch, spacy, PIL and many more. You will also need to download the en_core_web_sm language model using:

```
python -m spacy download en_core_web_sm
```

Next, to run the example code, simply use:

```
python train.py
```
If you want to use our already trained models just load the checkpoint we have prepared in a folder from drive. Here you can found checkpoints for different models, you can download from here to use them for testing the model or training it a little more:

[Checkpoints of different models](https://drive.google.com/drive/folders/1ada905qZpaIdcILrhixOt4uVA2xHtHzt?usp=sharing)

and run evaluation:

```
python evaluation.py
```
## Code Structure

The code in this repository primarily consists of a model implementation (CNNtoRNN), a dataset loading function (get_loader), a utils.py file that is used to save and load the models and a train file for training the model and saving the model with in a pth file called checkpoint; the training uses all previous files. We also have an evaluation.py file that loads a model already trained to do some test and show some examples, it provides utility functions for generating and visualizing image captions and calculating BLEU scores, a popular metric for evaluating the quality of generated text in comparison to reference text. 

## Model Evaluation

Our evaluation focuses on two areas: the BLEU score and a visual inspection of the generated captions. The BLEU score gives us a quantitative measure of our model's performance, while the visual inspection of the generated captions lets us qualitatively assess the model's output. The original and generated captions are printed, and the average BLEU score across all images is computed.

## Final Words

This project is an exciting journey into the intersection of computer vision and natural language processing. We hope that this project can serve as a helpful resource for those interested in image captioning, deep learning, and AI in general.

It´s worth to note that our main goal has been always learning as much as we can, that´s why we chose this project, we really thought that would be the most challenging as well as self-rewarding, and so it has been, we put a lot of effort but it paid off completely, regardless of te mark.

We appreciate having had this experience that has helped us a lot to understand how complicated a Deep Learning project can be, but also how exciting it is to get results after having done hard work.



## References
[CS 152 NN—25: Attention: Image Captioning by Neil Rhodes](https://youtu.be/JTXPrjvhLl8)

[Show, Attend and Tell](https://arxiv.org/abs/1502.03044)


## Contributors

Authors: Neil De La Fuente, Maiol Sabater, Daniel Vidal

Subject: Neural Networks and Deep Learning.

Degreee in Artificial Intelligence, 2n course.

UAB, 2023.
