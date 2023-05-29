# Image Captioning

## Content

This folder contains the following files:

- *model.py*: this file contains the structure of the model in different classes. The class that form the structure of our model are the following:
  - *encoderCNN class*: that consists in a VGG-16 CNN pretrained in imageNet that extracts the features from the images.
  - *decoderRNN class*: that consists in a LSTM network that takes the fature vectors of the images that is concatenated with the feature vector from the captions that was obtained using an embeding layer, this information is pased to the LSTM to generate words recurrently keeping the previous information until the sentence is compleated when de model predicts an EOS (End of Sentence).
  - *AdditiveAttention class*: that performs the additive attention between the feature vector of the image with the previous hidden state using a linear layer with the feature vector and the previous state concatenated. This process returns the attention weights for that state (word).
  - *CNNtoRNN class*: that joints the functions of the previous classes in one class that receives the images and the captions and returns the codified sentence.

- *get_loader.py*: this file contains a function to create the data loaders that we are going to use in this project. There are three different classes in this file:
  - *Vocabulary*: this class creates the vocabulary using the language model from spacy 'en_core_web_sm' to tokenize the words in the captions and saving each token in a dictionary and assign an index to each token. There is a dictionary to get the index given the token and a dictionary to get the token given the index.
  - *FlickrDataset*: this class generates the dataloader loading the file with the captions to get the filenames of the images, the corresponding captions and the vocabulary.
  - *Padding*: this class is for adding padding to the captions so that all captions in a single batch have the same length.

- *utils.py*: this file contains two functions:
  - *save_checkpoint*: function to save the model in a pth file.
  - *load_checkpoint*: function to load a model already trained and saved in a pth file.
  Note: The utils.py for the model with attention also saves and loads the scheduler of the model. In the model without attention there is no scheduler.

- *train.py*: This script is used to train an image captioning model using a CNN-to-RNN architecture. It loads the dataset of images and captions, creates the necessary data loader, initializes the model, loss function, scheduler and optimizer. The script then performs the training loop for the specified number of epochs, logging the training loss using TensorBoard. At the end of training, it saves the final model checkpoint and plots the training loss curve.

- *evaluation.py*: this script is used to plot images along ith their predicted captions and actual captions as qualitative evaluation and BLEU score as a quantiative evaluation

## Requirements

You will need to download the images and the captions using the following link that will take you to a floder in google drive with the images and the captions splited. For this model you should use the file with all captions and all the images, not the validation nor the train.

[flickr Dataset](https://drive.google.com/drive/folders/1skoIZFClsh_Ol-wiwG_Foo53BQF8KOMW?usp=sharing)

You will need and environment that contains the following libaries, packages and frameworks: 
```
pytorch, spacy, os, pandas, numpy, PIL (pillow).
```
You will also need to download the 'en_core_web_sm' language model from spacy to get the vocabulary from the captions using:

```
python -m spacy download en_core_web_sm
```

## - Using different datasets

In order to make the model more robust and get better results, we tried to train our model with different dataset such as COCO or Flickr30k but we experimented some problems when loading the data and the formats of the data were different. We tried to change the format of the data and upload all images, but during the training we got problems when loading some images.

## - Trying different configurations

We tried different embedding and hidden size for the models, different learning rates, different batch size, different number of layers, we tried to train the model with scheduler and without scheduler and the best results we obtained has this configuration:
 - embedding size = 128
 - hidden size = 512
 - batch size = 32
 - learning rate = 1e-4
 - num layers = 1
 - scheduler = with gamma = 0.9 and step_size = 1
 - dropout = There are dropouts in many layers in the RNN
 - Batch normalization = False

## Python Notebook  

To make your life easier we mount all python files in a notebook. So you can go running cell by cell and seeing the results (We deliver it already runned, in case you dont want to run it all again). All files needed will be requested while running the ipynb.
