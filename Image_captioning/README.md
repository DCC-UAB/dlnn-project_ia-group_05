# Image Captioning

## - Content

This folder contains the following files:

- *model.py*: this file contains the structure of the model in different classes. The first class is the encoderCNN that consists in a ResNet CNN pretrained in imageNet that extracts the features from the images; the second class is the decoderRNN that consists in a LSTM network that takes the fature vectors of the images that is concatenated with the feature vector from the captions that was obtained using an embeding layer, this information is pased to the LSTM to generate words recurrently keeping the previous information until the sentence is compleated when de model predicts an EOS (End of Sentence); finally, the class CNNtoRNN that joints functions of the previous to classes in one class that receives the images and the captions and returns the codified sentence.

- *get_loader.py*: this file contains a function to create the data loaders that we are going to use in this project. There are three different classes in this file:
  - *Vocabulary*: this class creates the vocabulary using the language model from spacy 'en_core_web_sm' to tokenize the words in the captions and saving each token in a dictionary and assign an index to each token. There is a dictionary to get the index given the token and a dictionary to get the token given the index.
  - *FlickrDataset*: this class generates the dataloader loading the file with the captions to get the filenames of the images, the corresponding captions and the vocabulary.
  - *Padding*: this class is for adding padding to the captions so that all captions in a single batch have the same length.

- *utils.py*: this file contains two functions:
  - *save_checkpoint*: function to save the model in a pth file.
  - *load_checkpoint*: function to load a model already trained and saved in a pth file.

- *train.py*: This script is used to train an image captioning model using a CNN-to-RNN architecture. It loads the dataset of images and captions, creates the necessary data loader, initializes the model, loss function, and optimizer. The script then performs the training loop for the specified number of epochs, logging the training loss using TensorBoard. At the end of training, it saves the final model checkpoint and plots the training loss curve.

- *evaluation.py*: ...

## - Requirements

You will need to download the images and the captions using the following link that will take you to a floder in google drive with the images and the captions splited.

[flickr Dataset](link)

You will need and environment that contains the following libaries, packages and frameworks: 
```
pytorch, spacy, os, pandas, numpy, PIL (pillow).
```
You will also need to download the 'en_core_web_sm' language model from spacy to get the vocabulary from the captions using:

```
python -m spacy download en_core_web_sm
```

