# Image Captioning

## Contents

This repository contains the code and files for the Image Captioning project, using a CNN and RNN.

### Files

- **model.py**: This file contains the structure of the model in different classes. The classes include:
  - `encoderCNN`: A ResNet CNN pretrained on ImageNet that extracts features from the images.
  - `decoderRNN`: An LSTM network that generates words recursively using the image features and captions as inputs.
  - `CNNtoRNN`: Combines the functionality of `encoderCNN` and `decoderRNN` to encode images and captions and produce codified sentences.

- **get_loader.py**: This file contains a function to create data loaders for the project. It includes the following classes:
  - `Vocabulary`: Creates the vocabulary by tokenizing words in the captions using the "en_core_web_sm" language model from spacy.
  - `FlickrDataset`: Generates a data loader by loading image filenames, captions, and the vocabulary.
  - `Padding`: Adds padding to captions in a batch to ensure uniform length.

- **utils.py**: This file contains two utility functions:
  - `save_checkpoint`: Saves the model in a .pth file.
  - `load_checkpoint`: Loads a pre-trained model from a .pth file.

- **train.py**: This script trains an image captioning model using a CNN-to-RNN architecture. It loads the image and caption datasets, creates data loaders, initializes the model, loss function, and optimizer. The script performs the training loop for a specified number of epochs, logs the training loss using TensorBoard, and saves the final model checkpoint and training loss curve.

- **evaluation.py**: This script evaluates the trained model. It prompts the user to enter the path to the trained model checkpoint, generates captions for sample images, and displays the generated captions, BLEU scores, and ground truth captions for comparison.

### Requirements

To run this project, you need to download the image and caption datasets. The datasets are available for download using the following link: [flickr8k Dataset](https://drive.google.com/drive/folders/1skoIZFClsh_Ol-wiwG_Foo53BQF8KOMW?usp=sharing). Download the training images and captions, as well as the validation images and captions as separate files.

You will also need the following libraries, packages, and frameworks:
- PyTorch
- Spacy
- OS
- Pandas
- NumPy
- PIL (Pillow)

Download the "en_core_web_sm" language model from spacy by running the following command:
```
python -m spacy download en_core_web_sm
```

### Execution

After setting up the environment and downloading the language model, execute the following command in the terminal:
```
python train.py
```

The script will prompt you to enter the paths for the training and validation images, as well as the training and validation captions.

Once the training starts, you will see the progress in the terminal, including loss evolution, epoch, and step information.

At the end of each epoch, the model will be saved in a .pth file. After training completes, a plot showing the training and validation loss curves will be displayed.

Use the saved checkpoint to evaluate the model by running the `evaluation.py` script. It will prompt you to enter the path to the model checkpoint and display generated captions along with their corresponding BLEU scores and ground truth captions.

### Using Different Datasets

We attempted to train our model with different datasets such as COCO or Flickr30k to improve its performance and robustness. However, we encountered issues with data loading and different data formats. Despite attempts to modify and upload the images in different formats, we faced problems during training when loading certain images.

### Trying Different Configurations

During the project, we experimented with various configurations to optimize the model's performance. We tried different embedding and hidden sizes, learning rates, batch sizes, number of layers, and training with or without a scheduler. After exploring these options, we found the following configuration to yield the best results:

- Embedding size: 256
- Hidden size: 256
- Batch size: 32
- Learning rate: 3e-4
- Number of layers: 2
- Scheduler: False
- Dropout: 0.5
- Batch normalization: True

Feel free to modify these parameters according to your specific needs and dataset characteristics to achieve the desired results.

## APP


