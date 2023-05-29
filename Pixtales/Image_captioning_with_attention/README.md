# Image Captioning: Transforming Visuals into Text

This folder offers an attention based solution for Image Captioning, using a powerful combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), in this case also adding attention, that makes our model much more flexible and interpretable (Also handling better long sentences). Is worth noting that we wanted to implement our own attention mechanism without recurring to third´s pretrained models. We did this knowing that the quality could not be as good as with a pretrained model like BERT or GPT beacuse we wanted to see where we could get to.

## Repository Structure

This repository includes the following key files:

- **model.py**: Defines the architecture of the model, featuring:
  - **encoderCNN**: Utilizes a VGG-16 model pre-trained on ImageNet to extract image features.
  - **decoderRNN**: Employs an LSTM network for decoding feature vectors into comprehensive captions.
  - **AdditiveAttention**: Executes additive attention between image feature vectors and previous hidden states.
  - **CNNtoRNN**: A unified model class that combines the functionality of the encoder and decoder.

- **get_loader.py**: Constructs DataLoader for the project, containing classes for vocabulary creation, dataset processing, and input padding.

- **utils.py**: Contains utility functions for model checkpoint saving and loading.

- **train.py**: Implements the training pipeline including model initialization, training loop, loss tracking, and model saving.

- **evaluation.py**: Provides both qualitative (image and predicted/actual captions) and quantitative (BLEU score) model evaluation.

## Prerequisites

Prior to running the code, please ensure you have the following:

1. Download the required images and captions from the provided [Flickr Dataset](https://drive.google.com/drive/folders/1x2pDFbpIsA8fCUomwBpo6wBy0Yw9LFF_?usp=sharing).
2. Prepare an environment with necessary libraries including `pytorch`, `spacy`, `os`, `pandas`, `numpy`, `PIL (pillow)`.
3. Download Spacy's 'en_core_web_sm' model using `python -m spacy download en_core_web_sm`.

## Extending the Model

We encourage exploration and modifications to improve the model. We attempted different configurations and data sources in pursuit of optimal performance. Our optimal configuration was found to be:

- Embedding size: 128
- Hidden size: 512
- Batch size: 32
- Learning rate: 1e-4
- Number of LSTM layers: 1
- Scheduler: With gamma = 0.9 and step_size = 1
- Dropout: Included in RNN layer
- Batch normalization: Not included

## Guided Run-through

For a detailed walkthrough of the entire process, refer to our Python Notebook. It provides step-by-step instructions, from initialization to model evaluation, facilitating an effective understanding of the Image Captioning model. 
[Full guided notebook](https://colab.research.google.com/drive/1oNQmnL9KO0TSOi4bNFqvnAgi3FdRgI8V?authuser=4#scrollTo=TIU_x5x68vuw)

## Qualitative evaluation

Qualitative evaluation for our image captioning model with attention can be found on these slides: [Qualitative evaluation attention](https://docs.google.com/presentation/d/12JIvFaUerl8xQFJnY5fT7xTXw3kX638slbluTsOe_fc/edit?usp=sharing)
<img width="537" alt="Captura de pantalla 2023-05-30 010402" src="https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/87651732/39ac7196-de89-4172-8618-3a55e7ad2a04">

<img width="567" alt="image" src="https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/87651732/320be3f8-4d90-4ac4-88be-6678c25a6335">

<img width="564" alt="image" src="https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/87651732/d7f0b38c-075e-4493-9a65-303b7c79c6ae">
