import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import CNNtoRNN
from utils import load_checkpoint


def display_image_with_caption(model, device, image_path):
    """
    Load an image, generate a caption using the trained model, and display the image with its caption.
    
    Args:
        model: The trained image captioning model.
        device: The device (e.g., 'cpu' or 'cuda') to perform computations on.
        image_path (str): The path to the image.
    """

	# Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize the image to a specific size
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image tensor
    ])

    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    # Generate caption using the model's caption_image method
    with torch.no_grad():
        model.eval()
        generated_caption = model.caption_image(image)
        model.train()

    # Convert the list of tokens into a string caption
    generated_caption = ' '.join(generated_caption)

    # Load the image (without normalizing transformation)
    original_image = Image.open(image_path).convert("RGB")
    
    # Plot image with caption
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.title(generated_caption)
    plt.axis('off')
    plt.show()


def test():
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set model parameters (these should match the ones used during training)
    embed_size = 256
    hidden_size = 256
    vocab_size = 5000  # You should use the actual vocab size from your data
    num_layers = 1

    # Initialize the model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    # Load the trained model checkpoint
    checkpoint = torch.load("checkpoint.pth.tar")  # Use your actual checkpoint path
    load_checkpoint(checkpoint, model, None)

    # List of image paths to test
    image_paths = ["/home/xnmaster/Documents/PIXTALES-1/Images/17273391_55cfc7d3d4.jpg",
                   "/home/xnmaster/Documents/PIXTALES-1/Images/3759230208_1c2a492b12.jpg",
                   "/home/xnmaster/Documents/PIXTALES-1/Images/3726168984_1fa2c8965b.jpg"]  # Use your actual image paths

    # Generate captions and display images with captions
    for image_path in image_paths:
        display_image_with_caption(model, device, image_path)


if __name__ == "__main__":
    test()
