import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim, nn
from torchvision import transforms
import torch.utils.data as data
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
import lightning as L
from omegaconf import DictConfig, OmegaConf
import torchvision.models as models
import rootutils
import hydra
import lightning as L
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import pandas as pd
from model.ResNetModel import ResNet
from model.cnn import CNN
import numpy as np
from PIL import Image, ImageDraw
from data_loading.dataset_png import PngDataset
import json
from data_loading.data_module import DataModule, DataModule2
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as TF
from yoloface.face_detector import YoloDetector
from data_loading.transform import Transforms

## import from src after this line
root_path = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "conf")

def load_image(file_path):
    """Load an image from the specified file path."""
    return Image.open(file_path)

def crop_image(image, box):
    """Crop the image based on the provided bounding box."""
    top = int(box[0])
    left = int(box[1])
    height = int(box[3])
    width = int(box[2])
    image = TF.crop(image, top, left, height, width)
    return image, top, left

def apply_predictions(cropped_image, landmarks=None):
    """Mock function to apply predictions on the cropped image.
       Replace this with actual model inference logic."""
    # Convert the cropped image from PIL to NumPy array
    image_cv2 = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

    original_shape = (image_cv2.shape[1], image_cv2.shape[0])  # (width, height)

    # Resize ảnh về kích thước 224x224
    image_cv2 = cv2.resize(image_cv2, (224, 224))
       
    if landmarks is not None:
        for i in range(len(landmarks)):
            # Get the landmark coordinates
            x = (landmarks[i][0] + 0.5) * 224  # Use size to get width
            y = (landmarks[i][1] + 0.5) * 224  # Use size to get height

            # Draw a circle for each landmark
            cv2.circle(image_cv2, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red color in BGR
    
    # Sau khi xử lý, resize ảnh trở lại kích thước gốc
    image_cv2 = cv2.resize(image_cv2, original_shape)

    # Convert back to PIL for further processing if needed
    processed_image = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    
    return processed_image

def paste_on_original(original_image, processed_cropped_image, top, left):
    """Paste the processed cropped image back onto the original image."""
    original_image.paste(processed_cropped_image, (left, top))
    return original_image

def save_image(image, output_path):
    """Save the processed image to the specified output path."""
    image.save(output_path)

def pipeline(xml_file_path="./data/labels_ibug_300W_test.xml", output_dir="image", model=None):
    """Main pipeline to load image, crop, apply predictions, and save."""
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    root_dir = './data'
    yolo_model = YoloDetector(target_size=720, device="cuda:0", min_face=90)

    for idx, filename in enumerate(root[2]):
        file_path = os.path.join(root_dir, filename.attrib['file'])
            
        # Load the image
        image = load_image(file_path)

        image1 = np.array(image)
        if (image1.ndim == 2):
            continue
        
        # Check if the image has 4 channels (RGBA)
        if image1.shape[-1] == 4:
            # Drop the alpha channel, keep only the first 3 channels (RGB)
            image1 = image1[:, :, :3]
            
       
        # if image.ndim == 3:  # RGB image (H, W, C)
        #     image1 = np.expand_dims(np.array(image), axis=0)  # Expand to (1, H, W, C)

        bboxes, points = yolo_model.predict(image1)
        # print(bboxes[0])
        # print(points)

        test_transform = Transforms(training=False, predict=True)

        for b in bboxes[0]:
            x_min = b[0]
            y_min = b[1]
            x_max = b[2]
            y_max = b[3]
            # Expand the box by 5 pixels on all sides
            expand_by = 5
            box = [
                [max(y_min-expand_by, 0), 
                max(x_min-expand_by, 0), 
                x_max - x_min + 2 * expand_by, 
                y_max - y_min + 2 * expand_by] 
            ]

            # Ensure the correct device
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # model = model.to(device)
            # start
            img = cv2.imread(file_path, 0)

            img, _ = test_transform(img, None, box[0])
            img = np.expand_dims(np.array(img), axis=0)
            if image1.ndim == 4:
                img = img[:, :3, :, :]

            # Get the corresponding prediction
            # print("pred")
            with torch.no_grad():  # Disable gradient calculation
                pred = model(torch.tensor(img, device=model.device))
            # print(pred[0].shape)

            image_pred = pred[0]  # Adjust index based on your prediction structure

            coordinates = image_pred.cpu().numpy()

            landmarks = []

            for i in range(0, len(coordinates), 2):
                x = coordinates[i]
                y = coordinates[i + 1]
                landmarks.append((x, y))

            cropped_image, top, left = crop_image(image, box[0])

            # Apply predictions to the cropped image
            processed_image = apply_predictions(cropped_image, landmarks=landmarks)

            new_image = paste_on_original(image, processed_image, top, left)

            # Save the processed image
            output_file_path = os.path.join(output_dir, os.path.basename(file_path))
            save_image(new_image, output_file_path)
            print("saved")

@hydra.main(version_base=None, config_path=config_path, config_name="train")
def main(cfg: DictConfig) -> None:
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)
    model = ResNet.load_from_checkpoint("/data/hpc/minhdd/filter/results_resnet501/checkpoints/epoch=199-mae=0.02.ckpt", strict=False)
    # data_module = DataModule2()

    # pred = trainer.predict(model, data_module)
    # print(len(pred))
    # print(len(pred[0][0]))
    # print(pred[0].shape)
    model.eval()
    pipeline(model=model)


    # # Read the JSON file
    # with open(json_file_path, 'r') as json_file:
    #     data = json.load(json_file)
    
    # # Access the data
    # imgs = data.get('imgs', [])  # Access the 'imgs' key or return an empty list if 'imgs' is not found

    # # Ensure the lengths match
    # if len(imgs) != len(pred[0]):
    #     raise ValueError("The number of images does not match the number of predictions")

    # # Iterate through images and corresponding predictions
    # for idx, item in enumerate(imgs):
    #     filename = item.get('filename', None)
    #     if filename:
    #         # Load the image
    #         image = Image.open(filename).resize((96, 96))
            
    #         # Get the corresponding prediction
    #         image_pred = pred[0][idx]  # Adjust index based on your prediction structure

    #         coordinates = image_pred.cpu().numpy()

    #         # Create a draw object
    #         draw = ImageDraw.Draw(image)

    #         for i in range(0, len(coordinates), 2):
    #             x = coordinates[i]
    #             y = coordinates[i + 1]
    #             # Draw a small circle at (x, y)
    #             draw.ellipse((x-2, y-2, x+2, y+2), fill='red', outline='red')
            

    #         # Define the directory and file path to save the output
    #         output_dir = './image'
    #         output_file = os.path.join(output_dir, 'image_with_keypoints4.png')

    #         # Create the directory if it does not exist
    #         os.makedirs(output_dir, exist_ok=True)

    #         # Save the image with keypoints
    #         image.save(output_file)
            
    #         # Print or process the image and prediction
    #         print(f'Image index: {idx}, Filename: {filename}')
    #         print(f'Prediction shape for this image: {image_pred.shape}')
    #         # Process or visualize image_pred as needed

if __name__ == "__main__":
    main()

