import torchvision
import torch
import cv2
import numpy as np
from collections import Counter
import torchvision.transforms as transforms
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights


def create_model(num_classes=2, size=300):
    
    model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    # print(model)

    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    # print("in_channel   ",in_channels)
    num_anchors = model.anchor_generator.num_anchors_per_location()
    # print("num_anchors   ",num_anchors)

    # Replace the classification head for 1 class (pothole) + background
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    # Adjust transform sizes
    model.transform.min_size = (size,)
    model.transform.max_size = size

    return model



def draw_boxes(image, boxes, labels, scores, class_names, color_sample, score_threshold=0.8):

    image_with_boxes = image.copy()

    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            xmin, ymin, xmax, ymax = map(int, box)

            # Ensure correct indexing
            color = color_sample[label % len(color_sample)]  

            # Draw bounding box
            cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw label text
            text = f"{class_names[label]}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_w, text_h = text_size

            # Adjust text position to prevent out-of-bounds
            ymin_text = max(ymin - text_h - 5, 0)  

            # Background rectangle for text
            cv2.rectangle(image_with_boxes, (xmin, ymin_text-5), (xmin + text_w + 10, ymin), color, -1)

            # Put text on the image with white color for better visibility
            cv2.putText(
                image_with_boxes,
                text,
                (xmin + 5, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA,
            )

    return image_with_boxes

def detection(image, class_names, color_sample):
    
    # Load the model
    model = create_model(num_classes=len(class_names), size=640)
    model.load_state_dict(torch.load("blood_cell.pth", map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    
    transform = transforms.ToTensor()
    
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0) # Add a batch dimension

    # Get predictions
    with torch.no_grad():
        prediction = model(image_tensor)[0]  # Get the first image's predictions

    # Extract predictions
    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    # Filter labels where score > 0.8
    filtered_labels = [label for label, score in zip(labels, scores) if score > 0.8]
    # filtered_labels = [0, 2, 1, 3, 2, 3, 3]

    # Count occurrences of each class
    counter = Counter(filtered_labels)

    # Convert image to numpy array for OpenCV
    image = np.array(image)
    
    # Draw Predictions on the image
    image_with_boxes = draw_boxes(image, boxes, labels, scores, class_names, color_sample)

    return image_with_boxes, counter

