import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os
import numpy as np
from glob import glob
import xml.etree.ElementTree as ET

# Function to load the model
def load_model(model_path, device):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    # Since we're loading our trained model, we don't need pre-trained weights.
    model = fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # background and lego
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Function to perform inference on a single image and display the result
def run_inference(model, image_path, device, score_threshold=0.5, display=True)
    # Define transform
    transform = T.Compose([T.ToTensor()])
    # Open image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).to(device)

    # Run inference
    with torch.no_grad():
        prediction = model([img_tensor])
    
    # Get the first prediction result
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    pred_labels = prediction[0]['labels'].cpu().numpy()

    # Filter predictions by score threshold
    valid_idx = pred_scores >= score_threshold
    boxes = pred_boxes[valid_idx]
    scores = pred_scores[valid_idx]
    labels = pred_labels[valid_idx]
    
    # Display results if requested
    if display:
        # Plot the image with bounding boxes
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.title(f"Detected {len(boxes)} LEGO piece(s)")
        plt.show()
    
    return {
        'boxes': boxes, 
        'scores': scores, 
        'labels': labels, 
        'image_size': img.size
    }

# Function to parse XML annotations
def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    labels = []
    for obj in root.findall("object"):
        # All objects will be labeled as "lego" (label = 1)
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)  # 1 for lego
    
    return {
        'boxes': np.array(boxes), 
        'labels': np.array(labels)
    }

# Function to calculate IoU between two boxes
def calculate_iou(box1, box2):
    # box format: [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

# Function to calculate Average Precision
def calculate_ap(predictions, ground_truths, iou_threshold=0.5):
    TP = []
    FP = []
    num_predictions = len(predictions['boxes'])
    num_ground_truths = len(ground_truths['boxes'])
    
    # Sort predictions by confidence score in descending order
    if num_predictions > 0:
        sorted_indices = np.argsort(predictions['scores'])[::-1]
        sorted_boxes = predictions['boxes'][sorted_indices]
        sorted_scores = predictions['scores'][sorted_indices]
    else:
        sorted_boxes = np.array([])
        sorted_scores = np.array([])
    
    # Create a list to track which ground truth boxes have been matched
    matched_gt = [False] * num_ground_truths
    
    # For each predicted box
    for i in range(num_predictions):
        if num_ground_truths == 0:
            TP.append(0)
            FP.append(1)
            continue
        
        max_iou = 0
        max_idx = -1
        
        # Find the ground truth box with highest IoU
        for j in range(num_ground_truths):
            if not matched_gt[j]:
                iou = calculate_iou(sorted_boxes[i], ground_truths['boxes'][j])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
        
        if max_iou >= iou_threshold and max_idx != -1:
            matched_gt[max_idx] = True  # Mark as matched
            TP.append(1)
            FP.append(0)
        else:
            TP.append(0)
            FP.append(1)
    
    # Calculate precision and recall at each threshold
    TP_cumsum = np.cumsum(TP)
    FP_cumsum = np.cumsum(FP)
    
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum)
    recalls = TP_cumsum / num_ground_truths if num_ground_truths > 0 else np.zeros_like(TP_cumsum)
    
    # Add sentinel values for calculation
    precisions = np.concatenate(([1], precisions))
    recalls = np.concatenate(([0], recalls))
    
    # Calculate Average Precision using the 11-point interpolation
    average_precision = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        average_precision += p / 11
    
    return average_precision

# Function to evaluate the model on a test set
def evaluate_model(model, test_dir, device, score_threshold=0.5):
    image_dir = os.path.join(test_dir, "images")
    annot_dir = os.path.join(test_dir, "annotations")
    
    all_ap_values = []
    image_files = glob(os.path.join(image_dir, "*.jpg"))
    
    total_detections = 0
    total_gt = 0
    
    for img_path in image_files:
        img_filename = os.path.basename(img_path)
        xml_filename = os.path.splitext(img_filename)[0] + ".xml"
        xml_path = os.path.join(annot_dir, xml_filename)
        
        # Skip if annotation file doesn't exist
        if not os.path.exists(xml_path):
            print(f"Warning: Annotation not found for {img_filename}, skipping...")
            continue
        
        # Run inference
        predictions = run_inference(model, img_path, device, score_threshold, display=False)
        
        # Parse ground truth
        ground_truth = parse_annotation(xml_path)
        
        # Convert predictions to numpy arrays
        pred_boxes = np.array(predictions['boxes'])
        pred_scores = np.array(predictions['scores'])
        pred_labels = np.array(predictions['labels'])
        
        # Prepare data for AP calculation
        predictions_formatted = {
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': pred_labels
        }
        
        # Calculate AP for this image
        ap = calculate_ap(predictions_formatted, ground_truth, iou_threshold=0.5)
        all_ap_values.append(ap)
        
        # Count detections and ground truths
        total_detections += len(pred_boxes)
        total_gt += len(ground_truth['boxes'])
        
        print(f"Processed {img_filename}: AP = {ap:.4f}")
    
    # Calculate mAP
    mAP = np.mean(all_ap_values) if all_ap_values else 0
    
    print(f"\nEvaluation Results:")
    print(f"Number of test images: {len(all_ap_values)}")
    print(f"Total ground truth objects: {total_gt}")
    print(f"Total detected objects: {total_detections}")
    print(f"mAP@0.5: {mAP:.4f}")
    
    return mAP

# Main function
def main():
    parser = argparse.ArgumentParser(description='LEGO Detection Inference')
    parser.add_argument('--model', type=str, default="fasterrcnn_lego.pth",
                        help='Path to the trained model')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single test image')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Path to test directory containing images and annotations')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Score threshold for detection')
    args = parser.parse_args()
    
    # Use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    
    # If single image is provided
    if args.image and os.path.exists(args.image):
        print(f"Running inference on {args.image}")
        run_inference(model, args.image, device, score_threshold=args.threshold)
    
    # If test directory is provided
    elif args.test_dir and os.path.exists(args.test_dir):
        print(f"Evaluating model on test set in {args.test_dir}")
        mAP = evaluate_model(model, args.test_dir, device, score_threshold=args.threshold)
        print(f"Final mAP@0.5: {mAP:.4f}")
    
    else:
        print("Error: Please provide either a valid image path or test directory.")

if __name__ == "__main__":
    main()
