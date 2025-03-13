import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
def run_inference(model, image_path, device, score_threshold=0.5):
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

    # Filter predictions by score threshold
    valid_idx = pred_scores >= score_threshold
    boxes = pred_boxes[valid_idx]
    
    # Plot the image with bounding boxes
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.title(f"Detected {len(boxes)} LEGO piece(s)")
    plt.show()

# Main function
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_path = "fasterrcnn_lego.pth"
    model = load_model(model_path, device)
    
    # Change this to the path of your test image
    test_image_path = r"D:\Downloads\lego_detection_project\reduced\val\images\fd0713e4-d9e2-11eb-89e9-3497f683a169.jpg"
    run_inference(model, test_image_path, device, score_threshold=0.5)