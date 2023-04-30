# model.py
import requests
from io import BytesIO
import numpy as np
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import cv2
from transformers import AutoImageProcessor, Swinv2ForImageClassification,AutoModelForImageClassification
import torch
# from datasets import load_dataset
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from ultralytics import YOLO

# from torchvision.models.segmentation import deeplabv3_resnet50
# from torchvision.models.segmentation.deeplabv3 import DeepLabV3Head, DeepLabV3, DeepLabV3Plus, DeepLabHead

# # Load pre-trained segmentation model
# model_segmentation = deeplabv3_resnet50(pretrained=True, progress=True, num_classes=21, aux_loss=None, 
#                                         backbone=None, replace_stride_with_dilation=None,
#                                         pretrained_backbone=True, trainable_backbone_layers=None,
#                                         norm_layer=None, deep_lab_v3_plus=False,
#                                         decoder_kernel_size=4, decoder_atrous_rate=(12, 24, 36),
#                                         output_stride=16, classifier=None, sigmoid_norm=True, **kwargs)

# # Set the weights argument explicitly to avoid the warning message
# model_segmentation = deeplabv3_resnet50(pretrained=True, weights=DeepLabV3_ResNet50_Weights.DEFAULT)


def load_image(image_path_or_stream, target_size=(224, 224)):
    if isinstance(image_path_or_stream, str) and (image_path_or_stream.startswith("http://") or image_path_or_stream.startswith("https://")):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
        }
        response = requests.get(image_path_or_stream, headers=headers)
        content_type = response.headers.get('content-type')

        if response.status_code != 200 or 'image' not in content_type:
            raise Exception(f"Failed to fetch image from URL, status code: {response.status_code}, content type: {content_type}")

        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_stream)
    
    image = image.resize(target_size)
    return image, image
    #return np.asarray(image), image
    #return preprocess_image(image), image


# def load_model():
#     return hub.load(model_handle)




def classify_image(model, image):
    # Convert the image to float32 and normalize pixel values
    image = np.asarray(image, dtype=np.float32) / 255.0

    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)

    # Run the model on the input image
    logits = model(image)
    probabilities = tf.nn.softmax(logits)
    return probabilities.numpy()



# Load pre-trained classification model
# model_classification = torchvision.models.resnet50(pretrained=True)
# model_classification.eval()
# Load pre-trained segmentation model
model_segmentation = deeplabv3_resnet50(pretrained=True)
model_segmentation.eval()

# Load pre-trained YOLOv6 model
# Load a pretrained YOLO model (recommended for training)
model_yolo = YOLO('yolov8l.pt')

# Preprocessing function
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

# Classification function
def classify_image(image):

    # dataset = load_dataset("huggingface/cats-image")
    # image = dataset["test"]["image"][0]

    # image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    # model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window16-256")
    model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window16-256")

    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, k=5)

    # print the top five predicted classes and their probabilities
    label = []
    prob = []

    # print the top five predicted classes and their probabilities
    for i in range(5):
        label.append(model.config.id2label[top5_indices[0][i].item()])
        prob.append(top5_probs[0][i].item())
        print(f"{label[i]}: {prob[i]:.2f}")
    
    top_5_predictions = [(label[index], float(prob[index])) for index in range(len(top5_indices[0]))]
    return top_5_predictions

# Segmentation function
def segment_image(image):
    # Preprocess the image
    input_tensor = preprocess_image(image)
    input_batch = input_tensor.unsqueeze(0)

    # Run the segmentation model
    with torch.no_grad():
        output = model_segmentation(input_batch)['out'][0]
    segmentation_map = output.argmax(0).numpy()
    segmentation_map = segmentation_map.astype('uint8')*255
    segmentation_map = Image.fromarray(segmentation_map)
    buffer = BytesIO()
    segmentation_map.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer

# Object detection function
def detect_objects(image):
    with torch.no_grad():
        detections = model_yolo(image)
        obd_image = plot_bboxes(image, detections[0].boxes.data, score=False)
    obd_image = obd_image.astype('uint8')
    obd_image = Image.fromarray(obd_image)
    buffer_obd = BytesIO()
    obd_image.save(buffer_obd, format="PNG")
    buffer_obd.seek(0)

    return buffer_obd




#@markdown For that we use our function (short and simple) which allows us to display the bounding boxes with the label and the score :
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  image = np.asarray(image)
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)
    # cv2.imwrite('output.jpg', image)
    return image

def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
  #Define COCO Labels
  if labels == []:
    labels = {0: u'__background__', 1: u'person', 2: u'bicycle',3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog', 54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'}
  #Define colors
  if colors == []:
    #colors = [(6, 112, 83), (253, 246, 160), (40, 132, 70), (205, 97, 162), (149, 196, 30), (106, 19, 161), (127, 175, 225), (115, 133, 176), (83, 156, 8), (182, 29, 77), (180, 11, 251), (31, 12, 123), (23, 6, 115), (167, 34, 31), (176, 216, 69), (110, 229, 222), (72, 183, 159), (90, 168, 209), (195, 4, 209), (135, 236, 21), (62, 209, 199), (87, 1, 70), (75, 40, 168), (121, 90, 126), (11, 86, 86), (40, 218, 53), (234, 76, 20), (129, 174, 192), (13, 18, 254), (45, 183, 149), (77, 234, 120), (182, 83, 207), (172, 138, 252), (201, 7, 159), (147, 240, 17), (134, 19, 233), (202, 61, 206), (177, 253, 26), (10, 139, 17), (130, 148, 106), (174, 197, 128), (106, 59, 168), (124, 180, 83), (78, 169, 4), (26, 79, 176), (185, 149, 150), (165, 253, 206), (220, 87, 0), (72, 22, 226), (64, 174, 4), (245, 131, 96), (35, 217, 142), (89, 86, 32), (80, 56, 196), (222, 136, 159), (145, 6, 219), (143, 132, 162), (175, 97, 221), (72, 3, 79), (196, 184, 237), (18, 210, 116), (8, 185, 81), (99, 181, 254), (9, 127, 123), (140, 94, 215), (39, 229, 121), (230, 51, 96), (84, 225, 33), (218, 202, 139), (129, 223, 182), (167, 46, 157), (15, 252, 5), (128, 103, 203), (197, 223, 199), (19, 238, 181), (64, 142, 167), (12, 203, 242), (69, 21, 41), (177, 184, 2), (35, 97, 56), (241, 22, 161)]
    colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),(155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),(128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),(164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),(181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),(100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),(205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),(119, 24, 48),(73, 8, 110)]
  
  #plot each boxes
  for box in boxes:
    #add score in label if score=True
    if score :
      label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
    else :
      label = labels[int(box[-1])+1]
    #filter every box under conf threshold if conf threshold setted
    if conf :
      if box[-2] > conf:
        color = colors[int(box[-1])]
        obd_image = box_label(image, box, label, color)
    else:
      color = colors[int(box[-1])]
      obd_image = box_label(image, box, label, color)
  return obd_image







