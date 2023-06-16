# model.py
import requests
from io import BytesIO
import numpy as np
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import cv2
# from transformers import AutoImageProcessor, Swinv2ForImageClassification,AutoModelForImageClassification
import torch
# from datasets import load_dataset
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from ultralytics import YOLO
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
from transformers import AutoFeatureExtractor, CvtForImageClassification

######################## Image understanding #######################
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

# Load the model architecture
segModel = deeplabv3_resnet50(pretrained=False)

# Load the model weights from the local file
segModel.load_state_dict(torch.load("/app/models/ImageUnderstanding/Segmentation/deeplabv3_resnet50_weights.pth"))

# Put the model in evaluation mode
segModel.eval()


def load_image(image_path_or_stream, target_size=(480, 480)):
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
    
    # image = image.resize(target_size)
    return image, image
    #return np.asarray(image), image
    #return preprocess_image(image), image




# Load pre-trained classification model
# model_classification = torchvision.models.resnet50(pretrained=True)
# model_classification.eval()
# Load pre-trained segmentation model
# SEG_model = deeplabv3_resnet50(pretrained=True)
# SEG_model.eval()

# # Load pre-trained YOLOv6 model
# # Load a pretrained YOLO model (recommended for training)
# model_yolo = YOLO('yolov8l.pt')

# Preprocessing function
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)


clFeature_extractor = AutoFeatureExtractor.from_pretrained("/app/models/ImageUnderstanding/ImageClassification/microsoft/cvt-21-384-22k")

# Load the model from the local files
clModel = CvtForImageClassification.from_pretrained("/app/models/ImageUnderstanding/ImageClassification/microsoft/cvt-21-384-22k")



# Classification function
def classify_image(image):


    inputs = clFeature_extractor(images=image, return_tensors="pt")


    with torch.no_grad():
        logits = clModel(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, k=5)

    # print the top five predicted classes and their probabilities
    label = []
    prob = []

    # print the top five predicted classes and their probabilities
    for i in range(5):
        label.append(clModel.config.id2label[top5_indices[0][i].item()])
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
        output = segModel(input_batch)['out'][0]
    segmentation_map = output.argmax(0).numpy()
    segmentation_map = segmentation_map.astype('uint8')*255
    segmentation_map = Image.fromarray(segmentation_map)
    buffer = BytesIO()
    segmentation_map.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer
# Load the processor from the local files
processor_obd = DetrImageProcessor(
    # feature_extractor_config_file="/Users/skasmani/Downloads/feature_extractor_config.json",
    processor_config_file="/app/models/ImageUnderstanding/ObjectDetection/processor_config.json"
)

# Load the model from the local files
config_obd = DetrConfig.from_pretrained("/app/models/ImageUnderstanding/ObjectDetection/config.json")
model_obd = DetrForObjectDetection(config_obd)
model_obd.load_state_dict(torch.load("/app/models/ImageUnderstanding/ObjectDetection/pytorch_model.bin"))


# Object detection function
def detect_objects(image):


    inputs = processor_obd(images=image, return_tensors="pt")
    outputs = model_obd(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor_obd.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    obd_image =visualise_od(image, results,model_obd)
    obd_image = obd_image.astype('uint8')
    obd_image = Image.fromarray(obd_image)
    buffer_obd = BytesIO()
    obd_image.save(buffer_obd, format="PNG")
    buffer_obd.seek(0)

    return buffer_obd





# # Object detection function
# def detect_objects(image):
#     with torch.no_grad():
#         detections = model_yolo(image)
#         obd_image = plot_bboxes(image, detections[0].boxes.data, score=False)
#     obd_image = obd_image.astype('uint8')
#     obd_image = Image.fromarray(obd_image)
#     buffer_obd = BytesIO()
#     obd_image.save(buffer_obd, format="PNG")
#     buffer_obd.seek(0)

#     return buffer_obd



# Load the model from the local files
icModel = VisionEncoderDecoderModel.from_pretrained("/app/models/ImageUnderstanding/ImageCaptionning")

# Load the tokenizer and feature extractor from the local files
icTokenizer = AutoTokenizer.from_pretrained("/app/models/ImageUnderstanding/ImageCaptionning")
icFeature_extractor = ViTImageProcessor.from_pretrained("/app/models/ImageUnderstanding/ImageCaptionning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
icModel.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def caption_image(image):
    images = []
    # for image_path in image_paths:
    #     i_image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    images.append(image)

    pixel_values = icFeature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = icModel.generate(pixel_values, **gen_kwargs)

    preds = icTokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# pred = caption_image(['/Users/skasmani/Downloads/IBM/ibm-repo/image_classification_microservice/data/test/dog.jpg'])





# IC_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# IC_model.to(device)
# # Object detection function
# def caption_image(image):


#     max_length = 16
#     num_beams = 4
#     gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
#     images = []
#     # for image_path in image_paths:
#     #     i_image = Image.open(image_path)
#     if image.mode != "RGB":
       
#         image = image.convert(mode="RGB")

#     images.append(image)

#     pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device)

#     output_ids = IC_model.generate(pixel_values, **gen_kwargs)

#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     preds = [pred.strip() for pred in preds]
#     return preds


# predict_step(['/Users/skasmani/Downloads/IBM/ibm-repo/image_classification_microservice/data/test/dog.jpg']) # ['a woman in a hospital bed with a woman in a hospital bed']




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


def visualise_od(image, results,model):
    # Convert the PIL Image to an OpenCV-compatible format (BGR)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Set the font for the labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale =1.8
    font_thickness = 6
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence {score.item()}"
        )

        # Get the coordinates of the bounding box and convert them to integers
        x, y, width, height = map(int, box)

        # Draw the bounding box on the image
        cv2.rectangle(cv_image, (x, y), (x + width, y + height), (0, 0, 255), 2)

        # Add a label with the class name and confidence score
        label_text = f"{model.config.id2label[label.item()]} {score.item():.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        cv2.rectangle(cv_image, (x, y - text_height), (x + text_width, y), (255, 255, 255), -1)
        cv2.putText(cv_image, label_text, (x, y), font, font_scale, (0, 0, 0), font_thickness)

    # Convert the image back to RGB format
    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Return the modified image in RGB format
    return cv_image_rgb




######################## Text Processing #######################


from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification,AutoModelForSeq2SeqLM
from transformers import   GenerationConfig

# Replace the model name with the path to the downloaded model files
emoDetModel_path = "./models/TextProcessing/classification/emotion-english-distilroberta-base/"
# Load the tokenizer and model from the local directory
emoDetTokenizer = AutoTokenizer.from_pretrained(emoDetModel_path)
emoDetModel = AutoModelForSequenceClassification.from_pretrained(emoDetModel_path)
def emotion_detection(text):
    # Create the text classification pipeline with the local model
    classifier = pipeline("text-classification", model=emoDetModel, tokenizer=emoDetTokenizer, return_all_scores=True)

    # Use the pipeline to classify text
    pred = classifier(text)
    print(pred)
    return pred


# Replace the model name with the path to the downloaded model files
sumModel_path = "./models/TextProcessing/summarisation/MEETING_SUMMARY/"
# Load the tokenizer and model from the local directory
sumTokenizer = AutoTokenizer.from_pretrained(sumModel_path)
sumModel = AutoModelForSeq2SeqLM.from_pretrained(sumModel_path)

def text_summarization(text):
    # Create the text classification pipeline with the local model
    summariser = pipeline("summarization", model=sumModel, tokenizer=emoDetTokenizer)

    # Use the pipeline to classify text
    summary = summariser(text)
    print(summary)
    return summary



# Load tokenizer and model from local files
texGenTokenizer = AutoTokenizer.from_pretrained("./models/TextProcessing/TextGeneration/t5-base", local_files_only=True)
texGenModel = AutoModelForSeq2SeqLM.from_pretrained("./models/TextProcessing/TextGeneration/t5-base", local_files_only=True)

# # Define generation config and save to file
# translation_generation_config = GenerationConfig(
#     num_beams=4,
#     early_stopping=True,
#     decoder_start_token_id=0,
#     eos_token_id=model.config.eos_token_id,
#     pad_token_id=model.config.pad_token_id,
# )
# translation_generation_config.save_pretrained("./models/TextProcessing/TextGeneration/t5-base")

# Load generation config from file
texGengeneration_config = GenerationConfig.from_pretrained("./models/TextProcessing/TextGeneration/t5-base")
def generate_text(text):
    # Generate translation
    inputs = texGenTokenizer(text, return_tensors="pt")
    outputs = texGenModel.generate(**inputs, generation_config=texGengeneration_config)
    print(texGenTokenizer.batch_decode(outputs, skip_special_tokens=True))
    return texGenTokenizer.batch_decode(outputs, skip_special_tokens=True)

