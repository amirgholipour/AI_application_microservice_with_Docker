# inference.py
import tensorflow as tf
from model import classify_image as classifier
from model import load_model


def infer(image):
   model = load_model()
   probabilities = classifier(model, image)
   return probabilities

