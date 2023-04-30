import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def create_transformer_model():
    input_layer = Input(shape=(224, 224, 3))
    transformer_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/vit_base_patch16_224/1", trainable=True)
    x = transformer_layer(input_layer)
    model = Model(inputs=input_layer, outputs=x)
    return model



def create_classifier_model():
    input_layer = Input(shape=(768,))
    x = Dense(1000, activation='softmax')(input_layer)
    model = Model(inputs=input_layer, outputs=x)
    return model


transformer_model = create_transformer_model()


