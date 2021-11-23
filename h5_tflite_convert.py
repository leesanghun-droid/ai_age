# Life is incomplete without this statement!
import tensorflow as tf

from tensorflow import keras

# And this as well!
import numpy as np

model = tf.keras.models.load_model( 'my_model.h5' )
converter = tf.lite.TFLiteConverter.from_keras_model(model)

def representative_dataset_gen():
    for _ in range(10):
        input_array = np.random.random((1,200,200,3))
        input_array = np.array(input_array,dtype=np.float32)
        yield [input_array]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open('keras_model_2-3-0.tflite', 'wb') as f:
    f.write(tflite_model)