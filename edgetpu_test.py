# TensorFlow and tf.keras
from PIL import Image
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np

# Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path="./saved_model/my_model.quant.tflite")
#interpreter = tf.lite.Interpreter(model_path="./keras_model_2-3-0_edgetpu.tflite",experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter = tf.lite.Interpreter(model_path="./keras_model_2-3-0.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = Image.open('./year1.jpg')

#print(input_details)
#print(output_details)

image = image.resize([200, 200])
input_data =np.array(image)
input_data = np.array([input_data])
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()


output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
output_data = interpreter.get_tensor(output_details[1]['index'])
print(output_data)









# Test model on random input data.
#input_shape = input_details[0]['shape']
#input_data = np.array((1,200,200,3), dtype=np.float32)

#input_data = np.array([[1]], dtype=np.float32)
#print(input_data)

# interpreter.set_tensor(input_details[0]['index'], input_data)

# interpreter.invoke()

# output_data = interpreter.get_tensor(output_details[0]['index'])
# print("output : %s" % output_data)