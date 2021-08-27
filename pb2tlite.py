import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('models/1628217960')
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)