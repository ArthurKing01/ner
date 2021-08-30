import tensorflow as tf
import os

# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model_name = 'lstm.per_ner_1630047688'
converter = tf.lite.TFLiteConverter.from_saved_model('models/' + model_name)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)