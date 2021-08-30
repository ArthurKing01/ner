import tensorflow as tf

model_name = 'lstm.per_ner_1630047688'
converter = tf.lite.TFLiteConverter.from_saved_model('models/' + model_name)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)