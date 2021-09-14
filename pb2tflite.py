import tensorflow as tf
import argparse
import os
from dirs import pb_models_dir, tflite_models_dir

parser = argparse.ArgumentParser(description="your script description") 
parser.add_argument('--model_name', '-n', required=True, type=str)
parser.add_argument('--model_version', '-v', required=False, type=str)
args = parser.parse_args()

model_name = args.model_name
model_dir = os.path.join(pb_models_dir, model_name)

if args.model_version == None:
    versions = os.listdir(model_dir)
    versions.sort(key=lambda x: int(x))
    version = versions[-1]
else:
    version = args.model_version


current_model_path = os.path.join(model_dir, version)

converter = tf.lite.TFLiteConverter.from_saved_model(current_model_path)
tflite_model = converter.convert()

if not os.path.exists(tflite_models_dir):
    os.mkdir(tflite_models_dir)
open(os.path.join(tflite_models_dir, model_name + '.' + version + ".tflite"), "wb").write(tflite_model)