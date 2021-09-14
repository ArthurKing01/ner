
# Load saved model
import kashgari
from kashgari.utils import convert_to_saved_model
import argparse
import os
from dirs import pb_models_dir, keras_models_dir

parser = argparse.ArgumentParser(description="your script description") 
parser.add_argument('--model_name', '-n', required=True, type=str)

args = parser.parse_args()

model_name = args.model_name
model_path = os.path.join(keras_models_dir, model_name)

loaded_model = kashgari.utils.load_model(model_path)


if not os.path.exists(pb_models_dir):
    os.mkdir(pb_models_dir)

save_model_dir = os.path.join(pb_models_dir, model_name)
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)

versions = os.listdir(save_model_dir)
versions.sort(key=lambda x: int(x))
print(versions)

new_version = 1
if len(versions) > 0:
    new_version = int(versions[-1]) + 1

convert_to_saved_model(loaded_model, save_model_dir, str(new_version))