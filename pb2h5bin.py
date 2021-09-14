import tensorflowjs as tfjs
import argparse
import os
from dirs import pb_models_dir, web_models_dir

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

tfjs.converters.convert_tf_saved_model(saved_model_dir=current_model_path, output_dir=os.path.join(web_models_dir, model_name+'.'+version))