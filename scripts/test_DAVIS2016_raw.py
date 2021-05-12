import os
import sys
from argparse import Namespace
sys.path.append(f'{os.environ["Codinria"]}/unsupervised_detection/')
from test_generator import main

CKPT_FILE=f"{os.environ['Dataria']}/Models/Contextual_GAN/davis_best_model/model.best"
DATASET_FILE=f"{os.environ['Dataria']}/Daphne_110120/" #Motion_Saliency_moving_camera_videos
PWC_CKPT_FILE='Not using for inference'#f"{os.environ['Dataria']}/Models/Contextual_GAN/pwc/checkpoint"
OUTPUT_FILE = f"{os.environ['Dataria']}/Daphne_110120/Analysis/Contextual_GAN_Selection"


argvs = {
 'dataset':'DAVIS2016',
 'ckpt_file' : CKPT_FILE,
 'flow_ckpt' : PWC_CKPT_FILE,
 'test_crop' : 0.9,
 'test_temporal_shift' : 1,
 'batch_size' : 1,
 'root_dir' : DATASET_FILE,
 'generate_visualization' : True,
 'test_save_dir' : OUTPUT_FILE
}

a = ['test_generator.py'] + [f'--{k}={v}' for k,v in argvs.items()]

main(a)
