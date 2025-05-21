import sys
sys.path.append(".")

import os
import torch.backends.cudnn as cudnn
import yaml
from step_train import train
from utils import AttrDict
import pandas as pd

import argparse

cudnn.benchmark = True
cudnn.deterministic = False

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, default="config_files/cyrillic-v1.yaml",
                        help="Path to the configuration file")
    args = parser.parse_args()

    opt = get_config(args.config)
    train(opt, amp=False)