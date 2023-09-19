# 线上测试：evaluating the performance of captioning model on official MS-COCO test server.
# 此处生成指定格式的官方测试集和验证集对应的captions文件，将这两个文件压缩后上传到[CodaLab](https://competitions.codalab.org/competitions/3221#participate)即可得到
# 线上测试的结果和排名

import torch
import argparse
import pickle
import numpy as np
import itertools
import json
import os

from tqdm import tqdm

from data import TextField, COCO_TestOnline
from models.transformer import Transformer, TransformerEncoder, TransformerDecoderLayer,  ScaledDotProductAttention, TransformerEnsemble

import random
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

from torch.utils.data import DataLoader


def gen_caps(captioning_model, dataset, batch_size=10, workers=0):
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=workers
    )

    outputs = []
    with tqdm(len(dataloader)) as pbar:
        for it, (image_ids, images) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = captioning_model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            caps_gen = [' '.join([k for k, g in itertools.groupby(gen_i)]).strip() for gen_i in caps_gen]
            for i in range(image_ids.size(0)):
                item = {}
                item['image_id'] = int(image_ids[i])
                item['caption'] = caps_gen[i]
                outputs.append(item)
            pbar.update()
    return outputs


def save_results(outputs, datasplit, dir_to_save_caps):
    if not os.path.exists(dir_to_save_caps):
        os.makedirs(dir_to_save_caps)
    #  命名规范：captions_test2014_XXX_results.json 和 captions_val2014_XXX_results.json
    output_path = os.path.join(dir_to_save_caps, 'captions_' + datasplit + '2014_ReCNet_results.json')
    with open(output_path, 'w') as f:
        json.dump(outputs, f)
    

if __name__ == '__main__':

    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Relationship-Sensitive Transformer Network')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    
    parser.add_argument('--datasplit', type=str, default='test')    # test, val

    # 测试集
    parser.add_argument('--test_features_path', type=str, default='./X101_grid_feats_coco_test.hdf5')
    parser.add_argument('--test_annotation_folder', type=str, default='./m2_annotations/image_info_test2014.json')
    
    # 验证集
    parser.add_argument('--val_features_path', type=str, default='/home/mjxy/Projects/S2copy/X101_grid_feats_coco_trainval.hdf5')
    parser.add_argument('--val_annotation_folder', type=str, default='/home/mjxy/Projects/S2copy/m2_annotations/captions_val2014.json')

    # 模型参数
    parser.add_argument('--models_path', type=list, default=[
    './saved_transformer_models/133.4/100memory/demo_best_test.pth',
    './saved_transformer_models/demo_best_test.pth',
    './saved_transformer_models/133.2/demo_best_test.pth',
    './saved_transformer_models/132.9/demo_best_test.pth',
    ])

    parser.add_argument('--dir_to_save_caps', type=str, default='./test_online/results/')    # 文件保存路径

    args = parser.parse_args()

    print('The Online Evaluation of RSTNet')

    # 加载数据集
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    text_field.vocab = pickle.load(open('./vocab.pkl', 'rb'))
    if args.datasplit == 'test':
        dataset = COCO_TestOnline(args.test_features_path, args.test_annotation_folder)
    else:
        dataset = COCO_TestOnline(feat_path=args.val_features_path, ann_file=args.val_annotation_folder)

    # 加载模型参数
    # 模型结构
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention) #attention_module_kwargs={'m': 40})
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder, 5, len(text_field.vocab), 54, text_field.vocab.stoi['<pad>'], 512).to(device)
    # 集成模型
    ensemble_model = TransformerEnsemble(model=model, weight_files=args.models_path)
    #data = torch.load(args.models_path)
    #model.load_state_dict({k.replace('module.',''):v for k,v in data['state_dict'].items()})
    # 生成结果
    outputs = gen_caps(ensemble_model, dataset, batch_size=args.batch_size, workers=args.workers)

    # 保存结果
    save_results(outputs, args.datasplit, args.dir_to_save_caps)
    
    print('finished!')
    
    
