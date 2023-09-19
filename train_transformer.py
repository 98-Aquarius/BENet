import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.nn.modules.loss import MSELoss 
from data import ImageDetectionsField, TextField, RawField, VisCtxField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider

from models.transformer import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention, ScaledDotProductAttentionMemory, T2IAttention


import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss, MSELoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse


import pickle
import numpy as np
import itertools
from shutil import copyfile

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import torch.nn.functional as F

from models.SimCLR_pre import Simclr_pre
from models.nt_xent import NT_Xent

def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    # temp = 0.25
    temp = 0.1
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    # infoNCE loss
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss

def evaluate_loss(model, dataloader, loss_fn, text_field, e, device):

    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader), disable=device!=0) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                out, encoder_out, img_out = model(mode='xe', images=detections, seq=captions)
                captions_gt = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                
                loss = loss_fn[0](out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss

def evaluate_metrics(model, dataloader, text_field, e, device):
    import itertools
    model.eval()
    seq_len = 20
    beam_size = 5
    gen = {}
    gts = {}

    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader), disable=device!=0) as pbar:
        for it, (images, caps_gt, captions) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model(mode='rl', images=images, max_len=seq_len, eos_idx=text_field.vocab.stoi['<eos>'], beam_size=beam_size, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    # temp = 0.25
    temp = 0.1
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    # infoNCE loss
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss

def train_xe(model, dataloader, optim, text_field,  scheduler, loss_fn, e, device):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    if device == 0:
        print('lr = ', optim.state_dict()['param_groups'][0]['lr'])
    
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader), disable=device!=0) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            out, encoder_out, img_out = model(mode='xe', images=detections, seq=captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()


            loss_cr = contrastive_loss(encoder_out, img_out)

            # print("out", out.shape)
            # print("out_view", (out.view(-1, len(text_field.vocab))).shape)
            # print("gt_view", (captions_gt.view(-1)).shape)
            # print("gt.shape", captions_gt.shape)

            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1)) + loss_cr

            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    # scheduler.step()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field,  scheduler_rl, e, device):
    # Training with self-critical
    # tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0

    model.train()
    scheduler_rl.step()
    if device == 0:
        print('lr = ', optim.state_dict()['param_groups'][0]['lr'])

    running_loss = .0
    seq_len = 20
    beam_size = 5
    # kwargs = {
    #     'text_flag': args.text2text
    # }
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader), disable=device!=0) as pbar:
        for it, (detections, caps_gt, captions) in enumerate(dataloader):
            detections = detections.to(device)
            text = captions.to(device)
            # kwargs['text'] = text
            outs, log_probs = model(mode='rl', images=detections, max_len=seq_len, eos_idx=text_field.vocab.stoi['<eos>'], beam_size=beam_size, out_size=beam_size)
            optim.zero_grad()
            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            # caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            caps_gen = evaluation.PTBTokenizer.tokenize(caps_gen)
            caps_gt = evaluation.PTBTokenizer.tokenize(caps_gt)
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            # pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
            #                  reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()
    # scheduler_rl.step()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    # tokenizer_pool.close()
    return loss, reward, reward_baseline


def _changeConfig(config, worldSize):
    batchSize = config.batch_size * worldSize
    # exponent = math.log2(batchSize)
    # scale = 3 - exponent / 2
    # config.xe_base_lr /= (2 ** scale)
    # config.rl_base_lr /= (2 ** scale)
    config.xe_base_lr *= worldSize
    config.rl_base_lr *= worldSize

def _generalConfig(rank: int, worldSize: int):
    #os.environ["MASTER_ADDR"] = "localhost"
    #os.environ["MASTER_PORT"] = "23482" #任意一个没被占用的端口号
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True
    random.seed(1244)
    torch.manual_seed(1244)
    np.random.seed(1244)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend = "nccl" if dist.is_nccl_available() else "gloo", world_size=worldSize, rank=rank,init_method="tcp://127.0.0.1:23459")


def train(rank, worldSize, args):
    _generalConfig(rank, worldSize)
    if rank == 0:
        print('Rank{}: Transformer Training'.format(rank))
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    #ResNext extract features
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=49, load_in_tmp=False)
    #CLIP extract features
    # image_field = VisCtxField(ctx_file=args.features_path)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits
    
    if not os.path.isfile('vocab.pkl'):
        print("Rank{}: Building vocabulary".format(rank))
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab.pkl', 'wb'))
    else:
        print('Rank{}: Loading from vocabulary'.format(rank))
        text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    # Model and dataloaders
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention,) #attention_module_kwargs={'m': args.m})
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])

    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder, args.num_clusters, len(text_field.vocab), 54, text_field.vocab.stoi['<pad>'], 512)
    model = torch.nn.parallel.DistributedDataParallel(model.to(rank), device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=True)
    
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'add_text':text_field})
    ref_caps_train = train_dataset.text()
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'add_text':text_field})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'add_text':text_field})

    '''
    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
    '''

    def lambda_lr(s):
        print("s:", s)
        if s <= 3:
            lr = args.xe_base_lr * s / 4
        elif s <= 10:
            lr = args.xe_base_lr
        elif s <= 12:
            lr = args.xe_base_lr * 0.2
        else:
            lr = args.xe_base_lr * 0.2 * 0.2
        return lr
    
    def lambda_lr_rl(s):
        refine_epoch = args.refine_epoch_rl
        print("rl_s:", s)
        if s <= refine_epoch:
            lr = args.rl_base_lr
        elif s <= refine_epoch + 3:
            lr = args.rl_base_lr * 0.2
        elif s <= refine_epoch + 6:
            lr = args.rl_base_lr * 0.2 * 0.2
        else:
            lr = args.rl_base_lr * 0.2 * 0.2 * 0.2
        return lr

    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)

    optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    loss_align = MSELoss()
    loss = (loss_fn, loss_align)
    use_rl = False
    best_cider = .0
    best_test_cider = 0.
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_transformer_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_transformer_models/%s_best.pth' % args.exp_name

        # fname = 'saved_transformer_models/align_share_K5_init_vlad_last.pth'

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            """
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            """
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            best_test_cider = data['best_test_cider']
            patience = data['patience']
            use_rl = data['use_rl']

            if use_rl:
                optim.load_state_dict(data['optimizer'])
                scheduler.load_state_dict(data['scheduler'])
            else:
                optim_rl.load_state_dict(data['optimizer'])
                scheduler_rl.load_state_dict(data['scheduler'])

            print('Resuming from epoch %d, validation loss %f, best cider %f, and best_test_cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))
            print('patience:', data['patience'])

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        trainSampler = DistributedSampler(train_dataset, worldSize, rank)
        trainSampler.set_epoch(e)
        dataloader_train = DataLoader(train_dataset, sampler=trainSampler, batch_size=args.batch_size, pin_memory=True, drop_last=False, num_workers=args.workers, persistent_workers=True)

        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        dict_trainSampler = DistributedSampler(dict_dataset_train, worldSize, rank)
        dict_trainSampler.set_epoch(e)
        dict_dataloader_train = DataLoader(dict_dataset_train, sampler=dict_trainSampler, batch_size=args.batch_size // 5,  pin_memory=True, drop_last=False, num_workers=args.workers, persistent_workers=True)

        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)
        
        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field, scheduler, loss_fn, e, rank)
            if rank == 0:
                writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim_rl, cider_train, text_field, scheduler_rl, e, rank)
            if rank == 0:
                writer.add_scalar('data/train_loss', train_loss, e)
                writer.add_scalar('data/reward', reward, e)
                writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss, text_field, e, rank)
        if rank == 0:
            writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field, e, rank)
        val_cider = scores['CIDEr']
        if rank == 0:
            print("Validation scores", scores)
            writer.add_scalar('data/val_cider', val_cider, e)
            writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
            writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
            writer.add_scalar('data/val_meteor', scores['METEOR'], e)
            writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field, e, rank)
        test_cider = scores['CIDEr']
        if rank == 0:
            print("Test scores", scores)
            writer.add_scalar('data/test_cider', test_cider, e)
            writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
            writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
            writer.add_scalar('data/test_meteor', scores['METEOR'], e)
            writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True

        switch_to_rl = False
        exit_train = False

        if patience == 5:
            if e < args.xe_least:   # xe stage train 15 epoches at least 
                if rank == 0:
                    print('special treatment, e = {}'.format(e))
                use_rl = False
                switch_to_rl = False
                patience = 0
            elif not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                
                optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
                
                for k in range(e-1):
                    scheduler_rl.step()
                if rank == 0:
                    print("Switching to RL")
            else:
                if rank == 0:
                    print('patience reached.')
                exit_train = True

        if e == args.xe_most:     # xe stage no more than 20 epoches
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                
                optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

                for k in range(e-1):
                    scheduler_rl.step()
                if rank == 0:
                    print("Switching to RL")
        if rank == 0:
            if switch_to_rl and not best:
                data = torch.load('saved_transformer_models/%s_best.pth' % args.exp_name)
                torch.set_rng_state(data['torch_rng_state'])
                torch.cuda.set_rng_state(data['cuda_rng_state'])
                np.random.set_state(data['numpy_rng_state'])
                random.setstate(data['random_rng_state'])
                model.load_state_dict(data['state_dict'])
                print('Resuming from epoch %d, validation loss %f, best_cider %f, and best test_cider %f' % (
                    data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))

            torch.save({
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'epoch': e,
                'val_loss': val_loss,
                'val_cider': val_cider,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict() if not use_rl else optim_rl.state_dict(),
                'scheduler': scheduler.state_dict() if not use_rl else scheduler_rl.state_dict(),
                'patience': patience,
                'best_cider': best_cider,
                'best_test_cider': best_test_cider,
                'use_rl': use_rl,
            }, 'saved_transformer_models/%s_last.pth' % args.exp_name)

            if best:
                copyfile('saved_transformer_models/%s_last.pth' % args.exp_name, 'saved_transformer_models/%s_best.pth' % args.exp_name)
            if best_test:
                copyfile('saved_transformer_models/%s_last.pth' % args.exp_name, 'saved_transformer_models/%s_best_test.pth' % args.exp_name)

        if exit_train:
            if rank==0:
                writer.close() 
            break


if __name__ == '__main__':
    # device = torch.device('cuda')
    
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--exp_name', type=str, default='demo')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')  
    parser.add_argument('--features_path', type=str, default="/home/mjxy/Projects/image_caption_datasets/X101_features/X101_grid_feats_coco_trainval.hdf5")
    # parser.add_argument('--features_path', type=str, default="/home/mjxy/Projects/HAAV-master/outputs/image_features/vis_ctx.hdf5")
    parser.add_argument('--annotation_folder', type=str, default="/home/mjxy/Projects/image_caption_datasets/mscoco/annotations/")

    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--xe_least', type=int, default=15)
    parser.add_argument('--xe_most', type=int, default=20) # 18
    parser.add_argument('--refine_epoch_rl', type=int, default=28)  # 35

    parser.add_argument('--xe_base_lr', type=float, default=0.0001)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--text2text', type=int, default=0)

    args = parser.parse_args()
    print(args)
    ## DDP Training
    worldSize = 1
    _changeConfig(args, worldSize)
    print('\nDistribute config', args)
    mp.spawn(train, args=(worldSize, args,), nprocs=worldSize,join=True)
