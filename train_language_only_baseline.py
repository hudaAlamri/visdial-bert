import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.optim as optim

from dataloader.dataloader_visdial import VisdialDataset
import options
from models.language_only_dialog_encoder import DialogEncoder
from utils.visualize import VisdomVisualize
from utils.visdial_metrics import SparseGTMetrics, NDCG, scores_to_ranks
from utils.data_utils import sequence_mask, batch_iter
from utils.optim_utils import WarmupLinearScheduleNonZero

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW
import os
import pprint
from time import gmtime, strftime
from timeit import default_timer as timer

def forward(dialog_encoder, batch, params, output_nsp_scores=False, output_lm_scores=False, sample_size=None, evaluation=False):

    tokens = batch['tokens']
    segments = batch['segments']
    sep_indices = batch['sep_indices'] 
    mask = batch['mask']
    hist_len = batch['hist_len']

    tokens = tokens.view(-1,tokens.shape[-1])
    segments = segments.view(-1, segments.shape[-1])
    sep_indices = sep_indices.view(-1,sep_indices.shape[-1])
    mask = mask.view(-1, mask.shape[-1])
    hist_len = hist_len.view(-1)

    if sample_size:
        # subsample a random set
        sample_indices = torch.randperm(hist_len.shape[0])
        sample_indices = sample_indices[:sample_size]
    else:
        sample_indices = torch.arange(hist_len.shape[0])

    tokens = tokens[sample_indices, :]
    segments = segments[sample_indices, :]
    sep_indices = sep_indices[sample_indices, :]
    mask = mask[sample_indices, :]
    hist_len = hist_len[sample_indices]

    next_sentence_labels = None

    if not evaluation:
        next_sentence_labels = batch['next_sentence_labels']
        next_sentence_labels = next_sentence_labels.view(-1)
        next_sentence_labels = next_sentence_labels[sample_indices]
        next_sentence_labels = next_sentence_labels.to(params['device'])

    tokens = tokens.to(params['device'])
    segments = segments.to(params['device'])
    sep_indices = sep_indices.to(params['device'])
    mask = mask.to(params['device'])
    hist_len = hist_len.to(params['device'])

    sequence_lengths = torch.gather(sep_indices,1,hist_len.view(-1,1)) + 1
    sequence_lengths = sequence_lengths.squeeze(1)
    attention_mask_lm_nsp = sequence_mask(sequence_lengths, max_len=tokens.shape[1])
    nsp_scores = None
    nsp_loss = None
    lm_loss = None
    loss = None
    lm_scores = None

    sep_len = hist_len + 1
    
    if output_nsp_scores and output_lm_scores:
        lm_nsp_loss, lm_loss, nsp_loss, nsp_scores, lm_scores = dialog_encoder(tokens,sep_indices=sep_indices, sep_len=sep_len\
                    ,token_type_ids=segments, masked_lm_labels=mask,attention_mask=attention_mask_lm_nsp\
                    ,next_sentence_label=next_sentence_labels,output_nsp_scores=output_nsp_scores, output_lm_scores=output_lm_scores)
    elif output_nsp_scores and not output_lm_scores:
        lm_nsp_loss, lm_loss, nsp_loss, nsp_scores = dialog_encoder(tokens,sep_indices=sep_indices, sep_len=sep_len\
                                ,token_type_ids=segments, masked_lm_labels=mask,attention_mask=attention_mask_lm_nsp\
                                ,next_sentence_label=next_sentence_labels,output_nsp_scores=output_nsp_scores, output_lm_scores=output_lm_scores)
    elif not output_nsp_scores and output_lm_scores:
        lm_nsp_loss, lm_loss, nsp_loss, lm_scores = dialog_encoder(tokens,sep_indices=sep_indices, sep_len=sep_len\
                                ,token_type_ids=segments, masked_lm_labels=mask,attention_mask=attention_mask_lm_nsp\
                                ,next_sentence_label=next_sentence_labels,output_nsp_scores=output_nsp_scores, output_lm_scores=output_lm_scores)
    else:
        lm_nsp_loss, lm_loss, nsp_loss = dialog_encoder(tokens,sep_indices=sep_indices, sep_len=sep_len\
                    ,token_type_ids=segments, masked_lm_labels=mask,attention_mask=attention_mask_lm_nsp\
                    ,next_sentence_label=next_sentence_labels,output_nsp_scores=output_nsp_scores, output_lm_scores=output_lm_scores)

    if not evaluation:
        lm_loss = lm_loss.mean()
        nsp_loss = nsp_loss.mean()
        loss = (params['lm_loss_coeff'] * lm_loss) + (params['nsp_loss_coeff'] * nsp_loss)
        lm_nsp_loss = loss

    if output_nsp_scores and output_lm_scores:
        return loss, lm_loss, nsp_loss, nsp_scores, lm_scores
    elif output_nsp_scores and not output_lm_scores:
        return loss, lm_loss, nsp_loss, nsp_scores
    elif not output_nsp_scores and output_lm_scores:
        return loss, lm_loss, nsp_loss, lm_scores
    else:
        return loss, lm_loss, nsp_loss

def visdial_evaluate(dataloader, params, eval_batch_size):
    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()
    dialog_encoder.eval()
    batch_idx = 0
    with torch.no_grad():
        batch_size = 500 * (params['n_gpus']/8)
        batch_size = min([1, 2, 4, 5, 100, 1000, 200, 8, 10, 40, 50, 500, 20, 25, 250, 125], \
             key=lambda x: abs(x-batch_size) if x <= batch_size else float("inf"))
        if params['overfit']:
            batch_size = 100
        for epoch_id, _, batch in batch_iter(dataloader, params):
            if epoch_id == 1:
                break
            tokens = batch['tokens']
            num_rounds = tokens.shape[1]
            num_options = tokens.shape[2]
            tokens = tokens.view(-1, tokens.shape[-1])                       
            segments = batch['segments']
            segments = segments.view(-1, segments.shape[-1])
            sep_indices = batch['sep_indices']
            sep_indices = sep_indices.view(-1, sep_indices.shape[-1])
            mask = batch['mask']
            mask = mask.view(-1, mask.shape[-1])
            hist_len = batch['hist_len']
            hist_len = hist_len.view(-1)
            gt_option_inds = batch['gt_option_inds']
            gt_relevance = batch['gt_relevance']
            gt_relevance_round_id = batch['round_id'].squeeze(1)

            assert tokens.shape[0] == segments.shape[0] == sep_indices.shape[0] == mask.shape[0] == \
                hist_len.shape[0] == num_rounds * num_options * eval_batch_size
            output = []
            assert (eval_batch_size * num_rounds * num_options)//batch_size == (eval_batch_size * num_rounds * num_options)/batch_size
            for j in range((eval_batch_size * num_rounds * num_options)//batch_size):
                # create chunks of the original batch
                item = {}
                item['tokens'] = tokens[j*batch_size:(j+1)*batch_size,:]
                item['segments'] = segments[j*batch_size:(j+1)*batch_size,:]
                item['sep_indices'] = sep_indices[j*batch_size:(j+1)*batch_size,:]
                item['mask'] = mask[j*batch_size:(j+1)*batch_size,:]
                item['hist_len'] = hist_len[j*batch_size:(j+1)*batch_size]
                _, _, _, nsp_scores = forward(dialog_encoder, item, params ,output_nsp_scores=True, evaluation=True)
                # normalize nsp scores
                nsp_probs = F.softmax(nsp_scores, dim=1)
                output.append(nsp_probs[:,0])

            output = torch.cat(output,0).view(eval_batch_size, num_rounds, num_options)
            sparse_metrics.observe(output, gt_option_inds)
            output = output[torch.arange(output.size(0)), gt_relevance_round_id - 1, :]
            ndcg.observe(output, gt_relevance)
            batch_idx += 1

    dialog_encoder.train()
    all_metrics = {}
    all_metrics.update(sparse_metrics.retrieve(reset=True))
    all_metrics.update(ndcg.retrieve(reset=True))

    return all_metrics

if __name__ == '__main__':

    params = options.read_command_line()
    os.makedirs('checkpoints', exist_ok=True)
    if not os.path.exists(params['save_path']):
        os.mkdir(params['save_path'])

    viz = VisdomVisualize(
        enable=bool(params['enable_visdom']),
        env_name=params['visdom_env'],
        server=params['visdom_server'],
        port=params['visdom_server_port'])
    pprint.pprint(params)
    viz.addText(pprint.pformat(params, indent=4))

    dataset = VisdialDataset(params)

    dataset.split = 'train'
    dataloader = DataLoader(
        dataset,
        batch_size= params['batch_size']//params['sequences_per_image'] if (params['batch_size']//params['sequences_per_image'])  \
            else 1 if not params['overfit'] else 5,
        shuffle=False,
        num_workers=params['num_workers'],
        drop_last=True,
        pin_memory=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['device'] = device
    dialog_encoder = DialogEncoder()

    param_optimizer = list(dialog_encoder.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=params['lr'])
    scheduler = WarmupLinearScheduleNonZero(optimizer, warmup_steps=10000, t_total=200000)
    start_iter_id = 0

    if params['start_path']:
        pretrained_dict = torch.load(params['start_path'])
        if not params['continue']:
            model_dict = dialog_encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print("pretrained dict", pretrained_dict)
            assert len(pretrained_dict.keys()) > 0
            model_dict.update(pretrained_dict)
            dialog_encoder.load_state_dict(model_dict)
        else:
            model_dict = dialog_encoder.state_dict()
            optimizer_dict = optimizer.state_dict()
            pretrained_dict_model = pretrained_dict['model_state_dict']
            pretrained_dict_optimizer = pretrained_dict['optimizer_state_dict']
            pretrained_dict_scheduler = pretrained_dict['scheduler_state_dict']
            pretrained_dict_model = {k: v for k, v in pretrained_dict_model.items() if k in model_dict}
            pretrained_dict_optimizer = {k: v for k, v in pretrained_dict_optimizer.items() if k in optimizer_dict}
            model_dict.update(pretrained_dict_model)
            optimizer_dict.update(pretrained_dict_optimizer)
            dialog_encoder.load_state_dict(model_dict)
            optimizer.load_state_dict(optimizer_dict)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            scheduler = WarmupLinearScheduleNonZero(optimizer, warmup_steps=10000, \
                 t_total=200000, last_epoch=pretrained_dict["iterId"])
            scheduler.load_state_dict(pretrained_dict_scheduler)
            start_iter_id = pretrained_dict['iterId']
            
    num_iter_per_epoch = dataset.numDataPoints['train'] // (params['batch_size'] // params['sequences_per_image'] if (params['batch_size'] // params['sequences_per_image']) \
         else 1 if not params['overfit'] else 5 )
    print('\n%d iter per epoch.' % num_iter_per_epoch)

    dialog_encoder = nn.DataParallel(dialog_encoder)
    dialog_encoder.to(device)

    start_t = timer()
    optimizer.zero_grad()
    for epoch_id, idx, batch in batch_iter(dataloader, params):
        iter_id = start_iter_id + idx + (epoch_id * num_iter_per_epoch)
        dialog_encoder.train()
        if not params['overfit']:
            loss, lm_loss, nsp_loss = forward(dialog_encoder, batch, params, sample_size=params['batch_size'])
        else:
            sample_size = 64
            loss, lm_loss, nsp_loss = forward(dialog_encoder, batch, params, sample_size=sample_size)

        lm_nsp_loss = None
        if lm_loss is not None and nsp_loss is not None:
            lm_nsp_loss = lm_loss + nsp_loss
        loss /= params['batch_multiply']
        loss.backward()        
        scheduler.step()

        if iter_id % params['batch_multiply'] == 0 and iter_id > 0:
            optimizer.step()
            optimizer.zero_grad()
                        
        if iter_id % 10 == 0:
            end_t = timer()
            curEpoch = float(iter_id) / num_iter_per_epoch
            timeStamp = strftime('%a %d %b %y %X', gmtime())
            print_lm_loss = 0
            print_nsp_loss = 0
            print_inconsistency_loss = 0
            print_lm_nsp_loss = 0
            if lm_loss is not None:
                print_lm_loss = lm_loss.item()
            if nsp_loss is not None:
                print_nsp_loss = nsp_loss.item()
            if lm_nsp_loss is not None:
                print_lm_nsp_loss = lm_nsp_loss.item()

            printFormat = '[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][NSP + LM Loss: %.3g][LM Loss: %.3g][NSP Loss: %.3g]'
            printInfo = [
                timeStamp, curEpoch, iter_id, end_t - start_t, print_lm_nsp_loss, print_lm_loss, print_nsp_loss
            ]
            print(printFormat % tuple(printInfo))

            start_t = end_t
            # Update line plots
            viz.linePlot(iter_id, loss.item(), 'loss', 'tot loss')
            if lm_nsp_loss is not None:
                viz.linePlot(iter_id, lm_nsp_loss.item(), 'loss', 'lm + nsp loss')
            if lm_loss is not None:
                viz.linePlot(iter_id, lm_loss.item(),'loss', 'lm loss')
            if nsp_loss is not None:
                viz.linePlot(iter_id, nsp_loss.item(), 'loss', 'nsp loss')

        old_num_iter_per_epoch = num_iter_per_epoch
        if params['overfit']:
            num_iter_per_epoch = 100
        if iter_id % num_iter_per_epoch == 0 and iter_id > 0:
            torch.save({'model_state_dict' : dialog_encoder.module.state_dict(),'scheduler_state_dict':scheduler.state_dict() \
                 ,'optimizer_state_dict': optimizer.state_dict(), 'iter_id':iter_id}, os.path.join(params['savePath'], 'visdial_dialog_encoder_%d.ckpt'%iter_id))

        if iter_id % num_iter_per_epoch == 0:
            viz.save()
        # fire evaluation
        print("num iteration for eval", num_iter_per_epoch * (8 // params['sequences_per_image']))
        if  ((iter_id % (num_iter_per_epoch * (8 // params['sequences_per_image']))) == 0):
            eval_batch_size = 2
            if params['overfit']:
                eval_batch_size = 5
            
            dataset.split = 'val'
            # each image will need 1000 forward passes, (100 at each round x 10 rounds).
            dataloader = DataLoader(
                dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=params['num_workers'],
                drop_last=True,
                pin_memory=False)
            all_metrics = visdial_evaluate(dataloader, params, eval_batch_size)
            for metric_name, metric_value in all_metrics.items():
                print(f"{metric_name}: {metric_value}")
                if 'round' in metric_name:
                    viz.linePlot(iter_id, metric_value, 'Retrieval Round Val Metrics Round -' + metric_name.split('_')[-1], metric_name)
                else:
                    viz.linePlot(iter_id, metric_value, 'Retrieval Val Metrics', metric_name)
            
            dataset.split = 'train'

        num_iter_per_epoch = old_num_iter_per_epoch        