import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
#from pytorch_transformers.modeling_bert import BertPredictionHeadTransform
from transformers.modeling_bert import BertPredictionHeadTransform
from models.language_and_video_dialog import BertForPretrainingDialog
from transformers import BertConfig

class DialogEncoder(nn.Module):

    def __init__(self):
        super(DialogEncoder, self).__init__()
        config = BertConfig.from_json_file('config/bert_base_baseline.json')
        self.bert_pretrained = BertForPretrainingDialog.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bert_pretrained.train()
        # add additional layers for the inconsistency loss
        assert self.bert_pretrained.config.output_hidden_states == True
        
    def forward(self,
                input_ids=None,
                inputs_embeds=None,
                sep_indices=None,
                sep_len=None,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                next_sentence_label=None,
                head_mask=None,
                return_dict=None,
                output_nsp_scores=False,
                output_lm_scores=False,
                num_frames=0,
                mode=0
        ):
    
        transformer_outputs = self.bert_pretrained.bert(
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )

        hidden_states = transformer_outputs['last_hidden_state']
        lm_scores, nsp_scores = self.bert_pretrained.cls(sequence_output=transformer_outputs['last_hidden_state'],
                                                         pooled_output=transformer_outputs['pooler_output'])
        
        outputs = (lm_scores, nsp_scores) + transformer_outputs['hidden_states']
        if mode == 1:
          
            lm_scores.detach()
            nsp_scores.detach()
            loss = None
            
            if next_sentence_label is not None and labels is not None:
                
                loss_lm_fct = CrossEntropyLoss(ignore_index=-1)
                nsp_loss = loss_lm_fct(nsp_scores, next_sentence_label)
                lm_loss = loss_lm_fct(lm_scores.view(-1, lm_scores.size(2)), labels[0].view(-1))
                outputs = (lm_loss, nsp_loss)

                return outputs
            
            return outputs

        else:

            
            #regress_input_shifted = list(map(lambda last_frame: hidden_states[:, :int(last_frame) - 1, :], num_frames))
            lm_video_regs = self.bert_pretrained.video_inverse_ff(hidden_states[:, :labels[1].size(1), :])
            #shifted_labels = list(map(lambda last_frame: labels[1][:, int(last_frame) - 1, :], num_frames))
            #losses = list(map(lambda idx: loss_video_fct(lm_video_regs[idx, int(num_frames[idx]), :], shifted_labels[idx]), num_frames))
            
            lm_video_regs.detach()
           
            if labels is not None:
                
                loss_video_fct = MSELoss(reduce=True, reduction='mean')
                loss = None
                shifted_video_labels = labels[1][..., :-1, :].contiguous()
                shifted_video_regs = lm_video_regs[..., :-1, :].contiguous()
                shifted_video_labels = shifted_video_labels.expand(shifted_video_regs.size(0), -1, -1)
                video_loss = loss_video_fct(shifted_video_regs, shifted_video_labels)
            
                outputs = video_loss

            return outputs
            
