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
        config = BertConfig.from_json_file('/nethome/halamri3/visdial-bert/config/bert_base_baseline.json')
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
                mode=0
        ):



        if mode == 0: #here we will only pass the language input and use the lanugage losses

            outputs = self.bert_pretrained(
                input_ids=input_ids,
                sep_indices=sep_indices,
                sep_len=sep_len,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
                next_sentence_label=next_sentence_label,
                head_mask=head_mask,
                return_dict=return_dict
            )
            loss = None

            if next_sentence_label is not None:
                loss, lm_scores, nsp_scores, hidden_states = outputs
            else: #evaluation!
                lm_scores, nsp_scores, hidden_states = outputs

            loss_fct = CrossEntropyLoss(ignore_index=-1)

            lm_loss = None
            nsp_loss = None

            if next_sentence_label is not None:
                nsp_loss = loss_fct(nsp_scores, next_sentence_label)
            if labels is not None:
                lm_loss = loss_fct(lm_scores.view(-1,lm_scores.shape[-1]), labels.view(-1))

            out = (loss,lm_loss, nsp_loss)
            if output_nsp_scores:
                out  = out + (nsp_scores,)
            if output_lm_scores:
                out = out + (lm_scores,)
            return out

        elif mode == 1:

            transformer_outputs = self.bert_pretrained.bert(
                inputs_embeds=inputs_embeds,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=return_dict
            )

            lm_scores, nsp_scores = self.bert_pretrained.cls(sequence_output= transformer_outputs['last_hidden_state'],
                                                             pooled_output=transformer_outputs ['pooler_output'])
            lm_scores.detach()
            nsp_scores.detach()

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss =None

            if next_sentence_label is not None:
                nsp_loss = loss_fct(nsp_scores, next_sentence_label)
            if labels is not None:
                lm_loss = loss_fct(lm_scores.view(-1,lm_scores.size(2)), labels.view(-1))

            out = (loss, lm_loss, nsp_loss)
            if output_nsp_scores:
                out = out + (nsp_scores,)
            if output_lm_scores:
                out = out + (lm_scores,)

            return out

        else:

            transformer_outputs = self.bert_pretrained.bert(
                inputs_embeds=inputs_embeds,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=return_dict
            )
            hidden_states= transformer_outputs['last_hidden_state']
            lm_logits = self.bert_pretrained.cls(hidden_states)

            outputs = (lm_logits,) + transformer_outputs['pooler_output'] + transformer_outputs['hidden_states']

            lm_video_regs = self.bert_pretrained.video_inverse_ff(hidden_states[:, :labels[1].size(1), :])
            shifted_video_regs = lm_video_regs[..., :-1, :].contiguous()
            shifted_video_labels = labels[1][..., :-1, :].contiguous()

            loss_video_fct = MSELoss(reduce=True, reduction='mean')
            loss_video = loss_video_fct(shifted_video_regs, shifted_video_labels)
            lm_video_regs = self.bert_pretrained.video_inverse_ff(hidden_states[:,:labels.size(1),:])
            loss = loss_video

            outputs = (loss,) + outputs
            return outputs
