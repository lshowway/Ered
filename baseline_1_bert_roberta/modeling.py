from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from torch import nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import \
    RobertaModel, RobertaPreTrainedModel, RobertaForSequenceClassification

logger = logging.getLogger(__name__)




class RoBERTaForEntityTyping(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RoBERTaForEntityTyping, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.classifier = RobertaClassification(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.hidden_size, self.num_labels, False)
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_ids=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(outputs[0])
        start_ids = start_ids.unsqueeze(1)  # batch 1 L
        entity_vec = torch.bmm(start_ids, sequence_output).squeeze(1)
        logits = self.typing(entity_vec)

        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss, logits
        else:
            return logits



class RoBERTaForRelationClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RoBERTaForRelationClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relation_classifier = nn.Linear(config.hidden_size * 2, self.num_labels, False)
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, start_ids=None, labels=None):

        outputs = self.roberta(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(outputs[0])

        if len(start_ids.shape) == 3:
            sub_start_id, obj_start_id = start_ids.split(1, dim=1) # split to 2, each is 1
            sub_start_id = sub_start_id
            subj_output = torch.bmm(sub_start_id, sequence_output)

            obj_start_id = obj_start_id
            obj_output = torch.bmm(obj_start_id, sequence_output)
            entity_vec = torch.cat([subj_output.squeeze(1), obj_output.squeeze(1)], dim=1)
            logits = self.relation_classifier(entity_vec)

            if labels is not None:
                loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1).to(torch.long))
                return loss, logits
            else:
                return logits
        else:
            raise ValueError("the entity index is wrong")









