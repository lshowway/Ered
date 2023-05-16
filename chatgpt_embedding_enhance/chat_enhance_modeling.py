import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN, gelu

from transformers.modeling_utils import (
    PreTrainedModel, apply_chunking_to_forward
)
from transformers.utils import logging
# from transformers.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaEmbeddings, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaAttention, RobertaIntermediate, RobertaOutput

logger = logging.get_logger(__name__)


class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

        self.add_chat = config.add_chat
        if config.add_chat:
            self.map = nn.Linear(1536, config.hidden_size, bias=False)

    def forward(
            self, des_embedding,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        if self.add_chat:
            des_embed = self.map(des_embedding)
            des_embed = des_embed.unsqueeze(1)
            layer_output = layer_output + des_embed
            outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        backbone_knowledge_dict = config.backbone_knowledge_dict
        module_list = []
        for i in range(config.num_hidden_layers):
            if i in backbone_knowledge_dict:
                config.add_chat = True
            else:
                config.add_chat = False
            module_list.append(RobertaLayer(config))
        self.layer = nn.ModuleList(module_list)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            des_embedding=None,
    ):
        for i, layer_module in enumerate(self.layer):
            # if i in backbone_knowledge_dict:
            #     hidden_states = hidden_states + des_embed
            layer_outputs = layer_module(
                des_embedding,
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]

        return layer_outputs


class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaPreTrainedModel(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class RobertaModel(RobertaPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.init_weights()

    def _compute_extended_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(
            self, input_ids=None, attention_mask=None, token_type_ids=None,
            start_id=None,
            des_embedding=None
    ):
        input_shape = input_ids.size()
        device = input_ids.device

        word_embeddings = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        attention_mask = self._compute_extended_attention_mask(attention_mask)
        encoder_outputs = self.encoder(word_embeddings, attention_mask, des_embedding=des_embedding)  # bld
        sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if self.pooler is None:
            return (sequence_output,)
        else:
            pooled_output = self.pooler(sequence_output)
            return (pooled_output,) + encoder_outputs[1:]


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# --------------
class SequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.loss = nn.CrossEntropyLoss(reduction='sum')

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, start_id=None,
                des_embedding=None,
                labels=None
                ):
        outputs = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            # start_id=start_id,
            des_embedding=des_embedding
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        if labels is None:
            return logits
        else:
            main_loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))

            return logits, main_loss


class EntityTyping(RobertaPreTrainedModel):
    def __init__(self, config):
        super(EntityTyping, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.hidden_size, self.num_labels, False)
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_id=None,
                des_embedding=None,
                labels=None):
        outputs = self.roberta(input_ids, attention_mask, token_type_ids, des_embedding=des_embedding)
        sequence_output = self.dropout(outputs[0])
        start_id = start_id.unsqueeze(1)  # batch 1 L
        entity_vec = torch.bmm(start_id, sequence_output).squeeze(1)
        logits = self.typing(entity_vec)

        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return logits, loss
        else:
            return logits


class RelationClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RelationClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relation_classifier = nn.Linear(config.hidden_size * 2, self.num_labels, False)
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, start_id=None,
                des_embedding=None,
                labels=None):

        outputs = self.roberta(input_ids, attention_mask, token_type_ids, des_embedding=des_embedding)
        sequence_output = self.dropout(outputs[0])

        if len(start_id.shape) == 3:
            sub_start_id, obj_start_id = start_id.split(1, dim=1) # split to 2, each is 1
            sub_start_id = sub_start_id
            subj_output = torch.bmm(sub_start_id, sequence_output)

            obj_start_id = obj_start_id
            obj_output = torch.bmm(obj_start_id, sequence_output)
            entity_vec = torch.cat([subj_output.squeeze(1), obj_output.squeeze(1)], dim=1)
            logits = self.relation_classifier(entity_vec)

            if labels is not None:
                loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1).to(torch.long))
                return logits, loss
            else:
                return logits
        else:
            raise ValueError("the entity index is wrong")

