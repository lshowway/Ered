import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertIntermediate,
    BertOutput,
    BertSelfOutput,
)
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

logger = logging.getLogger(__name__)




class LukeConfig(BertConfig):
    def __init__(self, vocab_size: int, entity_vocab_size: int, bert_model_name: str, entity_emb_size: int = None, **kwargs):
        super(LukeConfig, self).__init__(vocab_size, **kwargs)

        self.entity_vocab_size = entity_vocab_size
        self.bert_model_name = bert_model_name
        if entity_emb_size is None:
            self.entity_emb_size = self.hidden_size
        else:
            self.entity_emb_size = entity_emb_size



class IdentifierEntityEmbeddings(nn.Module):
    def __init__(self, config: LukeConfig):
        super(IdentifierEntityEmbeddings, self).__init__()
        self.config = config

        self.entity_identifier_embeddings = nn.Embedding(config.entity_identifier_size, config.entity_emb_size, padding_idx=0)  # 2*256
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)  # 514*768
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)  # 1*768

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self, entity_ids: torch.LongTensor, position_ids: torch.LongTensor, token_type_ids: torch.LongTensor = None
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(entity_ids)

        entity_identifier_embeddings = self.entity_identifier_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_identifier_embeddings = self.entity_embedding_dense(entity_identifier_embeddings)

        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = torch.sum(position_embeddings, dim=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_identifier_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings



class KnowledgeEntityEmbeddings(nn.Module):
    def __init__(self, config: LukeConfig):
        super(KnowledgeEntityEmbeddings, self).__init__()
        self.config = config

        self.k_entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)  # 2*256
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)  # 514*768
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)  # 1*768

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, entity_ids, position_ids=None, token_type_ids=None):
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros_like(entity_ids)

        k_entity_embeddings = self.k_entity_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.hidden_size:
            k_entity_embeddings = self.entity_embedding_dense(k_entity_embeddings)

        # position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        # position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        # position_embeddings = position_embeddings * position_embedding_mask
        # position_embeddings = torch.sum(position_embeddings, dim=-2)
        # position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=1e-7)

        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = k_entity_embeddings #+ position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings



class EntityAwareSelfAttention(nn.Module):
    def __init__(self, config):
        super(EntityAwareSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)  # Q1
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)  # Q2
        self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)  # Q3
        self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)  # Q4

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_size = word_hidden_states.size(1)

        w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))  # 2， 12， 39， 64
        w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
        e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))  # 2, 12, 2, 64
        e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

        key_layer = self.transpose_for_scores(self.key(torch.cat([word_hidden_states, entity_hidden_states], dim=1)))

        w2w_key_layer = key_layer[:, :, :word_size, :]  # key的不同部分使用不同的query
        e2w_key_layer = key_layer[:, :, :word_size, :]  # 2, 12, 39, 64
        w2e_key_layer = key_layer[:, :, word_size:, :]
        e2e_key_layer = key_layer[:, :, word_size:, :]  # 2, 12, 2, 64

        w2w_attention_scores = torch.matmul(w2w_query_layer, w2w_key_layer.transpose(-1, -2))  # 2, 12, 39， 39
        w2e_attention_scores = torch.matmul(w2e_query_layer, w2e_key_layer.transpose(-1, -2))  # 2, 12, 39, 2
        e2w_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2))  # 2, 12, 2, 39
        e2e_attention_scores = torch.matmul(e2e_query_layer, e2e_key_layer.transpose(-1, -2))  # 2, 12, 2, 2

        word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=3)  # 2, 12, 39, 41
        entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=3)  # 2, 12, 2, 41
        attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2)  # 2, 12, 41, 41

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)  # 2, 12, 41, 41

        value_layer = self.transpose_for_scores(
            self.value(torch.cat([word_hidden_states, entity_hidden_states], dim=1))
        )  # 2, 12, 41, 64
        context_layer = torch.matmul(attention_probs, value_layer)  # 2, 12, 41, 64

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # 2, 41, 768

        return context_layer[:, :word_size, :], context_layer[:, word_size:, :]  # 2, 39, 768   2, 2, 768



class EntityAwareAttention(nn.Module):
    def __init__(self, config):
        super(EntityAwareAttention, self).__init__()
        self.self = EntityAwareSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_self_output, entity_self_output = self.self(word_hidden_states, entity_hidden_states, attention_mask)
        hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)
        self_output = torch.cat([word_self_output, entity_self_output], dim=1)
        output = self.output(self_output, hidden_states)  # y是原封不动2, 62, 768
        return output[:, : word_hidden_states.size(1), :], output[:, word_hidden_states.size(1):, :]



class EntityAwareLayer(nn.Module):
    def __init__(self, config):
        super(EntityAwareLayer, self).__init__()

        self.attention = EntityAwareAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_attention_output, entity_attention_output = self.attention(
            word_hidden_states, entity_hidden_states, attention_mask
        )
        attention_output = torch.cat([word_attention_output, entity_attention_output], dim=1)  # 2, 62, 768
        intermediate_output = self.intermediate(attention_output)  # 2, 62, 3072
        layer_output = self.output(intermediate_output, attention_output)  # 2, 62, 768

        return layer_output[:, : word_hidden_states.size(1), :], layer_output[:, word_hidden_states.size(1):, :]



class EntityAwareEncoder(nn.Module):
    def __init__(self, config):
        super(EntityAwareEncoder, self).__init__()
        self.layer = nn.ModuleList([EntityAwareLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, word_hidden_states, k_ent_states=None, entity_hidden_states=None, attention_mask=None):
        if k_ent_states is not None:
            word_hidden_states = torch.cat([word_hidden_states, k_ent_states], dim=1)

        for layer_module in self.layer:
            word_hidden_states, entity_hidden_states = layer_module(
                word_hidden_states, entity_hidden_states, attention_mask
            )
        return word_hidden_states, entity_hidden_states



class LukeEntityAwareAttentionModel(nn.Module):
    def __init__(self, config):
        super(LukeEntityAwareAttentionModel, self).__init__()
        self.config = config

        if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
            self.embeddings = RobertaEmbeddings(config) # 50266*1024
            self.embeddings.token_type_embeddings.requires_grad = False  # why?
        else:
            self.embeddings = BertEmbeddings(config)
        self.entity_identifier_embeddings = IdentifierEntityEmbeddings(config) # 2*256
        self.entity_embeddings = KnowledgeEntityEmbeddings(config) # 50w*256

        self.encoder = EntityAwareEncoder(config)

    def _compute_extended_attention_mask(self, word_attention_mask: torch.LongTensor,
                                         k_ent_mask: torch.LongTensor,
                                         entity_attention_mask: torch.LongTensor):
        attention_mask = word_attention_mask
        if k_ent_mask is not None:
            attention_mask = torch.cat([attention_mask, k_ent_mask], dim=1)
        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(self,
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
            k_ent_ids,
            k_ent_mask,
    ):
        word_embeddings = self.embeddings(word_ids, word_segment_ids)
        entity_identifier_embeddings = self.entity_identifier_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        entity_embeddings = self.entity_embeddings(k_ent_ids, None, None)

        attention_mask = self._compute_extended_attention_mask(word_attention_mask, k_ent_mask, entity_attention_mask)  # 后两维度拼在一起

        output = self.encoder(word_embeddings, entity_embeddings, entity_identifier_embeddings, attention_mask)
        return output



class LukeForEntityTyping(nn.Module):
    def __init__(self, args, num_labels):
        super(LukeForEntityTyping, self).__init__()

        self.config = args.model_config
        self.num_labels = num_labels
        self.luke = LukeEntityAwareAttentionModel(args.model_config)
        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)
        self.typing = nn.Linear(args.model_config.hidden_size, num_labels)

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            if module.embedding_dim == 1:  # embedding for bias parameters
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        word_ids,
        word_segment_ids=None,
        word_attention_mask=None,
        entity_ids=None,
        entity_position_ids=None,
        entity_segment_ids=None,
        entity_attention_mask=None,
        k_entity_ids=None,
        k_entity_mask=None,
        labels=None,
    ):
        encoder_outputs = self.luke(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
            k_entity_ids,
            k_entity_mask,
        )  # 第一个是input的：2, 43, 768   第二个是entity identifier的： 2, 2, 768

        feature_vector = encoder_outputs[1][:, 0, :]  # 2*768
        feature_vector = self.dropout(feature_vector)
        logits = self.typing(feature_vector)
        if labels is None:
            return logits
        return (F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits)),)


