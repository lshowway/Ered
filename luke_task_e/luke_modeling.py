import torch.nn as nn
import torch.nn.functional as F

import logging
import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertIntermediate,
    BertOutput,
    BertPooler,
    BertSelfOutput,
)
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaEncoder
logger = logging.getLogger(__name__)


from data_utils import ENTITY_TOKEN, MASK_TOKEN



class LukeConfig(BertConfig):
    def __init__(
            self, vocab_size: int, entity_vocab_size: int, bert_model_name: str, entity_emb_size: int = None, **kwargs
    ):
        super(LukeConfig, self).__init__(vocab_size, **kwargs)

        self.entity_vocab_size = entity_vocab_size
        self.bert_model_name = bert_model_name
        if entity_emb_size is None:
            self.entity_emb_size = self.hidden_size
        else:
            self.entity_emb_size = entity_emb_size



class EntityEmbeddings(nn.Module):
    def __init__(self, config: LukeConfig):
        super(EntityEmbeddings, self).__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)  # 2*256
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

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = torch.sum(position_embeddings, dim=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings



class KnowledgeEntityEmbeddings(nn.Module):
    def __init__(self, config: LukeConfig):
        super(KnowledgeEntityEmbeddings, self).__init__()
        self.config = config

        self.k_entity_embeddings = nn.Embedding(config.k_entity_vocab_size, config.entity_emb_size, padding_idx=0)  # 2*256

        for k, v in self.k_entity_embeddings.named_parameters():
            v.requires_grad = False

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



class LukeModel(nn.Module):
    def __init__(self, config: LukeConfig):
        super(LukeModel, self).__init__()

        self.config = config

        self.encoder = BertEncoder(config)  # 这个的作用是什么？
        # self.encoder = RobertaEncoder(config)  # 这个的作用是什么？
        self.pooler = BertPooler(config)

        if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
            self.embeddings = RobertaEmbeddings(config)
            self.embeddings.token_type_embeddings.requires_grad = False  # why?
        else:
            self.embeddings = BertEmbeddings(config)
        self.entity_embeddings = EntityEmbeddings(config)

        self.k_entity_embeddings = KnowledgeEntityEmbeddings(config)  # 50w*256


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

    def _compute_extended_attention_mask(self, word_attention_mask: torch.LongTensor,
                                         k_ent_mask: torch.LongTensor,
                                         entity_attention_mask: torch.LongTensor):
        attention_mask = word_attention_mask

        attention_mask = torch.cat([attention_mask, k_ent_mask, entity_attention_mask], dim=1)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask



class LukeEntityAwareAttentionModel(LukeModel):
    def __init__(self, config):
        super(LukeEntityAwareAttentionModel, self).__init__(config)
        self.config = config
        self.encoder = EntityAwareEncoder(config)

    def forward(
            self,
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
            k_entity_ids=None,
            k_entity_mask=None,
    ):
        word_embeddings = self.embeddings(word_ids, word_segment_ids)
        entity_embeddings = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        k_entity_embeddings = self.k_entity_embeddings(k_entity_ids, None, None)

        if k_entity_mask is None:
            k_entity_mask = torch.ones(k_entity_ids.size(), device=k_entity_ids.device)

        attention_mask = self._compute_extended_attention_mask(word_attention_mask,
                                                               k_entity_mask,
                                                               entity_attention_mask,
                                                               )  # 后两维度拼在一起

        output = self.encoder(word_embeddings, k_entity_embeddings, entity_embeddings, attention_mask)
        return output

    def load_state_dict(self, input, strict):
        (state_dict, args) = input
        new_state_dict = state_dict.copy()

        for num in range(self.config.num_hidden_layers):
            for attr_name in ("weight", "bias"):
                if f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}" not in state_dict:
                    new_state_dict[f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}"] = state_dict[
                        f"encoder.layer.{num}.attention.self.query.{attr_name}"
                    ]
                if f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}" not in state_dict:
                    new_state_dict[f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}"] = state_dict[
                        f"encoder.layer.{num}.attention.self.query.{attr_name}"
                    ]
                if f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}" not in state_dict:
                    new_state_dict[f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}"] = state_dict[
                        f"encoder.layer.{num}.attention.self.query.{attr_name}"
                    ]

            # word embedding
            # args.model_config.vocab_size += 1
            word_emb = state_dict["embeddings.word_embeddings.weight"]  # 50265*768
            marker_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)  # 1*768
            new_state_dict["embeddings.word_embeddings.weight"] = torch.cat([word_emb, marker_emb])  # 后面拼一个marker_emb
            args.tokenizer.add_special_tokens(dict(additional_special_tokens=[ENTITY_TOKEN]))  # 用@的表征
            # entity embedding (knowledge)
            # args.model_config.k_entity_vocab_size = len(args.entity_vocab)
            entity_emb = state_dict["entity_embeddings.entity_embeddings.weight"]  # 50W*256

            new_state_dict['k_entity_embeddings.k_entity_embeddings.weight'] = entity_emb  # ???还要多拼一个吗？
            # new_state_dict['k_entity_embeddings.k_entity_embeddings.bias'] =  # ???还要多拼一个吗？
            new_state_dict['k_entity_embeddings.entity_embedding_dense.weight'] = state_dict['entity_embeddings.entity_embedding_dense.weight']
            new_state_dict['k_entity_embeddings.LayerNorm.weight'] = state_dict['entity_embeddings.LayerNorm.weight']
            new_state_dict['k_entity_embeddings.LayerNorm.bias'] = state_dict['entity_embeddings.LayerNorm.bias']

            # entity embedding (idenfifier)
            # args.entity_vocab = state_dict.entity_vocab
            mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0)  # 1*256
            # args.model_config.entity_vocab_size = 2
            new_state_dict["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb[:1], mask_emb])  # 2*256? pad+mask?

        super(LukeEntityAwareAttentionModel, self).load_state_dict(new_state_dict, strict=strict)



class LukeForEntityTyping(LukeEntityAwareAttentionModel):
    def __init__(self, args, num_labels):
        super(LukeForEntityTyping, self).__init__(args.model_config)

        self.num_labels = num_labels
        self.max_ent_num = args.max_ent_num
        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)

        self.typing = nn.Linear(args.model_config.hidden_size, num_labels)
        # self.typing_2 = nn.Linear(args.model_config.hidden_size, num_labels)


        self.apply(self.init_weights)

    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
            k_entity_ids=None,
            k_label=None,
        labels=None,
    ):
        encoder_outputs = super(LukeForEntityTyping, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
            k_entity_ids,
        )

        text_k_vector, entity_vector = encoder_outputs

        # main loss
        feature_vector = entity_vector[:, 0, :]  # 2*768
        feature_vector = self.dropout(feature_vector)
        logits = self.typing(feature_vector)
        if labels is None:
            return logits

        main_loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits))

        # aux loss
        text_vector = text_k_vector[:, :word_ids.size(1), :] # batch L d
        k_ent_vector = text_k_vector[:, -k_entity_ids.size(1):, :] # batch K d
        true_ent_vec = k_ent_vector[range(k_ent_vector.size(0)), k_label]
        logits_2 = self.typing(entity_vector[:, 0, :] + true_ent_vec) # ENT表征+true_e
        # logits_2 = self.typing_2(text_vector[:, 0, :] + true_ent_vec) # 文本表征+true_e

        loss_2 = F.binary_cross_entropy_with_logits(logits_2.view(-1), labels.view(-1).type_as(logits_2))

        # aux_logits = self.fc(text_vector[:, :1, :] + k_ent_vector)
        # aux_logits = self.fc(self.dropout(feature_vector.unsqueeze(1) + k_ent_vector))
        # aux_loss = self.cross_entropy_loss(aux_logits.view(-1, self.max_ent_num), k_label.view(-1))

        total_loss = (main_loss + loss_2, )

        return total_loss

