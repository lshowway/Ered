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
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

logger = logging.getLogger(__name__)





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

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        all_word_states, all_entity_states = (), ()
        for layer_module in self.layer:
            word_hidden_states, entity_hidden_states = layer_module(
                word_hidden_states, entity_hidden_states, attention_mask
            )
            all_word_states += (word_hidden_states, )
            all_entity_states += (entity_hidden_states, )
        # return word_hidden_states, entity_hidden_states
        return all_word_states, all_entity_states



class LukeEntityAwareAttentionModel(nn.Module):
    def __init__(self, config):
        super(LukeEntityAwareAttentionModel, self).__init__()

        self.encoder = BertEncoder(config) # 这个不能注释，会drop很多，why？？？？？参数量没增加，后面也重写了啊。

        self.config = config
        self.encoder = EntityAwareEncoder(config)

        self.pooler = BertPooler(config)

        self.embeddings = RobertaEmbeddings(config)
        self.embeddings.token_type_embeddings.requires_grad = False  # why?
        self.entity_embeddings = EntityEmbeddings(config)

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
    ):
        word_embeddings = self.embeddings(word_ids, word_segment_ids)
        entity_embeddings = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)  # 后两维度拼在一起

        return self.encoder(word_embeddings, entity_embeddings, attention_mask)

    def load_state_dict(self, state_dict, *args, **kwargs):
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

        kwargs["strict"] = False
        super(LukeEntityAwareAttentionModel, self).load_state_dict(new_state_dict, *args, **kwargs)


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

    def _compute_extended_attention_mask(
            self, word_attention_mask: torch.LongTensor, entity_attention_mask: torch.LongTensor
    ):
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask




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



class LukeForEntityTyping(LukeEntityAwareAttentionModel):
    def __init__(self, args, num_labels):
        super(LukeForEntityTyping, self).__init__(args.model_config)

        self.num_labels = num_labels
        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)
        self.typing = nn.Linear(args.model_config.hidden_size, num_labels)

        self.adapter = AdapterModel(args)

        self.task_dense = nn.Linear(args.model_config.hidden_size * 3, args.model_config.hidden_size)


        self.fac_adapter = load_pretrained_adapter(self.adapter, "/home/LAB/zhaoqh/phd4hh/K-Adapter-main/checkpoints/fac-adapter/pytorch_model.bin")
        self.lin_adapter = load_pretrained_adapter(self.adapter, "/home/LAB/zhaoqh/phd4hh/K-Adapter-main/checkpoints/lin-adapter/pytorch_model.bin")
        # self.fac_adapter = load_pretrained_adapter(self.adapter, "G:\D\MSRA\K-Adapter-main\checkpoints/fac-adapter/pytorch_model.bin")
        # self.lin_adapter = load_pretrained_adapter(self.adapter, "G:\D\MSRA\K-Adapter-main\checkpoints/fac-adapter/pytorch_model.bin")


    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
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
        )  # 第一个是input的：2, 43, 768   第二个是entity identifier的： 2, 2, 768
        # 为啥取第0个啊？就2个啊。因为这是一个entity typing问题，就是对entity分类

        t1 = encoder_outputs[1] # entity all
        combine_features = t1[-1][:, 0, :]

        t2 = self.fac_adapter(t1)
        t3 = self.lin_adapter(t1)

        task_features = self.task_dense(torch.cat([combine_features, t2[:, 0, :], t3[:, 0, :]], dim=1))

        feature_vector = self.dropout(task_features)
        logits = self.typing(feature_vector)
        if labels is None:
            return logits
        # We treat the task as
        # multi-label classification, and train the model using
        # binary cross-entropy loss averaged over all entity
        # types.
        return (F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits)),)



def load_pretrained_adapter(adapter, adapter_path):
    new_adapter = adapter
    model_dict = new_adapter.state_dict()

    adapter_meta_dict = torch.load(adapter_path, map_location='cpu')
    changed_adapter_meta = {}
    for key, v in adapter_meta_dict.items():
        if key in model_dict:
            # print(v.tolist())
            changed_adapter_meta[key] = v

    model_dict.update(changed_adapter_meta)
    new_adapter.load_state_dict(model_dict)
    return new_adapter



class AdapterModel(nn.Module):
    def __init__(self, args):
        super(AdapterModel, self).__init__()
        # self.config = pretrained_model_config
        self.args = args
        self.adapter_size = 768 # args.adapter_size

        class AdapterConfig:
            project_hidden_size: int = 1024 # self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = self.adapter_size  # 64
            adapter_initializer_range: float = 0.0002
            is_decoder: bool = False
            attention_probs_dropout_prob: float = 0.1
            hidden_dropout_prob: float = 0.1
            hidden_size: int = 768
            initializer_range: float = 0.02
            intermediate_size: int = 3072
            layer_norm_eps: float = 1e-05
            max_position_embeddings: int = 514
            num_attention_heads: int = 12
            num_hidden_layers: int = 2 # self.args.adapter_transformer_layers
            num_labels: int = 2
            output_attentions: bool = False
            output_hidden_states: bool = False
            torchscript: bool = False
            type_vocab_size: int = 1
            vocab_size: int = 50265
        self.adapter_config = AdapterConfig
        self.adapter_skip_layers = 0 # self.args.adapter_skip_layers
        self.adapter_list = [0,11,22] # args.adapter_list
        self.adapter_num = len(self.adapter_list)
        self.adapter = nn.ModuleList([Adapter(args, AdapterConfig) for _ in range(self.adapter_num)])

        for p in self.adapter.parameters():
            p.requires_grad = False

    def forward(self, hidden_states):
        t1 = hidden_states[0]
        hidden_states_last = torch.zeros(t1.size()).to(t1.device)

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapter):
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1
            if self.adapter_skip_layers >= 1:
                if adapter_hidden_states_count % self.adapter_skip_layers == 0:
                    hidden_states_last = hidden_states_last + adapter_hidden_states[
                        int(adapter_hidden_states_count / self.adapter_skip_layers)]


        return hidden_states_last  # (loss), logits, (hidden_states), (attentions)



class Adapter(nn.Module):
    def __init__(self, args, adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
        self.args = args
        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size,
        )

        import pytorch_transformers

        self.encoder = pytorch_transformers.modeling_bert.BertEncoder(self.adapter_config)
        self.up_project = nn.Linear(self.adapter_config.adapter_size, adapter_config.project_hidden_size)

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        batch, num, d = down_projected.size()
        # attention_mask = torch.ones(input_shape, device=down_projected.device, dtype=down_projected.dtype)
        extended_attention_mask = torch.ones(batch, 1, 1, num, device=down_projected.device, dtype=down_projected.dtype)
        # if attention_mask.dim() == 3:
        #     extended_attention_mask = attention_mask[:, None, :, :]
        # if attention_mask.dim() == 2:
        #     extended_attention_mask = attention_mask[:, None, None, :]
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # if encoder_attention_mask.dim() == 3:
        #     encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        # if encoder_attention_mask.dim() == 2:
        #     encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        head_mask = [None] * self.adapter_config.num_hidden_layers
        # print(hidden_states.dtype, down_projected.dtype, extended_attention_mask.dtype)
        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask)
        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-self.adapter_config.initializer_range, self.adapter_config.initializer_range)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-self.adapter_config.initializer_range, self.adapter_config.initializer_range)







