import math
import os.path

import torch.nn as nn
import torch
import torch.nn.functional as F

from parameters import parse_args

args = parse_args()

# if args.backbone_model_type == 'roberta':
#     from transformers.models.roberta.modeling_roberta import \
#         (
#         RobertaEmbeddings as BackboneEmbeddings,
#         RobertaLayer as BackboneLayer,
#         RobertaPreTrainedModel as BackbonePreTrainedModel,
#         RobertaPooler as BackbonePooler,
#     )  # 这个是Roberta
if args.backbone_model_type == 'luke':
    from luke_modeling import \
        (
        RobertaEmbeddings as BackboneEmbeddings,
        EntityAwareLayer as BackboneLayer,
        LukeEntityAwareAttentionModel as BackbonePreTrainedModel,
        # BertPooler as BackbonePooler,
        KnowledgeEntityEmbeddings,
        EntityEmbeddings,
        LukeModel
    )  # Bert
else:
    pass
from transformers.models.roberta.modeling_roberta import RobertaPooler as BackbonePooler
if args.knowledge_model_type == 'distilbert':
    from transformers.models.distilbert.modeling_distilbert import \
        (
        Embeddings as KEmbeddings,
        TransformerBlock as KnowledgeLayer,
    )  # 这个是k module




class GNN(nn.Module):
    def __init__(self, config, config_k=None):
        super(GNN, self).__init__()
        self.config_k = config_k
        if config_k is not None:
            self.projection = nn.Linear(config_k.hidden_size, config.hidden_size)

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

            self.map = nn.Linear(100, config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, k_hidden_states=None, entity_embed=None):
        if self.config_k is not None:
            # 处理original input text的表征
            # 第一种： 使用CLS的表征
            # center_states = hidden_states[:, :1, :]  # batch, 1, d=1024
            # 第二种：全部token的表征都使用
            center_states = hidden_states  # batch, L1, d1
            L1 = hidden_states.size(1)

            # 处理N条description的表征
            # 第一种使用CLS的表征
            # knowledge_states = k_hidden_states  # # batch, N, L2, d2=768
            batch, neighbour_num, description_len, d2 = k_hidden_states.size()
            knowledge_states = k_hidden_states[:, :, 0, :]  # batch, N, d2=768
            knowledge_states = self.projection(knowledge_states)  # batch, N, d1=1024
            knowledge_states = self.dropout(self.LayerNorm(knowledge_states))

            # 将original input text的表征和description的表征合起来
            entity_embed  = self.map(entity_embed)
            entity_embed = self.dropout(self.LayerNorm(entity_embed))
            center_knowledge_states = torch.cat([center_states, entity_embed + knowledge_states], dim=1)  # batch, L1+N+1, d1
            # center_knowledge_states = self.dropout(self.LayerNorm(center_knowledge_states))

            # 这个attention_mask是center和neighbour之间是否可见，也可以不加，默认就是互相可见
            # attention_mask = torch.ones(batch, L).unsqueeze(1).unsqueeze(1).to(hidden_states.device)  # batch, 1, 1, L1+K

            # query = self.query(center_knowledge_states[:, :1])  # batch, 1, d
            query = self.query(center_knowledge_states)  # batch, L1+N, d1
            key = self.key(center_knowledge_states)  # batch, L1+N, d1
            value = self.value(center_knowledge_states)  # batch, L1+N, d1

            query = self.transpose_for_scores(query)  # batch, d3, L1+K, d4
            key = self.transpose_for_scores(key)   # batch, d3, L1+K, d4
            value = self.transpose_for_scores(value)  # batch, d3, L1+K, d4

            attention_scores = torch.matmul(query, key.transpose(-1, -2))  # batch, d3, 1, L1+N
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            # if attention_mask is not None:
            #     attention_scores = attention_scores + attention_mask  # batch, d3, 1, L1+N
            attention_probs = nn.Softmax(dim=-1)(attention_scores)  # batch, d3, 1, L1+N

            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value)  # batch, d3, L1+N, d4

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)  # batch, L1+N, d1(d3*d4)

            # 使用受description影响后的original text的表征,为啥是-1
            # hidden_states[:, -1, :] = context_layer[:, 0, :]  # 替换掉 batch L1, d1
            hidden_states[:, 0, :] = context_layer[:, 0, :]  # 替换掉 batch L1, d1
            # hidden_states[:, :L1, :] = context_layer[:, :L1, :]

            return hidden_states  # batch, d
        else:
            return hidden_states



class KFormersLayer(nn.Module):
    def __init__(self, config, config_k=None):
        super(KFormersLayer, self).__init__()
        if config_k is not None:
            self.backbone_layer = BackboneLayer(config)  # 处理qk pair的backbone module，分类
            self.k_layer = KnowledgeLayer(config_k)  # 处理description的knowledge module，表示
            self.map = nn.Linear(config_k.hidden_size, config.hidden_size, bias=False)
        else:
            self.backbone_layer = BackboneLayer(config)  # 处理qk pair的backbone module，分类
            self.k_layer = None

    def forward(self, word_hidden_states, word_attention_mask,
                entity_hidden_states, entity_attention_mask,
                k_entity_hidden_states=None,
                k_des_hidden_states=None,  k_des_attention_mask=None, k_des_mask=None
                ):

        word_size = word_hidden_states.size(1)
        ent_size = k_entity_hidden_states.size(1)
        # des_size = k_des_hidden_states.size(1)
        # 谁在前，谁在后
        k_layer_outputs = None
        word_hidden_states = torch.cat([word_hidden_states, k_entity_hidden_states], dim=1)
        batch_size, ent_num, _ = k_entity_hidden_states.size()
        ent_attention_mask = torch.ones(batch_size, ent_num, device=word_hidden_states.device)
        word_attention_mask = torch.cat([word_attention_mask, ent_attention_mask], dim=-1)

        if self.k_layer is not None:
            batch_size, des_num, length, d = k_des_hidden_states.size()
            k_layer_outputs = self.k_layer(k_des_hidden_states.view(batch_size, des_num * length, d),
                                           k_des_attention_mask.view(batch_size, des_num * length))
            k_layer_outputs = k_layer_outputs[-1].reshape(batch_size, des_num, length, d)
            k_des_attention_output = k_layer_outputs[:, :, 0, :]  # batch K d2
            k_des_attention_output = self.map(k_des_attention_output)  # batch K d description representation

            # description_state, entity_state, original_input_state整合到一起
            word_hidden_states = torch.cat([word_hidden_states, k_des_attention_output], dim=1)  # batch L1+K1+K2 d
            word_attention_mask = torch.cat([word_attention_mask, k_des_mask], dim=-1) # batch 1 1 L1+K1+K2

        word_attention_output, entity_attention_output = self.backbone_layer(
            word_hidden_states, word_attention_mask, entity_hidden_states, entity_attention_mask
        ) # batch L1 d, batch 2 d

        return word_attention_output[:, :word_size, :], \
               word_attention_output[:, word_size: word_size+ent_size, :], \
               k_layer_outputs, \
               entity_attention_output




class KFormersEncoder(nn.Module):
    def __init__(self, config, config_k, backbone_knowledge_dict):
        super(KFormersEncoder, self).__init__()
        self.num_hidden_layers = config.num_hidden_layers

        module_list = []
        for i in range(config.num_hidden_layers):
            if i in backbone_knowledge_dict:
                module_list.append(KFormersLayer(config=config, config_k=config_k))
            else:
                module_list.append(KFormersLayer(config=config, config_k=None))
        self.layer = nn.ModuleList(module_list)

    def forward(self, word_hidden_states, attention_mask,
                entity_hidden_states, entity_attention_mask,
                k_entity_hidden_states=None,
                k_des_hidden_states=None, k_des_attention_mask=None, k_des_mask=None,):

        # k_ent_output, k_des_output = None, None
        last_des_hidden_states = k_des_hidden_states
        for i, layer_module in enumerate(self.layer):
            # print(i)
            layer_outputs = layer_module(word_hidden_states, attention_mask,
                                         entity_hidden_states, entity_attention_mask,
                                         k_entity_hidden_states=k_entity_hidden_states,
                                         k_des_hidden_states=k_des_hidden_states, k_des_attention_mask=k_des_attention_mask,
                                         k_des_mask=k_des_mask)
            word_hidden_states = layer_outputs[0]
            k_entity_hidden_states = layer_outputs[1]
            k_des_hidden_states = layer_outputs[2] if layer_outputs[2] is not None else last_des_hidden_states
            if layer_outputs[2] is not None:
                last_des_hidden_states = layer_outputs[2]
            entity_hidden_states = layer_outputs[-1]

        outputs = (word_hidden_states, k_entity_hidden_states, k_des_hidden_states, entity_hidden_states)
        return outputs



# class KFormersModel(nn.Module):
class KFormersModel(LukeModel):
    def __init__(self, config, config_k, backbone_knowledge_dict):
        # super(KFormersModel, self).__init__()
        super(KFormersModel, self).__init__(config)
        self.config = config
        self.config_k = config_k
        self.embeddings = BackboneEmbeddings(config)  # roberta
        self.embeddings.token_type_embeddings.requires_grad = False  # why?

        self.entity_embeddings = EntityEmbeddings(config)
        self.k_ent_embeddings = KnowledgeEntityEmbeddings(config)  # 50w*256
        self.k_des_embeddings = KEmbeddings(config_k) # distilbert-description

        self.encoder = KFormersEncoder(config, config_k, backbone_knowledge_dict)

        # self.pooler = BackbonePooler(config)
        self.pooler = None
    #
    def _compute_extended_attention_mask(self, word_attention_mask, entity_attention_mask, k_ent_mask=None,):
        attention_mask = word_attention_mask

        attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=1)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(self,input_ids=None, attention_mask=None, token_type_ids=None, start_id=None,
                entity_ids=None, entity_attention_mask=None, entity_segment_ids=None, entity_position_ids=None,
                k_ent_ids=None, k_label=None,
                k_des_ids=None, k_des_mask_one=None,  k_des_seg=None, k_des_mask=None):

        word_embeddings = self.embeddings(input_ids, token_type_ids)
        entity_embeddings = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        k_entity_embeddings = self.k_ent_embeddings(k_ent_ids)

        batch, des_num, length = k_des_ids.size()
        k_des_embeddings = self.k_des_embeddings(k_des_ids.view(-1, length)).reshape(batch, des_num, length, -1)


        # attention_mask = self._compute_extended_attention_mask(attention_mask,
        #                                                        entity_attention_mask,
        #                                                        )  # 后两维度拼在一起
        encoder_outputs = self.encoder(word_embeddings, attention_mask,
                                       entity_embeddings, entity_attention_mask,
                                       k_entity_hidden_states=k_entity_embeddings,
                                       k_des_hidden_states=k_des_embeddings, k_des_attention_mask=k_des_mask_one, k_des_mask=k_des_mask,
                                       )

        original_text_output, k_ent_output, k_des_output, entity_output = encoder_outputs
        if self.pooler is None:
            return (original_text_output, None, entity_output, k_entity_embeddings, k_ent_output, k_des_output)
        else:
            pooled_output = self.pooler(original_text_output)
            return (original_text_output, pooled_output, entity_output, k_entity_embeddings, k_ent_output, k_des_output)



# --------------------------------------------------------------------------
class RobertaOutputLayerEntityTyping(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.in_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features  # of entity
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class RobertaOutputLayerSequenceClassification(nn.Module):
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



class RobertaOutputLayerRelationClassification(nn.Module):
    """Head for two-entity classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features  # of entity
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



# ---------------------------------------------------------------------
class KFormersForEntityTyping(nn.Module):
    def __init__(self, args, config_k, backbone_knowledge_dict):
        super(KFormersForEntityTyping, self).__init__()
        self.num_labels = args.num_labels
        self.ent_num = args.max_ent_num
        self.alpha = args.alpha
        self.beta = args.beta


        args.add_pooling_layer = False
        config = args.model_config
        self.config = config

        self.kformers = KFormersModel(config, config_k, backbone_knowledge_dict)
        config.in_hidden_size = config.hidden_size


        # self.classifier = RobertaOutputLayerEntityTyping(config)
        # self.loss = nn.BCEWithLogitsLoss(reduction='mean')

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.hidden_size, self.num_labels)

        self.aug_pl_fc = nn.Linear(config.hidden_size, 1)

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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, start_id=None,
                entity_ids=None, entity_attention_mask=None, entity_segment_ids=None, entity_position_ids=None,
                k_ent_ids=None, k_label=None,
                k_des_ids=None, k_des_mask_one=None,  k_des_seg=None, k_des_mask=None, labels=None):


        outputs = self.kformers(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_id=start_id,
                entity_ids=entity_ids, entity_attention_mask=entity_attention_mask, entity_segment_ids=entity_segment_ids, entity_position_ids=entity_position_ids,
                k_ent_ids=k_ent_ids, k_label=k_label,
                k_des_ids=k_des_ids, k_des_mask_one=k_des_mask_one,  k_des_seg=k_des_seg, k_des_mask=k_des_mask, )

        original_text_output, pooled_output, entity_output, k_entity_embeddings, k_ent_output, k_des_output = outputs  # batch L d


        # 主loss
        feature_vector = self.dropout(entity_output[:, 0, :])
        logits = self.typing(feature_vector)

        # 第一个辅助loss：ent增强&des增强
        start_id = start_id.unsqueeze(1)  # batch 1 L
        anchor_vec = torch.bmm(start_id, original_text_output).squeeze(1)  # batch d
        true_ent_vec = k_ent_output[range(k_ent_output.size(0)), k_label]
        true_des_vec = torch.mean(k_ent_output, dim=1, keepdim=False)  # bach d
        aux_logits = self.typing(anchor_vec + true_ent_vec + true_des_vec)  # ENT表征+true_e

        if labels is None:
            return logits + aux_logits #+ aug_pl_logits
        else:
            # aux_logits = self.classifier(feature_vector + k_ent_output).squeeze()  # batch K d增强或者污染
            # aux_loss = torch.nn.CrossEntropyLoss()(aux_logits.view(input_ids.size(0), -1), k_label.view(-1))
            main_loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits),
                                                           reduction='mean', pos_weight=None)

            aux_loss = F.binary_cross_entropy_with_logits(aux_logits.view(-1), labels.view(-1).type_as(aux_logits))

            # 第二个辅助loss：区分污染和增强
            aug_pl_logits = self.aug_pl_fc(k_ent_output + anchor_vec.unsqueeze(1))  # batch K d -> batch K/batch K C
            aug_pl_loss = nn.CrossEntropyLoss()(aug_pl_logits.view(input_ids.size(0), -1), k_label.view(-1))

            return logits + aux_logits, main_loss + self.alpha * aux_loss + self.beta * aug_pl_loss



        # enhanced_vec = entity_vec
        # logits_2 = self.typing(enhanced_vec)
        # logits_2 += logits
        # aux_loss = F.binary_cross_entropy_with_logits(logits_2.view(-1), labels.view(-1).type_as(logits_2))
        #
        # if labels is None:
        #     return logits
        # else:
        #     return logits_2, main_loss + 0.01 * aux_loss



