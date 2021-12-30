import math
import torch.nn as nn
import torch

from parameters import parse_args

args = parse_args()

if args.backbone_model_type == 'roberta':
    from transformers.models.roberta.modeling_roberta import \
        (
        RobertaEmbeddings as BackboneEmbeddings,
        RobertaLayer as BackboneLayer,
        RobertaPreTrainedModel as BackbonePreTrainedModel,
        RobertaPooler as BackbonePooler,
        RobertaClassificationHead
    )  # 这个是Roberta
elif args.backbone_model_type == 'bert':
    from transformers.models.bert.modeling_bert import \
        (
        BertEmbeddings as BackboneEmbeddings,
        BertLayer as BackboneLayer,
        BertPreTrainedModel as BackbonePreTrainedModel,
        BertPooler as BackbonePooler,
    )  # Bert
else:
    pass

from transformers.models.distilbert.modeling_distilbert import \
    (
    Embeddings as KEmbeddings,
    TransformerBlock as KnowledgeLayer,
    DistilBertPreTrainedModel
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

            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, k_hidden_states=None, attention_mask=None, rel_pos=None):
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

            # 将original input text的表征和description的表征合起来
            center_knowledge_states = torch.cat([center_states, knowledge_states], dim=1)  # batch, L1+N, d1

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

            # 使用受description影响后的original text的表征
            hidden_states[:, -1, :] = context_layer[:, 0, :]  # 替换掉 batch L1, d1
            # 被neighbour全面影响的hidden_States，感觉这个的效果应该最好 (不收敛)
            # hidden_states = context_layer[:, :L1, :].to(dtype=hidden_states.dtype)

            return hidden_states  # batch, d
        else:
            return hidden_states


class KFormersLayer(nn.Module):
    # gnn可以整合到一起这个模块，也可以把gnn拿出来，现在是拿出来，因为还没写
    def __init__(self, config, config_k=None):
        super(KFormersLayer, self).__init__()
        if config_k is not None:
            self.backbone_layer = BackboneLayer(config)  # 处理qk pair的backbone module，分类
            self.k_layer = KnowledgeLayer(config_k)  # 处理description的knowledge module，表示
            # 在这儿禁止参数更新，但是embedding还是更新的，所以不如在最外面设置参数。
            # for p in self.k_layer.parameters():
            #     p.requires_grad = False
        else:
            self.backbone_layer = BackboneLayer(config)  # 处理qk pair的backbone module，分类
            self.k_layer = None
        self.gnn = GNN(config, config_k)

    def forward(self, hidden_states, attention_mask,
                k_hidden_states_list=None, k_attention_mask_list=None):
        layer_outputs = self.backbone_layer(hidden_states=hidden_states, attention_mask=attention_mask)
        hidden_states = layer_outputs[0]  # batch L d
        if self.k_layer is not None and k_hidden_states_list is not None:  # 现在先测试baseline没问题，之后用上面一行
            k_layer_outputs = self.k_layer(x=k_hidden_states_list, attn_mask=k_attention_mask_list)
            k_layer_outputs = k_layer_outputs[0]
            batch, neighbour_num, description_len = k_attention_mask_list.size()
            k_layer_outputs = k_layer_outputs.reshape(batch, neighbour_num, description_len, -1)  # batch, N, L2, d2
            hidden_states = self.gnn(hidden_states, k_layer_outputs)  # 这里对batch 0 d进行处理，使用neighbour的cls位？
            return hidden_states, k_layer_outputs.reshape(batch*neighbour_num, description_len, -1)
        else:
            return hidden_states, k_hidden_states_list


class KFormersEncoder(nn.Module):
    # knowledge module, small model, two tower, representation model
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

    def forward(self, hidden_states, attention_mask=None,
                k_hidden_states_list=None, k_attention_mask_list=None):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states=hidden_states, attention_mask=attention_mask,
                                         k_hidden_states_list=k_hidden_states_list,
                                         k_attention_mask_list=k_attention_mask_list)
            hidden_states = layer_outputs[0]
            k_hidden_states_list = layer_outputs[1]

        outputs = (hidden_states, k_hidden_states_list)
        return outputs


class KFormersModel(nn.Module):
    def __init__(self, config, config_k, backbone_knowledge_dict):
        super(KFormersModel, self).__init__()
        self.config = config
        self.embeddings = BackboneEmbeddings(config)
        self.k_embeddings = KEmbeddings(config_k)
        self.encoder = KFormersEncoder(config, config_k, backbone_knowledge_dict)

        if not config.add_pooling_layer:
            self.pooler = BackbonePooler(config)
        else:
            self.pooler = None

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape))
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                k_input_ids_list=None, k_attention_mask_list=None, k_token_type_ids_list=None, k_position_ids=None):

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape=input_ids.size())
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids)

        if k_input_ids_list is not None:
            batch, neighbour_num, description_len = k_input_ids_list.size()
            k_embedding_output = self.k_embeddings(input_ids=k_input_ids_list.reshape(-1, description_len))  # distilBert没有position和segment
        else:
            k_embedding_output = None
        encoder_outputs = self.encoder(hidden_states=embedding_output, attention_mask=extended_attention_mask,
                                       k_hidden_states_list=k_embedding_output,
                                       k_attention_mask_list=k_attention_mask_list)  # batch L d
        original_text_output, description_output = encoder_outputs
        if self.pooler is None:
            return encoder_outputs  # original_text_output, description_output
        else:
            pooled_output = self.pooler(original_text_output)
            return (original_text_output, pooled_output, description_output)


class RobertaClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
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


class KFormersRobertaForOpenEntity(BackbonePreTrainedModel):  # 这个不能继承一个类吧？两个？
    def __init__(self, config, config_k, backbone_knowledge_dict):
        super(KFormersRobertaForOpenEntity, self).__init__(config)
        self.num_labels = 9
        config.num_labels = 9
        config.add_pooling_layer = False

        self.kformers = KFormersModel(config, config_k, backbone_knowledge_dict)
        self.classifier = RobertaClassification(config)

        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, \
                k_input_ids_list=None, k_mask=None, k_attention_mask_list=None,
                k_token_type_ids_list=None, k_position_ids=None, label=None, start_id=None):
        outputs = self.kformers(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, position_ids=position_ids,
                                k_input_ids_list=k_input_ids_list, k_attention_mask_list=k_attention_mask_list,
                                k_token_type_ids_list=k_token_type_ids_list, k_position_ids=k_position_ids)
        original_text_output = outputs[0]  # batch L d
        start_id = start_id.unsqueeze(1)  # batch 1 L
        entity_vec = torch.bmm(start_id, original_text_output).squeeze(1)  # batch d
        logits = self.classifier(entity_vec)
        if label is not None:
            loss = self.loss(logits.view(-1, self.num_labels), label.view(-1, self.num_labels))
            return logits, loss
        else:
            return logits


class KFormersRobertaForFiger(BackbonePreTrainedModel):  # 这个不能继承一个类吧？两个？
    def __init__(self, config, config_k, backbone_knowledge_dict):
        super(KFormersRobertaForFiger, self).__init__(config)
        self.num_labels = 113
        config.num_labels = 113
        config.add_pooling_layer = False

        self.kformers = KFormersModel(config, config_k, backbone_knowledge_dict)
        self.classifier = RobertaClassification(config)

        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, \
                k_input_ids_list=None, k_mask=None, k_attention_mask_list=None,
                k_token_type_ids_list=None, k_position_ids=None, label=None, start_id=None):
        outputs = self.kformers(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, position_ids=position_ids,
                                k_input_ids_list=k_input_ids_list, k_attention_mask_list=k_attention_mask_list,
                                k_token_type_ids_list=k_token_type_ids_list, k_position_ids=k_position_ids)
        original_text_output = outputs[0]  # batch L d
        start_id = start_id.unsqueeze(1)  # batch 1 L
        entity_vec = torch.bmm(start_id, original_text_output).squeeze(1)  # batch d
        logits = self.classifier(entity_vec)
        if label is not None:
            loss = self.loss(logits.view(-1, self.num_labels), label.view(-1, self.num_labels))
            return logits, loss
        else:
            return logits
