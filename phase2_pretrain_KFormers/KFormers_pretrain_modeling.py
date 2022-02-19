import torch
import torch.nn as nn
# from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN

from transformers.models.roberta.modeling_roberta import RobertaModel


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx



class EntityEmbeddings(nn.Module):
    def __init__(self, config):
        super(EntityEmbeddings, self).__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)  # 2*256
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, entity_ids):
        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        embeddings = self.LayerNorm(entity_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings



class EntityPredictionHead(nn.Module):
    def __init__(self, config):
        super(EntityPredictionHead, self).__init__()
        self.config = config

        self.dense = nn.Linear(config.hidden_size, config.entity_emb_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.entity_emb_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.entity_emb_size, config.entity_vocab_size, bias=True)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        hidden_states = self.decoder(hidden_states)

        return hidden_states



class KModulePretrainingModel(nn.Module):
    '''
    RoBERTa Model with a `XXX` head on top
    '''
    def __init__(self, config):
        super(KModulePretrainingModel, self).__init__()

        self.roberta = RobertaModel(config)
        self.entity_embeddings = EntityEmbeddings(config)

        self.entity_predictions = EntityPredictionHead(config)
        # 为什么要这么做？使用entity embedding初始化？？
        # self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight

        self.loss_fn = nn.BCEWithLogitsLoss()  # 对比损失

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
        description_ids,
        description_attention_mask,
        description_segment_ids,
        entity_ids,
        entity_labels=None,  # 对比学习
    ):
        # model_dtype = next(self.parameters()).dtype  # for fp16 compatibility

        description_representation = self.roberta(description_ids, description_segment_ids, description_attention_mask)
        entity_embedding = self.entity_embeddings(entity_id)

        rep = torch.cat([entity_embedding, description_representation])
        logits = self.entity_predictions(rep)


        if entity_labels is not None:
            loss = self.loss_fn(logits, entity_labels)
            return logits, loss
        else:
            return logits




















