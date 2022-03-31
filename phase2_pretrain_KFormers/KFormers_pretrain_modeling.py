import torch
import torch.nn as nn
# from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN

from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.bert.modeling_bert import BertForMaskedLM


# def create_position_ids_from_input_ids(input_ids, padding_idx):
#     """
#     Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
#     are ignored. This is modified from fairseq's `utils.make_positions`.
#
#     Args:
#         x: torch.Tensor x:
#
#     Returns: torch.Tensor
#     """
#     # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
#     mask = input_ids.ne(padding_idx).int()
#     incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
#     return incremental_indices.long() + padding_idx



class EntityEmbeddings(nn.Module):
    def __init__(self, config):
        super(EntityEmbeddings, self).__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)  # 2*256
        # if config.entity_emb_size != config.hidden_size:
        #     self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        self.LayerNorm = nn.LayerNorm(config.entity_emb_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, entity_ids):
        entity_embeddings = self.entity_embeddings(entity_ids)
        # if self.config.entity_emb_size != self.config.hidden_size:
        #     entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        embeddings = self.LayerNorm(entity_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings



class EntityPredictionHead(nn.Module):
    def __init__(self, config):
        super(EntityPredictionHead, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.entity_emb_size)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.entity_emb_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.entity_emb_size, config.entity_vocab_size, bias=True)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        hidden_states = self.decoder(hidden_states)

        return hidden_states



class EntityPredictionHeadV2(nn.Module):
    def __init__(self, config):
        super(EntityPredictionHeadV2, self).__init__()
        self.config = config

        self.dense = nn.Linear(config.hidden_size, config.entity_emb_size)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.entity_emb_size, eps=config.layer_norm_eps)
        # self.decoder = nn.Linear(config.entity_emb_size, config.entity_vocab_size, bias=True)

    def forward(self, hidden_states, candidate):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        candidate = torch.squeeze(candidate, 0)
        # hidden_states [batch_size, max_seq, dim]
        # candidate [entity_num_in_the_batch, dim]
        # return [batch_size, max_seq, entity_num_in_the_batch]
        return torch.matmul(hidden_states, candidate.t())



class MLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_state):
        x = self.dense(hidden_state)
        x = self.act_fn(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x



class PreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(PreTrainingHeads, self).__init__()
        self.predictions = MLMHead(config)
        self.predictions_ent = EntityPredictionHeadV2(config)

    def forward(self, token_output,
                mention_output, mention_candidate_embedding,
                cls_output, des_candidate_embedding):

        mlm_score = self.predictions(token_output)
        m2e_socre = self.predictions_ent(mention_output, mention_candidate_embedding)
        d2e_score = self.predictions_ent(cls_output, des_candidate_embedding)
        return mlm_score, m2e_socre, d2e_score



class KModulePretrainingModel(nn.Module):
    '''
    RoBERTa Model with a `XXX` head on top
    '''
    def __init__(self, config):
        super(KModulePretrainingModel, self).__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.roberta = RobertaModel(config)
        self.cls = PreTrainingHeads(config)
        self.entity_embeddings = EntityEmbeddings(config)

        self.binary_CE_loss = nn.BCEWithLogitsLoss()  # 二值交叉熵
        # self.loss_fn = nn.BCEWithLogitsLoss()  # 对比损失
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')  # 对比损失

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

    def forward(self, input_ids, input_mask, input_segment, mention_token_label,
                mention_span_idx, mention_entity_candidates,
                des_entity_candidates,
                mention_entity_labels, des_entity_labels):
        """
        mention_span_idx: 也可以表达start，end这种两头的形式
        """

        main_output = self.roberta(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_segment)
        token_output = main_output.last_hidden_state
        cls_output = main_output.pooler_output # batch, 768

        mention_output = torch.bmm(mention_span_idx.unsqueeze(1), token_output).squeeze(1) # batch 1 L * batch L d = batch d
        # mention_output = torch.mean(mention_output, dim=1, keepdim=False)
        t1 = self.entity_embeddings(mention_entity_candidates)

        # 第三个损失，对比损失，description=>entity
        t2 = self.entity_embeddings(des_entity_candidates)

        mlm_logits, m2e_logits, d2e_logits = self.cls(token_output,
                                                       mention_output, t1,
                                                       cls_output, t2)

        # 第一个损失，交叉熵损失，类似于MLM
        # main_logits = self.classifier(main_output.last_hidden_state)
        mlm_loss = self.cross_entropy_loss(mlm_logits.view(-1, self.vocab_size), mention_token_label.view(-1))

        # 第二个loss，对比损失，mention=>entity

        # m2e_logits = torch.matmul(t1, mention_output).squeeze(-1) # batch N+1
        # m2e_loss = self.binary_CE_loss(m2e_logits.view(-1, 2), mention_entity_labels.view(-1, 2))
        m2e_loss = self.cross_entropy_loss(m2e_logits.view(-1, input_ids.size(0)), mention_entity_labels.view(-1))

        # 第三个损失，对比损失，description=>entity
        # pooler_output = main_output.pooler_output  # batch, 768
        # t2 = self.entity_embeddings(des_entity_candidates)
        # d2e_logits = torch.matmul(t2, pooler_output.unsqueeze(1).permute(0, 2, 1)).squeeze(-1)  # batch N+1
        # d2e_loss = self.binary_CE_loss(d2e_logits.view(-1, 2), des_entity_labels.view(-1, 2))
        d2e_loss = self.cross_entropy_loss(d2e_logits.view(-1, input_ids.size(0)), des_entity_labels.view(-1))

        total_loss = m2e_loss # + d2e_loss + mlm_loss

        return total_loss