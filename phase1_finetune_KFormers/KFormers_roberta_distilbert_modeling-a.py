import torch.nn as nn
import torch
import torch.nn.functional as F

from parameters import parse_args

args = parse_args()


if args.backbone_model_type == 'luke':
    from luke_modeling import \
        (
        RobertaEmbeddings as BackboneEmbeddings,
        EntityAwareLayer as BackboneLayer,
        KnowledgeEntityEmbeddings,
        EntityEmbeddings,
        LukeModel as BackboneModel
    )  # Bert
elif args.backbone_model_type == "roberta":
    from transformers.models.roberta.modeling_roberta import \
        (
        RobertaClassificationHead,
        RobertaEmbeddings as BackboneEmbeddings,
        RobertaLayer as BackboneLayer,
        RobertaModel as BackboneModel
    )
    from luke_modeling import \
        (
        KnowledgeEntityEmbeddings
    )
else:
    pass
if args.knowledge_model_type == 'distilbert':
    from transformers.models.distilbert.modeling_distilbert import \
        (
        Embeddings as KEmbeddings,
        TransformerBlock as KnowledgeLayer,
    )  # 这个是k module



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
        k_layer_outputs, k_des_to_backbone = None, None

        word_hidden_states = torch.cat([word_hidden_states, k_entity_hidden_states], dim=1)
        batch_size, ent_num, _ = k_entity_hidden_states.size()
        if word_attention_mask.dim() == 2:
            ent_attention_mask = torch.ones(batch_size, ent_num, device=word_hidden_states.device)
            word_attention_mask = torch.cat([word_attention_mask, ent_attention_mask], dim=-1)
        elif word_attention_mask.dim() == 4:
            ent_attention_mask = torch.ones(batch_size, 1, 1, ent_num, device=word_hidden_states.device)
            k_des_mask = k_des_mask[:, None, None, :              ]
            word_attention_mask = torch.cat([word_attention_mask, ent_attention_mask], dim=-1)

        if self.k_layer is not None:
            batch_size, des_num, length, d = k_des_hidden_states.size()
            k_layer_outputs = self.k_layer(k_des_hidden_states.view(batch_size, des_num * length, d),
                                           k_des_attention_mask.view(batch_size, des_num * length))
            k_layer_outputs = k_layer_outputs[-1].reshape(batch_size, des_num, length, d)
            k_des_to_backbone = k_layer_outputs[:, :, 0, :]  # batch K d2
            k_des_to_backbone = self.map(k_des_to_backbone)  # batch K d description representation

            # description_state, entity_state, original_input_state整合到一起
            word_hidden_states = torch.cat([word_hidden_states, k_des_to_backbone], dim=1)  # batch L1+K1+K2 d
            word_attention_mask = torch.cat([word_attention_mask, k_des_mask], dim=-1) # batch 1 1 L1+K1+K2

        t = self.backbone_layer(
            word_hidden_states, word_attention_mask, entity_hidden_states, entity_attention_mask
        ) # batch L1 d, batch 2 d
        # [0]: word_vec, [1]:k_ent_vec
        # [2, 3] = des_vec and mapped des_vec # batch des_num 1024
        # [-1]: entity_vec
        if len(t) == 2:
            word_attention_output, entity_attention_output = t
            return word_attention_output[:, :word_size, :], \
                   word_attention_output[:, word_size: word_size+ent_size, :], \
                   k_layer_outputs, \
                   k_des_to_backbone, \
                   entity_attention_output # entity identifier
        else:
            word_attention_output,  = t
            return word_attention_output[:, :word_size, :], \
                   word_attention_output[:, word_size: word_size+ent_size, :], \
                   k_layer_outputs, \
                   k_des_to_backbone, \
                   None



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

        last_des_hidden_states = k_des_hidden_states  # batch des_num L d1
        last_des_to_backbone = None
        for i, layer_module in enumerate(self.layer):
            # [0]: word_vec, [1]:k_ent_vec
            # [2, 3] = des_vec and mapped des_vec # batch des_num 1024
            # [-1]: entity_vec
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
                last_des_to_backbone = layer_outputs[3]
            entity_hidden_states = layer_outputs[-1]

        outputs = (word_hidden_states, k_entity_hidden_states, last_des_hidden_states, last_des_to_backbone, entity_hidden_states)
        return outputs



class KFormersModel(BackboneModel):
    def __init__(self, config, config_k, backbone_knowledge_dict):
        super(KFormersModel, self).__init__(config)
        self.config = config
        self.config_k = config_k
        self.embeddings = BackboneEmbeddings(config)  # roberta
        self.embeddings.token_type_embeddings.requires_grad = False  # why?
        if config.backbone_model_type == "luke":
            self.entity_embeddings = EntityEmbeddings(config) # entity identifier, only LUKE needs
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
        if self.config.backbone_model_type == "luke":
            entity_embeddings = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        else:
            entity_embeddings = None
            entity_attention_mask = None
            attention_mask = attention_mask[:, None, None, :]
        k_entity_embeddings = self.k_ent_embeddings(k_ent_ids)

        batch, des_num, length = k_des_ids.size()
        k_des_embeddings = self.k_des_embeddings(k_des_ids.view(-1, length)).reshape(batch, des_num, length, -1)

        encoder_outputs = self.encoder(word_embeddings, attention_mask,
                                       entity_embeddings, entity_attention_mask,
                                       k_entity_hidden_states=k_entity_embeddings,
                                       k_des_hidden_states=k_des_embeddings, k_des_attention_mask=k_des_mask_one, k_des_mask=k_des_mask,
                                       )
        # word_hidden_states, k_entity_hidden_states, last_des_hidden_states, last_des_to_backbone, entity_hidden_states
        original_text_output, k_ent_output, _, mapped_k_des_output, entity_output = encoder_outputs
        if self.pooler is None:
            return (original_text_output, None, entity_output, k_entity_embeddings, k_ent_output, mapped_k_des_output)
        else:
            pooled_output = self.pooler(original_text_output)
            return (original_text_output, pooled_output, entity_output, k_entity_embeddings, k_ent_output, mapped_k_des_output)




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
        config.backbone_model_type = args.backbone_model_type

        self.config = config

        self.kformers = KFormersModel(config, config_k, backbone_knowledge_dict)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.hidden_size, self.num_labels)
        self.typing_2 = nn.Linear(config.hidden_size, self.num_labels)

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

        original_text_output, pooled_output, entity_output, k_entity_embeddings, k_ent_output, mapped_k_des_output = outputs  # batch L d
        original_text_output = self.dropout(original_text_output)

        # main loss
        if entity_output: # for luke
            entity_vec = self.dropout(entity_output[:, 0, :])
        else:
            start_ids = start_id.unsqueeze(1)  # batch 1 L
            entity_vec = torch.bmm(start_ids, original_text_output).squeeze(1)
        logits = self.typing(entity_vec)

        # first auxiliary loss: ent/des enhancement
        start_id = start_id.unsqueeze(1)  # batch 1 L
        anchor_vec = torch.bmm(start_id, original_text_output).squeeze(1)  # batch d
        true_ent_vec = k_ent_output[range(k_ent_output.size(0)), k_label]
        true_des_vec = torch.mean(mapped_k_des_output, dim=1, keepdim=False)  # bach d
        aux_logits = self.typing_2(anchor_vec + true_ent_vec + true_des_vec)  # ENT表征+true_e

        if labels is None:
            return logits + aux_logits
        else:
            main_loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits))

            aux_loss = F.binary_cross_entropy_with_logits(aux_logits.view(-1), labels.view(-1).type_as(aux_logits))

            return logits + aux_logits, main_loss + self.alpha * aux_loss



class KFormersForRelationClassification(nn.Module):
    def __init__(self, args, config_k, backbone_knowledge_dict):
        super(KFormersForRelationClassification, self).__init__()
        self.num_labels = args.num_labels
        self.ent_num = args.max_ent_num

        self.alpha = args.alpha
        self.beta = args.beta

        args.add_pooling_layer = False
        config = args.model_config
        config.backbone_model_type = args.backbone_model_type

        self.config = config

        self.kformers = KFormersModel(config, config_k, backbone_knowledge_dict)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(args.model_config.hidden_size * 2, self.num_labels, False)
        self.classifier_2 = nn.Linear(args.model_config.hidden_size * 2, self.num_labels, False)

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

        original_text_output, pooled_output, entity_output, k_entity_embeddings, k_ent_output, mapped_k_des_output = outputs  # batch L d
        sequence_output = self.dropout(original_text_output)

        # main loss
        if entity_output:  # for luke
            entity_vec = torch.cat([entity_output[:, 0, :], entity_output[:, 1, :]], dim=1)
            entity_vec = self.dropout(entity_vec)
        else:
            sub_start_id, obj_start_id = start_id.split(1, dim=1)  # split to 2, each is 1
            sub_start_id = sub_start_id
            subj_output = torch.bmm(sub_start_id, sequence_output)

            obj_start_id = obj_start_id
            obj_output = torch.bmm(obj_start_id, sequence_output)
            entity_vec = torch.cat([subj_output.squeeze(1), obj_output.squeeze(1)], dim=1)
        logits = self.classifier(entity_vec)

        # 第一个辅助loss：ent增强&des增强
        sub_start_id, obj_start_id = start_id.split(1, dim=1)  # split to 2, each is 1
        subj_output = torch.bmm(sub_start_id, sequence_output)
        obj_output = torch.bmm(obj_start_id, sequence_output)
        anchor_vec = torch.cat([subj_output, obj_output], dim=1) # batch 2 d

        true_ent_vec = k_ent_output[range(k_ent_output.size(0)), k_label].unsqueeze(1) # batch 1 d
        true_des_vec = torch.mean(mapped_k_des_output, dim=1, keepdim=True)  # bach 1 d

        t = anchor_vec + true_ent_vec + true_des_vec # batch 2 d
        aux_logits = self.classifier_2(t.view(t.size(0), -1))  # ENT表征+true_e


        if labels is None:
            return logits + aux_logits
        else:
            main_loss = F.cross_entropy(logits, labels)
            aux_loss = F.cross_entropy(aux_logits, labels)

            return logits + aux_logits, main_loss + self.alpha * aux_loss



class KFormersForSequenceClassification(nn.Module):
    def __init__(self, args, config_k, backbone_knowledge_dict):
        super(KFormersForSequenceClassification, self).__init__()
        self.num_labels = args.num_labels
        self.ent_num = args.max_ent_num

        self.alpha = args.alpha
        self.beta = args.beta

        args.add_pooling_layer = False
        config = args.model_config
        config.backbone_model_type = args.backbone_model_type

        self.config = config

        self.kformers = KFormersModel(config, config_k, backbone_knowledge_dict)

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        # self.classifier_2 = nn.Linear(config.hidden_size, self.num_labels)

        self.classifier = RobertaClassificationHead(config)
        self.classifier_2 = RobertaClassificationHead(config)
        self.loss = nn.CrossEntropyLoss(reduction='sum')

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

        original_text_output, pooled_output, entity_output, k_entity_embeddings, k_ent_output, mapped_k_des_output = outputs  # batch L d

        # 主loss
        # main loss
        if entity_output:  # for luke
            # feature_vector = self.dropout(entity_output[:, 0, :])
            logits = self.classifier(entity_output)
        else:
            # feature_vector = self.dropout(original_text_output)
            logits = self.classifier(original_text_output)

        # 第一个辅助loss：ent增强&des增强
        cls_vec = original_text_output[:, 0, :] # batch d
        true_ent_vec = k_ent_output[range(k_ent_output.size(0)), k_label]
        true_des_vec = torch.mean(mapped_k_des_output, dim=1, keepdim=False)  # bach d
        aux_logits = self.classifier_2(cls_vec + true_ent_vec + true_des_vec)  # ENT表征+true_e

        if labels is None:
            return logits + aux_logits
        else:
            main_loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))

            aux_loss = self.loss(aux_logits.view(-1, self.num_labels), labels.view(-1))

            return logits + aux_logits, main_loss + self.alpha * aux_loss



class KFormersForSequencePairClassification(nn.Module):
    def __init__(self, args, config_k, backbone_knowledge_dict):
        super(KFormersForSequencePairClassification, self).__init__()
        self.num_labels = args.num_labels
        self.ent_num = args.max_ent_num

        args.add_pooling_layer = False
        config = args.model_config
        self.config = config

        self.kformers = KFormersModel(config, config_k, backbone_knowledge_dict)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

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

        original_text_output, pooled_output, entity_output, k_entity_embeddings, k_ent_output, mapped_k_des_output = outputs  # batch L d

        # 主loss
        feature_vector = self.dropout(original_text_output[:, 0, :] )
        logits = self.classifier(feature_vector)

        if labels is None:
            return logits
        else:
            main_loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

            return logits, main_loss
