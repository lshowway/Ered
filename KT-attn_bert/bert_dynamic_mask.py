import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


from transformers.models.bert.modeling_bert import \
    (
        BertEmbeddings,
        BertPreTrainedModel,
        BertPooler,
        BertSelfAttention,
        BertEncoder,
     )
from transformers.modeling_outputs import \
    (
        SequenceClassifierOutput,
        BaseModelOutputWithPoolingAndCrossAttentions,
    )


class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def convert_idx_to_mask(self, input_ids, attention_mask=None):
        # detail
        batch_size, seq_length = input_ids.size(0), input_ids.size(1)
        input_mask = torch.zeros((batch_size, seq_length, seq_length)).to(input_ids.device)
        T0, T1, T2, A_idx, B_idx = torch.unbind(attention_mask, dim=1)
        for i in range(batch_size):
            t0 = T0[i]
            t1 = T1[i]
            t2 = T2[i]
            a_idx = A_idx[i]
            b_idx = B_idx[i]

            input_mask[i, :t0, :t0] = 1
            if t1 != -1:  # description A不为空（空可能是没数据，也可能是长度不够） -1
                input_mask[i, t0:t1, t0:t1] = 1  # here
            if t2 != -1:  # desc
                if t1 == -1:  # 没有description A，从qk算
                    input_mask[i, t0: t2, t0: t2] = 1
                else:
                    input_mask[i, t1:t2, t1:t2] = 1
            if a_idx != -1 and t1 != -1:
                input_mask[i, a_idx, t0: t1] = 1
            if b_idx != -1 and t2 != -1:  # 都存在
                if t1 > -1:
                    input_mask[i, b_idx, t1: t2] += 1
                else:
                    input_mask[i, b_idx, t0: t2] += 1

        return input_mask

    # def convert_idx_to_mask(self, input_ids, attention_mask=None):
    #     # t1 t2
    #     batch_size, seq_length = input_ids.size(0), input_ids.size(1)
    #     input_mask = torch.zeros((batch_size, seq_length, seq_length)).to(input_ids.device)
    #     T0, T1, T2, A_idx, B_idx = torch.unbind(attention_mask, dim=1)
    #     for i in range(batch_size):
    #         t0 = T0[i]
    #         t1 = T1[i]
    #         t2 = T2[i]
    #         a_idx = A_idx[i]
    #         b_idx = B_idx[i]
    #
    #         input_mask[i, :t0, :t0] = 1
    #         if t1 != -1:
    #             input_mask[i, t0:t1, t0:t1] = 1  # here
    #         if t2 != -1:  # desc
    #             input_mask[i, t0: t2, t0: t2] = 1
    #         if a_idx != -1:
    #             input_mask[i, a_idx, :] = 1
    #         if b_idx != -1:  # 都存在
    #             input_mask[i, b_idx, :] = 1
    #
    #     return input_mask

    # def convert_idx_to_mask(self, input_ids, attention_mask=None):
    #     batch_size, seq_length = input_ids.size(0), input_ids.size(1)
    #     input_mask = torch.ones((batch_size, seq_length, seq_length)).to(input_ids.device)
    #     # T0, T1, T2, A_idx, B_idx = torch.unbind(attention_mask, dim=1)
    #     # for i in range(batch_size):
    #     #     t0 = T0[i]
    #     #     t1 = T1[i]
    #     #     t2 = T2[i]
    #     #     a_idx = A_idx[i]
    #     #     b_idx = B_idx[i]
    #     #
    #     #     input_mask[:, t0:, t0:] = 0
    #     #     if t1 != -1:
    #     #         input_mask[:, t0:t1, t0:t1] = 1
    #     #     if t2 != -1:
    #     #         input_mask[:, t0: t2, t0: t2] = 1
    #
    #     return input_mask


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        else:
            # ============ dynamic mask operation ===========
            attention_mask = self.convert_idx_to_mask(input_ids, attention_mask)
            # ============ dynamic mask operation ===========
            attention_mask = attention_mask.unsqueeze(1)  # batch 1 L L
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_attention_mask = attention_mask.to(dtype=input_ids.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
