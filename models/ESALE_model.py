import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask(source_ids, target_ids, padding_id, with_ulm):

    """
        `input`
        source_ids: batch_size * code_len
        target_ids: batch_size * comment_len
        with_ulm: when the arg is True, the 11 mask will be a lower triangular matrix.

        `output`
        mask: batch_size * (code_len + comment_len) * (code_len + comment_len)
    """
    
    batch_size = source_ids.shape[0]
    code_len = source_ids.shape[-1]
    comment_len = target_ids.shape[-1]
    mask00 = source_ids.ne(padding_id)[
        :, None, :]*source_ids.ne(padding_id)[:, :, None]
    mask01 = torch.zeros(
        (batch_size, code_len, comment_len), dtype=torch.uint8).to(source_ids.device)
    mask0 = torch.cat((mask00, mask01), 2)
    mask10 = source_ids.ne(padding_id)[
        :, None, :]*target_ids.ne(padding_id)[:, :, None]
    if with_ulm:
        mask11 = torch.tril(target_ids.ne(padding_id)[
                            :, None, :]*target_ids.ne(padding_id)[:, :, None], 0)
    else:
        mask11 = target_ids.ne(padding_id)[
            :, None, :]*target_ids.ne(padding_id)[:, :, None]
    mask1 = torch.cat((mask10, mask11), 2)
    mask = torch.cat((mask0, mask1), 1)
    return mask


class ESALE(nn.Module):
    def __init__(self,
                 encoder,
                 config,
                 max_seq=256,
                 max_output_seq = 128,
                 awp_cls=40,
                 with_mlm=True,
                 with_awp=True,
                 with_ulm=True
                 ):
        super().__init__()
        self.max_seq = max_seq
        self.max_output_seq = max_output_seq
        self.with_mlm = with_mlm
        self.with_awp = with_awp
        self.with_ulm = with_ulm
        self.initializer_range = config.initializer_range
        self.padding_id = config.pad_token_id
        d_model = config.hidden_size
        layer_norm_eps = config.layer_norm_eps
        vocab_size = config.vocab_size
        self.transform_act_fn = F.gelu

        self.ulmDecoder = nn.Linear(
            d_model, vocab_size, bias=False)
        self.ulmBias = nn.Parameter(torch.zeros(vocab_size))
        self.ulmDecoder.bias = self.ulmBias
        self.ulmDense = nn.Linear(d_model, d_model)
        self.ulmLayerNorm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.mlmDecoder = nn.Linear(
            d_model, vocab_size, bias=False)
        self.mlmBias = nn.Parameter(torch.zeros(vocab_size))
        self.mlmDecoder.bias = self.mlmBias
        self.mlmDense = nn.Linear(d_model, d_model)
        self.mlmLayerNorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.pooler = nn.Linear(d_model, d_model)
        self.pooler_activation = nn.Tanh()
        self.awp_pooler = nn.Linear(d_model, awp_cls + 1) # 1 for <other>

        self.apply(self.init_model_weights)

        self.encoder = encoder
        self.embeddings = encoder.embeddings
        word_embeddings_weight = self.encoder.embeddings.word_embeddings.weight
        self.ulmDecoder.weight = word_embeddings_weight
        self.mlmDecoder.weight = word_embeddings_weight

    def forward(self, source_ids, target_ids=None, masked_target_ids = None):

        if target_ids == None:
            mask = source_ids.ne(self.padding_id)[
                :, None, :]*source_ids.ne(self.padding_id)[:, :, None]
            seq_output = self.encoder(
                source_ids, attention_mask=mask, use_cache=True).last_hidden_state
            ulm_output = None
            mlm_output = None
        else:
            if self.with_mlm:
                ids = torch.cat((source_ids, masked_target_ids), -1)
                mask = get_mask(source_ids, masked_target_ids, self.padding_id,False)
                seq_output = self.encoder(
                    ids, attention_mask=mask, use_cache=True).last_hidden_state
                mlm_hidden_state = self.mlmDense(seq_output)
                mlm_hidden_state = self.transform_act_fn(mlm_hidden_state)
                mlm_hidden_state = self.mlmLayerNorm(mlm_hidden_state)
                output = self.mlmDecoder(mlm_hidden_state)
                mlm_output = output[:, self.max_seq:]  # only take comments
            else:
                mlm_output = None

            if self.with_ulm:
                ids = torch.cat((source_ids, target_ids), -1)
                mask = get_mask(source_ids, target_ids, self.padding_id,True)
                seq_output = self.encoder(
                    ids, attention_mask=mask, use_cache=True).last_hidden_state
                ulm_hidden_state = self.ulmDense(seq_output)
                ulm_hidden_state = self.transform_act_fn(ulm_hidden_state)
                ulm_hidden_state = self.ulmLayerNorm(ulm_hidden_state)
                output = self.ulmDecoder(ulm_hidden_state)
                ulm_output = output[:, self.max_seq:]  # only take comments
            else:
                ulm_output = None

        pooled_output = self.pooler_activation(self.pooler(seq_output[:, 0]))

        if self.with_awp:
            awp_output = self.awp_pooler(pooled_output)
        else:
            awp_output = None

        return awp_output, ulm_output, mlm_output

    def init_model_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

