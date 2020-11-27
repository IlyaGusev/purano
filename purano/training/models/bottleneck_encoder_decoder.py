from typing import Optional

import torch
from transformers import EncoderDecoderModel, PreTrainedModel, PretrainedConfig

class BottleneckEncoderDecoderModel(EncoderDecoderModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config, encoder, decoder)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ):
        return EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path,
            decoder_pretrained_model_name_or_path,
            *model_args,
            **kwargs
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_encoder,
        )
        batch_size = encoder_outputs[0].size(0)
        seq_len = encoder_outputs[0].size(1)
        embeddings_dim = encoder_outputs[0].size(2)
        encoder_outputs[0][:, 1:, :] = torch.zeros((batch_size, seq_len-1, embeddings_dim))
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
