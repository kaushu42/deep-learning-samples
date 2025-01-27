from typing import Union

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertForSequenceClassification,
    BertPreTrainedModel,
    BertTokenizer,
    pipeline,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
)


class BertWithCustomHead(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

        # Define the custom classification head
        self.custom_dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.config.num_labels),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        return_dict=True,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        # Get outputs from the base BERT model
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )

        # Extract the [CLS] pooled output
        pooled_output = outputs.pooler_output

        # Pass pooled output through the custom classification head
        pooled_output = self.custom_dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Return logits and loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        if not return_dict:
            return (loss, logits) if loss is not None else logits

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES["bert_with_custom_head"] = (
    "BertWithCustomHead"
)
# del MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES["bert_with_custom_head"]
