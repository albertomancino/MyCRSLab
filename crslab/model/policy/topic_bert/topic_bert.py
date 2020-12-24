# @Time   : 2020/12/17
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail

# UPDATE
# @Time   : 2020/12/24
# @Author : Xiaolei Wang
# @email  : wxl1999@foxmail.com

import os

from torch import nn
from transformers import BertModel

from crslab.config import dataset_language_map, MODEL_PATH
from crslab.model.base_model import BaseModel
from .resource import resources


class TopicBERTModel(BaseModel):
    def __init__(self, opt, device, vocab, side_data):
        self.topic_class_num = vocab['n_topic']
        language = dataset_language_map[opt['dataset']]
        dpath = os.path.join(MODEL_PATH, "tgredial", language)
        resource = resources[language]
        super(TopicBERTModel, self).__init__(opt, device, dpath, resource)

    def build_model(self, *args, **kwargs):
        """build model"""
        self.topic_bert = BertModel.from_pretrained(os.path.join(self.dpath, 'bert'))

        self.bert_hidden_size = self.topic_bert.config.hidden_size
        self.state2topic_id = nn.Linear(self.bert_hidden_size,
                                        self.topic_class_num)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch, mode):
        # conv_id, message_id, context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask, y = batch
        context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask, y = batch

        topic_rep = self.topic_bert(
            topic_path_kw,
            tp_mask).pooler_output  # (bs, hiddensize)

        topic_scores = self.state2topic_id(topic_rep)

        topic_loss = self.loss(topic_scores, y)

        return topic_loss, topic_scores
