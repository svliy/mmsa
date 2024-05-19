import logging
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ['MMDataLoader']

logger = logging.getLogger('MMSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
        }
        # 调用数据集dataset_name的初始化函数
        DATASET_MAP[args['dataset_name']]()

    def __init_mosi(self):
        # 打开pkl文件，data是一个字典
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)

        # 获取特征
        if 'use_bert' in self.args and self.args['use_bert']:
            self.text = data[self.mode]['text_bert'].astype(np.float32)  # BERT feature
        else:
            self.text = data[self.mode]['text'].astype(np.float32)  # GLOVE feature
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        # use custom features
        # 都没有用到
        use_custom_features = (self.args['feature_T'] != "" or self.args['feature_A'] != "" or self.args['feature_V'] != "")
        if self.args['feature_T'] != "":
            print('自定义了feature_T')
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            if 'use_bert' in self.args and self.args['use_bert']:
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768# TODO: fix this
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2]
        if self.args['feature_A'] != "":
            print('自定义了feature_A')
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2]
        if self.args['feature_V'] != "":
            print('自定义了feature_V')
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]

        # 处理标签
        self.labels = {
            # 'M': data[self.mode][self.args['train_mode']+'_labels'].astype(np.float32)
            # 多模态标签 (batch_size, 1)
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
            # 'M': np.array(data[self.mode]['labels']).astype(np.float32)
        }
        # 中文数据集使用单模态标签
        if self.args['dataset_name'] == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode]['regression' + '_labels_' + m].astype(np.float32)
                
        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        # 非对齐数据处理
        if not self.args['need_data_aligned']:
            if self.args['feature_A'] != "":
                self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
            else:
                # sims dataset 执行
                self.audio_lengths = data[self.mode]['audio_lengths']
            if self.args['feature_V'] != "":
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            else:
                # sims dataset 执行
                self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0
        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __truncate(self):
        """truncate the feature to the same length"""
        def do_truncate(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
        
        text_length, audio_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # For visual and audio modality, we average across time:
        # The original data has shape (max_len, num_examples, feature_dim)
        # After averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        # (2722, 3, 50)
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            # 'M': 
            # 'A': 
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }
        # 非对齐数据集sims执行
        if not self.args['need_data_aligned']:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        return sample

def MMDataLoader(args, num_workers):

    # 构建数据集
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    # 通过 dataset 获取序列长度
    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    # 为什么训练数据都打乱了？
    # 构建dataloaders
    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys() if ds != 'test'
    }

    dataLoader['test'] = DataLoader(datasets['test'],
                                    batch_size=args['batch_size'],
                                    num_workers=num_workers,
                                    shuffle=False)
    
    return dataLoader
