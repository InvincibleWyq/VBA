# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from mmaction.datasets import DATASETS, build_dataset
from mmaction.utils import local_numpy_seed


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be ``times`` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (dict): The config of the dataset to be repeated.
        times (int): Repeat times.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self, dataset, times, test_mode=False):
        dataset['test_mode'] = test_mode
        self.dataset = build_dataset(dataset)
        self.times = times

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get data."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len


@DATASETS.register_module()
class ConcatDataset:
    """A wrapper of concatenated dataset.

    The length of concatenated dataset will be the sum of lengths of all
    datasets. This is useful when you want to train a model with multiple data
    sources.

    Args:
        datasets (list[dict]): The configs of the datasets.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self, datasets, test_mode=False):

        for item in datasets:
            item['test_mode'] = test_mode

        datasets = [build_dataset(cfg) for cfg in datasets]
        self.datasets = datasets
        self.lens = [len(x) for x in self.datasets]
        self.cumsum = np.cumsum(self.lens)

    def __getitem__(self, idx):
        """Get data."""
        dataset_idx = np.searchsorted(self.cumsum, idx, side='right')
        item_idx = idx if dataset_idx == 0 else idx - self.cumsum[dataset_idx]
        return self.datasets[dataset_idx][item_idx]

    def __len__(self):
        """Length after repetition."""
        return sum(self.lens)


@DATASETS.register_module()
class EpisodicDataset:
    """A wrapper of episodic dataset.

    It will generate a list of support and query images indices for each
    episode (support + query images). Every call of `__getitem__` will fetch
    and return (`num_ways` * `num_shots`) support images and (`num_ways` *
    `num_queries`) query images according to the generated images indices.
    Note that all the episode indices are generated at once using a specific
    random seed to ensure the reproducibility for same dataset.

    Args:
        dataset (:obj:`Dataset`): The dataset to be wrapped.
        num_episodes (int): Number of episodes. Noted that all episodes are
            generated at once and will not be changed afterwards. Make sure
            setting the `num_episodes` larger than your needs.
        num_ways (int): Number of ways for each episode.
        num_shots (int): Number of support data of each way for each episode.
        num_queries (int): Number of query data of each way for each episode.
        episodes_seed (int | None): A random seed to reproduce episodic
            indices. If seed is None, it will use runtime random seed.
            Default: None.
        subset (str): Subset of the dataset. Support 'train', 'val', 'test'.
            Used in few shot learning. Default: None.
        train_label (list): Only used in few shot learning.
            List of labels for training. Default: [].
        val_label (list): Only used in few shot learning.
            List of labels for validation. Default: [].
        test_label (list): Only used in few shot learning.
            List of labels for testing. Default: [].
    """

    def __init__(self,
                 dataset: Dataset,
                 num_episodes: int,
                 num_ways: int,
                 num_shots: int,
                 num_queries: int,
                 episodes_seed: int,
                 subset: str = None,
                 train_labels: list = None,
                 val_labels: list = None,
                 test_labels: list = None):
        self.dataset = dataset
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.num_episodes = num_episodes
        self._len = len(self.dataset)
        # using same episodes seed can generate same episodes for same dataset
        # it is designed for the reproducibility of meta train or meta test
        self.episodes_seed = episodes_seed
        self.subset = subset
        assert self.subset in ['train', 'val', 'test']
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels

        self.episode_idxes, self.episode_class_ids = \
            self.generate_episodic_idxes()

    def generate_episodic_idxes(self) -> tuple:
        """Generate batch indices for each episodic."""
        episode_idxes, episode_class_ids = [], []
        if self.subset == 'train':
            class_ids = self.train_labels
        elif self.subset == 'val':
            class_ids = self.val_labels
        elif self.subset == 'test':
            class_ids = self.test_labels
        # using same episodes seed can generate same episodes for same dataset
        # it is designed for the reproducibility of meta train or meta test
        with local_numpy_seed(self.episodes_seed):
            for _ in range(self.num_episodes):
                np.random.shuffle(class_ids)
                # sample classes
                sampled_cls = class_ids[:self.num_ways]
                episode_class_ids.append(sampled_cls)
                episodic_support_idx = []
                episodic_query_idx = []
                # sample instances of each class
                for i in range(self.num_ways):
                    shots = self.dataset.sample_shots_by_class_id(
                        class_id=sampled_cls[i],
                        num_shots=self.num_shots + self.num_queries)
                    episodic_support_idx += shots[:self.num_shots]
                    episodic_query_idx += shots[self.num_shots:]
                episode_idxes.append({
                    'support': episodic_support_idx,
                    'query': episodic_query_idx
                })
        return episode_idxes, episode_class_ids

    def __getitem__(self, idx: int) -> dict:
        """Return a episode data at the same time.

        For `EpisodicDataset`, this function would return num_ways *
        num_shots support images and num_ways * num_queries query image.
        """
        episode_idx = self.episode_idxes[idx]
        ret_dict = {
            'support_data': [self.dataset[i] for i in episode_idx['support']],
            'query_data': [self.dataset[i] for i in episode_idx['query']]
        }

        return ret_dict

    def __len__(self) -> int:
        """The length of the dataset is the number of generated episodes."""
        return self.num_episodes

    def evaluate(self, *args, **kwargs) -> list:
        """Evaluate prediction."""
        return self.dataset.evaluate(*args, **kwargs)

    def get_episode_class_ids(self, idx: int) -> list:
        """Return class ids in one episode."""
        return self.episode_class_ids[idx]


@DATASETS.register_module()
class MetaTestDataset(EpisodicDataset):
    """A wrapper of the episodic dataset for meta testing.

    During meta test, the `MetaTestDataset` will be copied and converted into
    three mode: `test_set`, `support`, and `test`. Each mode of dataset will
    be used in different dataloader, but they share the same episode and image
    information.

    - In `test_set` mode, the dataset will fetch all images from the
      whole test set to extract features from the fixed backbone, which
      can accelerate meta testing.
    - In `support` or `query` mode, the dataset will fetch images
      according to the `episode_idxes` with the same `task_id`. Therefore,
      the support and query dataset must be set to the same `task_id` in
      each test task.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mode = 'test_set'
        self._task_id = 0
        self._with_cache_feats = False

    def with_cache_feats(self) -> bool:
        return self._with_cache_feats

    def set_task_id(self, task_id: int) -> None:
        """Query and support dataset use same task id to make sure fetch data
        from same episode."""
        self._task_id = task_id

    def __getitem__(self, idx: int) -> dict:
        """Return data according to mode.

        For mode `test_set`, this function would return single image as regular
        dataset. For mode `support`, this function would return single support
        image of current episode. For mode `query`, this function would return
        single query image of current episode. If the dataset have cached the
        extracted features from fixed backbone, then the features will be
        return instead of image.
        """

        if self._mode == 'test_set':
            idx = idx
        elif self._mode == 'support':
            idx = self.episode_idxes[self._task_id]['support'][idx]
        elif self._mode == 'query':
            idx = self.episode_idxes[self._task_id]['query'][idx]

        if self._with_cache_feats:

            return {
                'feats': self.dataset.video_infos[idx]['feats'],
                'label': self.dataset.video_infos[idx]['label']
            }
        else:
            return self.dataset[idx]

    def get_task_class_ids(self) -> list:
        return self.get_episode_class_ids(self._task_id)

    def test_set(self):
        self._mode = 'test_set'
        return self

    def support(self):
        self._mode = 'support'
        return self

    def query(self):
        self._mode = 'query'
        return self

    def __len__(self) -> int:
        if self._mode == 'test_set':
            return len(self.dataset)
        elif self._mode == 'support':
            return self.num_ways * self.num_shots
        elif self._mode == 'query':
            return self.num_ways * self.num_queries

    def cache_feats(self, feats: Tensor, img_metas: dict) -> None:
        """Cache extracted feats into dataset."""

        if 'filename' in img_metas[0]:
            unique_str = 'filename'
        elif 'frame_dir' in img_metas[0]:
            unique_str = 'frame_dir'
        else:
            raise ValueError(
                'img_metas must have key `filename` or `frame_dir`. '
                'It will be used as unique id')

        idx_map = {
            video_info[unique_str]: idx
            for idx, video_info in enumerate(self.dataset.video_infos)
        }

        # # save inverse_idx_map to file
        # inverse_idx_map = {
        #     idx: video_info[unique_str]
        #     for idx, video_info in enumerate(self.dataset.video_infos)
        # }
        # import pickle
        # with open('inverse_idx_map.pkl', 'wb') as f:
        #     pickle.dump(inverse_idx_map, f)

        # use unique_str as unique id
        for i, img_meta in enumerate(img_metas):
            idx = idx_map[img_meta[unique_str]]
            temp_dict = {k: feats[k][i] for k in feats.keys()}
            self.dataset.video_infos[idx]['feats'] = temp_dict

        self._with_cache_feats = True
