"""
Utilization functions
"""

import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm
from speechbrain.nnet.pooling import StatisticsPooling
import pickle as pkl


class AudioDataset(Dataset):
	def __init__(self, directory):
		self.directory = directory
		self.audio_files = [filename for filename in os.listdir(directory) if filename.endswith(".wav")]

	def __len__(self):
		return len(self.audio_files)

	def __getitem__(self, idx):
		filename = self.audio_files[idx]
		filepath = os.path.join(self.directory, filename)
		audio_input, _ = torchaudio.load(filepath)
		return audio_input, filename


def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    >>> str_to_bool('YES')
    1
    >>> str_to_bool('FALSE')
    0
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError('invalid truth value {}'.format(val))


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    """Learning rate decay in Keras-style"""
    return 1. / (1. + decay * step)


class SGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """SGD with restarts scheduler"""
    def __init__(self, optimizer, T0, T_mul, eta_min, last_epoch=-1):
        self.Ti = T0
        self.T_mul = T_mul
        self.eta_min = eta_min

        self.last_restart = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T_cur = self.last_epoch - self.last_restart
        if T_cur >= self.Ti:
            self.last_restart = self.last_epoch
            self.Ti = self.Ti * self.T_mul
            T_cur = 0

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + np.cos(np.pi * T_cur / self.Ti)) / 2
            for base_lr in self.base_lrs
        ]


def _get_optimizer(model_parameters, optim_config):
    """Defines optimizer according to the given config"""
    optimizer_name = optim_config['optimizer']

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters,
                                    lr=optim_config['base_lr'],
                                    momentum=optim_config['momentum'],
                                    weight_decay=optim_config['weight_decay'],
                                    nesterov=optim_config['nesterov'])
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=optim_config['base_lr'],
                                     betas=optim_config['betas'],
                                     weight_decay=optim_config['weight_decay'],
                                     amsgrad=str_to_bool(
                                         optim_config['amsgrad']))
    else:
        print('Un-known optimizer', optimizer_name)
        sys.exit()

    return optimizer


def _get_scheduler(optimizer, optim_config):
    """
    Defines learning rate scheduler according to the given config
    """
    if optim_config['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config['milestones'],
            gamma=optim_config['lr_decay'])

    elif optim_config['scheduler'] == 'sgdr':
        scheduler = SGDRScheduler(optimizer, optim_config['T0'],
                                  optim_config['Tmult'],
                                  optim_config['lr_min'])

    elif optim_config['scheduler'] == 'cosine':
        total_steps = optim_config['epochs'] * \
            optim_config['steps_per_epoch']

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                optim_config['lr_min'] / optim_config['base_lr']))

    elif optim_config['scheduler'] == 'keras_decay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: keras_decay(step))
    else:
        scheduler = None
    return scheduler


def create_optimizer(model_parameters, optim_config):
    """Defines an optimizer and a scheduler"""
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, config = None):
    """
    set initial seed for reproduction
    """
    if config is None:
        raise ValueError("config should not be None")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = str_to_bool(config["cudnn_deterministic_toggle"])
        torch.backends.cudnn.benchmark = str_to_bool(config["cudnn_benchmark_toggle"])


def collate_fn(batch):
	batch.sort(key=lambda x: x[0].size(1), reverse=True)
	audio_batch, filenames_batch = zip(*batch)
	max_length = max(audio.size(1) for audio in audio_batch)
	audio_batch_padded = []

	# We pad the audios with 0s in order to be as long as the longest audio.
	for audio in audio_batch:
		if audio.size(1) < max_length:
			pad_amount = max_length - audio.size(1)
			audio = torch.nn.functional.pad(audio, (0, pad_amount))

		audio = audio.squeeze(0)
		audio_batch_padded.append(audio)

	audio_batch_padded = torch.stack(audio_batch_padded)

	return audio_batch_padded, filenames_batch


def get_embeddings_wav2vec(directory: str, file: str, dataloader, batch_size: int=4, device: str='cuda') -> None:
	"""The function computes the embeddings with wav2vec2 and writes them to
		pickle file. The data is stored as key-value pairs where the key is the
		base name of the file.

	Args:
		directory (str): The path to the directory that contains the data we want.
		file (str): The path to the pickle file where we write the results
		batch_size (int): The batch size. Defaults to 1.
	"""
	bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
	model = bundle.get_model().to(device)
	sp_layer = StatisticsPooling()
	embeddings_dict = {}
	dataset = AudioDataset(directory)

	for audio_batch, _, filenames_batch in tqdm(dataloader, desc="Computing embeddings", unit="batch"):
		audio_batch = audio_batch.to(device)

		with torch.inference_mode():
			features, _ = model.extract_features(audio_batch)

		stats_pooled_features = sp_layer(features[-1])

		for audio, filename in zip(stats_pooled_features, filenames_batch):
			filename += ".flac"
			embeddings_dict[filename] = audio.squeeze(0).squeeze(0).cpu().numpy()

	with open(file, "wb") as f:
		pkl.dump(embeddings_dict, f)
