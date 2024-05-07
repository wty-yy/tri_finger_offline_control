import bisect, torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import h5py

class DatasetBuilder:
  def __init__(self, path_dataset: str, n_step: int = None, seed=42):
    torch.manual_seed(seed)
    self.path_dataset = path_dataset
    self.n_step = n_step
    self.data = None
    self.preload()
  
  def preload(self):
    # obs, action, rtg (each step), done_idx (each trajectory)
    data = self.data = {}
    if Path(self.path_dataset).suffix == '.npy':
      self.replay = replay = np.load(str(self.path_dataset), allow_pickle=True).item()
      replay['rewards'] = replay['rewards'].reshape(-1)
    elif Path(self.path_dataset).suffix in ['.h5', '.hdf5']:
      self.replay = replay = h5py.File(str(self.path_dataset))
    replay = {k: np.array(v) for k, v in replay.items()}
    data['obs'], data['action'] = replay['observations'], replay['actions']
    n = self.datasize = len(data['obs'])
    print(replay.keys())
    new_done_idx, count = [], 0
    if 'timeouts' in replay:
      data['done_idx'] = np.where(replay['terminals'] | replay['timeouts'])[0]
    else:
      data['done_idx'] = np.where(replay['terminals'])[0]
    ### Build return-to-go ###
    st = -1
    rtg = data['rtg'] = np.zeros((n,), np.float32)
    timestep = data['timestep'] = np.zeros(n, np.int32)
    data_mask = np.zeros(len(timestep), np.bool_)
    for i in data['done_idx']:
      time_len = i - st
      if self.n_step is None or time_len >= self.n_step:
        for j in range(i, st, -1):
          rtg[j] = replay['rewards'][j] + (0 if j == i else rtg[j+1])
          timestep[j] = j - st
        data_mask[st+1:i+1] = True
        count += time_len
        new_done_idx.append(count-1)
      st = i
    for k, v in data.items():
      if k != 'done_idx':
        data[k] = v[data_mask]
      else:
        data[k] = np.array(new_done_idx, np.int32)
    episode_len = self.data['timestep'][self.data['done_idx']]
    data['info'] = f"\
Max rtg: {max(data['rtg'])}, Mean rtg: {np.mean(data['rtg'])}, Min rtg: {min(data['rtg'])}, Min episode len: {min(episode_len)}, Max timestep: {max(episode_len)},\
Vocab size: {data['action'].shape[-1]}, Total steps: {len(data['obs'])}"
    print(data['info'])
  
  def debug(self):
    import matplotlib.pyplot as plt
    # plt.subplot(1,2,1)
    episode_len = self.data['timestep'][self.data['done_idx']]
    print(self.data['rtg'])
    plt.figure()
    plt.hist(episode_len)
    plt.title("episode length")
    plt.figure()
    plt.hist(self.data['rtg'])
    plt.title("return to go")
    plt.show()
    for i in range(4):
      action = self.data['action'][...,i]
      print(action.min(), action.max())
    
  def get_dataset(self, n_step: int, batch_size: int, num_workers: int = 4):  # Only train dataset
    return DataLoader(
      StateActionReturnDataset(self.data, n_step),
      batch_size=batch_size,
      shuffle=True,
      persistent_workers=True,  # GOOD
      num_workers=num_workers,
      drop_last=True,
    )

class StateActionReturnDataset(Dataset):
  def __init__(self, data: dict, n_step: int):
    self.data, self.n_step = data, n_step
  
  def __len__(self):
    return np.sum(self.data['timestep']!=0) - self.n_step - 1
  
  def __getitem__(self, idx):
    n_step, data = self.n_step, self.data
    done_idx = idx + n_step - 1
    # bisect_left(a, x): if x in a, return left x index, else return index with elem bigger than x
    # minus one for building the target action
    done_idx = min(data['done_idx'][bisect.bisect_left(data['done_idx'], idx)], done_idx)
    idx = done_idx - n_step + 1
    s = data['obs'][idx:done_idx+1].astype(np.float32)            # (n_step, obs_dim)
    a = data['action'][idx:done_idx+1].astype(np.int32)         # (n_step, n_vocab)
    rtg = data['rtg'][idx:done_idx+1].astype(np.float32)          # (n_step,)
    timestep = data['timestep'][idx:done_idx+1].astype(np.int32)  # (n_step,)
    return s, a, rtg, timestep
  
if __name__ == '__main__':
  path_dataset = str(Path(__file__).parents[1] / "dataset/mydata_v4_1367861.npy")
  ds_builder = DatasetBuilder(path_dataset, n_step=5)
  ds_builder.debug()
  ds = ds_builder.get_dataset(5, 128)
  from tqdm import tqdm
  for s, a, rtg, timestep in tqdm(ds):
    print(s.shape, a.shape, rtg.shape, timestep.shape)
    break
