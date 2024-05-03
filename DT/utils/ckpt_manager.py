import numpy as np
from pathlib import Path
import shutil
from flax.training import train_state

class CheckpointManager:
  def __init__(self, path_save, max_to_keep=1, remove_old=False):
    self.path_save = Path(path_save).resolve()
    self.path_save.mkdir(exist_ok=True, parents=True)
    self.max_to_keep = max_to_keep
    if remove_old:
      shutil.rmtree(str(self.path_save), ignore_errors=True)
  
  def save(self, epoch: int, state: train_state.TrainState, config: dict, verbose: bool = True):
    for k, v in config.items():
      if isinstance(v, Path): config[k] = str(v)
    config['_step'] = int(state.step)
    weights = {'params': state.params, 'config': config}
    if verbose:
      print(f"Save weights at {self.path_save}/{epoch:03}.npy")
    path_old_save = self.path_save / f"{epoch-self.max_to_keep:03}.npy"
    path_old_save.unlink(missing_ok=True)
    np.save(str(self.path_save / f"{epoch:03}"), weights, allow_pickle=True)
  
  def restore(self, epoch: int):
    return np.load(str(Path(self.path_save) / f"{epoch:03}.npy"), allow_pickle=True).item()
  