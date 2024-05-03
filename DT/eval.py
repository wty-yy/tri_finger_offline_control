from env import Env
from dt_model import GPT
from flax.training import train_state
import jax
import numpy as np

class Evaluator:
  def __init__(self, model: GPT, seed: int = 42):
    self.model = model
    self.rng = jax.random.PRNGKey(seed)
    self.n_step = self.model.cfg.n_token // 3
    self.action_dim = len(self.model.cfg.act_dim)
    self.env = Env(seed=seed)
  
  def get_action(self):
    n_step = self.n_step
    def pad(x):
      delta = max(n_step - len(x), 0)
      x = np.expand_dims(np.stack(x[-n_step:]), 0)
      if x.ndim == 2:
        return np.pad(x, ((0, 0), (0, delta)))
      else:  # state: (1, N, 20) or action: (1, N, 4)
        return np.pad(x, ((0, 0), (0, delta), (0, 0)))
    mask_len = np.array([min(3 * len(self.s) - 1, n_step * 3 - 1)], np.int32)  # the last action is awalys padding
    rng, self.rng = jax.random.split(self.rng)
    action = self.model.predict(
      self.state,
      pad(self.s).astype(np.float32),
      pad(self.a).astype(np.int32),
      pad(self.rtg).astype(np.float32),
      pad(self.timestep).astype(np.int32),
      mask_len, rng, self.deterministic)[0]
    return action
  
  def __call__(self, state: train_state.TrainState, n_test: int = 10, rtg: int = 90, deterministic=False):
    self.state, self.deterministic = state, deterministic
    score = []
    for i in range(n_test):
      score.append(0)
      s = self.env.reset()
      done, timestep = False, 1
      self.s, self.a, self.rtg, self.timestep = [s], [np.zeros(self.action_dim)], [rtg], [1]
      while not done:
        a = self.get_action()
        s, r, done = self.env.step(a)
        self.s.append(s)
        self.a[-1] = a; self.a.append(np.zeros(self.action_dim))  # keep s, a, r in same length, but last action is padding
        self.rtg.append(max(self.rtg[-1] - r, 1))
        # self.rtg.append(max(self.rtg[-1] - r, 1))
        timestep = min(timestep + 1, self.model.cfg.max_timestep - 1)
        self.timestep.append(timestep)
        score[-1] += r
      print(f"epoch {i} with score {score[-1]}, timestep {len(self.s)}")
    return score

from utils.ckpt_manager import CheckpointManager
from dt_model import GPTConfig, GPT, TrainConfig
class LoadToEvaluate:
  def __init__(self, path_weights, load_step, auto_shoot: bool = True, show: bool = False, path_video_save_dir: str = None):
    ckpt_mngr = CheckpointManager(path_weights)
    load_info = ckpt_mngr.restore(load_step)
    params, cfg = load_info['params'], load_info['config']
    self.model = GPT(cfg=GPTConfig(**cfg))
    self.model.create_fns()
    state = self.model.get_state(TrainConfig(**cfg), train=False)
    self.state = state.replace(params=params, tx=None, opt_state=None)
    self.evaluator = Evaluator(self.model, game=cfg['game'], seed=cfg['seed'], auto_shoot=auto_shoot, show=show, path_video_save_dir=path_video_save_dir)
  
  def evaluate(self, n_test: int = 10, rtg: int = 90, deterministic: bool = True):
    result = self.evaluator(self.state, n_test=n_test, rtg=rtg, deterministic=deterministic)
    return result

if __name__ == '__main__':
  path_weights = r"../logs/DT_wty__Breakout__1__20240325_141559/ckpt"
  load_step = 10
  lte = LoadToEvaluate(path_weights, load_step)
  score = lte.evaluate(n_test=10, rtg=90, deterministic=False)
  print("score =", score)
  print(np.mean(score))
