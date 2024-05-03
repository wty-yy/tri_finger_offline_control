import numpy as np

class Env:
  def __init__(self, seed: int = 42):
    """
    This is a custom Simple Env use `reset()` and 
    `step(action)` to interact with it.
    Args:
      - seed (int): The env seed.
    """
    # TODO: SET seed
    self.reset()
  
  def reset(self):
    # TODO: Calcuate initial state
    # return s
    raise NotImplementedError()  # Remove this
  
  def step(self, action):
    # TODO: Do one step, return state, reward, terminal
    # return s, r, terminal
    raise NotImplementedError()  # Remove this
    
if __name__ == '__main__':
  ### Test Env ###
  env = Env(seed=42)
  s = env.reset()
  done = False
  while not done:
    action = np.random.randint(0, 2, (4,))
    s, r, done = env.step(action)
