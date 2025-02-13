class Animation:
  # allow for html embedded animations, and import matplotlib
  from matplotlib import rc
  rc('animation', html='jshtml')

  import matplotlib.pyplot as plt
  import matplotlib.animation as anim

  def __init__(self, names: list[str], xlim: tuple, ylim: tuple):
    if not isinstance(names, list):
      raise TypeError("names must be a list of strings")
    for name in names:
      if not isinstance(name, str):
        raise TypeError("names must be a list of strings")
    if not isinstance(xlim, tuple) or len(xlim) != 2:
      raise TypeError("xlim must be a 2-tuple of floats")
    if not isinstance(ylim, tuple) or len(ylim) != 2:
      raise TypeError("ylim must be a 2-tuple of floats")
    
    # set up plot
    self.fig, self.ax = Animation.plt.subplots()

    self.lines = []
    for name in names:
      line, = self.ax.plot([], [], lw = 3)
      self.lines.append(line)
    
    self.ax.legend([ line for line in self.lines ], [ name for name in names ])
    self.ax.set_xlim(xlim[0], xlim[1])
    self.ax.set_ylim(ylim[0], ylim[1])

    # for some reason subplots() is showing a plot, so hide as not required
    self.plt.close()
  
  def set_data(self, lineNum: int, x: list[float], y: list[float]):
    self.lines[lineNum].set_data(x, y)
  
  # func: (frame: int) -> a.lines
  def set_anim_func(self, func):
    self.func = func
  
  # frames = number of frames
  # interval = time between frames in ms
  def make_anim(self, frames: int, interval: int, blit: bool):
    self.anim = Animation.anim.FuncAnimation(self.fig,
                                             self.func,
                                             frames=frames,
                                             interval=interval,
                                             blit=blit)
    return self.anim

  # unable to have a "show()" function with html embed, so just save the output
  # of make_anim to a variable and type the name of that variable to show
  # 
  # ex:
  #   >>> ani = a.make_anim(frames=100, interval=20, blit=True)
  #   >>> ani

  def save(self, path: str):
    if not isinstance(path, str):
      raise TypeError("path must be a string")
    self.anim.save(path)