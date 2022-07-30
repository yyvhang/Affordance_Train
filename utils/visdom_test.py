from matplotlib.pyplot import title
import numpy as np
import visdom

vis = visdom.Visdom(env='loss')

x, y = 0, 0
window= vis.line(
    X = np.array([x]),
    Y = np.array([y]),
    opts=dict(title = 'test loss')
)

for i in range(10000):
    x += i
    y += (i+2)
    vis.line(
        X = np.array([x]),
        Y = np.array([y]),
        win = window,
        update='append'
    )
