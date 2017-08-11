import numpy as np

x = np.array([[1,2,3],[1,2,3],[1,2,3]])
y = np.array([[5,6,7,8,9,9,0,2,3,4],[5,6,7,8,9,9,0,2,3,4],[5,6,7,8,9,9,0,2,3,4]])

max_len = max(x.shape[1], y.shape[1])
print(max_len)

if max_len - x.shape[1]:
    x = np.pad(
        x,
        ((0,0), (0, max_len - x.shape[1])),
        'constant'
    )

if max_len - y.shape[1]:
    y = np.pad(
        y,
        (0, max_len - y.shape[1]),
        'constant'
    )

print(x)
print(y)