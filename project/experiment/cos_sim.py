import numpy as np

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

x = np.array([0.789, 0.515, 0.335, 0])
y = np.array([0.832, 0.555, 0, 0])
print(f'cos類似度: {cos_sim(x, y)}')