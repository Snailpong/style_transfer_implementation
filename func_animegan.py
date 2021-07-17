import torch


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())

    return G.div(a * b * c * d)

def color_y(input):
    return 0.299 * input[:, 0] + 0.587 * input[:, 1] + 0.114 * input[:, 2]

def color_u(input, y):
    return (input[:, 2] - y) * 0.565

def color_v(input, y):
    return (input[:, 0] - y) * 0.713