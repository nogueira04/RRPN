import pickle
import numpy as np

with open("/home/live/RRPNv2/RRPN/data/nucoco/proposals/proposals_mini_train.pkl", 'rb') as f:
    proposals = pickle.load(f)
    print(proposals.keys())  # Should contain 'boxes', 'scores', 'ids'
    print(len(proposals['boxes']), len(proposals['scores']), len(proposals['ids']))

with open("/home/live/RRPNv2/RRPN/data/nucoco/proposals/proposals_mini_train.pkl", 'rb') as f:
    proposals = pickle.load(f)
    for boxes in proposals['boxes']:
        if np.isnan(boxes).any() or np.isinf(boxes).any():
            raise ValueError("Proposals contain NaN or Inf values")
    print("Proposals are valid")

