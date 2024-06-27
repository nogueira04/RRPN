import pickle

def inspect_proposals(proposal_file):
    with open(proposal_file, 'rb') as f:
        proposals = pickle.load(f)
    
    print(f"Number of images with proposals: {len(proposals['ids'])}")
    print("First 5 image IDs:", proposals['ids'])

inspect_proposals("/home/live/RRPNv2/RRPN/data/nucoco/proposals/proposals_mini_train.pkl")

