import torch

def choose_device(device : str):

    if device == "cpu":
        return "cpu"
    
    elif device == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    
    elif device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    else:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"