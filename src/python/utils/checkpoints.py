import os
import torch

def save_checkpoint(model, optimizer, epoch, folder_path, model_name, **kwargs):

    filename = os.path.join(folder_path, f'{model_name}_{str(epoch)}.pth')
    
    state = {
        'epoc': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    for key, value in kwargs.items():
        state[key] = value

    torch.save(state, filename)


def load_checkpoint(model, optimizer, epoch, folder_path, model_name):

    filename = os.path.join(folder_path, f'{model_name}_{epoch}.pth')

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    kwargs = {key: checkpoint[key] for key in checkpoint if key not in ['model_state_dict', 'optimizer_state_dict']}
    
    return model, optimizer, kwargs