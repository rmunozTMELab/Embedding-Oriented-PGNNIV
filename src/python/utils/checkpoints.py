import os
import torch

def save_checkpoint(model, optimizer, epoch, folder_path, end_flag=False, **kwargs):
    if end_flag:
        filename = os.path.join(folder_path, f'epoch_final.pth')
    else:
        filename = os.path.join(folder_path, f'epoch_{str(epoch)}.pth')
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_architecture': model.__dict__,  
    }
    
    for key, value in kwargs.items():
        state[key] = value

    torch.save(state, filename)



def load_checkpoint(model, optimizer, epoch, folder_path):

    filename = os.path.join(folder_path, f'epoch_{str(epoch)}.pth')
    checkpoint = torch.load(filename, weights_only=False)

    # model_architecture = checkpoint.get('model_architecture')
    
    # if model_class and model_architecture:
    #     model = model_class(**model_architecture)
    # else:
    #     raise ValueError("No se ha proporcionado la clase del modelo o los datos de la arquitectura.")

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    kwargs = {key: checkpoint[key] for key in checkpoint if key not in ['model_state_dict', 'optimizer_state_dict', 'model_architecture']}
    
    return model, optimizer, kwargs

