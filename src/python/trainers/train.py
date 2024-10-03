import torch
import torch.optim as optim

from .eval import loss_function

def train_epoch(model, optimizer, X_train, y_train, D):

    model.train()

    y_pred, K_pred = model(X_train.values)

    loss, e_loss, pi1_loss, pi2_loss, pi3_loss = loss_function(X_train, y_train, y_pred, K_pred, D)

    optimizer.zero_grad() 
    loss.backward(retain_graph=True)
    optimizer.step()

    return loss, e_loss, pi1_loss, pi2_loss, pi3_loss
