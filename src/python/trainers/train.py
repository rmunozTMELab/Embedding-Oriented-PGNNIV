import torch
import torch.optim as optim
from utils.checkpoints import load_checkpoint, save_checkpoint
from vecopsciml.utils import Tensor

from .eval import loss_function

def train_epoch(model, optimizer, X_train, y_train, D):

    model.train()

    y_pred, K_pred = model(X_train.values)

    loss, e_loss, pi1_loss, pi2_loss, pi3_loss = loss_function(X_train, y_train, y_pred, K_pred, D)

    optimizer.zero_grad() 
    loss.backward(retain_graph=True)
    optimizer.step()

    return loss, e_loss, pi1_loss, pi2_loss, pi3_loss


def test_epoch(model, X_test, y_test, D):

    y_pred, K_pred = model(X_test.values)
    loss, e_loss, pi1_loss, pi2_loss, pi3_loss = loss_function(X_test, y_test, y_pred, K_pred, D)

    return loss, e_loss, pi1_loss, pi2_loss, pi3_loss


def train_loop(model, optimizer, X_train, y_train, X_test, y_test, D, start_epoch, n_epochs, batch_size, model_results_path):

    if start_epoch > 0:

        resume_epoch = start_epoch
        model, optimizer, lists = load_checkpoint(model, optimizer, resume_epoch, model_results_path)

        train_total_loss_list = lists['train_total_loss_list']
        train_e_loss_list = lists['train_e_loss_list']
        train_pi1_loss_list = lists['train_pi1_loss_list']
        train_pi2_loss_list = lists['train_pi2_loss_list']
        train_pi3_loss_list = lists['train_pi3_loss_list']

        test_total_loss_list = lists['test_total_loss_list']
        test_e_loss_list = lists['test_e_loss_list']
        test_pi1_loss_list = lists['test_pi1_loss_list']
        test_pi2_loss_list = lists['test_pi2_loss_list']
        test_pi3_loss_list = lists['test_pi3_loss_list']

    else:

        train_total_loss_list = []
        train_total_MSE_list = []
        train_e_loss_list = []
        train_pi1_loss_list = []
        train_pi2_loss_list = []
        train_pi3_loss_list = []

        test_total_loss_list = []
        test_total_MSE_list = []
        test_e_loss_list = []
        test_pi1_loss_list = []
        test_pi2_loss_list = []
        test_pi3_loss_list = []


    for epoch_i in range(start_epoch, n_epochs):
        for batch_start in range(0, X_train.values.shape[0], batch_size):

            X_batch = Tensor(X_train.values[batch_start:batch_start+batch_size], space_dimension=X_train.space_dim, order=X_train.order)
            y_batch = Tensor(y_train.values[batch_start:batch_start+batch_size], space_dimension=y_train.space_dim, order=y_train.order)

            loss, e_loss, pi1_loss, pi2_loss, pi3_loss = train_epoch(model, optimizer, X_batch, y_batch, D)

        train_total_loss_list.append(loss.item())
        train_e_loss_list.append(e_loss.item())
        train_pi1_loss_list.append(pi1_loss.item())
        train_pi2_loss_list.append(pi2_loss.item())
        train_pi3_loss_list.append(pi3_loss.item())
        
        loss_test, e_loss_test, pi1_loss_test, pi2_loss_test, pi3_loss_test = test_epoch(model, X_test, y_test, D)

        test_total_loss_list.append(loss_test.item())
        test_e_loss_list.append(e_loss_test.item())
        test_pi1_loss_list.append(pi1_loss_test.item())
        test_pi2_loss_list.append(pi2_loss_test.item())
        test_pi3_loss_list.append(pi3_loss_test.item())

        if epoch_i % 10 == 0:
            save_checkpoint(model, optimizer, epoch_i, model_results_path, 
                            train_total_loss_list=train_total_loss_list, train_e_loss_list=train_e_loss_list, 
                            train_pi1_loss_list=train_pi1_loss_list, train_pi2_loss_list=train_pi2_loss_list, train_pi3_loss_list=train_pi3_loss_list,
                            test_total_loss_list=test_total_loss_list, test_e_loss_list=test_e_loss_list,
                            test_pi1_loss_list=test_pi1_loss_list, test_pi2_loss_list=test_pi2_loss_list, test_pi3_loss_list=test_pi3_loss_list
                            )

    save_checkpoint(model, optimizer, epoch_i, model_results_path, end_flag=True,
                    train_total_loss_list=train_total_loss_list, train_e_loss_list=train_e_loss_list, 
                    train_pi1_loss_list=train_pi1_loss_list, train_pi2_loss_list=train_pi2_loss_list, train_pi3_loss_list=train_pi3_loss_list,
                    test_total_loss_list=test_total_loss_list, test_e_loss_list=test_e_loss_list,
                    test_pi1_loss_list=test_pi1_loss_list, test_pi2_loss_list=test_pi2_loss_list, test_pi3_loss_list=test_pi3_loss_list
                    )
    