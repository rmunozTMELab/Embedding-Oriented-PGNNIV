import torch
from utils.checkpoints import load_checkpoint, save_checkpoint
from vecopsciml.utils import TensOps

from .eval import loss_function

def train_epoch(model, optimizer, X_train, y_train, f_true, D):

    model.train()

    y_pred, u_pred, K_pred = model(X_train)

    loss, e_loss, pi1_loss, pi2_loss, pi3_loss = loss_function(X_train, y_train, y_pred, u_pred, K_pred, f_true, D)

    optimizer.zero_grad() 
    loss.backward(retain_graph=True)
    optimizer.step()

    return loss, e_loss, pi1_loss, pi2_loss, pi3_loss


def test_epoch(model, X_test, y_test, f_test, D):

    y_pred, u_pred, K_pred = model(X_test)
    loss, e_loss, pi1_loss, pi2_loss, pi3_loss = loss_function(X_test, y_test, y_pred, u_pred, K_pred, f_test, D)

    return loss, e_loss, pi1_loss, pi2_loss, pi3_loss


def train_loop(model, optimizer, n_checkpoints, X_train, y_train, X_test, y_test, f_train, f_test,
               D, start_epoch, n_epochs, batch_size, model_results_path, device, new_lr=None):

    print("Start training")

    if start_epoch > 0:
        
        print(f'Starting from a checkpoint. Epoch {start_epoch}.')
        resume_epoch = start_epoch
        model, optimizer, lists = load_checkpoint(model, optimizer, resume_epoch, model_results_path)

        if new_lr != None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

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

    
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    for epoch_i in range(start_epoch, n_epochs):
        for batch_start in range(0, N_train, batch_size):

            X_batch = X_train[batch_start:(batch_start+batch_size)].to(device)
            y_batch = TensOps(y_train.values[batch_start:(batch_start+batch_size)].to(device), space_dimension=y_train.space_dim, contravariance=y_train.order[0], covariance=y_train.order[1])
            f_batch = TensOps(f_train.values[batch_start:(batch_start+batch_size)].to(device), space_dimension=f_train.space_dim, contravariance=f_train.order[0], covariance=f_train.order[1])

            loss, e_loss, pi1_loss, pi2_loss, pi3_loss = train_epoch(model, optimizer, X_batch, y_batch, f_batch, D)

        train_total_loss_list.append(loss.item()/batch_size)
        train_e_loss_list.append(e_loss.item()/batch_size)
        train_pi1_loss_list.append(pi1_loss.item()/batch_size)
        train_pi2_loss_list.append(pi2_loss.item()/batch_size)
        train_pi3_loss_list.append(pi3_loss.item()/batch_size)
        
        loss_test, e_loss_test, pi1_loss_test, pi2_loss_test, pi3_loss_test = test_epoch(model, X_test, y_test, f_test, D)

        test_total_loss_list.append(loss_test.item()/N_test)
        test_e_loss_list.append(e_loss_test.item()/N_test)
        test_pi1_loss_list.append(pi1_loss_test.item()/N_test)
        test_pi2_loss_list.append(pi2_loss_test.item()/N_test)
        test_pi3_loss_list.append(pi3_loss_test.item()/N_test)

        if epoch_i % (1 if n_epochs < 100 else (10 if n_epochs <= 1000 else 100)) == 0:
            # memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            # print(f"Epoch {epoch_i}, Memory Usage: {memory_usage_mb:.2f} MB")
            print(f'Epoch {epoch_i}, Train loss: {loss.item()/batch_size:.3e}, Test loss: {loss_test.item()/N_test:.3e}, MSE(e): {e_loss.item()/batch_size:.3e}, MSE(pi1): {pi1_loss.item()/batch_size:.3e}, MSE(pi2): {pi2_loss.item()/batch_size:.3e}, MSE(pi3): {pi3_loss.item()/batch_size:.3e}')



        if epoch_i % (int(n_epochs/n_checkpoints)) == 0:
            save_checkpoint(model, optimizer, epoch_i, model_results_path, 
                            train_total_loss_list=train_total_loss_list, train_e_loss_list=train_e_loss_list, 
                            train_pi1_loss_list=train_pi1_loss_list, train_pi2_loss_list=train_pi2_loss_list, train_pi3_loss_list=train_pi3_loss_list,
                            test_total_loss_list=test_total_loss_list, test_e_loss_list=test_e_loss_list,
                            test_pi1_loss_list=test_pi1_loss_list, test_pi2_loss_list=test_pi2_loss_list, test_pi3_loss_list=test_pi3_loss_list
                            )

    print("\nProceso finalizado después de", n_epochs, "épocas\n")
    
    save_checkpoint(model, optimizer, epoch_i, model_results_path, end_flag=True,
                    train_total_loss_list=train_total_loss_list, train_e_loss_list=train_e_loss_list, 
                    train_pi1_loss_list=train_pi1_loss_list, train_pi2_loss_list=train_pi2_loss_list, train_pi3_loss_list=train_pi3_loss_list,
                    test_total_loss_list=test_total_loss_list, test_e_loss_list=test_e_loss_list,
                    test_pi1_loss_list=test_pi1_loss_list, test_pi2_loss_list=test_pi2_loss_list, test_pi3_loss_list=test_pi3_loss_list
                    )
    
    
    