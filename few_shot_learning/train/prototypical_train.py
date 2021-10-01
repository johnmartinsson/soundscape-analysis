import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np



'''
This code is present in the notebook as some sort of setup before training.
Some of these objects are passed into the training loop.


data_gen = Datagen(cfg)
X_train, Y_train, X_val, Y_val = data_gen.generate_train()
X_tr = torch.tensor(X_train)
Y_tr = torch.LongTensor(Y_train)
X_val = torch.tensor(X_val)
Y_val = torch.LongTensor(Y_val)
samples_per_cls = cfg.train.n_shot + cfg.train.n_query
batch_size_tr = samples_per_cls * cfg.train.k_way
batch_size_vd = batch_size_tr

num_batches_tr = len(Y_train)//batch_size_tr
num_batches_vd = len(Y_val)//batch_size_vd


samplr_train = RandomEpisodicSampler(Y_train,num_batches_tr,cfg.train.k_way, cfg.train.n_shot, cfg.train.n_query)
samplr_valid = RandomEpisodicSampler(Y_val,num_batches_vd,cfg.train.k_way,cfg.train.n_shot, cfg.train.n_query)

train_dataset = torch.utils.data.TensorDataset(X_tr,Y_tr)
valid_dataset = torch.utils.data.TensorDataset(X_val,Y_val)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_sampler=samplr_train,num_workers=0,pin_memory=True,shuffle=False)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_sampler=samplr_valid,num_workers=0,pin_memory=True,shuffle=False)

'''

#Should the loaders be created in here instead? Based on some config stuff?
#Talk about this
def train(model, optimizer, loss_function, train_loader, val_loader, config):
    
    if config.set.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    #Should this be done here or passed into this function?
    #Could be configs for more terminal flexibility
    optim = optimizer
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=config.train.scheduler_gamma,
                                                  step_size=config.train.scheduler_step_size)
    num_epochs = config.train.epochs
    
    best_model_path = config.path.best_model
    last_model_path = config.path.last_model
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_acc = 0.0
    model.to(device)
    
    num_batches_tr = len(train_loader)
    num_batches_val = len(val_loader)

    writer = SummaryWriter(log_dir=config.train.artifacts_path)

    for epoch in range(num_epochs):
        
        print('Epoch {}'.format(epoch))
        train_iterator = iter(train_loader)
        batch_nr = 0
        for batch in tqdm(train_iterator):
            optim.zero_grad()
            model.train()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x_out = model(x)
            #TODO: We should possibly handle the loss handle here differently?
            #This should most likely be some kind of argument no?
            #Or is this OK since we already are in an application specific training loop?
            tr_loss, tr_acc = loss_function(x_out, y, config.train.n_shot)
            train_loss.append(tr_loss.item())
            train_acc.append(tr_acc.item())
            
            #Did not end up like i wanted it to.
            #Wanted to plot the loss/acc per batch in an epoch.
            #Instead got every batch as a separate card.
            #Also it overwrites every epoch

            #writer.add_scalar('Loss/Batch'+str(batch_nr), tr_loss, batch_nr)
            #writer.add_scalar('Accuracy/Batch'+str(batch_nr), tr_loss, batch_nr)

            batch_nr += 1

            tr_loss.backward()
            optim.step()
            
        avg_loss_tr = np.mean(train_loss[-num_batches_tr:])
        avg_acc_tr = np.mean(train_acc[-num_batches_tr:])
        print('Average train loss: {}  Average training accuracy: {}'.format(avg_loss_tr,avg_acc_tr))

        writer.add_scalar('Loss/train', avg_loss_tr, epoch)
        writer.add_scalar('Accuracy/train', avg_acc_tr, epoch)
        
        lr_scheduler.step()
        
        #No dropouts in model for now, I think there is no difference between train and eval mode
        model.eval()
        val_iterator = iter(val_loader)
        for batch in tqdm(val_iterator):
            x, y = batch
            x = x.to(device)
            x_val = model(x)
            #TODO: ditto as above
            valid_loss, valid_acc = loss_function(x_val, y, config.train.n_shot)
            val_loss.append(valid_loss.item())
            val_acc.append(valid_acc.item())
        avg_loss_val = np.mean(val_loss[-num_batches_val:])
        avg_acc_val = np.mean(val_acc[-num_batches_val:])

        writer.add_scalar('Loss/val', avg_loss_val, epoch)
        writer.add_scalar('Accuracy/val', avg_acc_val, epoch)
        
        print ('Epoch {}, Validation loss {:.4f}, Validation accuracy {:.4f}'.format(epoch,avg_loss_val,avg_acc_val))
        if avg_acc_val > best_val_acc:
            print("Saving the best model with valdation accuracy {}".format(avg_acc_val))
            best_val_acc = avg_acc_val
            best_state = model.state_dict()
            torch.save(model.state_dict(),best_model_path)
    torch.save(model.state_dict(),last_model_path)

    return best_val_acc, model, best_state

def load():
    return train
