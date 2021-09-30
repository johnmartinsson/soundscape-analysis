import torch
from tqdm import tqdm

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
def train(model, train_loader, val_loader, config, num_batches_tr, num_batches_val):
    
    if config.set.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    #Should this be done here or passed into this function?
    #Could be configs for more terminal flexibility
    optim = torch.optim.Adam(model.parameters(), lr=config.train.lr_rate)
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
    
    for epoch in range(num_epochs):
        
        print('Epoch {}'.format(epoch))
        train_iterator = iter(train_loader)
        for batch in tqdm(train_iterator):
            optim.zero_grad()
            model.train()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x_out = model(x)
            tr_loss, tr_acc = prototypical_loss(x_out, y, config.train.n_shot)
            train_loss.append(tr_loss.item())
            train_acc.append(tr_acc.item())
            
            tr_loss.backward()
            optim.step()
            
        avg_loss_tr = np.mean(train_loss[-num_batches_tr:])
        avg_acc_tr = np.mean(train_acc[-num_batches_tr:])
        print('Average train loss: {}  Average training accuracy: {}'.format(avg_loss_tr,avg_acc_tr))
        
        lr_scheduler.step()
        
        #No dropouts in model for now, I think there is no difference between train and eval mode
        model.eval()
        val_iterator = iter(val_loader)
        for batch in tqdm(val_iterator):
            x, y = batch
            x = x.to(device)
            x_val = model(x)
            valid_loss, valid_acc = prototypical_loss(x_val, y, config.train.n_shot)
            val_loss.append(valid_loss.item())
            val_acc.append(valid_acc.item())
        avg_loss_val = np.mean(val_loss[-num_batches_val:])
        avg_acc_val = np.mean(val_acc[-num_batches_val:])
        
        print ('Epoch {}, Validation loss {:.4f}, Validation accuracy {:.4f}'.format(epoch,avg_loss_val,avg_acc_val))
        if avg_acc_val > best_val_acc:
            print("Saving the best model with valdation accuracy {}".format(avg_acc_val))
            best_val_acc = avg_acc_val
            best_state = model.state_dict()
            torch.save(model.state_dict(),best_model_path)
    torch.save(model.state_dict(),last_model_path)

    return best_val_acc, model, best_state