import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import utils


'''
Possibly write some helper function which takes the entire dataset and creates one prototype per class in the dataset based on a subset of
the examples per class or from all examples. I think it could be of interest to see if these prototypes gets closer or further away from eachother.
Still trying to build intuition for this and if this idea even makes any sense.

What is most likely happening at validation/test time as we train the network further is that more and more segments gets closer to the positive prototype
regardless of if it is a segment corresponding to an event or not.
Why is this happening given our training procedure? 
'''

#Should the loaders be created in here instead? Based on some config stuff?
#Talk about this
def train(model, optimizer, loss_function, train_loader, val_loader, config, writer):
    
    if config.experiment.set.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    #Should this be done here or passed into this function?
    #Could be configs for more terminal flexibility
    optim = optimizer
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=config.experiment.train.scheduler_gamma,
                                                  step_size=config.experiment.train.scheduler_step_size)
    num_epochs = config.experiment.train.epochs
    
    best_model_path = config.experiment.path.best_model
    last_model_path = config.experiment.path.last_model
    train_loss = []
    #val_loss = []
    train_acc = []
    #val_acc = []
    #best_val_acc = 0.0
    model.to(device)
    
    eval_module = utils.load_module(config.experiment.eval.script_path)
    val_function = eval_module.load_help()
    val_precision = []
    val_recall = []
    val_fmeasure = []
    best_val_fmeasure = 0.0
    
    num_batches_tr = len(train_loader)
    #num_batches_val = len(val_loader)
    
    #As this grows just create a list i think over all diferent active styles
    if config.experiment.train.sampler == 'activequery':
        train_loader.batch_sampler.set_model(model)
        train_loader.batch_sampler.set_writer(writer)
    
    for epoch in range(num_epochs):
        
        if config.experiment.train.sampler == 'activequery':
            train_loader.batch_sampler.set_epoch(epoch)
        
        print('Epoch {}'.format(epoch))
        train_iterator = iter(train_loader)
        batch_nr = 0
        for batch in tqdm(train_iterator):
            optim.zero_grad()
            model.train()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            emb, x_out = model(x)
            #TODO: We should possibly handle the loss handle here differently?
            #This should most likely be some kind of argument no?
            #Or is this OK since we already are in an application specific training loop?
            tr_loss = loss_function(x_out, emb, y, config.experiment.train.n_shot, config)
            train_loss.append(tr_loss.item())
            smax = torch.nn.functional.softmax(x_out)
            y_pred = torch.argmax(smax, 1)
            tr_acc = torch.sum(y == y_pred)/config.experiment.train.batch_size
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
        
        if config.experiment.train.sampler == 'activequery':
            print('Maximum query entropy: {}  Minimum query entropy: {}'.format(
            train_loader.batch_sampler.max_entropy, train_loader.batch_sampler.min_entropy))
            writer.add_scalar('Entropy/minimum', train_loader.batch_sampler.min_entropy, epoch)
            writer.add_scalar('Entropy/maximum', train_loader.batch_sampler.max_entropy, epoch)
            
        lr_scheduler.step()
        
        #No dropouts in model for now, I think there is no difference between train and eval mode
        model.eval()
        scores = val_function(model, None, config, None, 'VAL')
        writer.add_scalar('Fmeasure/val', scores['fmeasure (percentage)'], epoch)
        writer.add_scalar('precision/val', scores['precision'], epoch)
        writer.add_scalar('recall/val', scores['recall'], epoch)
        if scores['fmeasure (percentage)'] > best_val_fmeasure:
            best_val_fmeasure = scores['fmeasure (percentage)']
            print("Saving the best model with validation fmeasure {}".format(best_val_fmeasure))
            best_state = model.state_dict()
            torch.save(best_state, best_model_path)
        
        
        '''
        Old validation routine
        
        val_iterator = iter(val_loader)
        for batch in tqdm(val_iterator):
            x, y = batch
            x = x.to(device)
            x_val = model(x)
            #TODO: ditto as above
            valid_loss, valid_acc = loss_function(x_val, y, config.experiment.train.n_shot)
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
        '''
    torch.save(model.state_dict(),last_model_path)

    return best_val_fmeasure, model, best_state

def load():
    return train
