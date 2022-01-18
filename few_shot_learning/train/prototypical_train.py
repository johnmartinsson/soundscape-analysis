import torch
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import torchaudio.transforms as Transforms
from tqdm import tqdm
import numpy as np
import utils
import datasets.semisupervised as semi
import datasets.semisupervised_lazy as semi_lazy
import datasets.background as background
import csv

def train(model, optimizer, loss_function, train_loader, val_loader, config, writer):
    
    if config.experiment.set.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    #Should this be done here or passed into this function?
    #Could be configs for more terminal flexibility
    optim = optimizer
    
    #TODO: Add config option
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
    
    if config.experiment.train.semi_supervised:
        if config.experiment.train.semi_lazy:
            semi_iterator = iter(semi_lazy.get_semi_loader(config))
        else:
            semi_iterator = iter(semi.get_semi_loader(config))
        
    if config.experiment.train.background_other_source:
        background_iterator = iter(background.get_background_loader(config))
        
    if config.experiment.train.specaugment:
        timeStretch = Transforms.TimeStretch(n_freq = config.experiment.features.n_mels)
        if config.experiment.train.specaugment_iid_filters:
            timeMask = Transforms.TimeMasking(config.experiment.train.time_mask_range, iid_masks = True)
            freqMask = Transforms.FrequencyMasking(config.experiment.train.freq_mask_range, iid_masks = True)
        else:
            timeMask = Transforms.TimeMasking(config.experiment.train.time_mask_range)
            freqMask = Transforms.FrequencyMasking(config.experiment.train.freq_mask_range)
    
    #
    
    for epoch in range(num_epochs):
        
        if config.experiment.train.sampler == 'activequery':
            train_loader.batch_sampler.set_epoch(epoch)
        
        print('Epoch {}'.format(epoch))
        train_iterator = iter(train_loader)
        
        
        
        
        for batch in tqdm(train_iterator):
            
            optim.zero_grad()
            model.train()
            
            x, y = batch
            
            semi_x = None
            #Can VRAM handle this?
            if config.experiment.train.semi_supervised:
                
                semi_x, _ = next(semi_iterator)
                
                '''
                I am not so sure that using this energy multipler we got right now is that good of an idea!
                '''
                #Mix before specaugment.
                if config.experiment.train.mix_background:
                    
                    if config.experiment.train.background_other_source:
                        
                        '''
                        So how should this be done?
                        (i): Draw one background sample and augment all of the data in the batch with said sample.
                        (ii): Draw one background per labeled and unlabeled sample in the batch respectively.
                        '''
                        
                        #Type (i):
                        
                        bgr_sample, _ = next(background_iterator)
                        
                        for i in range(len(x)):
                            if config.experiment.train.background_ediff:
                                E_x = torch.sum(x[i]**2).item()
                                E_s = torch.sum(bgr_sample**2).item()
                                E_diff = E_x/E_s
                            else:
                                E_diff = 1
                            #TODO: Add this to config
                            alpha = 0.1
                            bgr_lambda = np.random.uniform(low=0, high=alpha)
                            x[i] = (1-bgr_lambda)*x[i] + bgr_lambda*E_diff*bgr_sample
                        
                        if semi_x is not None:
                            for i in range(len(semi_x)):
                                if config.experiment.train.background_ediff:
                                    E_x = torch.sum(semi_x[i]**2).item()
                                    E_s = torch.sum(bgr_sample**2).item()
                                    E_diff = E_x/E_s
                                else:
                                    E_diff = 1
                                #TODO: Add this to config
                                alpha = 0.1
                                bgr_lambda = np.random.uniform(low=0, high=alpha)
                                semi_x[i] = (1-bgr_lambda)*semi_x[i] + bgr_lambda*E_diff*bgr_sample
                                
                                
                    else:
                    
                        for i in range(len(x)):

                            bgr_sample = semi_x[np.random.choice(len(semi_x))]
                            #TODO: Do this for real
                            if config.experiment.train.background_ediff:
                                E_x = torch.sum(x[i]**2).item()
                                E_s = torch.sum(bgr_sample**2).item()
                                E_diff = E_x/E_s
                            else:
                                E_diff = 1
                            #TODO: Add this to config
                            alpha = 0.1
                            bgr_lambda = np.random.uniform(low=0, high=alpha)
                            x[i] = (1-bgr_lambda)*x[i] + bgr_lambda*E_diff*bgr_sample
                        
                if config.experiment.train.specaugment:
                    semi_x = torch.transpose(semi_x, 1, 2)
                    time_stretch_range = config.experiment.train.time_stretch_range
                    stretch_factor = 1 + np.random.uniform(-time_stretch_range, time_stretch_range)
                    semi_x = timeStretch(semi_x.type(torch.complex64), stretch_factor).type(torch.float)
                    if config.experiment.train.specaugment_iid_filters:
                        semi_x = semi_x.reshape(semi_x.shape[0], 1, semi_x.shape[1], semi_x.shape[2])
                    semi_x = freqMask(timeMask(semi_x))
                    if config.experiment.train.specaugment_iid_filters:
                        semi_x = semi_x.squeeze()
                    semi_x = torch.transpose(semi_x, 1, 2)
                semi_x = semi_x.to(device, dtype=torch.float)
                semi_x = model(semi_x)
            
            #Give more control over what from specaugment we want to use I think.
            if config.experiment.train.specaugment:
                x = torch.transpose(x, 1, 2)
                time_stretch_range = config.experiment.train.time_stretch_range
                stretch_factor = 1 + np.random.uniform(-time_stretch_range, time_stretch_range)
                x = timeStretch(x.type(torch.complex64), stretch_factor).type(torch.float)
                if config.experiment.train.specaugment_iid_filters:
                    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
                x = freqMask(timeMask(x))
                if config.experiment.train.specaugment_iid_filters:
                    x = x.squeeze()
                x = torch.transpose(x, 1, 2)
            x = x.to(device)
            y = y.to(device)
            x_out = model(x)
            
            if config.experiment.train.embedding_propagation:
                x_out = torch.mm(global_consistency(get_similarity_matrix(x_out, 1), alpha=0.5), x_out)
                   
            tr_loss, tr_acc = loss_function(x_out, y, config.experiment.train.n_shot, config, semi_x)
            train_loss.append(tr_loss.item())
            train_acc.append(tr_acc.item())

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
            
            #Save best validation model predictions.
            val_file = open('VAL_out.csv', newline='')
            pp_val_file = open('PP_VAL_out.csv', newline='')
            best_file = open('BEST_VAL_out.csv', 'w', newline='')
            pp_best_file = open('PP_BEST_VAL_out.csv', 'w', newline='')
            
            csv_reader = csv.reader(val_file, delimiter=',')
            pp_csv_reader = csv.reader(pp_val_file, delimiter=',')
            csv_writer = csv.writer(best_file, delimiter=',')
            pp_csv_writer = csv.writer(pp_best_file, delimiter=',')
            
            for row in csv_reader:
                csv_writer.writerow(row)
            for row in pp_csv_reader:
                pp_csv_writer.writerow(row)
            
            val_file.close()
            pp_val_file.close()
            best_file.close()
            pp_best_file.close()
        
    torch.save(model.state_dict(),last_model_path)

    return best_val_fmeasure, model, best_state

'''
Embedding propagation code.
I am somewhat unsure about this. The code is taken from the repo posten in the paper.
Is the code a reflection of the procedure posted in the paper? I don't know.
Should we write our own code. Have been playing around with this in a notebook.
'''

def get_similarity_matrix(x, rbf_scale):
    b, c = x.size()
    sq_dist = ((x.view(b, 1, c) - x.view(1, b, c))**2).sum(-1) / np.sqrt(c)
    mask = sq_dist != 0
    sq_dist = sq_dist / sq_dist[mask].std()
    weights = torch.exp(-sq_dist * rbf_scale)
    mask = torch.eye(weights.size(1), dtype=torch.bool, device=weights.device)
    weights = weights * (~mask).float()
    return weights

def global_consistency(weights, alpha=1, norm_prop=False):
    """Implements D. Zhou et al. "Learning with local and global consistency". (Same as in TPN paper but without bug)
    Args:
        weights: Tensor of shape (n, n). Expected to be exp( -d^2/s^2 ), where d is the euclidean distance and
            s the scale parameter.
        labels: Tensor of shape (n, n_classes)
        alpha: Scaler, acts as a smoothing factor
    Returns:
        Tensor of shape (n, n_classes) representing the logits of each classes
    """
    n = weights.shape[1]
    identity = torch.eye(n, dtype=weights.dtype, device=weights.device)
    isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(weights, dim=-1))
    # checknan(laplacian=isqrt_diag)
    S = weights * isqrt_diag[None, :] * isqrt_diag[:, None]
    # checknan(normalizedlaplacian=S)
    propagator = identity - alpha * S
    propagator = torch.inverse(propagator[None, ...])[0]
    # checknan(propagator=propagator)
    if norm_prop:
        propagator = F.normalize(propagator, p=1, dim=-1)
    return propagator

def load():
    return train
