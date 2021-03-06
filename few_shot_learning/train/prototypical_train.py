import torch
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import torchaudio.transforms as Transforms
from tqdm import tqdm
import numpy as np
import utils
import datasets.semisupervised as semi
import datasets.semi_augment as semi_augment
import datasets.semisupervised_lazy as semi_lazy
import datasets.semisupervised_trainvalcv as semi_trainvalcv
import datasets.episodicchain as ec
import datasets.background as background
import csv

def train(model, optimizer, loss_function, train_loader, val_loader, config, writer, fold=None, class_map=None, class_dict=None, tr_cls_keys=None, val_cls_keys=None):
    
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
    
    if config.experiment.set.trainvalcv:
        val_function = eval_module.load_help_TrainValCV()
    else:
        val_function = eval_module.load_help()
    
    val_precision = []
    val_recall = []
    val_fmeasure = []
    best_val_fmeasure = 0.0
    
    num_batches_tr = len(train_loader)
    #num_batches_val = len(val_loader)
    if config.experiment.train.log_chain:
        chain_logger = ec.EpisodicChain(config, model, train_loader.dataset)
    
    #As this grows just create a list i think over all diferent active styles
    if config.experiment.train.sampler == 'activequery':
        train_loader.batch_sampler.set_model(model)
        train_loader.batch_sampler.set_writer(writer)
    
    if config.experiment.train.semi_supervised:
        if config.experiment.set.trainvalcv:
            semi_iterator = iter(semi_trainvalcv.get_semi_loader_TrainValCV(config, class_map, class_dict, tr_cls_keys, val_cls_keys))
        else:
            if config.experiment.train.semi_lazy:
                semi_iterator = iter(semi_lazy.get_semi_loader(config))
            else:
                semi_iterator = iter(semi.get_semi_loader(config))
                
        if config.experiment.train.semi_augment:
            semi_augment_loader = iter(semi_augment.get_augment_loader(config, class_map, class_dict, tr_cls_keys, val_cls_keys))
        
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
    
    if config.experiment.train.save_init_state:
        random_init_state = model.state_dict()
        torch.save(random_init_state, 'random_init.pth')
    
    #Mark TrainValCV
    
    for epoch in range(num_epochs):
        
        if config.experiment.train.sampler == 'activequery':
            train_loader.batch_sampler.set_epoch(epoch)
        
        print('Epoch {}'.format(epoch))
        train_iterator = iter(train_loader)
        
        
        
        
        for batch in tqdm(train_iterator):
            
            optim.zero_grad()
            model.train()
            
            x, y = batch
            
            #Can we simply do like this?
            if config.experiment.train.rotate:
                if np.random.random() > 0.5:
                    x = torch.transpose(x, 1, 2)
                    
            
            semi_x = None
            
            #Could potentially do more stuff here.
            #Can we do some bandit idea, exploration explotation kinda wibe. Perhaps reversed?
            #If we no longer get a lot of signals from the training data, how bout excluding that and just start doing unlabeled updates?
            #Can still choose unlabeled based on labeled.
            
            semi_delay_flag = False
            if epoch < config.experiment.train.semi_delay:
                semi_delay_flag = True
            else:
                semi_delay_flag = False
            

            if config.experiment.train.semi_supervised and not semi_delay_flag:
                
                semi_x, _ = next(semi_iterator)
                
                '''
                I am not so sure that using this energy multipler we got right now is that good of an idea!
                '''
                #Mix before specaugment.
                if config.experiment.train.mix_background:
                    
                    if config.experiment.train.background_other_source:
                        
                        '''
                        
                        TODO: Revisit this part. Perhaps include augmentation for support sets during training?
                        
                       
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
                
                
                '''
                TODO: Implement idea here about mixing on the few-shot samples on unlabeled samples.
                      (i): We will require some loader to get the samples for this I think.
                            (-): Should be able to sample from val, test, or both by config.
                      (ii): We should be able to include the unlabeled prototypes and queris augmented this way into the whole loss if we so desire
                            (-): How can we do this easily, just 'white wash' the u/p u/q pair (simply make it look like normal training data).
                                 I think we can earmark the pair/s we augment and just concat them to x and remove them from semi_x and we good.
                                 However, look at the order of samples in x and match that (if it even matters considering the loss).
                      (iii): We should be able to influence the parameters of a distribution from which we sample to determine how many of the u/p and u/q pairs we augment this way.
                '''
                
                if config.experiment.train.semi_augment:
                    #Just do something here for now, might wanna check into this more later on. Like what happens if we use all for example.
                    
                    
                    num_unlabeled = config.experiment.train.n_shot + config.experiment.train.n_query
                    num_semi = int(len(semi_x)/num_unlabeled)
                    num_augment = int(min(np.random.poisson(lam=int((num_semi)/5)), num_unlabeled))
                    to_augment = np.random.choice(list(range(num_semi)), num_augment, replace=False)
                    
                    
                    for i in to_augment:
                        
                        samples = next(semi_augment_loader)
                        for j in range(len(samples)):

                            if config.experiment.train.semi_augment_ediff:
                                E_x = torch.sum(semi_x[i*num_unlabeled+j]**2).item()
                                E_s = torch.sum(samples[j]**2).item()
                                E_diff = E_x/E_s
                            else:
                                E_diff = 1
                            if config.experiment.train.semi_augment_rw:
                                l = np.random.uniform(low=0.2, high=0.8)
                                w_x = l
                                w_s = 1 - l
                            else:
                                w_x = 0.5
                                w_s = 0.5
                            #TODO: Introduce some energy checking / matching here possibly.
                            semi_x[i*num_unlabeled+j] = w_x*semi_x[i*num_unlabeled+j] + w_s*E_diff*samples[j]
                            
                    if config.experiment.train.semi_augment_fl and num_augment != 0:
                        
                        semi_x = semi_x.numpy()
                        
                        tmp = []
                        to_delete = []
                        
                        for ix in to_augment:
                            to_delete.append(list(range(ix*num_unlabeled, ix*num_unlabeled+num_unlabeled)))
                            tmp.append(semi_x[ix*num_unlabeled:ix*num_unlabeled+num_unlabeled].copy())
                        
                        #shape (num_augment, num_unlabeled, frames, bins)
                        to_delete = np.array(to_delete)
                        to_delete = to_delete.reshape(-1)
                        tmp = np.array(tmp)
                        
                        semi_x = np.delete(semi_x, to_delete, 0)
                        semi_x = torch.tensor(semi_x)
                        
                        #Now to actually insert the augmented unlabeled samples into the labeled data and labels.
                        x = x.numpy()
                        y = y.numpy()
                        
                        num_samples = num_unlabeled
                        for i in range(num_augment):
                            
                            ix = [(config.experiment.train.k_way+i) * (j+1) for j in range(num_samples)]
                            x = np.insert(x, ix, tmp[i], axis=0)
                            y = np.insert(y, ix, 100+i, axis=0)
                            
                        x = torch.tensor(x)
                        y = torch.tensor(y)
                        
                        
                        
                if config.experiment.train.specaugment:
                    if config.experiment.train.specaugment_prototypes:
                        ix = np.array(list(range(len(semi_x))))
                    else:
                        n_shot = config.experiment.train.n_shot
                        n_query = config.experiment.train.n_query
                        n_samples = n_shot + n_query
                        tmp = [list(range(i*n_samples+n_shot, i*n_samples+n_samples)) for i in range(int(len(semi_x)/n_samples))]
                        ix = []
                        for l in tmp:
                            for e in l:
                                ix.append(e)
                        ix = np.array(ix)
                    
                    #Turn in to (batchsize, mel_bins, time_frames)
                    semi_x = torch.transpose(semi_x, 1, 2)
                    time_stretch_range = config.experiment.train.time_stretch_range
                    stretch_factor = 1 + np.random.uniform(-time_stretch_range, time_stretch_range)
                    if config.experiment.train.specaugment_timestretch:
                        semi_x = timeStretch(semi_x.type(torch.complex64), stretch_factor).type(torch.float)
                    if config.experiment.train.specaugment_iid_filters:
                        semi_x = semi_x.reshape(semi_x.shape[0], 1, semi_x.shape[1], semi_x.shape[2])
                    
                    '''
                    I think we currently let both the samples meant for prototype/query be time stretched as to make this change easier to start.
                    However, do no 
                    '''
                    
                    semi_x[ix] = freqMask(timeMask(semi_x[ix]))
                    if config.experiment.train.specaugment_iid_filters:
                        semi_x = semi_x.squeeze()
                    semi_x = torch.transpose(semi_x, 1, 2)
                semi_x = semi_x.to(device, dtype=torch.float)
                if config.experiment.train.multi:
                    semi_x = model(semi_x)
                else:
                    semi_x = model(semi_x)
            
            #Give more control over what from specaugment we want to use I think.
           # print(y)
            if config.experiment.train.specaugment:
                '''
                OBS: Order of samples no the same for the labeled data need some attention on this!
                     This should be simple though, just split it in half.
                '''
                if config.experiment.train.specaugment_prototypes:
                    ix = np.array(list(range(len(x))))
                else:
                    ix = np.array(list(range(config.experiment.train.n_shot*config.experiment.train.k_way, len(x))))
                
                x = torch.transpose(x, 1, 2)
                time_stretch_range = config.experiment.train.time_stretch_range
                stretch_factor = 1 + np.random.uniform(-time_stretch_range, time_stretch_range)
                if config.experiment.train.specaugment_timestretch:
                    x = timeStretch(x.type(torch.complex64), stretch_factor).type(torch.float)
                if config.experiment.train.specaugment_iid_filters:
                    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
                x[ix] = freqMask(timeMask(x[ix]))
                if config.experiment.train.specaugment_iid_filters:
                    x = x.squeeze()
                x = torch.transpose(x, 1, 2)
            x = x.to(device)
            y = y.to(device)
            if config.experiment.train.multi:
                x_out = model(x)
            else:
                x_out = model(x)
            
            if config.experiment.train.embedding_propagation:
                x_out = torch.mm(global_consistency(get_similarity_matrix(x_out, 1), alpha=0.5), x_out)
                   
            tr_loss, tr_acc = loss_function(x_out, y, config.experiment.train.n_shot, config, semi_x)
            train_loss.append(tr_loss.item())
            train_acc.append(tr_acc.item())

            tr_loss.backward()
            optim.step()
            
            '''
            Engage episodic logger here
            '''
            if config.experiment.train.log_chain:
                chain_logger.log_episode(y, x_out)
            
            if config.experiment.train.save_first_batch_state:
                first_back_state = model.state_dict()
                torch.save(first_back_state, 'first_back.pth')
            
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
        
        if config.experiment.set.trainvalcv:
            scores = val_function(model, None, config, None, class_map, class_dict, tr_cls_keys, val_cls_keys, fold, 'VAL')
        else:
            scores = val_function(model, None, config, None, 'VAL')
            
        if config.experiment.train.eval_test:
            test_scores = val_function(model, None, config, None, 'TEST')
            writer.add_scalar('Fmeasure_te/test', test_scores['fmeasure (percentage)'], epoch)
            writer.add_scalar('precision_te/test', test_scores['precision'], epoch)
            writer.add_scalar('recall_te/test', test_scores['recall'], epoch)
            
        #Save every epoch  
        if config.experiment.train.log_chain:
            chain_logger.save_chain(scores)
        
        writer.add_scalar('Fmeasure/val', scores['fmeasure (percentage)'], epoch)
        writer.add_scalar('precision/val', scores['precision'], epoch)
        writer.add_scalar('recall/val', scores['recall'], epoch)
        if scores['fmeasure (percentage)'] > best_val_fmeasure:
            best_val_fmeasure = scores['fmeasure (percentage)']
            print("Saving the best model with validation fmeasure {}".format(best_val_fmeasure))
            best_state = model.state_dict()
            torch.save(best_state, best_model_path)
            
            #TODO: Work with new fold file names here!!!!!!!
            #Save best validation model predictions.
            if config.experiment.set.trainvalcv:
                val_file = open('VAL_out_fold_'+str(fold)+'.csv', newline='')
                pp_val_file = open('PP_VAL_out_fold_'+str(fold)+'.csv', newline='')
                best_file = open('BEST_VAL_out_fold_'+str(fold)+'.csv', 'w', newline='')
                pp_best_file = open('PP_BEST_VAL_out_fold_'+str(fold)+'.csv', 'w', newline='')
            else:
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
