import torch 
from ..gnn import * 
from ..comparm import * 
import pickle,os,tempfile, shutil, zipfile, time, math, time, tqdm 
from rdkit import Chem
from rdkit.Chem import rdmolfiles 
from torch.utils.data import Dataset,DataLoader
from datetime import datetime      
from .Dataset import *
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
import torch.nn.utils as utils
from .Model import * 
from ..scores.Scoringfunction import * 

class MolGen_Model_RL:
    def __init__(self,prior_modelname='',agent_modelname='',loadtype='Minloss',**kwargs):
        self.prior_modelname=prior_modelname
        self.agent_modelname=prior_modelname+'_RL'
        self.training_history=[]
        self.batch_train_loss_dict={''}
        self.n_train_steps=None
        self.min_rl_loss={'scores':0,'loss':1e20}
        self.epochs=0 
        self.node_add_model_prior=None
        self.ring_model_prior=None
        self.node_connect_model_prior=None
        self.node_add_model_agent=None
        self.node_connect_model_agent=None
        self.ring_model_agent=None

        self.optimizer=None
        self.scheduler=None

        if not os.path.exists(f'./{self.agent_modelname}/model'):
            os.system(f'mkdir -p ./{self.agent_modelname}/model')
            pickle.dump(self.__dict__,open(self.agent_modelname+"/modelsetting.pickle", "wb"))

        self.__build_model()
        if loadtype:
            self.load_prior(self.prior_modelname,loadtype)
            if 'agent_modelname' not in kwargs:
                self.load_agent(self.prior_modelname,loadtype)
            else:
                self.load_agent(self.agent_modelname,loadtype)
        self.batchsize=GP.trainsetting.batchsize

        if GP.modelsetting.device=='cuda':
            
            self.node_add_model_prior.to('cuda')
            self.ring_model_prior.to('cuda')
            self.node_connect_model_prior.to('cuda')

            self.node_add_model_agent.to('cuda')
            self.node_connect_model_agent.to('cuda')
            self.ring_model_agent.to('cuda')

        self.node_add_model_prior.eval()
        self.node_connect_model_prior.eval()
        self.ring_model_prior.eval()
        self.node_type_dict=GP.syssetting.node_type_dict
        self.f_node_dict=GP.syssetting.f_node_dict

        self.logger=open(f'./{self.agent_modelname}/Training.log','a')
        self.logger.write('='*40+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n') 
        self.logger.flush()
        return

    def __build_model(self,):
        self.node_add_model_prior=Node_adder()
        self.ring_model_prior=Ring_generator()
        self.node_connect_model_prior=Node_connecter()
        self.node_add_model_agent=Node_adder()
        self.ring_model_agent=Ring_generator()
        self.node_connect_model_agent=Node_connecter()        
        return 

    def Save(self,modelname=''):
        self.node_add_model_prior=None
        self.ring_model_prior=None
        self.node_connect_model_prior=None
        self.node_add_model_agent=None
        self.node_connect_model_agent=None
        self.ring_model_agent=None
        
        self.scheduler=None
        self.logger.close()
        self.logger=None

        self.optimizer=None
        pickle.dump(self.__dict__,open(self.agent_modelname+"/modelsetting.pickle", "wb"))
        shutil.make_archive(self.agent_modelname,"zip",self.agent_modelname)
        return 

    def load_prior(self,modelname,loadtype):
        with tempfile.TemporaryDirectory() as dirpath:
            with zipfile.ZipFile(modelname + ".zip", "r") as zip_ref:
                zip_ref.extractall(dirpath)
            # Load metadata
            metadata = pickle.load(open(dirpath + "/modelsetting.pickle", "rb"))
            #print (metadata.keys())
            #print (metadata)
            self.__dict__.update(metadata)

            if loadtype=="Perepoch":
                node_add_modelcpkt=torch.load(dirpath+"/model/node_add_model_perepoch.cpk")
                self.node_add_model_prior.load_state_dict(node_add_modelcpkt["state_dict"])
                ring_modelcpkt=torch.load(dirpath+"/model/ring_model_perepoch.cpk")
                self.ring_model_prior.load_state_dict(ring_modelcpkt["state_dict"])
                node_connect_modelcpkt=torch.load(dirpath+"/model/node_connect_model_perepoch.cpk")
                self.node_connect_model_prior.load_state_dict(node_connect_modelcpkt["state_dict"]) 
            elif loadtype=="Finetune":
                node_add_modelcpkt=torch.load(dirpath+"/model/node_add_model_finetune.cpk")
                self.node_add_model_prior.load_state_dict(node_add_modelcpkt["state_dict"])
                ring_modelcpkt=torch.load(dirpath+"/model/ring_model_finetune.cpk")
                self.ring_model_prior.load_state_dict(ring_modelcpkt["state_dict"])
                node_connect_modelcpkt=torch.load(dirpath+"/model/node_connect_model_finetune.cpk")
                self.node_connect_model_prior.load_state_dict(node_connect_modelcpkt["state_dict"])
            else:
                node_add_modelcpkt=torch.load(dirpath+"/model/node_add_model_minloss.cpk")
                self.node_add_model_prior.load_state_dict(node_add_modelcpkt["state_dict"])
                ring_modelcpkt=torch.load(dirpath+"/model/ring_model_minloss.cpk")
                self.ring_model_prior.load_state_dict(ring_modelcpkt["state_dict"])
                node_connect_modelcpkt=torch.load(dirpath+"/model/node_connect_model_minloss.cpk")
                self.node_connect_model_prior.load_state_dict(node_connect_modelcpkt["state_dict"])

            if 'epochs' not in node_add_modelcpkt.keys():
                node_add_modelcpkt['epochs']=0
            if 'epochs' not in ring_modelcpkt.keys():
                ring_modelcpkt['epochs']=0
            if 'epochs' not in node_connect_modelcpkt.keys():
                node_connect_modelcpkt['epochs']=0

            self.epochs=max((node_add_modelcpkt['epochs'],ring_modelcpkt['epochs'],node_connect_modelcpkt['epochs']))
            print ("Load prior model successfully!")
        return 

    def load_agent(self,modelname,loadtype):
        with tempfile.TemporaryDirectory() as dirpath:
            with zipfile.ZipFile(modelname + ".zip", "r") as zip_ref:
                zip_ref.extractall(dirpath)
            # Load metadata
            metadata = pickle.load(open(dirpath + "/modelsetting.pickle", "rb"))
            #print (metadata.keys())
            #print (metadata)
            self.__dict__.update(metadata)
            if loadtype=="Perepoch":
                node_add_modelcpkt=torch.load(dirpath+"/model/node_add_model_perepoch.cpk")
                self.node_add_model_agent.load_state_dict(node_add_modelcpkt["state_dict"])
                ring_modelcpkt=torch.load(dirpath+"/model/ring_model_perepoch.cpk")
                self.ring_model_agent.load_state_dict(ring_modelcpkt["state_dict"])
                node_connect_modelcpkt=torch.load(dirpath+"/model/node_connect_model_perepoch.cpk")
                self.node_connect_model_agent.load_state_dict(node_connect_modelcpkt["state_dict"])
            elif loadtype=="Finetune":
                node_add_modelcpkt=torch.load(dirpath+"/model/node_add_model_finetune.cpk")
                self.node_add_model_agent.load_state_dict(node_add_modelcpkt["state_dict"])
                ring_modelcpkt=torch.load(dirpath+"/model/ring_model_finetune.cpk")
                self.ring_model_agent.load_state_dict(ring_modelcpkt["state_dict"])
                node_connect_modelcpkt=torch.load(dirpath+"/model/node_connect_model_finetune.cpk")
                self.node_connect_model_agent.load_state_dict(node_connect_modelcpkt["state_dict"]) 
            elif loadtype=="RL":
                node_add_modelcpkt=torch.load(dirpath+"/model/node_add_model_RL.cpk")
                self.node_add_model_agent.load_state_dict(node_add_modelcpkt["state_dict"])
                ring_modelcpkt=torch.load(dirpath+"/model/ring_model_RL.cpk")
                self.ring_model_agent.load_state_dict(ring_modelcpkt["state_dict"])
                node_connect_modelcpkt=torch.load(dirpath+"/model/node_connect_model_RL.cpk")
                self.node_connect_model_agent.load_state_dict(node_connect_modelcpkt["state_dict"])  
            else:
                node_add_modelcpkt=torch.load(dirpath+"/model/node_add_model_minloss.cpk")
                self.node_add_model_agent.load_state_dict(node_add_modelcpkt["state_dict"])
                ring_modelcpkt=torch.load(dirpath+"/model/ring_model_minloss.cpk")
                self.ring_model_agent.load_state_dict(ring_modelcpkt["state_dict"])
                node_connect_modelcpkt=torch.load(dirpath+"/model/node_connect_model_minloss.cpk")
                self.node_connect_model_agent.load_state_dict(node_connect_modelcpkt["state_dict"])
            if 'epochs' not in node_add_modelcpkt.keys():
                node_add_modelcpkt['epochs']=0
            if 'epochs' not in ring_modelcpkt.keys():
                ring_modelcpkt['epochs']=0
            if 'epochs' not in node_connect_modelcpkt.keys():
                node_connect_modelcpkt['epochs']=0
            self.epochs=max((node_add_modelcpkt['epochs'],ring_modelcpkt['epochs'],node_connect_modelcpkt['epochs']))
            print ("Load agent model successfully!")
        return 

    def kl_loss(self,output,target):
        """
        The graph generation loss is the KL divergence between the target and
        predicted actions.

        Args:
        ----
            output (torch.Tensor)        : Predicted APD tensor.
            target_output (torch.Tensor) : Target APD tensor.

        Returns:
        -------
            loss (torch.Tensor) : Average loss for this output.
        """
        # define activation function; note that one must use the softmax in the
        # KLDiv, never the sigmoid, as the distribution must sum to 1
        LogSoftmax = torch.nn.LogSoftmax(dim=1)
        output     = LogSoftmax(output)
        # normalize the target output (as can contain information on > 1 graph)
        target_output = target/torch.sum(target, dim=1, keepdim=True)
        # define loss function and calculate the los
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        loss      = criterion(target=target_output, input=output)
        return loss 

    def rl_step(self,score_func,step,picpath='./RL_mols',score_path='./score_gen'):
        self.node_add_model_agent.zero_grad()
        self.ring_model_agent.zero_grad()
        self.node_connect_model_agent.zero_grad()
        nodes,edges,atomnums,agent_loglikelihoods,prior_loglikelihoods=self.sample_batch() 
        mols,smis=self.nodes_edges_to_mol(nodes,edges,atomnums)
        #nodes,edges,atomnums,agent_loglikelihoods,prior_loglikelihoods,mols,smis=self.sample(sample_num=self.batchsize)
        scores,validity,uniqueness,unknown=score_func.compute_score(mols,output_path=score_path)
        os.system(f'mkdir -p {picpath}')
        img=Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(250,250),legends=list([str(float(a)) for a in scores.clone().detach().cpu().numpy()]))
        img.save(f'{picpath}/RL_{step}.png')
        augmented_prior_loglikelihoods = (
            prior_loglikelihoods + GP.rlsetting.sigma * scores
        )

        difference =  augmented_prior_loglikelihoods - agent_loglikelihoods
        loss       = difference * difference
        mask       = (uniqueness != 0).int()
        loss       = torch.mean(loss[torch.nonzero(uniqueness)])
        #loss       = torch.mean(loss * mask)
        if torch.all(torch.isfinite(loss)):
            loss.backward()
            self.optimizer.step()

            # update the learning rate
            self.scheduler.step()
            self.lr=self.optimizer.state_dict()['param_groups'][0]['lr']
            
            if torch.mean(scores)> self.min_rl_loss['scores']:
                self.min_rl_loss['scores']=torch.mean(scores).item()
                print (f'-- Save New check point for Node-Add model at RL steps: {step}!')
                savepath=f'{self.agent_modelname}/model/node_add_model_RL.cpk'
                savedict={'step':step,'lossmin':self.min_rl_loss['scores'],'state_dict':self.node_add_model_agent.state_dict()}
                torch.save(savedict,savepath)
           
                print (f'-- Save New check point for Ring model at Epoch: {step}!')
                savepath=f'{self.agent_modelname}/model/ring_model_RL.cpk'
                savedict={'step':step,'lossmin':self.min_rl_loss['scores'],'state_dict':self.ring_model_agent.state_dict()}
                torch.save(savedict,savepath)
           
                print (f'-- Save New check point for Node connect model at Epoch: {step}!')
                savepath=f'{self.agent_modelname}/model/node_connect_model_RL.cpk'
                savedict={'step':step,'lossmin':self.min_rl_loss['scores'],'state_dict':self.node_connect_model_agent.state_dict()}
                torch.save(savedict,savepath)

        lstr=f'Lr: {self.lr:.3E} Loss: {loss.item():.3E} Scores: {torch.mean(scores).item():.3E} Valid_scores: {torch.sum(scores).item()/torch.sum(validity):.3E} Validity: {torch.sum(validity).item()/float(len(validity)):.3E} Uniqueness:  {torch.sum(uniqueness).item()/float(len(uniqueness)):.3E}\n'
        return lstr

    def rl_step_acc_grad(self,score_func,step,picpath='./RL_mols',acc_steps=4,temp=1.0,score_path='./score_gen'):
        self.node_add_model_agent.zero_grad()
        self.ring_model_agent.zero_grad()
        self.node_connect_model_agent.zero_grad()
        nvalid=0
        nunique=0
        nunknown=0
        ntotal=0
        nacc=0
        total_scores=torch.zeros(self.batchsize-1,device=GP.modelsetting.device)
        valid_mols=[]
        valid_smis=[]
        valid_scores=[]
        valid_score_components=[[] for i in range(len(GP.rlsetting.score_components))]+[[]]
        #print (valid_score_components)
        while nacc<acc_steps:
            nodes,edges,atomnums,agent_loglikelihoods,prior_loglikelihoods=self.sample_batch(temp=temp)
            mols,smis,vids=self.nodes_edges_to_mol(nodes,edges,atomnums)
            #nodes,edges,atomnums,agent_loglikelihoods,prior_loglikelihoods,mols,smis=self.sample(sample_num=self.batchsize)
            #print (mols)
            scores,validity,uniqueness,unknown,score_components=score_func.compute_score(mols,output_path=f'{score_path}/{nacc}')
            
            nv=torch.sum(validity).item()
            nu=torch.sum(uniqueness).item()
            nk=torch.sum(unknown).item()
            #print (step,scores)
            os.system(f'mkdir -p {picpath}')
            augmented_prior_loglikelihoods = (
                prior_loglikelihoods + GP.rlsetting.sigma * scores + GP.rlsetting.vsigma * validity + GP.rlsetting.ksigma*unknown
            )
       
            difference =  augmented_prior_loglikelihoods - agent_loglikelihoods
            loss       = difference * difference
            mask       = (uniqueness != 0).int()
            loss       = torch.mean(loss[torch.nonzero(uniqueness)])/acc_steps
            #loss       = torch.mean(loss * mask)
            if torch.all(torch.isfinite(loss)):
                valid_mols+=[mols[i] for i in vids]
                valid_smis+=[smis[i] for i in vids]
                valid_scores+=[scores.clone().detach().cpu().numpy()[i] for i in vids]
                #print (score_components)
                for i in range(len(GP.rlsetting.score_components)):
                    #print (vids,valid_score_components[i])
                    valid_score_components[i]+=[score_components[i][j].clone().detach().cpu().numpy() for j in vids]                    
                valid_score_components[-1]+=[scores.clone().detach().cpu().numpy()[i] for i in vids]
                total_scores+=scores
                loss.backward()
                nvalid+=nv
                nunique+=nu
                nunknown+=nk
                ntotal+=self.batchsize-1
                nacc+=1
               
        total_scores=total_scores/acc_steps        
        self.optimizer.step()
        # update the learning rate
        self.scheduler.step()
        self.lr=self.optimizer.state_dict()['param_groups'][0]['lr']
        try:
            valid_score_str=[]
            for i in range(len(valid_mols)):
                score_str=','.join([f'{valid_score_components[j][i]:.3F}' for j in range(len(GP.rlsetting.score_components)+1)])
                valid_score_str.append(score_str)
            with open(f'{picpath}/RL_{step}.smi','w') as f:
                for mid,mol in enumerate(valid_mols):
                    f.write(Chem.MolToSmiles(mol)+','+valid_score_str[mid]+'\n')
            if GP.rlsetting.save_pic:
                img=Draw.MolsToGridImage(valid_mols,molsPerRow=5,subImgSize=(250,250),legends=valid_score_str)
                img.save(f'{picpath}/RL_{step}.png')

        except Exception as e:
            print (f'save png of valid molecules failed due to {e}')
            
        if torch.mean(total_scores)> self.min_rl_loss['scores']:
            self.min_rl_loss['scores']=torch.mean(total_scores).item()
            print (f'-- Save New check point for Node-Add model at RL steps: {step}!')
            savepath=f'{self.agent_modelname}/model/node_add_model_RL.cpk'
            savedict={'step':step,'lossmin':self.min_rl_loss['scores'],'state_dict':self.node_add_model_agent.state_dict()}
            torch.save(savedict,savepath)
           
            print (f'-- Save New check point for Ring model at Epoch: {step}!')
            savepath=f'{self.agent_modelname}/model/ring_model_RL.cpk'
            savedict={'step':step,'lossmin':self.min_rl_loss['scores'],'state_dict':self.ring_model_agent.state_dict()}
            torch.save(savedict,savepath)
           
            print (f'-- Save New check point for Node connect model at Epoch: {step}!')
            savepath=f'{self.agent_modelname}/model/node_connect_model_RL.cpk'
            savedict={'step':step,'lossmin':self.min_rl_loss['scores'],'state_dict':self.node_connect_model_agent.state_dict()}
            torch.save(savedict,savepath)

        if nvalid > 0:
            valid_scores=torch.sum(total_scores).item()*acc_steps/nvalid 
        else:
            valid_scores=0
        if nunknown>0:
            unknown_scores=torch.sum(total_scores).item()*acc_steps/nunknown
        else:
            unknown_scores=0

        #print (nunknown,nvalid)
        if GP.rlsetting.unknown_fielter:
            lstr=f'Temp: {temp} Lr: {self.lr:.3E} Loss: {loss.item():.3E} Scores: {torch.mean(total_scores).item():.3E} Valid_scores: {valid_scores:.3E} Unknown_scores: {unknown_scores:.3E} Validity: {float(nvalid)/float(ntotal):.3E} Uniqueness:  {nunique/float(ntotal):.3E} Unknown:{nunknown/float(ntotal):.3E} \n'
        else:
            lstr=f'Temp: {temp} Lr: {self.lr:.3E} Loss: {loss.item():.3E} Scores: {torch.mean(total_scores).item():.3E} Valid_scores: {valid_scores:.3E} Validity: {float(nvalid)/float(ntotal):.3E} Uniqueness:  {nunique/float(ntotal):.3E}\n' 
        
        return lstr 

    def RL_fit(self,steps=100,score_func=None,iter=1):
        if GP.trainsetting.optimizer=='Adam':
            print("-- Defining optimizer.", flush=True)
            self.optimizer=torch.optim.Adam([{'params':self.node_add_model_agent.parameters()},{'params':self.ring_model_agent.parameters()},{'params':self.node_connect_model_agent.parameters()}],lr=GP.trainsetting.initlr)
            print("-- Defining scheduler.", flush=True)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr= GP.trainsetting.max_rel_lr * GP.trainsetting.initlr,
                div_factor= 1. /GP.trainsetting.max_rel_lr,
                final_div_factor = 1. /GP.trainsetting.min_rel_lr,
                pct_start = 0.05,
                total_steps=steps,
                epochs=steps
            )

        init_temp=GP.rlsetting.temp_range[0]
        final_temp=GP.rlsetting.temp_range[1]
        for step in range(steps):
            if GP.rlsetting.temp_scheduler=="same":
                temp=init_temp
            elif GP.rlsetting.temp_scheduler=="linear":
                temp=init_temp+(final_temp-init_temp)/steps*step
            lstr=self.rl_step_acc_grad(score_func,step,picpath=f'./RL_mols_iter{iter}',acc_steps=GP.rlsetting.acc_steps,temp=temp,score_path=f'./dock/{step}')
            print (step,lstr)
            logstr=f'Reinforcement Learning Steps: {step}: '+lstr
            self.logger.write(logstr)
            self.logger.flush()
            self.__tmprecord_clean()
        return
    def CL_fit(self,target_mol_steps):
        n_rl_steps=len(target_mol_steps)
        sfs=[ScoringFunction() for i in range(n_rl_steps)]
        for i in range(n_rl_steps):
            sfs[i].target_smiles=[Chem.MolToSmiles(target_mol_steps[i])]
            logstr=f'Curriculum Learning Iteration {i}; Target: {sfs[i].target_smiles[0]} :\n'
            self.logger.write(logstr)
            self.logger.flush()
            self.RL_fit(steps=GP.clsetting.max_clsteps_per_iter,score_func=sfs[i],iter=i)        
        return 
    def __tmprecord_clean(self):
        self.rl_loss={'nadd':0,'rgen':0,'njoint':0,'total':0}
        return 

    def __init_mol_graphs(self,idx=-1):
        dummy_af=onek_encoding_unk(6,GP.syssetting.possible_atom_types)+\
                onek_encoding_unk(0,GP.syssetting.formal_charge_types)

        if idx==-1:
            self.rw_mol_nodes = torch.zeros((self.batchsize,GP.syssetting.max_atoms,GP.syssetting.n_node_features),device=GP.modelsetting.device).long()
            self.rw_mol_adjs = torch.zeros((self.batchsize,GP.syssetting.max_atoms,GP.syssetting.max_atoms,GP.syssetting.n_edge_features),device=GP.modelsetting.device).long()
            self.rw_mol_atomnums=torch.zeros(self.batchsize,device=GP.modelsetting.device).long()
            self.rw_mol_atomics=torch.zeros((self.batchsize,GP.syssetting.max_atoms),device=GP.modelsetting.device).long()
            self.rw_mol_pt=torch.zeros((self.batchsize,GP.sample_constrain_setting.max_node_steps+1,2),device=GP.modelsetting.device).long()
            self.rw_mol_round=torch.zeros(self.batchsize,device=GP.modelsetting.device).long()
            self.rw_mol_saturation_list=[[] for i in range(self.batchsize)]
            self.rw_mol_nodes[0,0]=torch.Tensor(dummy_af)
            self.rw_mol_adjs[0,0,0,0]=1
            self.rw_mol_atomnums[0]=1
            self.rw_mol_atomics[0,0]=6
            self.rw_mol_pt[0,0,0]=0
            self.rw_mol_pt[0,0,1]=1

            self.rw_mol_likelihoods_agent=torch.zeros((self.batchsize,GP.syssetting.max_atoms),device=GP.modelsetting.device)
            self.rw_mol_likelihoods_prior=torch.zeros((self.batchsize,GP.syssetting.max_atoms),device=GP.modelsetting.device)

        elif idx>0:
            self.rw_mol_nodes[idx]=0
            self.rw_mol_adjs[idx]=0
            self.rw_mol_atomnums[idx]=0
            self.rw_mol_atomics[idx]=0
            self.rw_mol_pt[idx]=0
            self.rw_mol_round[idx]=0
            self.rw_mol_likelihoods_agent[idx]=0
            self.rw_mol_likelihoods_prior[idx]=0
            self.rw_mol_saturation_list[idx]=[]

        else:
            self.rw_mol_nodes[0]=0
            self.rw_mol_adjs[0]=0
            self.rw_mol_atomnums[0]=1 
            self.rw_mol_atomics[0]=0
            self.rw_mol_pt[0]=0
            self.rw_mol_round[0]=0
            self.rw_mol_likelihoods_agent[0]=0
            self.rw_mol_likelihoods_prior[0]=0
            self.rw_mol_nodes[0,0]=torch.Tensor(dummy_af)
            self.rw_mol_adjs[0,0,0,0]=1
            self.rw_mol_atomics[0,0]=6
            self.rw_mol_pt[0,0,0]=0
            self.rw_mol_pt[0,0,1]=1
            self.rw_mol_saturation_list[0]=[]
        return
    
    def __init_ring_graphs(self,idx=-1):
        dummy_af=onek_encoding_unk(6,GP.syssetting.possible_atom_types)+\
            onek_encoding_unk(0,GP.syssetting.formal_charge_types)
        if idx==-1:
            self.rw_ring_nodes = torch.zeros((self.batchsize,GP.syssetting.max_ring_size,GP.syssetting.n_node_features),device=GP.modelsetting.device).long()
            self.rw_ring_adjs = torch.zeros((self.batchsize,GP.syssetting.max_ring_size,GP.syssetting.max_ring_size,GP.syssetting.n_edge_features),device=GP.modelsetting.device).long()
            self.rw_ring_atomnums=torch.zeros(self.batchsize,device=GP.modelsetting.device).long()
            self.rw_ring_atomics=torch.zeros((self.batchsize,GP.syssetting.max_atoms),device=GP.modelsetting.device).long()
            self.rw_ring_ftypes = torch.zeros((self.batchsize,GP.syssetting.n_clique_features),device=GP.modelsetting.device).long()
            self.rw_ring_focused_ids = torch.zeros(self.batchsize,device=GP.modelsetting.device).long()
            self.rw_ring_round=torch.zeros(self.batchsize,device=GP.modelsetting.device).long()

            self.rw_ring_likelihoods_agent=torch.zeros((self.batchsize,GP.syssetting.max_ring_size),device=GP.modelsetting.device)
            self.rw_ring_likelihoods_prior=torch.zeros((self.batchsize,GP.syssetting.max_ring_size),device=GP.modelsetting.device) 

            self.rw_ring_nodes[0,0]=torch.Tensor(dummy_af)
            self.rw_ring_adjs[0,0,0,0]=1
            self.rw_ring_atomnums[0]=1
            self.rw_ring_atomics[0,0]=6
            
        elif idx>0:
            self.rw_ring_nodes[idx]=0
            self.rw_ring_adjs[idx]=0
            self.rw_ring_atomnums[idx]=0
            self.rw_ring_atomics[idx]=0
            #self.rw_ring_ftypes[idx]=0
            self.rw_ring_focused_ids[idx]=0
            self.rw_ring_round[idx]=0
            self.rw_ring_likelihoods_agent[idx]=0
            self.rw_ring_likelihoods_prior[idx]=0
        else:
            self.rw_ring_nodes[0]=0
            self.rw_ring_adjs[0]=0
            self.rw_ring_atomnums[0]=0
            self.rw_ring_atomics[0]=0
            self.rw_ring_focused_ids[0]=0 
            #self.rw_ring_ftypes[0]=0
            self.rw_ring_round[0]=0

            self.rw_ring_likelihoods_agent[0]=0
            self.rw_ring_likelihoods_prior[0]=0
            self.rw_ring_nodes[0,0]=torch.Tensor(dummy_af)
            self.rw_ring_adjs[0,0,0,0]=1
            self.rw_ring_atomnums[0]=1
            self.rw_ring_atomics[0,0]=6
        return 

    def __init_failed_ring_graphs(self,ids):
        self.rw_ring_nodes[ids]=0
        self.rw_ring_adjs[ids]=0
        self.rw_ring_atomnums[ids]=0
        self.rw_ring_atomics[ids]=0
        self.rw_ring_focused_ids[ids]=0
        self.rw_ring_round[ids]=0
        self.rw_ring_likelihoods_agent[ids]=0
        self.rw_ring_likelihoods_prior[ids]=0
        return 

    def __init_failed_mol_graphs(self,ids):
        self.rw_mol_nodes[ids]=0
        self.rw_mol_adjs[ids]=0
        self.rw_mol_atomnums[ids]=0
        self.rw_mol_atomics[ids]=0
        self.rw_mol_pt[ids]=0
        self.rw_mol_round[ids]=0
        self.rw_ring_nodes[ids]=0
        self.rw_ring_adjs[ids]=0
        self.rw_ring_atomnums[ids]=0
        self.rw_ring_atomics[ids]=0
        self.rw_ring_focused_ids[ids]=0
        self.rw_ring_ftypes[ids]=0
        self.rw_ring_round[ids]=0
        for idx in list(ids.view(-1).clone().detach().cpu().numpy()):
            self.rw_mol_saturation_list[idx]=[]
        return 

    def __init_nconn(self,idx=-1):
        self.rw_conn_likelihoods_agent=torch.zeros(self.batchsize,device=GP.modelsetting.device)
        self.rw_conn_likelihoods_prior=torch.zeros(self.batchsize,device=GP.modelsetting.device)

    def select_mol_graphs(self,ids):
        mol_nodes=self.rw_mol_nodes[ids]
        mol_adjs=self.rw_mol_adjs[ids]
        #print (ids,self.rw_mol_atomnums)
        mol_atomnums=self.rw_mol_atomnums[ids]
        mol_atomics=self.rw_mol_atomics[ids]
        mol_pt=self.rw_mol_pt[ids]
        mol_round=self.rw_mol_round[ids]
        return mol_nodes,mol_adjs,mol_atomnums,mol_atomics,mol_pt,mol_round

    def select_ring_graphs(self,ids):
        ring_nodes=self.rw_ring_nodes[ids]
        ring_adjs=self.rw_ring_adjs[ids]
        ring_atomnums=self.rw_ring_atomnums[ids]
        ring_ftypes=self.rw_ring_ftypes[ids]
        ring_focused_ids=self.rw_ring_focused_ids[ids]
        ring_atomics=self.rw_ring_atomics[ids]
        ring_round=self.rw_ring_round[ids]
        return ring_nodes,ring_adjs,ring_atomnums,ring_atomics,ring_ftypes,ring_focused_ids,ring_round

    def __unstop_mol_graphs(self):
        mol_unstop_ids=torch.where(self.mol_stop_mask>0)[0]
        mol_nodes,mol_adjs,mol_atomnums,mol_atomics,mol_pt,mol_round=self.select_mol_graphs(mol_unstop_ids)
        return mol_unstop_ids,mol_nodes,mol_adjs,mol_atomnums,mol_atomics,mol_pt,mol_round

    def __unstop_ring_graphs(self):
        ring_unstop_ids=torch.where(self.ring_stop_mask>0)[0]
        ring_nodes,ring_adjs,ring_atomnums,ring_atomics,ring_ftypes,ring_focused_ids,ring_round=self.select_ring_graphs(ring_unstop_ids)
        return ring_unstop_ids,ring_nodes,ring_adjs,ring_atomnums,ring_atomics,ring_ftypes,ring_focused_ids,ring_round
    
    def __reshape_nadd_apd(self,nadd_apd):
        f_nadd_shape=(nadd_apd.shape[0],GP.syssetting.num_node_types)
        f_add=torch.reshape(nadd_apd[:,:GP.syssetting.num_node_types],f_nadd_shape)
        f_term=nadd_apd[:,-1]
        return f_add,f_term

    def sample_nadd_action(self,rw_mol_nodes,rw_mol_adjs,nadd_masks,temp):
        softmax=torch.nn.Softmax(dim=1)
        #print (GP.samplesetting.avaliable_node_mask)
        nadd_apd_predict_agent_out = self.node_add_model_agent(rw_mol_nodes,rw_mol_adjs)
        nadd_apd_predict_agent = torch.mul((softmax(nadd_apd_predict_agent_out)+1e-8),nadd_masks.to(GP.modelsetting.device).long())
        nadd_apd_predict_agent_with_temp=torch.exp(torch.log(nadd_apd_predict_agent)/temp)
        nadd_apd_predict_prior_out = self.node_add_model_prior(rw_mol_nodes,rw_mol_adjs).detach()
        nadd_apd_predict_prior = torch.mul(softmax(nadd_apd_predict_prior_out)+1e-8,nadd_masks.to(GP.modelsetting.device).long())

        action_probability_distribution = torch.distributions.Multinomial(
                1,
                probs=nadd_apd_predict_agent_with_temp
            )
        nadd_apd_one_hot=action_probability_distribution.sample()

        f_add,f_term=self.__reshape_nadd_apd(nadd_apd_one_hot)
        agent_likelihoods = torch.log(nadd_apd_predict_agent[nadd_apd_one_hot==1])
        prior_likelihoods = torch.log(nadd_apd_predict_prior[nadd_apd_one_hot==1]).detach()
        add_idc = torch.nonzero(f_add,as_tuple=True)
        term_idc = torch.nonzero(f_term,as_tuple=True)
        return add_idc,term_idc,agent_likelihoods,prior_likelihoods

    def __reshape_rgen_apd(self,rgen_apd):
        f_radd_shape=(rgen_apd.shape[0],*GP.syssetting.f_ring_add_dim)
        f_rconn_shape=(rgen_apd.shape[0],*GP.syssetting.f_ring_connect_dim)
        f_radd_size=np.prod(GP.syssetting.f_ring_add_dim)
        f_radd=torch.reshape(rgen_apd[:,:f_radd_size],f_radd_shape)
        f_rconn=torch.reshape(rgen_apd[:,f_radd_size:-1],f_rconn_shape)
        f_rterm=rgen_apd[:,-1]
        return f_radd,f_rconn,f_rterm

    def sample_rgen_action(self,rw_mol_nodes,rw_mol_adjs,rw_ring_nodes,rw_ring_adjs,rw_ring_ftype):
        softmax=torch.nn.Softmax(dim=1)
        rgen_apd_pred_agent_out = self.ring_model_agent(rw_mol_nodes,rw_mol_adjs,rw_ring_nodes,rw_ring_adjs,rw_ring_ftype)
        rgen_apd_pred_agent = softmax(rgen_apd_pred_agent_out+1e-8)
        rgen_apd_pred_prior_out = self.ring_model_prior(rw_mol_nodes,rw_mol_adjs,rw_ring_nodes,rw_ring_adjs,rw_ring_ftype)
        rgen_apd_pred_prior = softmax(rgen_apd_pred_prior_out+1e-8).detach()

        action_probability_distribution = torch.distributions.Multinomial( 1, probs=rgen_apd_pred_agent)
        
        rgen_apd_one_hot = action_probability_distribution.sample()

        f_radd,f_rconn,f_rterm = self.__reshape_rgen_apd(rgen_apd_one_hot)
        agent_likelihoods=torch.log(rgen_apd_pred_agent[rgen_apd_one_hot==1])
        prior_likelihoods=torch.log(rgen_apd_pred_prior[rgen_apd_one_hot==1]).detach()

        add_idc=torch.nonzero(f_radd,as_tuple=True)
        conn_idc=torch.nonzero(f_rconn,as_tuple=True)
        term_idc=torch.nonzero(f_rterm,as_tuple=True)
        return add_idc,conn_idc,term_idc,agent_likelihoods,prior_likelihoods

    def rgen_invalid_actions(self,rgen_add_idc,rgen_conn_idc,rw_ring_nodes,rw_ring_adjs,rw_ring_atomnum,rw_ring_ftype):
        rw_ring_maxatoms=torch.sum(rw_ring_ftype[:,1:-2],dim=-1).long()
        rw_ring_maxatoms=torch.where(rw_ring_atomnum.long()>rw_ring_maxatoms.long(),rw_ring_atomnum.long(),rw_ring_maxatoms.long())
        
        f_add_empty_graphs = torch.nonzero(rw_ring_atomnum[rgen_add_idc[0]]==0)
        #print (f_add_empty_graphs)
        invalid_add_idx_tmp = torch.nonzero(rgen_add_idc[1]>=rw_ring_atomnum[rgen_add_idc[0]])
        #print (invalid_add_idx_tmp)
        combined = torch.cat((invalid_add_idx_tmp,f_add_empty_graphs)).squeeze(1)
        
        uniques,counts = combined.unique(return_counts=True)
        invalid_add_idc = uniques[counts==1].unsqueeze(dim=1)
        #print (invalid_add_idc)
        invalid_add_empty_idc = torch.nonzero(rgen_add_idc[1] != rw_ring_atomnum[rgen_add_idc[0]])
        combined              = torch.cat((invalid_add_empty_idc, f_add_empty_graphs)).squeeze(1)
        uniques, counts       = combined.unique(return_counts=True)

        invalid_add_empty_idc = uniques[counts > 1].unsqueeze(dim=1)
        invalid_madd_idc=torch.nonzero(rgen_add_idc[5]>=rw_ring_maxatoms[rgen_add_idc[0]])
        invalid_conn_idc=torch.nonzero(rgen_conn_idc[1]>=rw_ring_atomnum[rgen_conn_idc[0]])
        invalid_conn_nonex_idc=torch.nonzero(rw_ring_atomnum[rgen_conn_idc[0]]==0)
        invalid_sconn_idc=torch.nonzero(rgen_conn_idc[1]==rgen_conn_idc[3])

        invalid_dconn_idc=torch.nonzero(
                torch.sum(
                    rw_ring_adjs,
                    dim=-1
                    )[
                        rgen_conn_idc[0].long(),
                        rgen_conn_idc[1].long(),
                        rgen_conn_idc[-1].long()]==1
            )

        invalid_action_idc=torch.unique(
            torch.cat(
                (
                    rgen_add_idc[0][invalid_add_idc],
                    rgen_add_idc[0][invalid_add_empty_idc],
                    rgen_conn_idc[0][invalid_conn_idc],
                    rgen_conn_idc[0][invalid_conn_nonex_idc],
                    rgen_conn_idc[0][invalid_sconn_idc],
                    rgen_conn_idc[0][invalid_dconn_idc],
                    rgen_add_idc[0][invalid_madd_idc]
                )
            )
        )

        #invalid_action_idc_needing_reset = torch.unique(torch.cat((invalid_madd_idc,f_add_empty_graphs)))
        return  invalid_action_idc,invalid_madd_idc

    def __reshape_conn_apd(self,conn_apd):
        f_conn_shape=(conn_apd.shape[0],*GP.syssetting.f_node_joint_dim)
        f_conn=torch.reshape(conn_apd,f_conn_shape)
        return f_conn

    def conn_invalid_actions(self,conn_idc,rw_mol_atomnum):
        f_conn_empty_graphs = torch.nonzero(rw_mol_atomnum[conn_idc[0]]==0)
        #print (f_conn_empty_graphs)
        invalid_conn_idc_tmp=torch.nonzero(conn_idc[1]>=rw_mol_atomnum[conn_idc[0]])
        combined = torch.cat((invalid_conn_idc_tmp,f_conn_empty_graphs)).squeeze(1)
        uniques,counts = combined.unique(return_counts=True)
        invalid_conn_idc = uniques[counts==1].unsqueeze(dim=1)         

        invalid_action_idc=conn_idc[0][invalid_conn_idc].view(-1)

        return  invalid_action_idc

    def sample_conn_action(self,rw_mol_nodes,rw_mol_adjs,rw_ring_nodes,rw_ring_adjs,rw_ring_focused_ids,nconn_masks):
        softmax=torch.nn.Softmax(dim=1)
        conn_apd_pred_agent_out = self.node_connect_model_agent(rw_mol_nodes,rw_mol_adjs,rw_ring_nodes,rw_ring_adjs,rw_ring_focused_ids)
        conn_apd_pred_agent = torch.mul(softmax(conn_apd_pred_agent_out+1e-8),nconn_masks.to(GP.modelsetting.device).long())
        conn_apd_pred_prior_out = self.node_connect_model_prior(rw_mol_nodes,rw_mol_adjs,rw_ring_nodes,rw_ring_adjs,rw_ring_focused_ids)
        conn_apd_pred_prior = torch.mul(softmax(conn_apd_pred_prior_out+1e-8).detach(),nconn_masks.to(GP.modelsetting.device).long())

        action_probability_distribution = torch.distributions.Multinomial(1, probs=conn_apd_pred_agent)
        
        conn_apd_one_hot = action_probability_distribution.sample()
        
        f_conn= self.__reshape_conn_apd(conn_apd_one_hot)
        
        agent_likelihoods = torch.log(conn_apd_pred_agent[conn_apd_one_hot==1])
        prior_likelihoods = torch.log(conn_apd_pred_prior[conn_apd_one_hot==1]).detach()
        
        conn_idc=torch.nonzero(f_conn,as_tuple=True)

        return conn_idc, agent_likelihoods,prior_likelihoods

    def sample_prepare(self):
        self.mol_stop_mask=torch.ones(self.batchsize,device=GP.modelsetting.device)
        self.ring_stop_mask=torch.ones(self.batchsize,device=GP.modelsetting.device)

        self.mol_finished=torch.ones(self.batchsize,device=GP.modelsetting.device).long()
        self.Nodetypeid_To_Ringftype=[self.f_node_dict[Nodetypeid] for Nodetypeid in range(GP.syssetting.num_node_types)]
        self.Nodetypeid_To_Ringftype=torch.Tensor(self.Nodetypeid_To_Ringftype).long().to(GP.modelsetting.device)
        #print (self.Nodetypeid_To_Ringftype)
        self.Singlenode_To_Nodefeatures=torch.Tensor([onek_encoding_unk(atomicnum,GP.syssetting.possible_atom_types)+\
                        onek_encoding_unk(0,GP.syssetting.formal_charge_types) for atomicnum in GP.syssetting.possible_atom_types]).long().to(GP.modelsetting.device)
    
    def sample_clean(self):
        self.mol_stop_mask=None

    def gen_nadd_valid_masks(self,nadd_constrain_list):
        n_types=len(self.node_type_dict.keys())
        avaliable_nadd_mask=torch.ones(self.batchsize,n_types+1).long()
        #print (node_type_constrain.__dict__)
        for j in range(self.batchsize):
            nadd_constrain=nadd_constrain_list[j]
            for k,f in self.f_node_dict.items():
                rnum=f[0]
                Besnum=f[-2]
                ARnum=f[-1]
                anumlist=f[1:-2]
                tanum=np.sum(anumlist)
                if nadd_constrain.min_ring_num_per_node>rnum or rnum>nadd_constrain.max_ring_num_per_node:
                    avaliable_nadd_mask[j][k]=0               
                for aid,attype in enumerate(["C","N","O","F","P","S","Cl","Br","I"]):
                    if nadd_constrain.min_anum_per_atomtype[attype]>anumlist[aid] or anumlist[aid]>nadd_constrain.max_anum_per_atomtype[attype]:
                        avaliable_nadd_mask[j][k]=0
                if nadd_constrain.min_branches>Besnum or Besnum>nadd_constrain.max_branches:
                    avaliable_nadd_mask[j][k]=0
                if nadd_constrain.max_aromatic_rings<ARnum or ARnum<nadd_constrain.min_aromatic_rings:
                    avaliable_nadd_mask[j][k]=0
            if nadd_constrain.force_step:
                avaliable_nadd_mask[j][-1]=0
            avaliable_nadd_mask=avaliable_nadd_mask.view(-1,n_types+1).long()
        return avaliable_nadd_mask

    def gen_nconn_valid_masks(self,nconn_constrain_list,node_pt,mol_atom_nums,mol_atomic_nums,saturation_atomid_list):
        #print (node_conn_constrain.__dict__)
        avaliable_nconn_mask=torch.zeros(self.batchsize,*GP.syssetting.f_node_joint_dim).long()
        #avaliable_nconn_general_mask=torch.ones(self.batchsize,*GP.syssetting.f_node_joint_dim).long()
        #print ('*'*80)
        for j in range(self.batchsize):
            if len(nconn_constrain_list[j].constrain_connect_node_id)>0:
                for kid,k in enumerate(nconn_constrain_list[j].constrain_connect_node_id):
                    pt0=node_pt[j][k][0]
                    pt1=node_pt[j][k][1]
                    #print (k,pt0,pt1)
                    if len(nconn_constrain_list[j].constrain_connect_atom_id[kid])>0:
                        for nid,n in enumerate(range(pt0,pt1)):
                            if nid in nconn_constrain_list[j].constrain_connect_atom_id[kid]:
                                if mol_atomic_nums[j][n] in nconn_constrain_list[j].constrain_connect_atomic_type:
                                    avaliable_nconn_mask[j,n,nconn_constrain_list[j].constrain_connect_bond_type]=1
                    else:
                        for nid,n in enumerate(range(pt0,pt1)):
                            if mol_atomic_nums[j][n] in nconn_constrain_list[j].constrain_connect_atomic_type:
                                avaliable_nconn_mask[j,n,nconn_constrain_list[j].constrain_connect_bond_type]=1
            else:
                avaliable_nconn_mask[j]=1
        for j in range(self.batchsize):
            avaliable_nconn_mask[j,saturation_atomid_list[j]]=0
            if mol_atom_nums[j]==0:
                avaliable_nconn_mask[j]=1
        avaliable_nconn_mask=avaliable_nconn_mask.view(self.batchsize,np.prod(GP.syssetting.f_node_joint_dim))
        avaliable_nconn_mask[0]=1    
        return avaliable_nconn_mask

    def specific_node_to_ringgraph(self,sp_node_graph):
        sp_nodes,sp_adjs=MolToMolgraph(sp_node_graph,ifpadding=True,mode="ring")
        sp_nodes=torch.Tensor(sp_nodes).long().to(GP.modelsetting.device)
        sp_adjs=torch.Tensor(sp_adjs).long().to(GP.modelsetting.device).permute((1,2,0))
        sp_atomnums=len(sp_node_graph.GetAtoms())
        sp_atomics=torch.Tensor([atom.GetAtomicNum() for atom in sp_node_graph.GetAtoms()]).long().to(GP.modelsetting.device)
        
        return sp_nodes,sp_adjs,sp_atomnums,sp_atomics

    def __init_ring_graphs_with_sp_nodes(self,idx,sp_node_graph,anchor=0):
        sp_nodes,sp_adjs,sp_atomnums,sp_atomics=self.specific_node_to_ringgraph(sp_node_graph)
        self.rw_ring_nodes[idx]=sp_nodes
        self.rw_ring_adjs[idx]=sp_adjs
        self.rw_ring_atomnums[idx]=sp_atomnums
        self.rw_ring_atomics[idx,:sp_atomnums]=sp_atomics
        self.rw_ring_ftypes[idx]=0
        self.rw_ring_focused_ids[idx]=anchor
        self.rw_ring_likelihoods_agent[idx]=0
        self.rw_ring_likelihoods_prior[idx]=0
        return 
    
        
    def __update_mol_saturation_atomlist(self):
        for i in range(self.batchsize):
            nconn_constrain=self.nconn_clist[i]
            if len(nconn_constrain.saturation_atomid_list)>0:
                self.rw_mol_saturation_list[i]+=[aid+self.rw_mol_atomnums[i] for aid in nconn_constrain.saturation_atomid_list]    
        return 
    def __gen_constrains(self):
        self.nadd_clist=[]
        self.nconn_clist=[]
        for i in range(self.batchsize):
            step_for_graph=self.rw_mol_round[i].detach().cpu().numpy()
            nadd_constrain=GP.sample_constrain_setting.constrain_step_dict[str(step_for_graph)]["node add"]
            nconn_constrain=GP.sample_constrain_setting.constrain_step_dict[str(step_for_graph)]["node conn"]
            self.nadd_clist.append(nadd_constrain)
            self.nconn_clist.append(nconn_constrain)
        
        self.__update_mol_saturation_atomlist()
        self.rw_mol_nadd_mask=self.gen_nadd_valid_masks(self.nadd_clist)
        self.rw_mol_nconn_mask=self.gen_nconn_valid_masks(nconn_constrain_list=self.nconn_clist,
                                                      node_pt=self.rw_mol_pt,
                                                      mol_atom_nums=self.rw_mol_atomnums,
                                                      mol_atomic_nums=self.rw_mol_atomics,
                                                      saturation_atomid_list=self.rw_mol_saturation_list)
        return 
    

    def select_mol_nadd_constrains(self,ids):
        return self.rw_mol_nadd_mask[ids]

    def select_mol_nconn_constrains(self,ids):
        return self.rw_mol_nconn_mask[ids]
            
    def sample_batch(self,temp=1.0):
        self.node_add_model_prior.requires_grad=False
        self.ring_model_prior.requires_grad=False 
        self.node_connect_model_prior.requires_grad=False
        #print (GP.sample_constrain_setting.max_node_steps)

        self.sample_prepare()
        self.__init_mol_graphs()
        rings=[]
        while torch.sum(self.mol_stop_mask[1:])>=1:   

            self.mol_stop_mask[0]=1  
            self.__init_ring_graphs()
            self.__init_mol_graphs(idx=0)
            self.__gen_constrains()

            all_ids=torch.arange(self.batchsize,device=GP.modelsetting.device).long()
            for i in range(self.batchsize):
                self.rw_mol_pt[i,self.rw_mol_round[i],0]=self.rw_mol_atomnums[i]
            #print('pt',self.rw_mol_pt[1])
            mol_unstop_ids,mol_nodes,mol_adjs,mol_atomnums,mol_atomics,mol_pt,mol_round=self.__unstop_mol_graphs()
            mol_nadd_masks=self.select_mol_nadd_constrains(mol_unstop_ids)
            #print (mol_nadd_masks.shape)
            nadd_add_ids,nadd_term_ids,nadd_llh_agent,nadd_llh_prior=self.sample_nadd_action(mol_nodes,mol_adjs,mol_nadd_masks,temp=temp)
            self.mol_stop_mask[mol_unstop_ids[nadd_term_ids]]=0
            #print ("stop ids",mol_unstop_ids[nadd_term_ids])
            #print ("mol_stop_mask",self.mol_stop_mask)
            self.rw_ring_likelihoods_agent[mol_unstop_ids,0]=nadd_llh_agent
            self.rw_ring_likelihoods_prior[mol_unstop_ids,0]=nadd_llh_prior
            
            self.ring_stop_mask=self.mol_stop_mask.clone().detach()

            #print ('mol_stop_mask',self.mol_stop_mask) 
            nadd_mol_ids=mol_unstop_ids[nadd_add_ids[0]]
            #print (nadd_mol_ids)
            nadd_type_ids=nadd_add_ids[1]
            #print ("nadd_type_ids",nadd_type_ids)
            single_node_ids=torch.where(nadd_type_ids<len(GP.syssetting.possible_atom_types))

            self.rw_ring_nodes[nadd_mol_ids[single_node_ids],0]=self.Singlenode_To_Nodefeatures[nadd_type_ids[single_node_ids]]
            self.rw_ring_atomnums[nadd_mol_ids[single_node_ids]]+=1
            self.rw_ring_atomics[nadd_mol_ids[single_node_ids],0]=torch.Tensor(GP.syssetting.possible_atom_types).to(GP.modelsetting.device).long()[nadd_type_ids[single_node_ids]]

            self.ring_stop_mask[nadd_mol_ids[single_node_ids]]=0
            self.rw_ring_ftypes[nadd_mol_ids]=self.Nodetypeid_To_Ringftype[nadd_type_ids]

            for i in range(self.batchsize):
                if self.nadd_clist[i].specific_nodegraph:
                    self.mol_stop_mask[i]=1
                    self.__init_ring_graphs_with_sp_nodes(i,self.nadd_clist[i].specific_nodegraph,anchor=self.nconn_clist[i].anchor_before)
                    self.ring_stop_mask[i]=0
            #print (self.rw_ring_focused_ids)

            #print ('rw_ring_ftypes',self.rw_ring_ftypes[1])
            while torch.sum(self.ring_stop_mask[1:])>=1:
                self.ring_stop_mask[0]=1
                self.__init_ring_graphs(idx=0)
                #print ('ring_stop_mask',self.ring_stop_mask)
                ring_unstop_ids,ring_nodes,ring_adjs,ring_atomnums,ring_atomics,ring_ftypes,ring_focused_ids,ring_round=self.__unstop_ring_graphs()
                ring_mol_nodes,ring_mol_adjs,ring_mol_atomnums,ring_mol_atomics,ring_mol_pt,ring_mol_round=self.select_mol_graphs(ring_unstop_ids)
                self.rw_ring_round[ring_unstop_ids]+=1
                
                rgen_add_ids,rgen_conn_ids,rgen_term_ids,likelihoods_agent,likelihoods_prior=\
                    self.sample_rgen_action(ring_mol_nodes,ring_mol_adjs,ring_nodes,ring_adjs,ring_ftypes)

                #print ('rgen_term_ids',rgen_term_ids)
                rgen_add_froms = self.rw_ring_atomnums[ring_unstop_ids[rgen_add_ids[0]]]
                rgen_add_ids = (*rgen_add_ids,rgen_add_froms)
                rgen_conn_froms = self.rw_ring_atomnums[ring_unstop_ids[rgen_conn_ids[0]]]-1
                rgen_conn_ids = (*rgen_conn_ids,rgen_conn_froms)
                #print (rgen_add_idc,rgen_conn_idc,rgen_term_idc)

                invalid_ids,rgen_full_ids=self.rgen_invalid_actions(rgen_add_ids,rgen_conn_ids,ring_nodes,ring_adjs,ring_atomnums,ring_ftypes) 
                self.ring_stop_mask[ring_unstop_ids[rgen_term_ids]]=0
                self.ring_stop_mask[ring_unstop_ids[rgen_full_ids]]=0

                self.__init_failed_ring_graphs(ring_unstop_ids[invalid_ids])
                #print ('ring_stop_mask',self.ring_stop_mask)
                ring_valid_mask=self.ring_stop_mask.detach()
                #print ('invalid_ids',invalid_ids)
                #print ('ring_unstop_ids',ring_unstop_ids)
                ring_valid_mask[ring_unstop_ids[invalid_ids]]=0
                #ring add action
                #print ('ring_valid_mask',ring_valid_mask)
                rgen_add_valid_ids=torch.nonzero(ring_valid_mask[ring_unstop_ids[rgen_add_ids[0]]]>0)
                #print('rgen_add_valid_ids',rgen_add_valid_ids)
                rgen_conn_valid_ids=torch.nonzero(ring_valid_mask[ring_unstop_ids[rgen_conn_ids[0]]]>0)

                (batch,bond_to,atom_type,charge,bond_type,bond_from)=rgen_add_ids
                #print ('batch',batch)
                batch=ring_unstop_ids[batch[rgen_add_valid_ids]]
                #print ('batch',batch)
                bond_to=bond_to[rgen_add_valid_ids]
                bond_from=bond_from[rgen_add_valid_ids]
                atom_type=atom_type[rgen_add_valid_ids].detach().cpu().numpy()
                atom_type=torch.Tensor(GP.syssetting.ringat_to_molat[atom_type]).long().to(GP.modelsetting.device)
                #print ('atom_type',atom_type)
                #print ('ring_round',self.rw_ring_round)
                
                charge=charge[rgen_add_valid_ids]
                bond_type=bond_type[rgen_add_valid_ids]
                self.rw_ring_nodes[batch,bond_from,atom_type]=1
                self.rw_ring_nodes[batch,bond_from,charge+len(GP.syssetting.possible_atom_types)]=1
                self.rw_ring_atomics[batch,self.rw_ring_atomnums[batch]]=torch.Tensor(GP.syssetting.possible_atom_types).to(GP.modelsetting.device).long()[atom_type]
                #print ('ring_atomics',self.rw_ring_atomics)
                # ring likelihoods

                self.rw_ring_likelihoods_agent[batch,self.rw_ring_round[batch]]=likelihoods_agent[rgen_add_valid_ids]                
                self.rw_ring_likelihoods_prior[batch,self.rw_ring_round[batch]]=likelihoods_prior[rgen_add_valid_ids]  

                non_zero_ring_graph_ids=torch.nonzero(self.rw_ring_atomnums[batch]!=0)
                batch_masked=batch[non_zero_ring_graph_ids]
                bond_to_masked = bond_to[non_zero_ring_graph_ids]
                bond_from_masked = bond_from[non_zero_ring_graph_ids]
                bond_type_masked = bond_type[non_zero_ring_graph_ids]
                self.rw_ring_adjs[batch_masked,bond_to_masked,bond_from_masked,bond_type_masked]=1
                self.rw_ring_adjs[batch_masked,bond_from_masked,bond_to_masked,bond_type_masked]=1
                self.rw_ring_atomnums[batch]+=1

                (batch,bond_to,bond_type,bond_from)=rgen_conn_ids
                batch=ring_unstop_ids[batch[rgen_conn_valid_ids]]
                bond_to=bond_to[rgen_conn_valid_ids]
                bond_from=bond_from[rgen_conn_valid_ids]
                bond_type=bond_type[rgen_conn_valid_ids]

                self.rw_ring_adjs[batch,bond_from,bond_to,bond_type]=1
                self.rw_ring_adjs[batch,bond_to,bond_from,bond_type]=1

                self.rw_ring_likelihoods_agent[batch,self.rw_ring_round[batch]]=likelihoods_agent[rgen_conn_valid_ids]
                self.rw_ring_likelihoods_prior[batch,self.rw_ring_round[batch]]=likelihoods_prior[rgen_conn_valid_ids]

            self.__init_nconn()
            self.mol_stop_mask[0]=1
            self.__init_ring_graphs(idx=0)
            fulling_ids=torch.where(self.rw_mol_atomnums+self.rw_ring_atomnums>GP.syssetting.max_atoms)
            self.mol_stop_mask[fulling_ids]=0

            self.nconn_stop_mask=self.mol_stop_mask.clone().detach()
            null_ring_ids=torch.where(self.rw_ring_atomnums==0)
            self.nconn_stop_mask[null_ring_ids]=0
            nconn_unstop_ids=torch.where(self.nconn_stop_mask>0)[0]
            mol_nconn_masks=self.select_mol_nconn_constrains(nconn_unstop_ids)
            nconn_mol_nodes,nconn_mol_adjs,nconn_mol_atomnums,nconn_mol_atomics,nconn_mol_pt,nconn_mol_round=self.select_mol_graphs(nconn_unstop_ids)
            nconn_ring_nodes,nconn_ring_adjs,nconn_ring_atomnums,nconn_ring_atomics,nconn_ring_ftypes,nconn_ring_focused_ids,nconn_ring_round=self.select_ring_graphs(nconn_unstop_ids)
            #print ('nconn',nconn_ring_focused_ids)
            nconn_ids,likelihoods_agent,likelihoods_prior=self.sample_conn_action(nconn_mol_nodes,nconn_mol_adjs,nconn_ring_nodes,nconn_ring_adjs,nconn_ring_focused_ids,mol_nconn_masks)

            nconn_from = self.rw_mol_atomnums[nconn_unstop_ids[nconn_ids[0]]]
            for nconn_from_id,idx in enumerate(nconn_unstop_ids[nconn_ids[0]]):
                if self.rw_ring_focused_ids[idx]!=0:
                    nconn_from[nconn_from_id]+=self.rw_ring_focused_ids[idx]
                    
            nconn_ids = (*nconn_ids,nconn_from)
            nconn_valid_mask=self.nconn_stop_mask.detach()
            
            invalid_ids=self.conn_invalid_actions(nconn_ids,nconn_mol_atomnums) 
            
            nconn_valid_mask[nconn_unstop_ids[invalid_ids]]=0
            nconn_valid_ids=torch.nonzero(nconn_valid_mask[nconn_unstop_ids[nconn_ids[0]]]>0).view(-1)

            (batch,bond_to,bond_type,bond_from)=nconn_ids
            #print ('batch0',batch)
            batch=nconn_unstop_ids[batch[nconn_valid_ids]].long()

            bond_to=bond_to[nconn_valid_ids].long()
            bond_type=bond_type[nconn_valid_ids].long()
            bond_from=bond_from[nconn_valid_ids].long()
            likelihoods_agent=likelihoods_agent[nconn_valid_ids]
            likelihoods_prior=likelihoods_prior[nconn_valid_ids]

            non_zero_mol_graph_ids=torch.nonzero(self.rw_mol_atomnums[batch]!=0)
            batch_masked=batch[non_zero_mol_graph_ids]
            bond_to_masked = bond_to[non_zero_mol_graph_ids]
            bond_from_masked = bond_from[non_zero_mol_graph_ids]
            bond_type_masked = bond_type[non_zero_mol_graph_ids]
            likelihoods_agent_masked=likelihoods_agent[non_zero_mol_graph_ids]
            likelihoods_prior_masked=likelihoods_prior[non_zero_mol_graph_ids]

            self.rw_conn_likelihoods_agent[batch_masked]=likelihoods_agent_masked 
            self.rw_conn_likelihoods_prior[batch_masked]=likelihoods_prior_masked
            self.rw_mol_adjs[batch_masked,bond_from_masked,bond_to_masked,bond_type_masked]=1
            self.rw_mol_adjs[batch_masked,bond_to_masked,bond_from_masked,bond_type_masked]=1
            
            for j in range(self.batchsize):
                if self.rw_mol_round[j]==0 or j in batch_masked.view(-1):
                    if self.nconn_stop_mask[j]==1:
                        natom_mol=self.rw_mol_atomnums[j].long()
                        natom_ring=self.rw_ring_atomnums[j].long()
                        self.rw_mol_nodes[j,natom_mol:natom_mol+natom_ring]=self.rw_ring_nodes[j,:natom_ring]
                        self.rw_mol_adjs[j,natom_mol:natom_mol+natom_ring,natom_mol:natom_mol+natom_ring]=self.rw_ring_adjs[j,:natom_ring,:natom_ring]
                        self.rw_mol_atomics[j,natom_mol:natom_mol+natom_ring]=self.rw_ring_atomics[j,:natom_ring]
                        self.rw_mol_atomnums[j]+=self.rw_ring_atomnums[j]
                        self.rw_conn_likelihoods_agent[j]+=torch.sum(self.rw_ring_likelihoods_agent[j])
                        self.rw_conn_likelihoods_prior[j]+=torch.sum(self.rw_ring_likelihoods_prior[j])
                        self.rw_mol_likelihoods_agent[j,self.rw_mol_round[j]]=self.rw_conn_likelihoods_agent[j]
                        self.rw_mol_likelihoods_prior[j,self.rw_mol_round[j]]=self.rw_conn_likelihoods_prior[j]
                        self.rw_mol_pt[j,self.rw_mol_round[j],1]=self.rw_mol_atomnums[j]
                        self.rw_mol_round[j]+=1
                if self.rw_mol_round[j]>=GP.sample_constrain_setting.max_node_steps:
                    self.mol_stop_mask[j]=0
            #print ('rw_mol_pt',self.rw_mol_pt[1]) 
        self.rw_mol_likelihoods_agent=torch.sum(self.rw_mol_likelihoods_agent,dim=1)
        self.rw_mol_likelihoods_prior=torch.sum(self.rw_mol_likelihoods_prior,dim=1)
        return self.rw_mol_nodes[1:],self.rw_mol_adjs[1:],self.rw_mol_atomnums[1:],self.rw_mol_likelihoods_agent[1:],self.rw_mol_likelihoods_prior[1:]

    def sample(self,sample_num):
        sampled_num=0
        sample_mols=[]
        sample_smis=[]
        while sampled_num<sample_num:
            mol_nodes,mol_adjs,mol_atomnums,ll_agent,ll_prior=self.sample_batch()
            mols,smis,valid_ids=self.nodes_edges_to_mol(mol_nodes,mol_adjs,mol_atomnums) 
            valid_num=len(valid_ids)
            valid_mols=[mols[i] for i in valid_ids]
            valid_smis=[smis[i] for i in valid_ids]
            valid_ids=torch.Tensor(valid_ids).long().to(GP.modelsetting.device)
            valid_mol_nodes,valid_mol_adjs,valid_mol_atomnums,valid_ll_agent,valid_ll_prior=mol_nodes[valid_ids],mol_adjs[valid_ids],mol_atomnums[valid_ids],ll_agent[valid_ids],ll_prior[valid_ids]
            if sampled_num+valid_num<sample_num:
                sample_mols+=valid_mols
                sample_smis+=valid_smis
            else:
                sample_mols+=valid_mols[:sample_num-sampled_num]
                sample_smis+=valid_smis[:sample_num-sampled_num]
            sampled_num+=valid_num

        return sample_mols,sample_smis

    

    def node_feature_to_atom(self,node_features,node_idx):
        nonzero_idc=torch.nonzero(node_features[node_idx])
        atom_idx = nonzero_idc[0]
        atom_type = GP.syssetting.possible_atom_types[atom_idx]
        new_atom=Chem.Atom(atom_type)
        fc_idx=nonzero_idc[1]-len(GP.syssetting.possible_atom_types)
        formal_charge = GP.syssetting.formal_charge_types[fc_idx]
        new_atom.SetFormalCharge(formal_charge)
        return new_atom

    def nodes_edges_to_mol(self,nodes_batch,edges_batch,atomnums):
        mols=[]
        smis=[]
        valid_ids=[]
        ngraphs=nodes_batch.shape[0]
        for i in range(ngraphs):
            #print (f'Dealing with {i+1}th graphs')
            natoms=atomnums[i]
            graph_nodes=nodes_batch[i]
            graph_edges=edges_batch[i]
            #print (graph_edges[:natoms,:natoms].permute(2,0,1))
            #print (graph_edges.permute(2,0,1))
            node_to_idx={}
            molecule=Chem.RWMol()
            for j in range(natoms.long()):
                atom_to_add=self.node_feature_to_atom(graph_nodes,j)
                molecule_idx=molecule.AddAtom(atom_to_add)
                node_to_idx[j]=molecule_idx
            edge_mask = torch.triu(
                    torch.ones((GP.syssetting.max_atoms, GP.syssetting.max_atoms), device=GP.modelsetting.device),
                    diagonal=1
                    ).view(GP.syssetting.max_atoms,GP.syssetting.max_atoms,1)
            edges_idc=torch.nonzero(graph_edges*edge_mask)
            for node_idx1,node_idx2,bond_idx in edges_idc:
                try:
                    molecule.AddBond(node_to_idx[node_idx1.item()],node_to_idx[node_idx2.item()],GP.syssetting.bond_types[bond_idx.item()])
                except:
                    pass
            try:
                mol=molecule.GetMol()
                Chem.SanitizeMol(mol)
                mols.append(mol)
                smi=Chem.MolToSmiles(mol)
                smis.append(smi)
                valid_ids.append(i)
            except Exception as e:
                print (e)
                mols.append(None)
                smis.append(None)

        return mols,smis,valid_ids
    


        




