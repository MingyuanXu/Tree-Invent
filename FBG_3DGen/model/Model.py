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
from rdkit import DataStructs

def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """
    Clip the norm of the gradients for all parameters under `optimizer`.
    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)

class Node_adder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if GP.modelsetting.graphnet_type=='GGNN':
            self.gnn=GGNN()
            self.node_add_net_1 = MLP(
                in_features=GP.modelsetting.hidden_node_features,
                hidden_layer_sizes=[GP.modelsetting.mlp1_hidden_dim] * GP.modelsetting.mlp1_depth,
                out_features=64,
                dropout_p=GP.modelsetting.mlp1_dropout_p
                )
            self.node_add_net_2 = MLP(
                in_features=64*GP.syssetting.max_atoms+GP.modelsetting.gather_width,
                hidden_layer_sizes=[GP.modelsetting.mlp2_hidden_dim] * GP.modelsetting.mlp2_depth,
                out_features=GP.syssetting.num_node_types,
                dropout_p=GP.modelsetting.mlp2_dropout_p
                )
            self.node_terminate_net2= MLP(
            in_features=GP.modelsetting.gather_width,
            hidden_layer_sizes=[GP.modelsetting.mlp2_hidden_dim] * GP.modelsetting.mlp2_depth,
            out_features=1,
            dropout_p=GP.modelsetting.mlp2_dropout_p
            )
        return

    def forward(self,nodes,edges):
        graph_embedding,node_embedding=self.gnn(nodes,edges)
        f_node_add1=self.node_add_net_1(node_embedding)
        #print ('++++++++++++++++++++++++++++++++++++++++')
        #print (f_node_add1.shape)
        f_node_add1=f_node_add1.view(-1,GP.syssetting.max_atoms*64)
        #print (f_node_add1.shape,graph_embedding.shape)
        #print ('++++++++++++++++++++++++++++++++++++++++')
        f_node_add2=torch.cat((f_node_add1,graph_embedding),dim=1)
        add_output=self.node_add_net_2(f_node_add2)
        terminate_output=self.node_terminate_net2(graph_embedding)
        action_output=torch.cat((add_output,terminate_output),dim=1)
        return action_output

class Ring_generator(torch.nn.Module):
    def __init__(self,graphnet_type='GGNN'):
        super().__init__()
        if GP.modelsetting.graphnet_type=='GGNN':
            self.molgnn=GGNN()
            self.ringgnn=GGNN()
            self.ring_node_add_net_1 = MLP(
                in_features=GP.modelsetting.hidden_node_features,
                hidden_layer_sizes=[GP.modelsetting.mlp1_hidden_dim] * GP.modelsetting.mlp1_depth,
                out_features=GP.syssetting.f_ring_add_per_node,
                dropout_p=GP.modelsetting.mlp1_dropout_p)

            self.ring_node_add_net_2 = MLP(
                in_features=GP.syssetting.f_ring_add_per_node*GP.syssetting.max_ring_size+GP.modelsetting.gather_width*2+GP.syssetting.n_clique_features,
                hidden_layer_sizes=[GP.modelsetting.mlp2_hidden_dim] * GP.modelsetting.mlp2_depth,
                out_features=np.prod(GP.syssetting.f_ring_add_dim),
                dropout_p=GP.modelsetting.mlp2_dropout_p)
            #print ([GP.modelsetting.mlp1_hidden_dim] * GP.modelsetting.mlp1_depth,GP.syssetting.f_ring_connect_dim)
            self.ring_node_connect_net_1 = MLP(
                in_features=GP.modelsetting.hidden_node_features,
                hidden_layer_sizes=[GP.modelsetting.mlp1_hidden_dim] * GP.modelsetting.mlp1_depth,
                out_features=GP.syssetting.f_ring_connect_per_node,
                dropout_p=GP.modelsetting.mlp1_dropout_p)

            self.ring_node_connect_net_2 = MLP(
                in_features=GP.syssetting.f_ring_connect_per_node*GP.syssetting.max_ring_size+GP.modelsetting.gather_width*2+GP.syssetting.n_clique_features,
                hidden_layer_sizes=[GP.modelsetting.mlp2_hidden_dim] * GP.modelsetting.mlp2_depth,
                out_features=np.prod(GP.syssetting.f_ring_connect_dim),
                dropout_p=GP.modelsetting.mlp2_dropout_p)

            self.ring_node_terminate_net_2 = MLP(
                in_features=GP.modelsetting.gather_width+GP.syssetting.n_clique_features,
                hidden_layer_sizes=[GP.modelsetting.mlp2_hidden_dim] * GP.modelsetting.mlp2_depth,
                out_features=1,
                dropout_p=GP.modelsetting.mlp2_dropout_p
                )
        else:
            pass 
        return 

    def forward(self,molnodes,moledges,ringnodes,ringedges,f_t):
        molgraph_embedding,molnode_embedding=self.molgnn(molnodes,moledges)
        ringgraph_embedding,ringnode_embedding=self.ringgnn(ringnodes,ringedges)
        #print (ringnode_embedding.shape)
        f_ring_node_add1=self.ring_node_add_net_1(ringnode_embedding)
        f_ring_node_add1=f_ring_node_add1.view(-1,GP.syssetting.max_ring_size*GP.syssetting.f_ring_add_per_node)
        #print (f_ring_node_add1.shape,ringgraph_embedding.shape,molgraph_embedding.shape,f_t.shape)
        f_ring_node_add2=torch.cat((f_ring_node_add1,ringgraph_embedding,molgraph_embedding,f_t),dim=1)
        add_output=self.ring_node_add_net_2(f_ring_node_add2)

        f_ring_node_connect1=self.ring_node_connect_net_1(ringnode_embedding)
        f_ring_node_connect1=f_ring_node_connect1.view(-1,GP.syssetting.max_ring_size*GP.syssetting.f_ring_connect_per_node)
        f_ring_node_connect2=torch.cat((f_ring_node_connect1,ringgraph_embedding,molgraph_embedding,f_t),dim=1)
        connect_output=self.ring_node_connect_net_2(f_ring_node_connect2)
        f_ring_node_terminate=torch.cat((ringgraph_embedding,f_t),dim=1)
        terminate_output=self.ring_node_terminate_net_2(f_ring_node_terminate)
        action_output=torch.cat((add_output,connect_output,terminate_output),dim=1)
        return action_output

class Node_connecter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if GP.modelsetting.graphnet_type=='GGNN':
            self.molgnn=GGNN()
            self.ringgnn=GGNN()

            self.node_connect_net_1 = MLP(
                in_features=GP.modelsetting.hidden_node_features,
                hidden_layer_sizes=[GP.modelsetting.mlp1_hidden_dim] * GP.modelsetting.mlp1_depth,
                out_features=64,
                dropout_p=GP.modelsetting.mlp1_dropout_p)

            self.node_connect_net_2 = MLP(
                in_features=64*GP.syssetting.max_atoms+GP.modelsetting.gather_width*2+GP.modelsetting.hidden_node_features,
                hidden_layer_sizes=[GP.modelsetting.mlp2_hidden_dim] * GP.modelsetting.mlp2_depth,
                out_features=np.prod(GP.syssetting.f_node_joint_dim),
                dropout_p=GP.modelsetting.mlp2_dropout_p)

    def forward(self,molnodes,moledges,ringnodes,ringedges,focused_ids):
        molgraph_embedding,molnode_embedding=self.molgnn(molnodes,moledges)
        ringgraph_embedding,ringnode_embedding=self.ringgnn(ringnodes,ringedges)
        batch_ids=torch.arange(molnodes.shape[0]).long().to(focused_ids) 
        focused_node_embedding=ringnode_embedding[batch_ids,focused_ids,:]
        
        f_node_connect1=self.node_connect_net_1(molnode_embedding)
        f_node_connect1=f_node_connect1.view(-1,GP.syssetting.max_atoms*64)
        f_ring_node_connect2=torch.cat((f_node_connect1,ringgraph_embedding,molgraph_embedding,focused_node_embedding),dim=1)
        connect_output=self.node_connect_net_2(f_ring_node_connect2)
        return connect_output

class MolGen_Model:
    def __init__(self,**kwargs):
        epochs=kwargs.get('start')
        if "modelname" not in kwargs:
            self.mode="train"
            self.modelname='MFTs_Gen_Model'
            self.training_history=[]
            self.batchsize=GP.trainsetting.batchsize
            self.batch_train_loss_dict={''}
            self.n_train_steps=None
            self.n_valid_steps=None
            self.record_num={'nadd':0,'rgen':0,'njoint':0,'total':0}
            self.record_vnum={'nadd':0,'rgen':0,'njoint':0,'total':0}
            self.train_loss={'nadd':0,'rgen':0,'njoint':0,'total':0}
            self.min_train_loss={'nadd':1e20,'rgen':1e20,'njoint':1e20,'total':1e20}
            self.valid_loss={'nadd':0,'rgen':0,'njoint':0,'total':0}
            self.min_valid_loss={'nadd':1e20,'rgen':1e20,'njoint':1e20,'total':1e20}
            self.training_history=[]
            self.epochs=0
            self.node_add_model=None
            self.ring_model=None
            self.node_connect_model=None
            self.optimizer=None
            self.lr_scheduler=None
        else:
            self.mode="test"
            self.modelname=kwargs.get('modelname')
        if not os.path.exists(f'./{self.modelname}/model'):
            os.system(f'mkdir -p ./{self.modelname}/model')

        if self.mode=="train":
            pickle.dump(self.__dict__,open(self.modelname+"/modelsetting.pickle", "wb"))

            self.__build_model()
            if GP.modelsetting.device=='cuda':
                self.node_add_model.to('cuda')
                self.ring_model.to('cuda')
                self.node_connect_model.to('cuda')

            self.logger=open(f'./{self.modelname}/Training.log','a')
            self.logger.write('='*40+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n') 
            self.logger.flush()
        else:
            self.modelname=kwargs.get("modelname")
            self.loadtype=kwargs.get("loadtype")
            mode=self.mode 
            self.load(self.modelname,self.loadtype)
            self.mode=mode
            self.batchsize=GP.trainsetting.batchsize

            self.logger=open(f'./{self.modelname}/Training.log','a')
            now = datetime.now()
            self.logger.write('='*40+now.strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n')
            self.logger.flush()
        if epochs:
            self.epochs=epochs 
        self.node_type_dict=GP.syssetting.node_type_dict
        self.f_node_dict=GP.syssetting.f_node_dict
        return
        
    def __build_model(self,):
        self.node_add_model=Node_adder()
        self.ring_model=Ring_generator()
        self.node_connect_model=Node_connecter()
        return

    def trainable_params(self):
        nadd_param_num=sum(p.numel() for p in self.node_add_model.parameters())
        nadd_trainable_num=sum(p.numel() for p in self.node_add_model.parameters() if p.requires_grad)
        rgen_param_num=sum(p.numel() for p in self.ring_model.parameters())
        rgen_trainable_num=sum(p.numel() for p in self.ring_model.parameters() if p.requires_grad)
        njoint_param_num=sum(p.numel() for p in self.node_connect_model.parameters())
        njoint_trainable_num=sum(p.numel() for p in self.node_connect_model.parameters() if p.requires_grad)
        return f' nadd: {nadd_param_num} / {nadd_trainable_num}\n rgen: {rgen_param_num} / {rgen_trainable_num} \n njoint: {njoint_param_num} / {njoint_trainable_num}'

    def Save(self,modelname=''):
        self.node_add_model=None
        self.ring_model=None
        self.node_connect_model=None
        self.lr_scheduler=None
        self.logger.close()
        self.logger=None
        self.optimizer=None
        self.traindataloader=None
        self.validdataloader=None
        pickle.dump(self.__dict__,open(self.modelname+"/modelsetting.pickle", "wb"))
        shutil.make_archive(self.modelname,"zip",self.modelname)
        return 

    def load(self,modelname,loadtype):
        with tempfile.TemporaryDirectory() as dirpath:
            with zipfile.ZipFile(modelname + ".zip", "r") as zip_ref:
                zip_ref.extractall(dirpath)
            # Load metadata
            metadata = pickle.load(open(dirpath + "/modelsetting.pickle", "rb"))
            #print (metadata.keys())
            #print (metadata)
            self.__dict__.update(metadata)
            self.__build_model()
            if loadtype=='Minloss':
                node_add_modelcpkt=torch.load(dirpath+"/model/node_add_model_minloss.cpk")
                self.node_add_model.load_state_dict(node_add_modelcpkt["state_dict"])
                ring_modelcpkt=torch.load(dirpath+"/model/ring_model_minloss.cpk")
                self.ring_model.load_state_dict(ring_modelcpkt["state_dict"])
                node_connect_modelcpkt=torch.load(dirpath+"/model/node_connect_model_minloss.cpk")
                self.node_connect_model.load_state_dict(node_connect_modelcpkt["state_dict"])
                #self.epochs=max((node_add_modelcpkt['epochs'],ring_modelcpkt['epochs'],node_connect_modelcpkt['epochs']))
                self.epochs=node_add_modelcpkt['epochs']
                print ("Load model successfully!")
            elif loadtype=='Finetune':
                node_add_modelcpkt=torch.load(dirpath+"/model/node_add_model_finetune.cpk")
                self.node_add_model.load_state_dict(node_add_modelcpkt["state_dict"])
                ring_modelcpkt=torch.load(dirpath+"/model/ring_model_finetune.cpk")
                self.ring_model.load_state_dict(ring_modelcpkt["state_dict"])
                node_connect_modelcpkt=torch.load(dirpath+"/model/node_connect_model_finetune.cpk")
                self.node_connect_model.load_state_dict(node_connect_modelcpkt["state_dict"])
                self.epochs=max((node_add_modelcpkt['epochs'],ring_modelcpkt['epochs'],node_connect_modelcpkt['epochs']))
            else:
                node_add_modelcpkt=torch.load(dirpath+"/model/node_add_model_perepoch.cpk")
                self.node_add_model.load_state_dict(node_add_modelcpkt["state_dict"])
                ring_modelcpkt=torch.load(dirpath+"/model/ring_model_perepoch.cpk")
                self.ring_model.load_state_dict(ring_modelcpkt["state_dict"])
                node_connect_modelcpkt=torch.load(dirpath+"/model/node_connect_perepoch.cpk")
                self.node_connect_model.load_state_dict(node_connect_modelcpkt["state_dict"])
                self.epochs=max((node_add_modelcpkt['epochs'],ring_modelcpkt['epochs'],node_connect_modelcpkt['epochs']))
                
            #self.training_history=modelcpkt['training_history']
            if GP.modelsetting.device=='cuda':
                self.node_add_model.cuda()
                self.ring_model.cuda()
                self.node_connect_model.cuda()
        return 

    def kl_loss(self,output,target,termination_weight=1):
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
        #criterion = torch.nn.KLDivLoss(reduction="batchmean")
        criterion = torch.nn.KLDivLoss(reduction="none")
        weight=torch.ones_like(target_output).cuda()
        if termination_weight!=1:
            weight[:,-1]=termination_weight
        loss = criterion(target=target_output, input=output)*weight
        #print (target_output.shape,output.shape,loss.shape)
        loss = loss.sum() / output.size(0)
        #print (loss.shape)
        return loss 

    def train_step(self,datas):
        self.node_add_model.zero_grad()
        self.ring_model.zero_grad()
        self.node_connect_model.zero_grad()
        
        nadd_mg_nodes=datas['Nadd_mg_nodes'].view(-1,GP.syssetting.max_atoms,GP.syssetting.n_node_features)
        nadd_mg_edges=datas['Nadd_mg_edges'].view(-1,GP.syssetting.n_edge_features,GP.syssetting.max_atoms,GP.syssetting.max_atoms).permute(0,2,3,1)
        nadd_apd=datas['Nadd_apd'].view(-1,GP.syssetting.f_graph_add_dim+GP.syssetting.f_graph_termination_dim)
        nadd_step_mask=datas['Nadd_step_masks'].view(-1)
        nadd_valid_ids=torch.where(nadd_step_mask)
        nadd_mg_nodes=nadd_mg_nodes[nadd_valid_ids]
        nadd_mg_edges=nadd_mg_edges[nadd_valid_ids]
        nadd_apd=nadd_apd[nadd_valid_ids]

        rgen_step_mask=datas['Rgen_step_masks'].view(-1)
        if torch.sum(rgen_step_mask)>0:
            rgen_valid_ids=torch.where(rgen_step_mask)
            rgen_mg_nodes=datas['Rgen_mg_nodes'].view(-1,GP.syssetting.max_atoms,GP.syssetting.n_node_features)
            rgen_mg_edges=datas['Rgen_mg_edges'].view(-1,GP.syssetting.n_edge_features,GP.syssetting.max_atoms,GP.syssetting.max_atoms).permute(0,2,3,1)
            rgen_rg_nodes=datas['Rgen_rg_nodes'].view(-1,GP.syssetting.max_ring_size,GP.syssetting.n_node_features)
            rgen_rg_edges=datas['Rgen_rg_edges'].view(-1,GP.syssetting.n_edge_features,GP.syssetting.max_ring_size,GP.syssetting.max_ring_size).permute(0,2,3,1)
            rgen_rg_ftype=datas['Rgen_rg_ftype'].view(-1,GP.syssetting.n_clique_features)
            rgen_apd=datas['Rgen_apd'].view(-1,np.prod(GP.syssetting.f_ring_add_dim)+np.prod(GP.syssetting.f_ring_connect_dim)+np.prod(GP.syssetting.f_ring_termination_dim))

            rgen_mg_nodes=rgen_mg_nodes[rgen_valid_ids]
            rgen_mg_edges=rgen_mg_edges[rgen_valid_ids]
            rgen_rg_nodes=rgen_rg_nodes[rgen_valid_ids]
            rgen_rg_edges=rgen_rg_edges[rgen_valid_ids]
            rgen_rg_ftype=rgen_rg_ftype[rgen_valid_ids]
            rgen_apd=rgen_apd[rgen_valid_ids]
        njoint_step_mask=datas['Njoint_step_masks'].view(-1) 
        #print (torch.sum(rgen_mg_edges))
        #print (torch.sum(rgen_step_mask),torch.sum(njoint_step_mask))
        if torch.sum(njoint_step_mask)>0:           
            njoint_mg_nodes=datas['Njoint_mg_nodes'].view(-1,GP.syssetting.max_atoms,GP.syssetting.n_node_features)
            njoint_mg_edges=datas['Njoint_mg_edges'].view(-1,GP.syssetting.n_edge_features,GP.syssetting.max_atoms,GP.syssetting.max_atoms).permute(0,2,3,1)
            njoint_rg_nodes=datas['Njoint_rg_nodes'].view(-1,GP.syssetting.max_ring_size,GP.syssetting.n_node_features)
            njoint_rg_edges=datas['Njoint_rg_edges'].view(-1,GP.syssetting.n_edge_features,GP.syssetting.max_ring_size,GP.syssetting.max_ring_size).permute(0,2,3,1)
            njoint_apd=datas['Njoint_apd'].view(-1,np.prod(GP.syssetting.f_node_joint_dim))
            njoint_focused_ids=datas['Njoint_focused_ids'].view(-1)
            njoint_valid_ids=torch.where(njoint_step_mask)
            njoint_mg_nodes=njoint_mg_nodes[njoint_valid_ids]
            njoint_mg_edges=njoint_mg_edges[njoint_valid_ids]
            njoint_rg_nodes=njoint_rg_nodes[njoint_valid_ids]
            njoint_rg_edges=njoint_rg_edges[njoint_valid_ids]
            njoint_apd=njoint_apd[njoint_valid_ids]
            njoint_focused_ids=njoint_focused_ids[njoint_valid_ids]

        if GP.modelsetting.device=='cuda':
            nadd_mg_nodes,nadd_mg_edges,nadd_apd=nadd_mg_nodes.cuda(),nadd_mg_edges.cuda(),nadd_apd.cuda()
            if torch.sum(rgen_step_mask)>0:
                rgen_mg_nodes,rgen_mg_edges,rgen_rg_nodes,rgen_rg_edges,rgen_rg_ftype,rgen_apd=\
                    rgen_mg_nodes.cuda(),rgen_mg_edges.cuda(),rgen_rg_nodes.cuda(),rgen_rg_edges.cuda(),rgen_rg_ftype.cuda(),rgen_apd.cuda()
            if torch.sum(njoint_step_mask)>0:
                njoint_mg_nodes,njoint_mg_edges,njoint_rg_nodes,njoint_rg_edges,njoint_apd,njoint_focused_ids=\
                    njoint_mg_nodes.cuda(),njoint_mg_edges.cuda(),njoint_rg_nodes.cuda(),njoint_rg_edges.cuda(),njoint_apd.cuda(),njoint_focused_ids.cuda()
        
        nadd_apd_predict=self.node_add_model(nadd_mg_nodes,nadd_mg_edges)
        nadd_apd_loss=self.kl_loss(nadd_apd_predict,nadd_apd,termination_weight=50)
        total_loss=nadd_apd_loss*nadd_apd_predict.shape[0]
        total_loss_dim=nadd_apd_predict.shape[0]
        if torch.sum(njoint_step_mask)>0:
            njoint_apd_predict=self.node_connect_model(njoint_mg_nodes,njoint_mg_edges,njoint_rg_nodes,njoint_rg_edges,njoint_focused_ids)
            njoint_apd_loss=self.kl_loss(njoint_apd_predict,njoint_apd)
            total_loss+=njoint_apd_loss*njoint_apd_predict.shape[0] 
            total_loss_dim+=njoint_apd_predict.shape[0]

        if torch.sum(rgen_step_mask)>0:
            rgen_apd_predict=self.ring_model(rgen_mg_nodes,rgen_mg_edges,rgen_rg_nodes,rgen_rg_edges,rgen_rg_ftype)
            rgen_apd_loss=self.kl_loss(rgen_apd_predict,rgen_apd)
            total_loss+=rgen_apd_loss*rgen_apd_predict.shape[0]
            total_loss_dim+=rgen_apd_predict.shape[0]

        total_loss=total_loss/total_loss_dim

        #print (nadd_apd_loss.shape,rgen_apd_loss.shape,njoint_apd_loss.shape)
        #total_loss=(nadd_apd_loss*nadd_apd_predict.shape[0]+rgen_apd_loss*rgen_apd_predict.shape[0]+njoint_apd_loss*njoint_apd_predict.shape[0])/(nadd_apd_predict.shape[0]+rgen_apd_predict.shape[0]+njoint_apd_predict.shape[0])
        if not torch.isnan(total_loss):
            total_loss.backward()
            if GP.trainsetting.max_grad_norm>0:
                clip_grad_norm(self.optimizer,GP.trainsetting.max_grad_norm)
            self.optimizer.step()
            self.lr=self.optimizer.state_dict()['param_groups'][0]['lr']
            lstr=f'Lr: {self.lr} ;'
            self.train_loss['nadd']+=nadd_apd_loss.item()*nadd_apd_predict.shape[0]
            self.record_num['nadd']+=nadd_apd.shape[0]
            lstr+=f'Node Add: {nadd_apd_loss.item():.3E} ;'
            if torch.sum(rgen_step_mask)>0:
                self.train_loss['rgen']+=rgen_apd_loss.item()*rgen_apd_predict.shape[0]
                self.record_num['rgen']+=rgen_apd.shape[0]
                lstr+=f' Ring Gen: {rgen_apd_loss.item():.3E} ;'

            if torch.sum(njoint_step_mask)>0:
                self.train_loss['njoint']+=njoint_apd_loss.item()*njoint_apd_predict.shape[0]
                self.record_num['njoint']+=njoint_apd.shape[0]
                lstr+=f'Node Joint: {njoint_apd_loss.item():.3E} ;'

        else:
            lstr=f'Loss is unnormal: {nadd_apd_loss.item():.3E}/'
            if torch.sum(rgen_step_mask)>0:
                lstr+=f'{rgen_apd_loss.item():.3E}/'
            if torch.sum(njoint_step_mask)>0:
                lstr+=f'{njoint_apd_loss.item():.3E}'

        return lstr
    
    def evaluate_step(self,datas):
        self.node_add_model.zero_grad()
        self.ring_model.zero_grad()
        self.node_connect_model.zero_grad()
        nadd_step_mask=datas['Nadd_step_masks'].view(-1)
        nadd_mg_nodes=datas['Nadd_mg_nodes'].view(-1,GP.syssetting.max_atoms,GP.syssetting.n_node_features)
        nadd_mg_edges=datas['Nadd_mg_edges'].view(-1,GP.syssetting.n_edge_features,GP.syssetting.max_atoms,GP.syssetting.max_atoms).permute(0,2,3,1)
        nadd_apd=datas['Nadd_apd'].view(-1,GP.syssetting.f_graph_add_dim+GP.syssetting.f_graph_termination_dim)
        nadd_valid_ids=torch.where(nadd_step_mask)
        nadd_mg_nodes=nadd_mg_nodes[nadd_valid_ids]
        nadd_mg_edges=nadd_mg_edges[nadd_valid_ids]
        nadd_apd=nadd_apd[nadd_valid_ids]
        
        rgen_step_mask=datas['Rgen_step_masks'].view(-1)
        if torch.sum(rgen_step_mask)>0:
            rgen_valid_ids=torch.where(rgen_step_mask)
            rgen_mg_nodes=datas['Rgen_mg_nodes'].view(-1,GP.syssetting.max_atoms,GP.syssetting.n_node_features)
            rgen_mg_edges=datas['Rgen_mg_edges'].view(-1,GP.syssetting.n_edge_features,GP.syssetting.max_atoms,GP.syssetting.max_atoms).permute(0,2,3,1)
            rgen_rg_nodes=datas['Rgen_rg_nodes'].view(-1,GP.syssetting.max_ring_size,GP.syssetting.n_node_features)
            rgen_rg_edges=datas['Rgen_rg_edges'].view(-1,GP.syssetting.n_edge_features,GP.syssetting.max_ring_size,GP.syssetting.max_ring_size).permute(0,2,3,1)
            rgen_rg_ftype=datas['Rgen_rg_ftype'].view(-1,GP.syssetting.n_clique_features)
            rgen_apd=datas['Rgen_apd'].view(-1,np.prod(GP.syssetting.f_ring_add_dim)+np.prod(GP.syssetting.f_ring_connect_dim)+np.prod(GP.syssetting.f_ring_termination_dim))

            rgen_mg_nodes=rgen_mg_nodes[rgen_valid_ids]
            rgen_mg_edges=rgen_mg_edges[rgen_valid_ids]
            rgen_rg_nodes=rgen_rg_nodes[rgen_valid_ids]
            rgen_rg_edges=rgen_rg_edges[rgen_valid_ids]
            rgen_rg_ftype=rgen_rg_ftype[rgen_valid_ids]
            rgen_apd=rgen_apd[rgen_valid_ids]

        njoint_step_mask=datas['Njoint_step_masks'].view(-1) 
        if torch.sum(njoint_step_mask)>0:           
            njoint_mg_nodes=datas['Njoint_mg_nodes'].view(-1,GP.syssetting.max_atoms,GP.syssetting.n_node_features)
            njoint_mg_edges=datas['Njoint_mg_edges'].view(-1,GP.syssetting.n_edge_features,GP.syssetting.max_atoms,GP.syssetting.max_atoms).permute(0,2,3,1)
            njoint_rg_nodes=datas['Njoint_rg_nodes'].view(-1,GP.syssetting.max_ring_size,GP.syssetting.n_node_features)
            njoint_rg_edges=datas['Njoint_rg_edges'].view(-1,GP.syssetting.n_edge_features,GP.syssetting.max_ring_size,GP.syssetting.max_ring_size).permute(0,2,3,1)
            njoint_apd=datas['Njoint_apd'].view(-1,np.prod(GP.syssetting.f_node_joint_dim))
            njoint_focused_ids=datas['Njoint_focused_ids'].view(-1)
            njoint_valid_ids=torch.where(njoint_step_mask)
            njoint_mg_nodes=njoint_mg_nodes[njoint_valid_ids]
            njoint_mg_edges=njoint_mg_edges[njoint_valid_ids]
            njoint_rg_nodes=njoint_rg_nodes[njoint_valid_ids]
            njoint_rg_edges=njoint_rg_edges[njoint_valid_ids]
            njoint_apd=njoint_apd[njoint_valid_ids]
            njoint_focused_ids=njoint_focused_ids[njoint_valid_ids]


        if GP.modelsetting.device=='cuda':
            nadd_mg_nodes,nadd_mg_edges,nadd_apd=nadd_mg_nodes.cuda(),nadd_mg_edges.cuda(),nadd_apd.cuda()
            if torch.sum(rgen_step_mask)>0:
                rgen_mg_nodes,rgen_mg_edges,rgen_rg_nodes,rgen_rg_edges,rgen_rg_ftype,rgen_apd=\
                    rgen_mg_nodes.cuda(),rgen_mg_edges.cuda(),rgen_rg_nodes.cuda(),rgen_rg_edges.cuda(),rgen_rg_ftype.cuda(),rgen_apd.cuda()
            if torch.sum(njoint_step_mask)>0:
                njoint_mg_nodes,njoint_mg_edges,njoint_rg_nodes,njoint_rg_edges,njoint_apd,njoint_focused_ids=\
                    njoint_mg_nodes.cuda(),njoint_mg_edges.cuda(),njoint_rg_nodes.cuda(),njoint_rg_edges.cuda(),njoint_apd.cuda(),njoint_focused_ids.cuda()
            
        
        nadd_apd_predict=self.node_add_model(nadd_mg_nodes,nadd_mg_edges)
        nadd_apd_loss=self.kl_loss(nadd_apd_predict,nadd_apd)
        total_loss=nadd_apd_loss*nadd_apd_predict.shape[0]
        total_loss_dim=nadd_apd_predict.shape[0]

        if torch.sum(njoint_step_mask)>0:
            njoint_apd_predict=self.node_connect_model(njoint_mg_nodes,njoint_mg_edges,njoint_rg_nodes,njoint_rg_edges,njoint_focused_ids)
            njoint_apd_loss=self.kl_loss(njoint_apd_predict,njoint_apd)
            total_loss+=njoint_apd_loss*njoint_apd_predict.shape[0] 
            total_loss_dim+=njoint_apd_predict.shape[0]

        if torch.sum(rgen_step_mask)>0:
            rgen_apd_predict=self.ring_model(rgen_mg_nodes,rgen_mg_edges,rgen_rg_nodes,rgen_rg_edges,rgen_rg_ftype)
            rgen_apd_loss=self.kl_loss(rgen_apd_predict,rgen_apd)
            total_loss+=rgen_apd_loss*rgen_apd_predict.shape[0]
            total_loss_dim+=rgen_apd_predict.shape[0]

        total_loss=total_loss/total_loss_dim
        
        if not torch.isnan(total_loss):
            self.lr=self.optimizer.state_dict()['param_groups'][0]['lr']
            self.valid_loss['nadd']+=nadd_apd_loss.item()*nadd_apd_predict.shape[0]
            self.record_vnum['nadd']+=nadd_apd.shape[0]
            lstr=f'Node Add: {nadd_apd_loss.item():.3E} ;'
            if torch.sum(rgen_step_mask)>0:
                self.valid_loss['rgen']+=rgen_apd_loss.item()*rgen_apd_predict.shape[0]
                self.record_vnum['rgen']+=rgen_apd.shape[0]
                lstr+=f'Ring Gen: {rgen_apd_loss.item():.3E} ;'
            if torch.sum(njoint_step_mask)>0:
                self.valid_loss['njoint']+=njoint_apd_loss.item()*njoint_apd_predict.shape[0]
                self.record_vnum['njoint']+=njoint_apd.shape[0]
                lstr+=f'Node Joint: {njoint_apd_loss.item():.3E} ;'

        else:
            lstr=f'Valid Loss is unnormal: {nadd_apd_loss.item():.3E}/'
            if torch.sum(rgen_step_mask)>0:
                lstr+=f'{rgen_apd_loss.item():.3E}/'
            if torch.sum(njoint_step_mask)>0:
                lstr=f'{njoint_apd_loss.item():.3E}' 
        return lstr

    def log_training(self,index,epochnum,steps,part=1):
        logstr=f'Step:{index}/{epochnum} -- {part}/{steps} ; Lr: {self.lr:.3E} ; Total Noad add / Ring / Node connect: {self.train_loss["nadd"]/self.record_num["nadd"]:.3E} / {(self.train_loss["rgen"]/self.record_num["rgen"]):.3E} / {self.train_loss["njoint"]/(self.record_num["njoint"]):.3E} Valid: {self.valid_loss["nadd"]/(self.record_vnum["nadd"]):.3E} / {self.valid_loss["rgen"]/(self.record_vnum["rgen"]):.3E} / { self.valid_loss["njoint"]/(self.record_vnum["njoint"]):.3E}\n'
        return logstr
        
    def fit(self,mfts,epochnum=100,save_freq=5,mini_epoch=0,split_rate=0.95):
        nmols=len(mfts)
        splitnum=math.ceil(nmols*split_rate)
        trainset=MFTree_Dataset(mfts[:splitnum],name='Train')
        validset=MFTree_Dataset(mfts[splitnum:],name='Valid')
        trainloader=DataLoader(trainset,batch_size=self.batchsize,shuffle=GP.trainsetting.shuffle,num_workers=GP.trainsetting.n_workers)
        validloader=DataLoader(validset,batch_size=self.batchsize,shuffle=GP.trainsetting.shuffle,num_workers=GP.trainsetting.n_workers)
        n_train_steps=trainset.nmols/self.batchsize
        n_valid_steps=validset.nmols/self.batchsize

        if GP.trainsetting.optimizer=='Adam':
            self.optimizer=torch.optim.Adam([{'params':self.node_add_model.parameters()},{'params':self.ring_model.parameters()},{'params':self.node_connect_model.parameters()}],lr=GP.trainsetting.initlr)
            self.lr_scheduler= ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=0.8, patience=GP.trainsetting.lr_patience*n_train_steps,
                verbose=True, threshold=0.0001, threshold_mode='rel',
                cooldown=GP.trainsetting.lr_cooldown*n_train_steps,
                min_lr=1e-06, eps=1e-06)
        
        step=0
        for epoch in range(epochnum):
            trainbar=tqdm(enumerate(trainloader)) 
            validbar=tqdm(enumerate(validloader))
            try:
                for bid,Datas in trainbar:
                    step+=1             
                    try:   
                        lstr=self.train_step(Datas)
                        self.lr_scheduler.step(metrics=self.valid_loss['total'])
                        trainbar.set_description(f'Epoch {epoch} |'+lstr)
                    except Exception as e:
                        print (f'Training breaked due to {e}')
                for vid,vDatas in validbar:
                    try:
                        lstr=self.evaluate_step(vDatas)
                        validbar.set_description(f'Epoch {epoch} |Valid: '+lstr)
                    except Exception as e:
                        print (f'Validation breaked due to {e}')
         
                bid=0
                logstr=self.log_training(epoch,epochnum,bid)
                self.logger.write(logstr)
                self.logger.flush()
                self.training_history.append([epoch,self.train_loss,self.valid_loss])
                self.epochs+=1
                print (self.valid_loss['nadd'],self.min_valid_loss['nadd'])
                if self.valid_loss['nadd'] < self.min_valid_loss['nadd']:
                    self.min_valid_loss['nadd']=self.valid_loss['nadd']
                    print (f'Save New check point for Node-Add model at Epoch: {epoch}!')
                    savepath=f'{self.modelname}/model/node_add_model_minloss.cpk'
                    savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_valid_loss['nadd'],'state_dict':self.node_add_model.state_dict(),'training_history':self.training_history}
                    torch.save(savedict,savepath)
         
                print (self.valid_loss['rgen'],self.min_valid_loss['rgen'])
                if self.valid_loss['rgen'] < self.min_valid_loss['rgen']:
                    self.min_valid_loss['rgen']=self.valid_loss['rgen']
                    print (f'Save New check point for Ring model at Epoch: {epoch}!')
                    savepath=f'{self.modelname}/model/ring_model_minloss.cpk'
                    savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_valid_loss['rgen'],'state_dict':self.ring_model.state_dict(),'training_history':self.training_history}
                    torch.save(savedict,savepath)
         
                print (self.valid_loss['njoint'],self.min_valid_loss['njoint'])
                if self.valid_loss['njoint'] < self.min_valid_loss['njoint']:
                    self.min_valid_loss['njoint']=self.valid_loss['njoint']
                    print (f'Save New check point for Node connect model at Epoch: {epoch}!')
                    savepath=f'{self.modelname}/model/node_connect_model_minloss.cpk'
                    savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_valid_loss['njoint'],'state_dict':self.node_connect_model.state_dict(),'training_history':self.training_history}
                    torch.save(savedict,savepath)
                self.__tmprecord_clean()
            except Exception as e:
                print (f'Training process breaked due to {e}, turned to next epoch!')
        return

    def fit_for_big_datasets(self,mftsfilelist,epochnum=100,save_freq=5,mini_epoch=0,split_rate=0.95,n_train_steps=100000,init_epoch=0):
        if GP.trainsetting.optimizer=='Adam':
            self.optimizer=torch.optim.Adam([{'params':self.node_add_model.parameters()},{'params':self.ring_model.parameters()},{'params':self.node_connect_model.parameters()}],lr=GP.trainsetting.initlr)
            patience=math.ceil(GP.trainsetting.lr_patience*GP.trainsetting.nmols_per_pickle/self.batchsize)
            cooldown=math.ceil(GP.trainsetting.lr_cooldown*GP.trainsetting.nmols_per_pickle/self.batchsize)
            self.lr_scheduler= ReduceLROnPlateau(
                    self.optimizer, mode='min',
                    factor=0.8, patience=patience,
                    verbose=True, threshold=0.000001, threshold_mode='rel',
                    cooldown=cooldown,
                    min_lr=1e-08, eps=1e-08)
        for epoch in range(epochnum):
            step=0
            for mid,mftsf in enumerate(mftsfilelist):
                try:
                    with open(mftsf,'rb') as f:
                        mfts=pickle.load(f)
                    #mfts=mfts[:2000]
                    nmols=len(mfts)
                    splitnum=math.ceil(nmols*split_rate)
                    trainset=MFTree_Dataset(mfts[:splitnum],name='Train')
                    validset=MFTree_Dataset(mfts[splitnum:],name='Valid')
                    trainloader=DataLoader(trainset,batch_size=self.batchsize,shuffle=GP.trainsetting.shuffle,num_workers=GP.trainsetting.n_workers)
                    validloader=DataLoader(validset,batch_size=self.batchsize,shuffle=GP.trainsetting.shuffle,num_workers=GP.trainsetting.n_workers)
                    #n_train_steps=trainset.nmols/self.batchsize
                    #n_valid_steps=validset.nmols/self.batchsize
             
                    trainbar=tqdm(enumerate(trainloader)) 
                    validbar=tqdm(enumerate(validloader))
                    for bid,Datas in trainbar:
                        step+=1             
                        try:   
                            lstr=self.train_step(Datas)
                            self.lr_scheduler.step(metrics=self.valid_loss['total'])
                            trainbar.set_description(f'Epoch {init_epoch+epoch}/{mid} |'+lstr)
                        except Exception as e:
                            print (f'Training breaked due to {e}')
                    for vid,vDatas in validbar:
                        try:
                            lstr=self.evaluate_step(vDatas)
                            validbar.set_description(f'Epoch {init_epoch+epoch}/{mid} |Valid: '+lstr)
                        except Exception as e:
                            print (f'Validation breaked due to {e}')
             
                    bid=0
                    logstr=self.log_training(init_epoch+epoch,epochnum,bid,mid)
                    self.logger.write(logstr)
                    self.logger.flush()
                    self.training_history.append([epoch,self.train_loss,self.valid_loss])
                    self.epochs+=1
             
                    print (self.valid_loss['nadd'],self.min_valid_loss['nadd'])
                    if self.valid_loss['nadd'] < self.min_valid_loss['nadd']:
                        self.min_valid_loss['nadd']=self.valid_loss['nadd']
                        print (f'Save New check point for Node-Add model at Epoch: {epoch}!')
                        savepath=f'{self.modelname}/model/node_add_model_minloss.cpk'
                        savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_valid_loss['nadd'],'state_dict':self.node_add_model.state_dict(),'training_history':self.training_history}
                        torch.save(savedict,savepath)
             
                    print (self.valid_loss['rgen'],self.min_valid_loss['rgen'])
                    if self.valid_loss['rgen'] < self.min_valid_loss['rgen']:
                        self.min_valid_loss['rgen']=self.valid_loss['rgen']
                        print (f'Save New check point for Ring model at Epoch: {epoch}!')
                        savepath=f'{self.modelname}/model/ring_model_minloss.cpk'
                        savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_valid_loss['rgen'],'state_dict':self.ring_model.state_dict(),'training_history':self.training_history}
                        torch.save(savedict,savepath)
             
                    print (self.valid_loss['njoint'],self.min_valid_loss['njoint'])
                    if self.valid_loss['njoint'] < self.min_valid_loss['njoint']:
                        self.min_valid_loss['njoint']=self.valid_loss['njoint']
                        print (f'Save New check point for Node connect model at Epoch: {epoch}!')
                        savepath=f'{self.modelname}/model/node_connect_model_minloss.cpk'
                        savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_valid_loss['njoint'],'state_dict':self.node_connect_model.state_dict(),'training_history':self.training_history}
                        torch.save(savedict,savepath)


                    savepath=f'{self.modelname}/model/node_add_model_perepoch.cpk'
                    savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_valid_loss['nadd'],'state_dict':self.node_add_model.state_dict(),'training_history':self.training_history}
                    torch.save(savedict,savepath)
             
                    savepath=f'{self.modelname}/model/ring_model_perepoch.cpk'
                    savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_valid_loss['rgen'],'state_dict':self.ring_model.state_dict(),'training_history':self.training_history}
                    torch.save(savedict,savepath)
             
                    savepath=f'{self.modelname}/model/node_connect_perepoch.cpk'
                    savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_valid_loss['njoint'],'state_dict':self.node_connect_model.state_dict(),'training_history':self.training_history}
                    torch.save(savedict,savepath) 
                    self.__tmprecord_clean()
                except Exception as e:
                    print (f'Training process breaked due to {e}, turn to next part!')
        return

    def finetune(self,mfts,epochnum=100,save_freq=5,mini_epoch=0,split_rate=0.95,picpath='./Finetune_mols',sample_num=32,temp_range=(1.0,1.01,0.1)):
        target_smiles=list(set([mft.smi for mft in mfts]))
        target_fps=np.array([AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useCounts=True, useFeatures=True) for smi in target_smiles])
        nmols=len(mfts)
        splitnum=math.ceil(nmols*split_rate)
        trainset=MFTree_Dataset(mfts[:splitnum],name='Train')
        validset=MFTree_Dataset(mfts[splitnum:],name='Valid')
        trainloader=DataLoader(trainset,batch_size=self.batchsize,shuffle=GP.trainsetting.shuffle,num_workers=GP.trainsetting.n_workers)
        validloader=DataLoader(validset,batch_size=self.batchsize,shuffle=GP.trainsetting.shuffle,num_workers=GP.trainsetting.n_workers)
        n_train_steps=trainset.nmols/self.batchsize
        n_valid_steps=validset.nmols/self.batchsize

        if GP.trainsetting.optimizer=='Adam':
            self.optimizer=torch.optim.Adam([{'params':self.node_add_model.parameters()},{'params':self.ring_model.parameters()},{'params':self.node_connect_model.parameters()}],lr=GP.trainsetting.initlr)
            self.lr_scheduler= ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=0.8, patience=GP.trainsetting.lr_patience*n_train_steps,
                verbose=True, threshold=0.0001, threshold_mode='rel',
                cooldown=GP.trainsetting.lr_cooldown*n_train_steps,
                min_lr=1e-08, eps=1e-08)
         
        step=0
        os.system(f'mkdir -p {picpath}')
        
        for epoch in range(epochnum):
            trainbar=tqdm(enumerate(trainloader)) 
            validbar=tqdm(enumerate(validloader))
            for bid,Datas in trainbar:
                step+=1                
                lstr=self.train_step(Datas)
                self.lr_scheduler.step(metrics=self.valid_loss['total'])
                trainbar.set_description(f'Epoch {self.epochs} |'+lstr)
            
            for vid,vDatas in validbar:
                lstr=self.evaluate_step(vDatas)
                validbar.set_description(f'Epoch {self.epochs} |Valid: '+lstr)
            
            if epoch%1==0:
                sample_num=128
                addlogstr=' '
                for temp in np.arange(*temp_range):
                    mols,smis,validity=self.sample(sample_num,temp=temp)
                    sims,tsim,nsim,unique=self.evaluate_mols_similarity_to_given_fps(mols,target_fps)
                    lstr=f'| temp: {temp} tsmi: {tsim:.3E} nsim: {nsim:.3E} unknown: {unique:.3E} validity: {validity:.3E}' 
                    addlogstr.join(lstr)
                    print (f'Epoch: {self.epochs} nsamples: {sample_num}'+lstr)
                    Save_smiles_list(smis,f'{picpath}/ft_{self.epochs}_temp_{temp:.2E}.smi')
                    img=Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(250,250),legends=list([f'{float(a):.3E}' for a in sims]))
                    img.save(f'{picpath}/ft_{self.epochs}_temp_{temp:.2E}.png')
            
            bid=0
            logstr=self.log_training(epoch,epochnum,bid)

            self.logger.write(logstr.strip()+addlogstr)
            self.logger.flush()
            self.training_history.append([self.epochs,self.train_loss,self.valid_loss])
            self.epochs+=1

            if self.valid_loss['nadd'] < self.min_valid_loss['nadd']:
                self.min_valid_loss['nadd']=self.valid_loss['nadd']
                print (f'Save New check point for Node-Add model at Epoch: {self.epochs}!')
                savepath=f'{self.modelname}/model/node_add_model_finetune.cpk'
                savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_valid_loss['nadd'],'state_dict':self.node_add_model.state_dict(),'training_history':self.training_history}
                torch.save(savedict,savepath)

            if self.valid_loss['rgen'] < self.min_valid_loss['rgen']:
                self.min_valid_loss['rgen']=self.valid_loss['rgen']
                print (f'Save New check point for Ring model at Epoch: {self.epochs}!')
                savepath=f'{self.modelname}/model/ring_model_finetune.cpk'
                savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_valid_loss['rgen'],'state_dict':self.ring_model.state_dict(),'training_history':self.training_history}
                torch.save(savedict,savepath)

            if self.valid_loss['njoint'] < self.min_valid_loss['njoint']:
                self.min_valid_loss['njoint']=self.valid_loss['njoint']
                print (f'Save New check point for Node connect model at Epoch: {self.epochs}!')
                savepath=f'{self.modelname}/model/node_connect_model_finetune.cpk'
                savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_valid_loss['njoint'],'state_dict':self.node_connect_model.state_dict(),'training_history':self.training_history}
                torch.save(savedict,savepath)

            self.__tmprecord_clean()
        return
    def best_fp_similarity(self,fp,target_fps):
        #fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
        similarities=[]
        for target_fp in target_fps:
            score = DataStructs.TanimotoSimilarity(target_fp, fp)
            similarities.append(score)
        return max(similarities)

    def evaluate_mols_similarity_to_given_fps(self,mols,target_fps):
        tarnimoto_sim=[]
        new_sim=[]
        for mol in mols:
                fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
                best_similarity=self.best_fp_similarity(fp,target_fps)
                tarnimoto_sim.append(best_similarity)
                if best_similarity <1:
                    new_sim.append(best_similarity)
        return tarnimoto_sim,np.mean(tarnimoto_sim),np.mean(new_sim),len(new_sim)/len(mols)

    def __tmprecord_clean(self):
        self.record_num={'nadd':0,'rgen':0,'njoint':0,'total':0}
        self.record_vnum={'nadd':0,'rgen':0,'njoint':0,'total':0}
        self.train_loss={'nadd':0,'rgen':0,'njoint':0,'total':0}
        self.valid_loss={'nadd':0,'rgen':0,'njoint':0,'total':0}
        return 

    def __reshape_nadd_apd(self,nadd_apd):
        f_nadd_shape=(nadd_apd.shape[0],GP.syssetting.num_node_types)
        f_add=torch.reshape(nadd_apd[:,:GP.syssetting.num_node_types],f_nadd_shape)
        f_term=nadd_apd[:,-1]
        return f_add,f_term

    def sample_nadd_action(self,rw_mol_nodes,rw_mol_adjs,node_constrain_mask,temp=1.0):
        softmax=torch.nn.Softmax(dim=1)
        nadd_apd_predict = self.node_add_model(rw_mol_nodes,rw_mol_adjs)
        nadd_apd_predict=softmax(nadd_apd_predict)*node_constrain_mask.to(GP.modelsetting.device).long()
        nadd_apd_predict_with_temp=torch.exp(torch.log(nadd_apd_predict)/temp)
        if temp==1:
            action_probability_distribution = torch.distributions.Multinomial(
                1,
                probs=nadd_apd_predict
            )
        else:
            action_probability_distribution = torch.distributions.Multinomial(1,probs=nadd_apd_predict_with_temp)

        nadd_apd_one_hot=action_probability_distribution.sample()
        f_add,f_term=self.__reshape_nadd_apd(nadd_apd_one_hot)
        if temp==1:
            likelihoods = torch.log(nadd_apd_predict[nadd_apd_one_hot==1])
        else:
            likelihoods = torch.log(nadd_apd_predict_with_temp[nadd_apd_one_hot==1])
        add_idc = torch.nonzero(f_add,as_tuple=True)
        term_idc = torch.nonzero(f_term,as_tuple=True)

        return add_idc,term_idc,likelihoods

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

        rgen_apd_pred = softmax(self.ring_model(rw_mol_nodes,rw_mol_adjs,rw_ring_nodes,rw_ring_adjs,rw_ring_ftype))

        action_probability_distribution = torch.distributions.Multinomial( 1, probs=rgen_apd_pred )
        
        rgen_apd_one_hot = action_probability_distribution.sample()
        #rgen_apd_one_hot=rgen_apd_one_hot*ring_stop_mask

        #print ('onehot',rgen_apd_one_hot.shape)
        f_radd,f_rconn,f_rterm = self.__reshape_rgen_apd(rgen_apd_one_hot)
        #print (f_radd.shape,f_rconn.shape,f_rterm.shape)
        likelihoods=torch.log(rgen_apd_pred[rgen_apd_one_hot==1])#*(ring_stop_mask.view(-1))

        add_idc=torch.nonzero(f_radd,as_tuple=True)
        conn_idc=torch.nonzero(f_rconn,as_tuple=True)
        term_idc=torch.nonzero(f_rterm,as_tuple=True)
        return add_idc,conn_idc,term_idc,likelihoods

    def rgen_invalid_actions(self,rgen_add_idc,rgen_conn_idc,rw_ring_nodes,rw_ring_adjs,rw_ring_atomnum,rw_ring_ftype):
        rw_ring_maxatoms=torch.sum(rw_ring_ftype[:,1:-2],dim=-1).long()
        rw_ring_maxatoms=torch.where(rw_ring_atomnum.long()>rw_ring_maxatoms.long(),rw_ring_atomnum.long(),rw_ring_maxatoms.long())
        #print ('rw_ring_atomnum',rw_ring_atomnum)
        f_add_empty_graphs = torch.nonzero(rw_ring_atomnum[rgen_add_idc[0]]==0)
        #print (f_add_empty_graphs)
        invalid_add_idx_tmp = torch.nonzero(rgen_add_idc[1]>=rw_ring_atomnum[rgen_add_idc[0]])

        combined = torch.cat((invalid_add_idx_tmp,f_add_empty_graphs)).squeeze(1)

        uniques,counts = combined.unique(return_counts=True)
        invalid_add_idc = uniques[counts==1].unsqueeze(dim=1)

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

    def sample_conn_action(self,rw_mol_nodes,rw_mol_adjs,rw_ring_nodes,rw_ring_adjs,rw_ring_focused_ids,node_conn_constrain_mask):
        softmax=torch.nn.Softmax(dim=1)
        conn_apd_pred = self.node_connect_model(rw_mol_nodes,rw_mol_adjs,rw_ring_nodes,rw_ring_adjs,rw_ring_focused_ids)
        #print (conn_apd_pred)
        conn_apd_pred = torch.mul(softmax(conn_apd_pred),node_conn_constrain_mask.to(GP.modelsetting.device).long())
        #conn_apd_pred = softmax(conn_apd_pred)
        action_probability_distribution = torch.distributions.Multinomial(1, probs=conn_apd_pred)
        conn_apd_one_hot = action_probability_distribution.sample()
        f_conn= self.__reshape_conn_apd(conn_apd_one_hot)
        likelihoods=torch.log(conn_apd_pred[conn_apd_one_hot==1])#*mol_stop_mask
        conn_idc=torch.nonzero(f_conn,as_tuple=True)
        return conn_idc, likelihoods
    
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
            natoms=atomnums[i]
            graph_nodes=nodes_batch[i]
            graph_edges=edges_batch[i]
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
    
    def ring_graph_check(self,ring_nodes,ring_edges,ring_ftype,ring_atomnum,ring_atomic_nums):
        flag=True
        molecule=Chem.RWMol()
        node_to_idx={}
        for j in range(ring_atomnum):
            atom_to_add=self.node_feature_to_atom(ring_nodes,j)
            molecule_idx=molecule.AddAtom(atom_to_add)
            node_to_idx[j]=molecule_idx
        edge_mask = torch.triu(
                torch.ones((GP.syssetting.max_ring_size, GP.syssetting.max_ring_size), device=GP.modelsetting.device),
                diagonal=1
                ).view(GP.syssetting.max_ring_size,GP.syssetting.max_ring_size,1)

        edges_idc=torch.nonzero(ring_edges*edge_mask)
        for node_idx1,node_idx2,bond_idx in edges_idc:
            try:
                molecule.AddBond(node_to_idx[node_idx1.item()],node_to_idx[node_idx2.item()],GP.syssetting.bond_types[bond_idx.item()])
            except Exception as e:
                flag=False
        try:
        #if True:
            molecule.GetMol()
            Chem.SanitizeMol(molecule)
            if GP.sample_constrain_setting.ring_check_mode=="strict":
                ssrlist=[list(x) for x in Chem.GetSymmSSSR(molecule)]
                rnum=len(ssrlist)
                if rnum!=ring_ftype[0]:
                    flag=False
                num_aromatic_rings=Chem.rdMolDescriptors.CalcNumAromaticRings(molecule)
                if num_aromatic_rings!=ring_ftype[-1]:
                    flag=False
                for jid,j in enumerate(GP.syssetting.possible_atom_types):
                    jnum=list(ring_atomic_nums.detach().cpu().numpy()).count(j)
                    if jnum!=ring_ftype[1+jid]:
                        flag=False
                ring_atoms=[]
                for ring in ssrlist:
                    for i in ring:
                        if i not in ring_atoms:
                            ring_atoms.append(i)
                branches_num=ring_atomnum-len(ring_atoms)
                if branches_num!=ring_ftype[-2]:
                    flag=False
            else:
                ssrlist=[list(x) for x in Chem.GetSymmSSSR(molecule)]
                #print (ssrlist)
                if len(ssrlist)<1:
                    flag=False 
        except Exception as e:
            flag=False
        return flag

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
            self.rw_mol_likelihoods=torch.zeros((self.batchsize,GP.syssetting.max_atoms),device=GP.modelsetting.device)

        elif idx>0:
            self.rw_mol_nodes[idx]=0
            self.rw_mol_adjs[idx]=0
            self.rw_mol_atomnums[idx]=0
            self.rw_mol_atomics[idx]=0
            self.rw_mol_pt[idx]=0
            self.rw_mol_round[idx]=0
            self.rw_mol_likelihoods[idx]=0
            self.rw_mol_saturation_list[idx]=[]

        else:
            self.rw_mol_nodes[0]=0
            self.rw_mol_adjs[0]=0
            self.rw_mol_atomnums[0]=1 
            self.rw_mol_atomics[0]=0
            self.rw_mol_pt[0]=0
            self.rw_mol_round[0]=0
            self.rw_mol_likelihoods[0]=0
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
            self.rw_ring_likelihoods=torch.zeros((self.batchsize,GP.syssetting.max_ring_size),device=GP.modelsetting.device)

            self.rw_ring_nodes[0,0]=torch.Tensor(dummy_af)
            self.rw_ring_adjs[0,0,0,0]=1
            self.rw_ring_atomnums[0]=1
            self.rw_ring_atomics[0,0]=6
            
        elif idx>0:
            self.rw_ring_nodes[idx]=0
            self.rw_ring_adjs[idx]=0
            self.rw_ring_atomnums[idx]=0
            self.rw_ring_atomics[idx]=0
            self.rw_ring_focused_ids[idx]=0
            self.rw_ring_round[idx]=0
            self.rw_ring_likelihoods[idx]=0
        else:
            self.rw_ring_nodes[0]=0
            self.rw_ring_adjs[0]=0
            self.rw_ring_atomnums[0]=0
            self.rw_ring_atomics[0]=0
            self.rw_ring_focused_ids[0]=0 
            self.rw_ring_round[0]=0
            self.rw_ring_likelihoods[0]=0
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
        self.rw_ring_likelihoods[ids]=0
        #self.rw_ring_likelihoods_prior[ids]=0
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

    def __init_wrapper(self):
        self.rw_mol_nodes_wrap = torch.zeros((self.batchsize,100,GP.syssetting.n_node_features),device=GP.modelsetting.device).long()
        self.rw_mol_adjs_wrap = torch.zeros((self.batchsize,100,100,GP.syssetting.n_edge_features),device=GP.modelsetting.device).long()

    def __update_wrapper(self,idx):
        curr_atoms=self.rw_mol_atomnums[idx]
        self.rw_mol_nodes_wrap[idx,:curr_atoms]=self.rw_mol_nodes[idx,:curr_atoms]
        self.rw_mol_adjs_wrap[idx,:curr_atoms,:curr_atoms]=self.rw_mol_adjs[idx,:curr_atoms,:curr_atoms]
        return 

    def __init_nconn(self,idx=-1):
        self.rw_conn_likelihoods=torch.zeros(self.batchsize,device=GP.modelsetting.device)
        #self.rw_conn_likelihoods_prior=torch.zeros(self.batchsize,device=GP.modelsetting.device)

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
        avaliable_nconn_mask=torch.zeros(self.batchsize,*GP.syssetting.f_node_joint_dim).long()
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
        self.rw_ring_likelihoods[idx]=0
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
            
    def sample_batch(self,ring_apd_resample_times=1,temp=1.0):
        self.node_add_model.eval()
        self.ring_model.eval()
        self.node_connect_model.eval()
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
            nadd_add_ids,nadd_term_ids,nadd_llh=self.sample_nadd_action(mol_nodes,mol_adjs,mol_nadd_masks,temp=temp)
            self.mol_stop_mask[mol_unstop_ids[nadd_term_ids]]=0
            #print ("stop ids",mol_unstop_ids[nadd_term_ids])
            #print ("mol_stop_mask",self.mol_stop_mask)
            self.rw_ring_likelihoods[mol_unstop_ids,0]=nadd_llh
            
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

            while torch.sum(self.ring_stop_mask[1:])>=1:
                self.ring_stop_mask[0]=1
                self.__init_ring_graphs(idx=0)
                #print ('ring_stop_mask',self.ring_stop_mask)
                ring_unstop_ids,ring_nodes,ring_adjs,ring_atomnums,ring_atomics,ring_ftypes,ring_focused_ids,ring_round=self.__unstop_ring_graphs()
                ring_mol_nodes,ring_mol_adjs,ring_mol_atomnums,ring_mol_atomics,ring_mol_pt,ring_mol_round=self.select_mol_graphs(ring_unstop_ids)
                self.rw_ring_round[ring_unstop_ids]+=1
                
                rgen_add_ids,rgen_conn_ids,rgen_term_ids,likelihoods=\
                    self.sample_rgen_action(ring_mol_nodes,ring_mol_adjs,ring_nodes,ring_adjs,ring_ftypes)

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

                self.rw_ring_likelihoods[batch,self.rw_ring_round[batch]]=likelihoods[rgen_add_valid_ids]                

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

                self.rw_ring_likelihoods[batch,self.rw_ring_round[batch]]=likelihoods[rgen_conn_valid_ids]

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
            nconn_ids,likelihoods=self.sample_conn_action(nconn_mol_nodes,nconn_mol_adjs,nconn_ring_nodes,nconn_ring_adjs,nconn_ring_focused_ids,mol_nconn_masks)

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
            likelihoods=likelihoods[nconn_valid_ids]

            non_zero_mol_graph_ids=torch.nonzero(self.rw_mol_atomnums[batch]!=0)
            batch_masked=batch[non_zero_mol_graph_ids]
            bond_to_masked = bond_to[non_zero_mol_graph_ids]
            bond_from_masked = bond_from[non_zero_mol_graph_ids]
            bond_type_masked = bond_type[non_zero_mol_graph_ids]
            likelihoods_masked=likelihoods[non_zero_mol_graph_ids]

            self.rw_conn_likelihoods[batch_masked]=likelihoods_masked 

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
                        self.rw_conn_likelihoods[j]+=torch.sum(self.rw_ring_likelihoods[j])
                        self.rw_mol_likelihoods[j,self.rw_mol_round[j]]=self.rw_conn_likelihoods[j]
                        self.rw_mol_pt[j,self.rw_mol_round[j],1]=self.rw_mol_atomnums[j]
                        self.rw_mol_round[j]+=1
                if self.rw_mol_round[j]>=GP.sample_constrain_setting.max_node_steps:
                    self.mol_stop_mask[j]=0
            #print ('rw_mol_pt',self.rw_mol_pt[1]) 
        self.rw_mol_likelihoods=torch.sum(self.rw_mol_likelihoods,dim=1)

        return self.rw_mol_nodes[1:],self.rw_mol_adjs[1:],self.rw_mol_atomnums[1:],self.rw_mol_likelihoods[1:]

    def sample(self,sample_num,temp=1.0):
        sampled_num=0
        sample_mols=[]
        sample_smis=[]
        validities=[]
        while sampled_num<sample_num:
            mol_nodes,mol_adjs,mol_atomnums,ll_agent=self.sample_batch(temp=temp)
            mols,smis,valid_ids=self.nodes_edges_to_mol(mol_nodes,mol_adjs,mol_atomnums)
            temp_validity=len(valid_ids)/len(smis)
            valid_num=len(valid_ids)
            valid_mols=[mols[i] for i in valid_ids]
            valid_smis=[smis[i] for i in valid_ids]
            valid_ids=torch.Tensor(valid_ids).long().to(GP.modelsetting.device)
            valid_mol_nodes,valid_mol_adjs,valid_mol_atomnums,valid_ll_agent=mol_nodes[valid_ids],mol_adjs[valid_ids],mol_atomnums[valid_ids],ll_agent[valid_ids]
            if sampled_num+valid_num<sample_num:
                sample_mols+=valid_mols
                sample_smis+=valid_smis
            else:
                sample_mols+=valid_mols[:sample_num-sampled_num]
                sample_smis+=valid_smis[:sample_num-sampled_num]
            sampled_num+=valid_num
            validities.append(temp_validity)
        validity=np.mean(validities)
        return sample_mols,sample_smis,validity
    
    




    


        




