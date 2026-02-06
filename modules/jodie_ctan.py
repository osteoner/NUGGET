import torch
import math
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
from torch_geometric.nn import TransformerConv, TGNMemory
from typing import Callable, Optional, Dict, Tuple, Optional

class GenericModel(torch.nn.Module):
    
    def __init__(self, num_nodes, memory=None, gnn=None, gnn_act=None, readout=None, predict_dst=False):
        super(GenericModel, self).__init__()
        self.memory = memory
        self.gnn = gnn
        self.gnn_act = gnn_act
        self.readout = readout
        self.num_gnn_layers = 1
        self.num_nodes = num_nodes
        self.predict_dst = predict_dst

    def reset_memory(self):
        if self.memory is not None: self.memory.reset_state()

    def zero_grad_memory(self):
        if self.memory is not None: self.memory.zero_grad_memory()

    def update(self, src, pos_dst, t, msg, *args, **kwargs):
        # FIX 1: Explicitly cast src and dst to LongTensor before passing to memory.
        # This prevents 'IndexError' when creating n_id inside TGNMemory.
        if self.memory is not None: 
            self.memory.update_state(src.long(), pos_dst.long(), t, msg)

    def detach_memory(self):
        if self.memory is not None: self.memory.detach()
    
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        # GenericModel inherits from nn.Module, calling super().reset_parameters() might fail if not defined
        if hasattr(self.memory, 'reset_parameters'):
            self.memory.reset_parameters()
        if hasattr(self.gnn, 'reset_parameters'):
            self.gnn.reset_parameters()
        if hasattr(self.readout, 'reset_parameters'):
            self.readout.reset_parameters()

    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        src, pos_dst = batch.src, batch.dst
        
        neg_dst = batch.neg_dst if hasattr(batch, 'neg_dst') else None

        # Get updated memory of all nodes involved in the computation.
        m, last_update = self.memory(n_id)

        if hasattr(batch, 'x'):
            if len(batch.x.shape) == 3: # sequence classification case
                x = batch.x.squeeze(0)
            elif len(batch.x.shape) == 2: # link-based predictions
                x = batch.x
            else:
                raise ValueError(f"Unexpected node feature shape. Got {batch.x.shape}")
            z = torch.cat((m, x[n_id]), dim=-1)

        if self.gnn is not None:
            for gnn_layer in self.gnn:
                z = gnn_layer(z, last_update, edge_index, t, msg)
                z = self.gnn_act(z)

        if self.predict_dst:
            pos_out = self.readout(z[id_mapper[pos_dst]])
            neg_dst = None
        else:
            pos_out = self.readout(z[id_mapper[src]], z[id_mapper[pos_dst]])
            neg_out = self.readout(z[id_mapper[src]], z[id_mapper[neg_dst]]) if neg_dst is not None else None

        return pos_out, neg_out, m[id_mapper[src]], m[id_mapper[pos_dst]]


class IdentityLayer(torch.nn.Module):
    # NOTE: this object is used to implement those models that do not have a RNN-based memory
    def __init__(self):
          super().__init__()
          self.I = torch.nn.Identity()
    
    def forward(self, x, *args, **kwargs):
         return self.I(x)

class NormalLinear(torch.nn.Linear):
    # From Jodie code
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


class JodieEmbedding(torch.nn.Module):
    def __init__(self, out_channels: int,
                 mean_delta_t: float = 0., std_delta_t: float = 1.):
        super().__init__()
        
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        self.projector = NormalLinear(1, out_channels)

    def forward(self, x, last_update, t):
        t_ = torch.cat([t, t]) if len(last_update) == 2*len(t) else t
        rel_t = (last_update - t_).abs()
        if rel_t.shape[0] > 0:
            rel_t = (rel_t - self.mean_delta_t) / self.std_delta_t # delta_t normalization
            return x * (1 + self.projector(rel_t.view(-1, 1).to(x.dtype))) 
    
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None, out_channels: int = 1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.lin_src = torch.nn.Linear(in_channels, hidden_channels)
        self.lin_dst = torch.nn.Linear(in_channels, hidden_channels)
        self.lin_final = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)

class SequencePredictor(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None, out_channels: int = 1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.lin = torch.nn.Linear(in_channels, hidden_channels)
        self.lin_final = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, z_dst):
        h = self.lin(z_dst)
        h = h.relu()
        return self.lin_final(h)

TGNMessageStoreType = Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


class GeneralMemory(TGNMemory):
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable,
                 rnn: Optional[str] = None,
                 non_linearity: str = 'tanh',
                 init_time: int = 0,
                 message_batch: int = 10000):
        
        super().__init__(num_nodes, raw_msg_dim, memory_dim, time_dim, message_module, aggregator_module)

        # === FIX: Override PyG default (Long) with Float for continuous time ===
        # This prevents "RuntimeError: Index put requires... got Long for destination and Float for source"
        self.register_buffer("last_update", torch.empty(num_nodes, dtype=torch.float))
        # =======================================================================

        self.message_batch = message_batch
        if rnn is None:
             self.gru = IdentityLayer()
        else:
            rnn_instance = getattr(torch.nn, rnn)
            if 'RNN' in rnn:
                self.gru = rnn_instance(message_module.out_channels, memory_dim, nonlinearity=non_linearity)
            else:
                self.gru = rnn_instance(message_module.out_channels, memory_dim)

        self.memory[:] = torch.zeros(num_nodes, memory_dim).type_as(self.memory)
        self.last_update[:] = torch.ones(num_nodes).type_as(self.last_update) * init_time

        if hasattr(self.gru, 'reset_parameters'):
            self.gru.reset_parameters()

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            # Do it in batches of nodes, otherwise CUDA runs out of memory for datasets with millions of nodes
            for i in range(0, self.num_nodes, self.message_batch):
                self._update_memory(
                    torch.arange(i, min(self.num_nodes, i + self.message_batch), device=self.memory.device))
            self._reset_message_store()
        super(TGNMemory, self).train(mode)
        

class JODIE(GenericModel):
    def __init__(self, 
                 # Memory params
                 num_nodes: int,
                 edge_dim: int, 
                 memory_dim: int,
                 time_dim: int,
                 node_dim: int = 0, 
                 non_linearity: str = 'tanh',
                 # Link predictor
                 readout_hidden: Optional[int] = None,
                 readout_out: Optional[int] = 1,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0,
                 # Distinguish between link prediction and sequence prediction
                 predict_dst: bool = False
        ):
        # Define memory
        memory = GeneralMemory(
            num_nodes,
            edge_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(edge_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
            rnn='RNNCell',
            non_linearity=non_linearity,
            init_time = init_time
        )

        # Define the link predictor
        readout = (LinkPredictor(memory_dim + node_dim, readout_hidden, readout_out) if not predict_dst 
                   else SequencePredictor(memory_dim + node_dim, readout_hidden, readout_out))

        super().__init__(num_nodes, memory, readout=readout, predict_dst=predict_dst)
        self.num_gnn_layers = 1
        self.projector_src = JodieEmbedding(memory_dim + node_dim, mean_delta_t=mean_delta_t, std_delta_t=std_delta_t)
        self.projector_dst = JodieEmbedding(memory_dim + node_dim, mean_delta_t=mean_delta_t, std_delta_t=std_delta_t)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self.projector_src, 'reset_parameters'):
            self.projector_src.reset_parameters()
        if hasattr(self.projector_dst, 'reset_parameters'):
                    self.projector_dst.reset_parameters()
    
    
    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        src, pos_dst = batch.src, batch.dst
        
        neg_dst = batch.neg_dst if hasattr(batch, 'neg_dst') else None

        # Get updated memory of all nodes involved in the computation.
        m, last_update = self.memory(n_id)

        if hasattr(batch, 'x'):
            if len(batch.x.shape) == 3: # sequence classification case
                x = batch.x.squeeze(0)
            elif len(batch.x.shape) == 2: # link-based predictions
                x = batch.x
            else:
                raise ValueError(f"Unexpected node feature shape. Got {batch.x.shape}")
            z = torch.cat((m, x[n_id]), dim=-1)

        # Compute the projected embeddings
        z_src =  self.projector_src(z[id_mapper[src]], last_update[id_mapper[src]], batch.t)
        try:
            z_pos_dst =  self.projector_dst(z[id_mapper[pos_dst]], last_update[id_mapper[pos_dst]], batch.t)
        except IndexError:
            # Debugging breakpoint (can be removed in production)
            pass

        if self.predict_dst:
            pos_out = self.readout(z_pos_dst)        
        else:
            pos_out = self.readout(z_src, z_pos_dst)

        if neg_dst is not None:
            z_neg_dst =  self.projector_dst(z[id_mapper[neg_dst]], last_update[id_mapper[neg_dst]], batch.t)
            neg_out = self.readout(z_src, z_neg_dst)
        else:
            neg_out = None
        if neg_out is None and len(pos_out) == 2*len(batch.t):
            neg_out = pos_out[len(batch.t):]
            pos_out = pos_out[:len(batch.t)]
        elif hasattr(batch, 'n_neg'):
            neg_out = pos_out[-batch.n_neg:]
            pos_out = pos_out[:-batch.n_neg]
            
        return pos_out, neg_out, m[id_mapper[src]], m[id_mapper[pos_dst]]