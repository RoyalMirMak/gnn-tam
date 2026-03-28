import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import sys
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from gnn import GNN_TAM

class NodeAffine(nn.Module):
    def __init__(self, n_nodes, scale=None, bias=None):
        super(NodeAffine, self).__init__()
        self.shape = (1, n_nodes, 1)
        
        if scale is None:
            scale = torch.ones(self.shape)
        if bias is None:
            bias = torch.zeros(self.shape)
            
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, x):
        return x * self.scale + self.bias

def fold_bn_to_affine(bn_layer):
    with torch.no_grad():
        gamma = bn_layer.weight
        beta = bn_layer.bias
        mean = bn_layer.running_mean
        var = bn_layer.running_var
        eps = bn_layer.eps

        std = torch.sqrt(var + eps)
        scale = gamma / std
        bias = beta - (mean * scale)

        n_nodes = bn_layer.num_features
        return NodeAffine(n_nodes, scale.view(1, n_nodes, 1), bias.view(1, n_nodes, 1))

class PrecomputedGCNLayer(nn.Module):
    def __init__(self, gsl_layer, conv_layer, idx_tensor, z_tensor):
        super().__init__()
        self.dense = copy.deepcopy(conv_layer.dense)
        gsl_layer.eval()
        with torch.no_grad():
            adj = gsl_layer(idx_tensor)
            
            adj = adj * z_tensor
            n_nodes = adj.size(0)
            I = torch.eye(n_nodes, device=adj.device)
            adj = adj + I
            
            degree = adj.sum(dim=1)
            degree_inv_sqrt = degree.pow(-0.5)
            degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
            
            d_mat = torch.diag(degree_inv_sqrt)
            a_hat = d_mat @ adj @ d_mat
            self.register_buffer('a_hat', a_hat)

    def forward(self, x):
        h = self.dense(x)
        return self.a_hat @ h

class GNN_TAM_Folded(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.n_nodes = original_model.bnorm1[0].num_features
        sample_dense = original_model.conv1[0].dense
        self.window_size = sample_dense.in_features
        self.n_gnn = len(original_model.conv1)

        idx = original_model.idx.cpu()
        z = original_model.z.cpu()

        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.affine1 = nn.ModuleList()
        self.affine2 = nn.ModuleList()
        self.fc = copy.deepcopy(original_model.fc)

        for i in range(self.n_gnn):
            self.conv1.append(PrecomputedGCNLayer(
                original_model.gsl[i], 
                original_model.conv1[i], 
                idx, z
            ))
            
            self.affine1.append(fold_bn_to_affine(original_model.bnorm1[i]))
            self.conv2.append(PrecomputedGCNLayer(
                original_model.gsl[i], 
                original_model.conv2[i], 
                idx, z
            ))
            self.affine2.append(fold_bn_to_affine(original_model.bnorm2[i]))

    def forward(self, x):
        h_concat_list = []
        
        for i in range(self.n_gnn):
            h = self.conv1[i](x) 
            h = F.relu(h)
            h = self.affine1[i](h)
            
            skip_val, _ = torch.min(h, dim=1)
            skip_val = skip_val.unsqueeze(1) 

            h = self.conv2[i](h)
            h = F.relu(h)
            h = self.affine2[i](h)
            
            h_val, _ = torch.min(h, dim=1)
            h_val = h_val.unsqueeze(1)
            
            h_reduced = h_val.view(h_val.size(0), -1)
            skip_reduced = skip_val.view(skip_val.size(0), -1)
            
            h_result = h_reduced + skip_reduced
            
            h_concat_list.append(h_result)
        
        h_final = torch.cat(h_concat_list, 1)
        output = self.fc(h_final)
        return output

def main():
    parser = argparse.ArgumentParser(description='Fold GNN-TAM model and export to ONNX')
    parser.add_argument('--model_path', type=str,
                        help='Path to the input .pt model file')
    parser.add_argument('--out_dir', type=str, default='onnx_models',
                        help='Output directory for the ONNX model')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Output filename')
    args = parser.parse_args()

    if args.output_name is None:
        base_name = os.path.basename(args.model_path)
        output_filename = os.path.splitext(base_name)[0] + '.onnx'
    else:
        output_filename = args.output_name

    model_path = args.model_path
    out_dir = args.out_dir
    out_path = os.path.join(out_dir, output_filename)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {model_path} ...")
    torch.serialization.add_safe_globals([GNN_TAM])
    
    orig_model = torch.load(model_path, map_location='cpu', weights_only=False)
    orig_model.eval()

    print("Converting Model...")
    folded_model = GNN_TAM_Folded(orig_model)
    folded_model.eval()

    dumb_input = torch.randn(1, folded_model.n_nodes, folded_model.window_size)
    print("Validating equivalence...")
    with torch.no_grad():
        out_orig = orig_model(dumb_input)
        out_fold = folded_model(dumb_input)
    
    diff = (out_orig - out_fold).abs().max().item()
    print(f"Maximum Absolute Difference: {diff}")
    if diff > 1e-3:
        print("WARNING! Difference is too big!")
    
    print(f"Exporting to {out_path}...")
    torch.onnx.export(
        folded_model,
        dumb_input,
        out_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['batch_ts'],
        output_names=['logits'],
        dynamic_axes={
            'batch_ts': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        dynamo=False, 
        verbose=False
    )

if __name__ == "__main__":
    main()
