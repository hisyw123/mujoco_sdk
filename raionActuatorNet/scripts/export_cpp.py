"""Export PyTorch GRU weights for the standalone Eigen C++ implementation."""
import json
from pathlib import Path


def export_cpp(net, cfg, path):
    codes={'pos_err':0,'vel':1,'torque':2}
    data={k:cfg[k] for k in ['n_joints','hidden','layers','x_mean','x_std','y_mean','y_std']}
    data.update(format='raion-gru-v1',features=[codes[x] for x in cfg['features']],
                weights={k:dict(shape=list(v.shape),data=v.detach().cpu().reshape(-1).tolist())
                         for k,v in net.state_dict().items()})
    text=json.dumps(data,allow_nan=False)+'\n'
    path=Path(path)
    if path.exists() and path.read_text()!=text:
        raise ValueError('C++ export exists; choose a new model directory')
    path.write_text(text)


if __name__=='__main__':
    import sys,torch
    from train import ActuatorNet
    folder=Path(sys.argv[1])
    cfg=json.loads((folder/'config.json').read_text())
    net=ActuatorNet(len(cfg['x_mean']),cfg['hidden'],cfg['layers'],cfg['n_joints'])
    net.load_state_dict(torch.load(folder/'actuator_net.pt',map_location='cpu',weights_only=True))
    export_cpp(net,cfg,folder/'actuator_net_cpp.json')
