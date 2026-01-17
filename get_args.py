import argparse

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()

    # Model name/experiment identifier - required for distinguishing different models
    parser.add_argument('-model_name', type=str, required=True,
                        help='Model identifier/experiment name (required) - used to distinguish different model runs')

    parser.add_argument('-dataset', default  = 'cifar10')

    # The ViT setting
    parser.add_argument('-n_patches', type=int, default  = 64)
    parser.add_argument('-n_feat', type=int, default  = 48)
    parser.add_argument('-hidden_size', type=int, default  = 256)
    parser.add_argument('-feedforward_size', type=int, default  = 1024)
    parser.add_argument('-n_layers', type=int, default  = 8)
    parser.add_argument('-dropout_prob', type=float, default  = 0.1)
    
    # The Swin setting
    parser.add_argument('-image_dims', default  = [32, 32])
    parser.add_argument('-depth', default  = [2, 4])
    parser.add_argument('-embed_size', type=int, default  = 256)
    parser.add_argument('-window_size', type=int, default  = 8)
    parser.add_argument('-mlp_ratio', type=float, default  = 4)

    parser.add_argument('-n_trans_feat', type=int, default  = 16)
    parser.add_argument('-n_heads', type=int, default  = 8)

    # The bandwidth adaption setting -> 2 types; one using different feats, one using different patches
    parser.add_argument('-min_trans_feat', type=int, default  = 1)
    parser.add_argument('-max_trans_feat', type=int, default  = 6)
    parser.add_argument('-unit_trans_feat', type=int, default  = 4)
    parser.add_argument('-trg_trans_feat', type=int, default  = 6)       # should be consistent with args.n_trans_feat
    

    parser.add_argument('-min_trans_patch', type=int, default  = 5)
    parser.add_argument('-max_trans_patch', type=int, default  = 8)
    parser.add_argument('-unit_trans_patch', type=int, default  = 8)
    parser.add_argument('-trg_trans_patch', type=int, default  = 5)       # should be consistent with args.n_trans_feat

    parser.add_argument('-n_adapt_embed', type=int, default  = 2)
    
    # channel
    # parser.add_argument('-channel_mode', default = 'awgn')
    parser.add_argument('-channel_mode', default = 'rayleigh')
    parser.add_argument('-link_qual', type=float, default  = 7.0) # wireless link quality in dB
    parser.add_argument('-link_rng',  type=float, default  = 3.0) # wireless link quality range in dB
    parser.add_argument('-plr', type=float, default  = 0.0) # package loss rate
    parser.add_argument('-min_plr', type=float, default  = 0.0) # minimum package loss rate
    parser.add_argument('-max_plr', type=float, default  = 0.2) # maximum package loss rate


    parser.add_argument('-adapt', default  = True)
    parser.add_argument('-full_adapt', default  = True)

    # dynamic weight adaption -- initial at 1; maximum 10; 
    parser.add_argument('-threshold', default  = 0.25)            # if it is smaller than 0.25 dB, then it's fine
    parser.add_argument('-min_clip', default  = 0)               # no smaller than 0
    parser.add_argument('-max_clip', default  = 10)              # no larger than 10
    parser.add_argument('-alpha', default  = 2)                  # weight[l] = 2**(alpha*delta[l])-1
    parser.add_argument('-freq', default  = 1)                   # The frequency of updating the weights

    # training setting
    parser.add_argument('-epoch', type=int, default  = 300)
    parser.add_argument('-lr', type=float, default  = 1e-4)
    parser.add_argument('-train_patience', type=int, default  = 80)
    parser.add_argument('-train_batch_size', type=int, default  = 32)

    parser.add_argument('-val_batch_size', type=int, default  = 32)
    parser.add_argument('-lpips_net', type=str, default='alex', choices=['alex', 'vgg'], help='LPIPS network type')
    parser.add_argument('-resume', default  = False)
    parser.add_argument('-path', default  = 'models/')
    parser.add_argument('-device', default  = 'cuda:4')

    # Computation-adaptive module switches
    parser.add_argument('-use_encoder_pruning', type=lambda x: x.lower() in ['true', '1', 'yes'], 
                        default=True, help='Enable encoder spatial pruning to reduce encoder computation (default: True)')
    parser.add_argument('-use_decoder_early_exit', type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=True, help='Enable decoder early exit heads to reduce decoder computation (default: True)')

    args = parser.parse_args()

    return args