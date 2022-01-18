# coding=utf8
import argparse


def get_parser():
    """get model args"""
    parser = argparse.ArgumentParser()
    # Mode
    parser.add_argument("--mode", type=str, default='train',
                        help="run mode")
    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument("--num_layers", type=int, default=2,
                           help="Number of LSTM layers")
    model_arg.add_argument("--hidden_size", type=int, default=512,
                           help="Hidden size")
    model_arg.add_argument("--emb_size", type=int, default=16,
                           help="embedding size")
    model_arg.add_argument("--dropout", type=float, default=0,
                           help="dropout between LSTM layers except for last")

    # Train
    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--master_ip', type=str, default='127.0.0.1',
                           help='dist ip')
    train_arg.add_argument('--master_port', type=str, default='23333',
                           help='dist port')
    train_arg.add_argument('--world_size', type=int, default=4,
                           help='GPUS')
    train_arg.add_argument('--local_rank', type=int, default=-1,
                           help='local rank')
    train_arg.add_argument('--device', type=str, default='cuda:0',
                           help='gpu cuda device')
    train_arg.add_argument('--num_epochs', type=int, default=100,
                           help='Number of epochs for model training')
    train_arg.add_argument('--batch_size', type=int, default=512,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                           help='Learning rate')
    train_arg.add_argument('--seed', type=int, default=2021,
                           help='random seed')
    train_arg.add_argument('--scheduler', type=str, default='step',
                           help='scheduler type')
    train_arg.add_argument('--step_size', type=int, default=6,
                           help='Period of learning rate decay')
    train_arg.add_argument('--gamma', type=float, default=0.97,
                           help='factor of learning rate decay')
    train_arg.add_argument('--clip_grad', type=float, default=5,
                           help='Period of learning rate decay')
    train_arg.add_argument('--num_workers', type=int, default=1,
                           help='Number of workers for DataLoaders')
    train_arg.add_argument('--trainset', type=str, default='../datasets/trainset.csv',
                           help='trainset filepath')
    train_arg.add_argument('--validset', type=str, default='../datasets/testset.csv',
                           help='validset filepath')

    # Sample
    sample_arg = parser.add_argument_group('Sample')
    sample_arg.add_argument('--model_load_path', type=str, default='./ckpt/model.pt',
                           help='model load path')
    sample_arg.add_argument('--max_length', type=int, default=100,
                           help='max smiles length')
    sample_arg.add_argument('--sample_num', type=int, default=10000,
                           help='sample smiles')
    sample_arg.add_argument('--sample_file', type=str, default='sampled_smiles.csv',
                           help='sample smiles file')
    sample_arg.add_argument('--temperature', type=float, default=1.0,
                           help='sample temperature')

    # Save and Log
    save_arg = parser.add_argument_group('Save_Log')
    save_arg.add_argument('--save_path', type=str, default='./ckpt/',
                           help='model save path')
    save_arg.add_argument('--save_freq', type=int, default=1,
                           help='model save freq')
    save_arg.add_argument('--log_file', type=str, default='log.csv',
                           help='log filename')

    args = parser.parse_args()
    return args
