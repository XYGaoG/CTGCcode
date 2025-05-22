def SSL_hyperpara(args):
    if args.dataset_name == 'citeseer':
        args.iter_num = 3
    if args.dataset_name == 'ogbn-arxiv':
        args.lr_ssl_spe = 0.1
        args.epoch_ssl=50
        args.iter_num=3
        args.epoch_cls=40
    if args.dataset_name == 'reddit':
        args.epoch_pretrain = 20
        args.epoch_ssl=40
        args.lr_pretrain = 0.0001
        args.lr_ssl_spa = 0.001
        args.lr_ssl_spe = 0.1
        args.iter_num=3
        args.alpha=10000
    return args


def SSL_reduction(args):
    if args.dataset_name == 'cora':
        args.reduction_rate = 0.5
    if args.dataset_name == 'citeseer':
        args.reduction_rate = 0.5
    if args.dataset_name == 'ogbn-arxiv':
        args.reduction_rate = 0.005
    if args.dataset_name == 'reddit':
        args.reduction_rate = 0.001
    return args


def generation_hyperpara(args):
    if args.dataset_name == 'citeseer':
        args.epoch_downstream = 800
    if args.dataset_name == 'ogbn-arxiv':
        args.loss_generation = 'clip'
    if args.dataset_name == 'reddit':
        args.loss_generation = 'clip'
    return args