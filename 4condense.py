from util.data_loader import *
from util.evaluate import *
from util.hyperpara import *
from util.models import *
from util.module import *
from util.training import *
from util.utils import *
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="cora", help= 'cora, citeseer, ogbn-arxiv, reddit')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--result_path', type=str, default="./results/")
parser.add_argument('--model_path', type=str, default="./save_pretrain_model/")
parser.add_argument('--eigen_path', type=str, default="./save_eigen/")
parser.add_argument('--condensed_path', type=str, default="./save_condensed_data/")
parser.add_argument('--data_dir', type=str, default="./data/")
parser.add_argument('--split_data_dir', type=str, default="./dataset_split/")

# generation
parser.add_argument('--epoch_u', type=int, default=600)
parser.add_argument('--lr_u', type=float, default=0.001)
parser.add_argument('--epoch_x', type=int, default=600)
parser.add_argument('--lr_x', type=float, default=0.001)
parser.add_argument('--epoch_downstream', type=int, default=600)
parser.add_argument('--lr_downstream', type=float, default=0.01)
parser.add_argument('--loss_generation', type=str, default='mse')

# downstream task
parser.add_argument('--reduction_rate', type=float, default=0.5)
parser.add_argument('--epoch_cls', type=int, default=200)
parser.add_argument('--epoch_lp', type=int, default=200)
parser.add_argument('--lr_cls', type=float, default=0.01)
parser.add_argument('--lr_lp', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--nrepeat', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_dim', type=int, default=256)
parser.add_argument('--eva_iter', type=int, default=1)
parser.add_argument('--test_gnn', type=str, default='GCN')
parser.add_argument('--shot', type=int, default=3)
args = parser.parse_args()
args = device_setting(args)
seed_everything(args.seed)


args.result_path =  f'./results_proposed/'
args.condensed_path +=  f'{args.dataset_name}/'
args.eigen_path +=  f'{args.dataset_name}/'
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
if not os.path.exists(args.condensed_path):
    os.makedirs(args.condensed_path)

args = generation_hyperpara(args)

acc_shot3_NC= []
acc_shot5_NC= []
auc_LP = []
acc_LP = []
nmi_CL = []
ari_CL = []

for i in range(args.nrepeat):
    args.seed += 1

    ## original data
    datasets = get_dataset(args)
    args, data, data_val, data_test = set_dataset(args, datasets)
    args.syn_num = int(data.train_num_original * args.reduction_rate)
    data = load_eigens(args, data)

    ## model
    model_spa = GCN(data.num_features, args.n_dim, args.syn_num, 2, args.dropout).to(args.device)
    model_spe = EigenMLP(args.n_dim, args.syn_num, args.syn_num).to(args.device)
    model_spa, ccenter_spa, model_spe, ccenter_spe = load_pre_train(args, model_spa, model_spe)
    
    ## condensed data generation
    adj_syn = adj_generation(args, data, model_spe, ccenter_spe)
    data_syn = feat_generation(args, data, model_spa, ccenter_spa, adj_syn)

    save_condensed_data(args, data_syn)

    ## downstream model training
    model = train_model_syn(args, data_syn)

    ## evaluation
    H, H_val, H_test, H_test_masked, labels_test, \
    H_train_shot_3, label_train_shot_3, \
    H_train_shot_5, label_train_shot_5 = eva_data(args, data, data_val, data_test, model)

    nmi, ari = evaluate_CL(H_test_masked, labels_test)
    acc_shot3, acc_shot5 = evaluate_NC(args, H_train_shot_3, label_train_shot_3, H_train_shot_5, label_train_shot_5, H_test_masked, labels_test)
    auc_lp, acc_lp = evaluate_LP(args, data, H, H_val, H_test, data_val, data_test)

    acc_shot3_NC.append(acc_shot3)
    acc_shot5_NC.append(acc_shot5)
    auc_LP.append(auc_lp)
    acc_LP.append(acc_lp) 
    nmi_CL.append(nmi)
    ari_CL.append(ari)
    print(f"Performance: acc_shot3: {acc_shot3*100:.2f}",  f"auc_lp: {auc_lp*100:.2f}", f"nmi: {nmi*100:.2f}")
    print()

downstream_record_caption(args, data_syn)
result_record_whole_NC(args, acc_shot3_NC, shot=3)
result_record_whole_NC(args, acc_shot5_NC, shot=5)
result_record_whole_LP(args, auc_LP, acc_LP)
result_record_whole_CL(args, nmi_CL, ari_CL)
print()
