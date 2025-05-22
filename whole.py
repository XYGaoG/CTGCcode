from util.data_loader import *
from util.evaluate import *
from util.hyperpara import *
from util.models import *
from util.module import *
from util.training import *
from util.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="cora", help= 'cora, citeseer, ogbn-arxiv, reddit')
parser.add_argument('--result_path', type=str, default="./results")
parser.add_argument('--epoch_cls', type=int, default=600)
parser.add_argument('--epoch_lp', type=int, default=200)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr_ssl', type=float, default=0)
parser.add_argument('--lr_cls', type=float, default=0.01)
parser.add_argument('--lr_lp', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--nrepeat', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n_dim', type=int, default=256)
parser.add_argument('--test_gnn', type=str, default='GCN')
parser.add_argument('--shot', type=int, default=3)
parser.add_argument('--data_dir', type=str, default="./data/")
parser.add_argument('--split_data_dir', type=str, default="./dataset_split/")
args = parser.parse_args()
args = device_setting(args)
seed_everything(args.seed)

args.result_path =  f'./results_whole/'
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
result_path = args.split_data_dir
if not os.path.exists(result_path):
    os.makedirs(result_path)

acc_NC= []
auc_LP= []
acc_LP= []
nmi_CL = []
ari_CL = []


for i in range(args.nrepeat):
    args.seed += i
    ## data
    datasets = get_dataset(args)
    args, data, data_val, data_test = set_dataset(args, datasets)
    print("train num:", int(data.train_mask.sum()))

    ## model
    model = GCN(data.num_features, args.n_dim, args.num_class, 2, args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.weight_decay)
    criterion = torch.nn.NLLLoss()

    ## train
    best_val_acc = 0
    best_loss = 1e6
    for epoch in range(1, args.epoch_cls):
        if epoch == args.epoch_cls // 2:
            lr = args.lr_cls*0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

        model.train()
        output = model(data)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.dataset_name in ['flickr', 'reddit']:
            train_acc, val_acc, tmp_test_acc = test_inductive(args, model, data_val, data_test)
        else:
            train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            weight = model.state_dict()

        if epoch % 1 == 0:
            print(f'NC Epoch: {epoch:03d}, Test: {tmp_test_acc:.4f}, Best Test: {test_acc:.4f}') 
    
    model.load_state_dict(weight)

    H, H_val, H_test, H_test_masked, labels_test, \
    H_train_shot_3, label_train_shot_3, \
    H_train_shot_5, label_train_shot_5 = eva_data(args, data, data_val, data_test, model)

    nmi, ari = evaluate_CL(H_test_masked, labels_test)
    auc_lp, acc_lp = evaluate_LP(args, data, H, H_val, H_test, data_val, data_test)

    acc_NC.append(test_acc)
    auc_LP.append(auc_lp)
    acc_LP.append(acc_lp) 
    nmi_CL.append(nmi)
    ari_CL.append(ari) 


result_record_whole_NC(args, acc_NC)
result_record_whole_LP(args, auc_LP, acc_LP)
result_record_whole_CL(args, nmi_CL, ari_CL)
print()






