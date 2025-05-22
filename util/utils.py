from util.module import *


def device_setting(args):
    if args.gpu != -1:
        args.device='cuda'
    else:
        args.device='cpu'  
    torch.cuda.set_device(args.gpu)
    return args


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_clu_idx(H,cls_num):
    device = H.device
    H = F.normalize(H, p=2, dim=-1)    
    H = H.cpu().detach().numpy()
    kmeans = faiss.Kmeans(int(H.shape[1]), cls_num, gpu=False) 
    kmeans.cp.min_points_per_centroid = 1
    kmeans.train(H.astype('float32'))
    _, I = kmeans.index.search(H.astype('float32'), 1)
    cluster_idx = I.flatten()
    return cluster_idx


def clustering(H, cls_num, labels):
    device = H.device
    H = H.cpu().detach().numpy()
    labels = labels.cpu()
    kmeans = faiss.Kmeans(int(H.shape[1]), cls_num, gpu=False) 
    kmeans.cp.min_points_per_centroid = 1
    kmeans.train(H.astype('float32'))
    _, I = kmeans.index.search(H.astype('float32'), 1)
    y_pred = I.flatten()
    nmi = nmi_score(labels, y_pred, average_method='arithmetic')
    ari = ari_score(labels, y_pred)
    return nmi, ari



def pretrain_record_caption(args):
    result_path_file = args.result_path + f"{args.dataset_name}.csv"
    with open(result_path_file ,'a+',newline='')as f:
        writer = csv.writer(f)
        writer.writerow( 
        ["******************teacher pretrain performance******************"])
        writer.writerow( [
        "syn_num:" f"{args.syn_num}",
        "reduction_rate:" f"{args.reduction_rate}",
        "epoch_pretrain:" f"{args.epoch_pretrain}",
        "lr_pretrain:" f"{args.lr_pretrain}",
        "epoch_ssl:" f"{args.epoch_ssl}",
        "iter_num:" f"{args.iter_num}",
        "lr_ssl_spa:" f"{args.lr_ssl_spa}",
        "lr_ssl_spe:" f"{args.lr_ssl_spe}",
        "alpha:" f"{args.alpha}"])

def result_record_whole_NC(args, ALL_ACCs, shot=None):
    result_path_file = args.result_path + f"{args.dataset_name}.csv"
    ALL_ACC = [np.mean(ALL_ACCs, axis=0)*100, np.std(ALL_ACCs, axis=0, ddof=1)*100] if len(ALL_ACCs) > 1 else [ALL_ACCs[0]*100, 0]
    if shot == None:
        shot = args.shot
    with open(result_path_file ,'a+',newline='')as f:
        writer = csv.writer(f)
        writer.writerow( [  
        "Node cla ACC",
        "shot:" f"{shot}",
        "test GNN:" f"{args.test_gnn}",
        "lr_cls:" f"{args.lr_cls}",
        f"{ALL_ACC[0]:.1f}+{ALL_ACC[1]:.1f}"])


        
def result_record_whole_LP(args, ALL_AUCs, ALL_ACCs):
    result_path_file = args.result_path + f"{args.dataset_name}.csv"

    ALL_ACC = [np.mean(ALL_ACCs, axis=0)*100, np.std(ALL_ACCs, axis=0, ddof=1)*100] if len(ALL_ACCs) > 1 else [ALL_ACCs[0]*100, 0]
    ALL_AUC = [np.mean(ALL_AUCs, axis=0)*100, np.std(ALL_AUCs, axis=0, ddof=1)*100] if len(ALL_AUCs) > 1 else [ALL_AUCs[0]*100, 0]  
    with open(result_path_file ,'a+',newline='')as f:
        writer = csv.writer(f)
        writer.writerow( [
        "Link pre AUC",
        "shot:" f"{args.shot}",
        "test GNN:" f"{args.test_gnn}", 
        "lr_lp:" f"{args.lr_lp}",
        f"{ALL_AUC[0]:.1f}+{ALL_AUC[1]:.1f}"    
        ])

    with open(result_path_file ,'a+',newline='')as f:
        writer = csv.writer(f)
        writer.writerow( [
        "Link pre ACC",
        "shot:" f"{args.shot}",
        "test GNN:" f"{args.test_gnn}",  
        "lr_lp:" f"{args.lr_lp}",
        f"{ALL_ACC[0]:.1f}+{ALL_ACC[1]:.1f}"       
        ])

        
def result_record_whole_CL(args, NMI, ARI):
    result_path_file = args.result_path + f"{args.dataset_name}.csv"

    ALL_ACC = [np.mean(NMI, axis=0)*100, np.std(NMI, axis=0, ddof=1)*100] if len(NMI) > 1 else [NMI[0]*100, 0]
    with open(result_path_file ,'a+',newline='')as f:
        writer = csv.writer(f)
        writer.writerow( [
        "Cluster  NMI",
        "shot:" f"{args.shot}",
        "test GNN:" f"{args.test_gnn}",
        "cluster:kmeans",
        f"{ALL_ACC[0]:.1f}+{ALL_ACC[1]:.1f}"])

    ALL_ACC = [np.mean(ARI, axis=0)*100, np.std(ARI, axis=0, ddof=1)*100] if len(ARI) > 1 else [ARI[0]*100, 0]
    with open(result_path_file ,'a+',newline='')as f:
        writer = csv.writer(f)
        writer.writerow( [
        "Cluster  ARI",
        "shot:" f"{args.shot}",
        "test GNN:" f"{args.test_gnn}", 
        "cluster:kmeans",
        f"{ALL_ACC[0]:.1f}+{ALL_ACC[1]:.1f}"])
        writer.writerow( ["****************************************************************"])
        writer.writerow( ["    "])

def teacher_record_caption(args):
    result_path_file = args.result_path + f"{args.dataset_name}.csv"
    with open(result_path_file ,'a+',newline='')as f:
        writer = csv.writer(f)
        writer.writerow( 
        ["******************teacher model performance******************"])
        writer.writerow( [
        "syn_num:" f"{args.syn_num}",
        "reduction_rate:" f"{args.reduction_rate}",
        "epoch_pretrain:" f"{args.epoch_pretrain}",
        "lr_pretrain:" f"{args.lr_pretrain}",
        "epoch_ssl:" f"{args.epoch_ssl}",
        "iter_num:" f"{args.iter_num}",
        "lr_ssl_spa:" f"{args.lr_ssl_spa}",
        "lr_ssl_spe:" f"{args.lr_ssl_spe}",
        "alpha:" f"{args.alpha}"])



def init_cc(data, model_spa, model_spe,cluster_idx):
    ccenter_spa,_ = get_cluster_center(model_spa, data, cluster_idx)
    ccenter_spe,_ = get_cluster_center(model_spe, data, cluster_idx)
    return ccenter_spa, ccenter_spe 

def get_cluster_center(model, data, cluster_idx):
    H = model.embedding(data).detach()
    H = F.normalize(H, p=2, dim=-1)
    H = H.cpu().detach().numpy()

    clusters = np.unique(cluster_idx)
    centroids = []
    stds = []
    id =[]
    for cluster in clusters:
        cluster_embeddings = H[cluster_idx == cluster]
        centroid = np.mean(cluster_embeddings, axis=0)
        std = np.std(cluster_embeddings, axis=0)
        centroids.append(centroid)
        stds.append(std)
        id.append(cluster)

    centroids=np.vstack(centroids)
    stds=np.vstack(stds)  
    cluster_centers = torch.nn.Parameter(torch.Tensor(centroids))
    stds = torch.nn.Parameter(torch.Tensor(stds))
    return cluster_centers, stds



def save_pre_train(args, model_spa, ccenter_spa, model_spe, ccenter_spe):
    model_spa_ = model_spa.cpu()
    ccenter_spa_ = ccenter_spa.clone().detach().cpu()
    model_spe_ = model_spe.cpu()
    ccenter_spe_ = ccenter_spe.clone().detach().cpu()
    save_path = args.model_path+f'{args.dataset_name}_rate_{args.reduction_rate}_seed_{args.seed}.pth'

    torch.save({
        'model_spa': model_spa_.state_dict(),
        'ccenter_spa': ccenter_spa_,
        'model_spe': model_spe_.state_dict(),
        'ccenter_spe': ccenter_spe_
    }, save_path)
    model_spa = model_spa.to(args.device)


def save_condensed_data(args, data_syn):
    feature = data_syn.x
    edge_index = data_syn.edge_index
    edge_weight = data_syn.edge_weight
    label = data_syn.y
    save_path = args.condensed_path+f'{args.dataset_name}_rate_{args.reduction_rate}_seed_{args.seed}.pth'

    torch.save({
        'feature': feature,
        'label': label,
        'edge_index': edge_index,
        'edge_weight': edge_weight
    }, save_path)


def downstream_record_caption(args, data_syn):
    result_path_file = args.result_path + f"{args.dataset_name}.csv"
    with open(result_path_file ,'a+',newline='')as f:
        writer = csv.writer(f)
        writer.writerow( 
        ["******************downstream model performance******************"])
        writer.writerow( [
        "syn_num:" f"{args.syn_num}",
        "reduction_rate:" f"{args.reduction_rate}",
        "epoch_u:" f"{args.epoch_u}",
        "lr_u:" f"{args.lr_u}",
        "epoch_x:" f"{args.epoch_x}",
        "lr_x:" f"{args.lr_x}",
        "epoch_downstream:" f"{args.epoch_downstream}",
        "lr_downstream:" f"{args.lr_downstream}"])
        

        sum_adj = data_syn.edge_weight.sum().item()
        num_adj= len(data_syn.edge_weight)
        sparsity = (num_adj*100/(len(data_syn.x)**2))
        writer.writerow([
        "adj sum:" f"{sum_adj:.2f}",
        "adj num:" f"{num_adj:.2f}",
        "adj sparsity:" f"{sparsity:.2f}"])


def clustering_learn(H, y, labels):
    labels = labels.cpu()
    cosine_similarity = torch.mm(H, y.T)
    y_pred = torch.argmax(cosine_similarity, dim=1)
    y_pred = y_pred.cpu().detach().numpy()
    nmi = nmi_score(labels, y_pred, average_method='arithmetic')
    ari = ari_score(labels, y_pred)
    return nmi, ari

def get_clu_idx_cc(H,ccenter):
    H_norm = F.normalize(H, p=2, dim=-1)
    cc_norm = F.normalize(ccenter, p=2, dim=-1)
    cosine_similarity = torch.mm(H_norm, cc_norm.T)
    y_pred = torch.argmax(cosine_similarity, dim=1)
    y_pred = y_pred.cpu().detach().numpy()
    return y_pred


def test_LP_inductive(args, model, data_val, data_test, k=2):
    with torch.no_grad():
        model.eval()

        logits = model.encode(data_val.x, data_val.val_edge_index)
        out = model.decode(logits, data_val.val_edge_label_index).view(-1).sigmoid()
        val_auc = roc_auc_score(data_val.val_edge_label.cpu().numpy(), out.cpu().numpy())
        pre = (out.cpu().numpy()>0.5).astype(np.int64)
        val_acc = accuracy_score(data_val.val_edge_label.cpu().numpy(), pre)

        logits = model.encode(data_test.x, data_test.test_edge_index)
        out = model.decode(logits, data_test.test_edge_label_index).view(-1).sigmoid()
        test_auc = roc_auc_score(data_test.test_edge_label.cpu().numpy(), out.cpu().numpy())
        pre = (out.cpu().numpy()>0.5).astype(np.int64)
        test_acc = accuracy_score(data_test.test_edge_label.cpu().numpy(), pre)

    return [0, val_auc, test_auc, val_acc, test_acc]


def test_LP(model, data):
    with torch.no_grad():
        model.eval()

        logits = model.encode(data.x, data.val_edge_index)
        out = model.decode(logits, data.val_edge_label_index).view(-1).sigmoid()
        val_auc = roc_auc_score(data.val_edge_label.cpu().numpy(), out.cpu().numpy())
        pre = (out.cpu().numpy()>0.5).astype(np.int64)
        val_acc = accuracy_score(data.val_edge_label.cpu().numpy(), pre)

        logits = model.encode(data.x, data.test_edge_index)
        out = model.decode(logits, data.test_edge_label_index).view(-1).sigmoid()
        test_auc = roc_auc_score(data.test_edge_label.cpu().numpy(), out.cpu().numpy())
        pre = (out.cpu().numpy()>0.5).astype(np.int64)
        test_acc = accuracy_score(data.test_edge_label.cpu().numpy(), pre)

    return [0, val_auc, test_auc, val_acc, test_acc]


def test_inductive(args, model, data_val, data_test, k=2):
    with torch.no_grad():
        model.eval()
        accs = []
        accs.append(0)
        for data in [data_val, data_test]:
            out = model(data)
            pred = out.argmax(1)
            acc = pred.eq(data.y).sum().item() / len(data.y)
            accs.append(acc)
    return accs

def test(model, data):
    with torch.no_grad():
        model.eval()
        out, accs = model(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = out[mask].argmax(1)
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return accs