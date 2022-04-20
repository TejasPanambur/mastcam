import os
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
from sklearn.metrics import confusion_matrix
from plot_tsne import getTSNEembed,visualize_scatter

def plotTSNE(embed,lbl,path,epoch,ps=None):
     """
        Produces a scatter plot TSNE
     """
     path1 = path+'ee_12_p_50/'
     path2 = path+'ee_20_p_50/'
     path3 = path+'ee_200_p_60/'
     path4 = path+'ee_5_p_80/'
     if not os.path.isdir(path+'ee_12_p_50/'):
         os.makedirs(path1)
         os.makedirs(path2)
         os.makedirs(path3)
         os.makedirs(path4)
     tsne_comp = getTSNEembed(embed,12.0,50.0)
     visualize_scatter(tsne_comp, ps, path1+'psl_'+str(epoch))
     visualize_scatter(tsne_comp, lbl, path1+'true_'+str(epoch))
     tsne_comp = getTSNEembed(embed,20.0,50.0)
     visualize_scatter(tsne_comp, ps, path2+'psl_'+str(epoch))
     visualize_scatter(tsne_comp, lbl, path2+'true_'+str(epoch))
     tsne_comp = getTSNEembed(embed,200.0,60.0)
     visualize_scatter(tsne_comp, ps, path3+'psl_'+str(epoch))
     visualize_scatter(tsne_comp, lbl, path3+'true_'+str(epoch))
     tsne_comp = getTSNEembed(embed,5.0,80.0)
     visualize_scatter(tsne_comp, ps, path4+'psl_'+str(epoch))
     visualize_scatter(tsne_comp, lbl, path4+'true_'+str(epoch))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr

def class_nmi(lbl_1,lbl_2):
    return normalized_mutual_info_score(lbl_1,lbl_2)

def get_acc(true_y,pred_y):
    cm = confusion_matrix(true_y,pred_y)
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    indexes = np.array(linear_assignment(_make_cost_m(cm))).T
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    return np.trace(cm2) / np.sum(cm2)

def eval_recall_K(embedding, label, K_list =None):
    if K_list is None:
        K_list = [1, 2, 4, 8]
    norm = np.sum(embedding*embedding,axis = 1)
    right_num = 0

    recall_list = np.zeros(len(K_list))

    for i in range(embedding.shape[0]):
        dis = norm[i] + norm - 2*np.squeeze(np.matmul(embedding[i],embedding.T))
        dis[i] = 1e10
        index = np.argsort(dis)
        list_index = 0
        for k in range(np.max(K_list)):
            if label[i]==label[index[k]]:
                recall_list[list_index] = recall_list[list_index]+1
                break
            if k>=K_list[list_index]-1:
                list_index = list_index + 1
    recall_list = recall_list/float(embedding.shape[0])
    for i in range(recall_list.shape[0]):
        if i == 0:
            continue
        recall_list[i] = recall_list[i]+recall_list[i-1]
    return recall_list
