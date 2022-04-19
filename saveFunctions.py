import pandas as pd
import csv
from os import path
import datetime 

def SaveClusters(pths,lbls,epoch,workpath):
   
        with open(workpath+'new_cluster.csv','w') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(["ImageName",str(epoch)])
            print(len(pths),len(lbls))
            #print(train_dataset.imgs[0][:2])
            for i in range(len(pths)):
                csv_out.writerow([pths[i],lbls[i]])

        df = pd.read_csv(workpath+'new_cluster.csv')        
        if not path.exists(workpath+'clustered.csv'):
            df.to_csv(workpath+'clustered.csv',index = False)
        else:
            exdf = pd.read_csv(workpath+'clustered.csv')   
            ndf = pd.merge(exdf,df,on='ImageName')
            ndf.to_csv(workpath+'clustered.csv',index = False) 
   
def SaveEpochInfo(epoch,clustering_loss, loss, nmi,workpath):
    if not path.exists(workpath+'EpochResults.csv'):
        with open(workpath+'EpochResults.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['time','epoch','clustering_loss', 'loss', 'nmi'])
            csvwriter.writerow([datetime.datetime.now(),epoch,clustering_loss, loss, nmi])
    else:
        with open(workpath+'EpochResults.csv','a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([datetime.datetime.now(),epoch,clustering_loss, loss, nmi])


def saveNMIRecall(epoch,nmi,recall,acc,workpath):
    if not path.exists(workpath+'nmi_recall.csv'):
        with open(workpath+'nmi_recall.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['time','epoch','r@1', 'r@2', 'r@4', 'r@8', 'nmi','acc'])
            csvwriter.writerow([datetime.datetime.now(),epoch,recall[0], recall[1],recall[2],recall[3], nmi, acc])
    else:
        with open(workpath+'nmi_recall.csv','a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([datetime.datetime.now(),epoch,recall[0],recall[1],recall[2],recall[3], nmi, acc])

