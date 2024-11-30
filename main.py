import pandas as pd
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import metrics  
import matplotlib.pyplot as plt

#get training data
file_names=["art_daily_flatmiddle.csv", "art_daily_jumpsdown.csv","art_daily_jumpsup.csv",
            "art_daily_no_noise.csv","art_daily_nojump.csv",
            "art_daily_perfect_square_wave.csv", "art_daily_small_noise.csv",
            'art_flatline.csv', "art_increase_spike_density.csv", 
            "art_load_balancer_spikes.csv","art_noisy.csv"]
file_labels=[1,1,1,0,1,0,0,0,1,1,0]

windows=[6,7,8,9,12,14,16,18,21,24,28,32,36,42,48,56,63,64,72,84,96,112,126,144,168,192,224,252,288,336,448,504,576,672]
best_metrics=[[],[],[],[]]
all_accuracy=[[],[],[],[]]

for window_size in windows:
    #split dataset into 80% training and 20% testing
    test_size=int((4032*0.2)/window_size)*window_size
    train_set=[]
    test_set=[]
    train_labels=[]
    test_labels=[]

    for f in range(0,len(file_names)):
        df = pd.read_csv(file_names[f])
        bound=4032-test_size
        for i in range(0,4032,window_size):
            if i<bound:
                train_set.append(df.to_numpy()[i:i+window_size,1])
                train_labels.append(file_labels[f])
            else:
                test_set.append(df.to_numpy()[i:i+window_size:,1])
                test_labels.append(file_labels[f])

    #normalize the data
    train_set=normalize(train_set)
    test_set=normalize(test_set)

    test_predictions=[[],[],[],[]]
    curr_accuracy=[0,0,0,0]

    #support vector machine
    alg_svm=svm.SVC()
    alg_svm.fit(train_set,train_labels)
    test_predictions[0]=alg_svm.predict(test_set)

    #random forest with 2000 trees
    alg_rf=RandomForestClassifier(n_estimators = 2000) 
    alg_rf.fit(train_set,train_labels) 
    test_predictions[1]=alg_rf.predict(test_set)

    #k-nearest neighbor
    alg_kn=KNeighborsClassifier(n_neighbors=2, algorithm='auto')
    alg_kn.fit(train_set,train_labels)
    test_predictions[2]=alg_kn.predict(test_set)

    #k-means clustering with 2 clusters
    alg_km=KMeans(n_clusters = 2, random_state = 0, n_init='auto')
    alg_km.fit(train_set)
    test_predictions[3]=alg_km.predict(test_set)

    #calcuate metrics of each algorithm and see it's better
    for i in range(0,4):
        curr_accuracy[i]=metrics.accuracy_score(test_labels,test_predictions[i])
        if len(best_metrics[i])==0:
            tn, fp, fn, tp = metrics.confusion_matrix(test_labels, test_predictions[i]).ravel()
            best_metrics[i]=[window_size, curr_accuracy[i], metrics.precision_score(test_labels, test_predictions[i]),
                        metrics.recall_score(test_labels, test_predictions[i]), metrics.f1_score(test_labels, test_predictions[i]),
                        metrics.roc_auc_score(test_labels, test_predictions[i]), 
                        metrics.average_precision_score(test_labels, test_predictions[i]),fp/(fp+tn), fn/(fn+tp)] 
        elif best_metrics[i][1]<curr_accuracy[i]:
            tn, fp, fn, tp = metrics.confusion_matrix(test_labels, test_predictions[i]).ravel()
            best_metrics[i]=[window_size, curr_accuracy[i], metrics.precision_score(test_labels, test_predictions[i]),
                        metrics.recall_score(test_labels, test_predictions[i]), metrics.f1_score(test_labels, test_predictions[i]),
                        metrics.roc_auc_score(test_labels, test_predictions[i]), 
                        metrics.average_precision_score(test_labels, test_predictions[i]),fp/(fp+tn), fn/(fn+tp)] 
        all_accuracy[i].append(curr_accuracy[i])

#Calclate best for each algorithm
alg_labels=["Support Machine Vector", "Random Forest", "K-Nearest Neighbor","K-Means Clustering"]
fig,axs=plt.subplots(2,2)
x=0
y=0
for p in range(0,len(best_metrics)):
    #graph window size versus accuracy
    axs[x, y].plot(windows,all_accuracy[p])
    axs[x, y].set_title(alg_labels[p])

    #print out best metrics
    print(alg_labels[p])
    print("Best Window Size:", best_metrics[p][0])
    print("Accuracy:", best_metrics[p][1])
    print("Precision:",best_metrics[p][2])
    print("Recall:",best_metrics[p][3])
    print("F1-score:",best_metrics[p][4])
    print("AUC-ROC:",best_metrics[p][5])
    print("AUC-PR:",best_metrics[p][6])
    print("False Postives:", best_metrics[p][7])
    print("False Negatives:", best_metrics[p][8],"\n")

    if x==0:
        if y==0:
            y=1
        else:
            x=1
            y=0
    elif y==0:
        y=1

#graph
fig.tight_layout()
plt.show()