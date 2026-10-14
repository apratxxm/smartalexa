import sys, json, time, numpy as np, torch
from torch import nn
import train_classifier as T
d=np.load(r"A:\wake-data\features\train_data.npz")
X,y,w,kind,val,grp=d["X"],d["y"],d["w"],d["kind"],d["val"],d["group"]
names={0:"real_pos",1:"synth_pos",2:"generic",3:"local",4:"synth_neg",5:"tail",6:"pad",7:"music"}
tr=~val
Xt=torch.tensor(X[tr]);yt=torch.tensor(y[tr]);wt=torch.tensor(w[tr]);Xv=torch.tensor(X[val]);yv=y[val];kv=kind[val];gv=grp[val]
def run(drop=0.1,dim=128,epochs=12,wd=1e-4,lr=1e-3,pcw=2.0,seed=0,label=""):
    torch.manual_seed(seed)
    m=T.Net(dim); m.drop.p=drop
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    n=len(Xt)
    for ep in range(epochs):
        m.train();perm=torch.randperm(n)
        for i in range(0,n,512):
            bi=perm[i:i+512];p=m(Xt[bi]).squeeze(-1)
            cw=torch.where(yt[bi]>0.5,torch.full_like(p,pcw),torch.ones_like(p))
            loss=(T.bce(p,yt[bi])*cw*wt[bi]).mean();opt.zero_grad();loss.backward();opt.step()
        if ep in (1,3,5,8,11,15,19):
            pv=T.predict(m,Xv).numpy()
            out={}
            for k in (0,1):
                s=kv==k; out[names[k]+"_rec"]=round(float((pv[s]>0.5).mean()),3)
                # clip level: max over group
                gs=np.unique(gv[s]); out[names[k]+"_clip"]=round(float(np.mean([pv[s&(gv==g)].max()>0.5 for g in gs])),3)
            for k in (2,3,4,5,6,7):
                s=kv==k; out[names[k]]=round(float((pv[s]>0.5).mean()),4) if s.any() else None
            print(label,"ep",ep,out,flush=True)
    return m
if __name__=="__main__":
    run(label="base")
    run(drop=0.5,label="drop.5")
    run(drop=0.5,dim=64,wd=1e-2,label="d64")
