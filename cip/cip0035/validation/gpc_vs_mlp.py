"""GPc CDeepGp (via gpc.models.DeepGP) on the prototype tasks, for comparison with MLP baselines."""
import numpy as np, json, time, sys
from scipy.stats import norm
from gpc.models import DeepGP, SparseGPRegression
from task_src import task   # same seeded data as the prototype and mlp_baseline.py

def evaluate(m,Xt,yt):
    mu,var=m.predict(Xt); mu=mu[:,0]; var=var[:,0]
    return float(np.sqrt(np.mean((mu-yt)**2))), float(-np.mean(norm.logpdf(yt,mu,np.sqrt(var))))

def run(tname,configs,seeds,iters,batch):
    X,y,Xt,yt=task(tname); Y=y[:,None]; out=[]
    for (L,kern,M,D) in configs:
        rs=[]
        for s in seeds:
            m=DeepGP(X,Y,layers=L,hidden=D,num_inducing=M,kernel=kern)
            t=time.time(); m.optimize(iters=iters,batch_size=min(batch,len(X)),adam_lr=0.01,seed=s); tt=time.time()-t
            r,n=evaluate(m,Xt,yt); rs.append((r,n,m.elbo()/len(X),tt/iters*1e3))
        a=np.array(rs)
        row=dict(task=tname,L=L,kernel=kern,M=M,hidden=D,rmse=a[:,0].mean(),rmse_sd=a[:,0].std(),nll=a[:,1].mean(),nll_sd=a[:,1].std(),elbo_per_n=a[:,2].mean(),ms_per_iter=a[:,3].mean(),per_seed=a.tolist())
        out.append(row)
        print(f"{tname:9s} L={L} {kern:3s} M={M:3d} D={D}: RMSE {row['rmse']:.3f}±{row['rmse_sd']:.3f}  NLL {row['nll']:6.3f}±{row['nll_sd']:.3f}  ELBO/N {row['elbo_per_n']:7.3f}  {row['ms_per_iter']:.1f} ms/iter",flush=True)
    return out

if __name__=="__main__":
    seeds=[0,1,2]; res=[]
    comp=[(L,k,64,5) for k in ["mlp","rbf"] for L in [1,2,3]]
    res+=run("composite",comp,seeds,iters=3000,batch=256)
    step=[(L,k,16,1) for k in ["mlp","rbf"] for L in [1,2,3]]
    res+=run("step",step,seeds,iters=3000,batch=256)
    json.dump(res,open("gpc_vs_mlp.json","w"),indent=1)
