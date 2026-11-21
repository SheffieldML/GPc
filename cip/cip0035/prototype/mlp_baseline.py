"""Classical deep MLPs on the same tasks as the deep-GP prototype (point estimates, Adam, early stopping)."""
import numpy as np, jax, jax.numpy as jnp, optax, json, time
from jax.scipy.special import erf
from dgp_train import task
jax.config.update("jax_enable_x64", True)
act={"relu":jax.nn.relu,"tanh":jnp.tanh,"erf":erf}

def init_plain(rng,d,M,k):
    dims=[d]+[M]*k+[1]
    return [dict(W=jnp.asarray(rng.normal(size=(a,b))*np.sqrt(2.0/a)),b=jnp.zeros(b)) for a,b in zip(dims[:-1],dims[1:])]
def f_plain(ps,X,a):
    h=X
    for p in ps[:-1]: h=act[a](h@p["W"]+p["b"])
    return (h@ps[-1]["W"]+ps[-1]["b"])[:,0]

def init_matched(rng,X,M,L,D=5):
    """Same block structure as the deep GP: h <- h P + erf([h,1] W1) W2, width-D bottleneck, PCA first mean."""
    d=X.shape[1]; _,_,Vt=np.linalg.svd(X-X.mean(0),full_matrices=False)
    ps=[];Din=d
    for l in range(L):
        last=l==L-1; Dout=1 if last else D
        P=np.zeros((Din,Dout)) if last else (Vt[:D].T if Din!=D else np.eye(D))
        ps.append(dict(W1=jnp.asarray(rng.normal(size=(Din+1,M))*np.sqrt(1.0/(Din+1))),W2=jnp.asarray(rng.normal(size=(M,Dout))*0.1/np.sqrt(M)),P=jnp.asarray(P)))
        Din=Dout
    return ps
def f_matched(ps,X,a):
    h=X
    for p in ps:
        z=erf(jnp.concatenate([h,jnp.ones((h.shape[0],1))],1)@p["W1"])@p["W2"]
        h=h@jax.lax.stop_gradient(p["P"])+z
    return h[:,0]

def train(f,ps,a,Xtr,ytr,Xva,yva,steps=8000,lr=1e-3,wd=1e-5,B=256,seed=0,fixed_steps=None):
    opt=optax.adamw(lr,weight_decay=wd); st=opt.init(ps); rng=np.random.default_rng(seed)
    loss=lambda ps,X,y: jnp.mean((f(ps,X,a)-y)**2)
    @jax.jit
    def step(ps,st,X,y):
        l,g=jax.value_and_grad(loss)(ps,X,y); u,st=opt.update(g,st,ps); return optax.apply_updates(ps,u),st
    va=jax.jit(loss); best=(np.inf,ps,0)
    for t in range(fixed_steps or steps):
        idx=rng.choice(len(Xtr),min(B,len(Xtr)),replace=False); ps,st=step(ps,st,Xtr[idx],ytr[idx])
        if fixed_steps is None and t%100==0:
            v=float(va(ps,Xva,yva))
            if v<best[0]: best=(v,ps,t+1)
    return best if fixed_steps is None else (None,ps,fixed_steps)

def evaluate(fs_and_ps,a,Xt,yt,noise_var):
    mus=np.array([np.asarray(f(ps,Xt,a)) for f,ps in fs_and_ps]); mu=mus.mean(0)
    var=noise_var+(mus.var(0) if len(mus)>1 else 0.0)
    nll=np.mean(0.5*np.log(2*np.pi*var)+0.5*(yt-mu)**2/var)
    return float(np.sqrt(np.mean((mu-yt)**2))),float(nll)

def run(tname,configs,seeds=range(5)):
    X,y,Xt,yt=task(tname); rng0=np.random.default_rng(123); perm=rng0.permutation(len(X)); nv=len(X)//5
    Xva,yva,Xtr,ytr=map(jnp.asarray,(X[perm[:nv]],y[perm[:nv]],X[perm[nv:]],y[perm[nv:]]))
    Xall,yall,Xtj=map(jnp.asarray,(X,y,Xt))
    out=[]
    for name,kind,a,M,k in configs:
        mk=lambda r:(f_plain,init_plain(r,X.shape[1],M,k)) if kind=="plain" else (f_matched,init_matched(r,X,M,k))
        grid=[]
        for lr in [1e-3,3e-3]:
            for wd in [1e-5,1e-3,1e-2]:
                f,ps=mk(np.random.default_rng(0)); v,_,tb=train(f,ps,a,Xtr,ytr,Xva,yva,lr=lr,wd=wd)
                grid.append((v,lr,wd,tb))
        v,lr,wd,tb=min(grid); t0=time.time(); members=[]
        for sd in seeds:   # retrain on all 2000 points (same data as the deep GPs) for the selected budget
            f,ps=mk(np.random.default_rng(100+sd)); _,ps,_=train(f,ps,a,Xall,yall,None,None,lr=lr,wd=wd,seed=sd,fixed_steps=max(tb,100))
            members.append((f,ps))
        singles=[evaluate([m],a,Xtj,yt,v) for m in members]   # noise var = validation MSE
        ens=evaluate(members,a,Xtj,yt,v)
        sr=np.array(singles)
        r=dict(name=name,M=M,depth=k,act=a,lr=lr,wd=wd,steps=tb,val_mse=v,single_rmse=sr[:,0].mean(),single_rmse_sd=sr[:,0].std(),single_nll=sr[:,1].mean(),ens_rmse=ens[0],ens_nll=ens[1],sec=(time.time()-t0)/len(seeds))
        out.append(r)
        print(f"  {name:32s} [lr {lr:g}, wd {wd:g}, {tb} steps] single RMSE {r['single_rmse']:.3f}±{r['single_rmse_sd']:.3f} NLL {r['single_nll']:6.3f} | ens-5 RMSE {r['ens_rmse']:.3f} NLL {r['ens_nll']:6.3f}",flush=True)
    return out

if __name__=="__main__":
    res={}
    cfg=[]
    for L in [2,3]:
        for M in [32,128]: cfg.append((f"DGP-shaped erf, L={L}, M={M}","matched","erf",M,L))
    for k in [2,3]:
        for a in ["relu","tanh"]: cfg.append((f"plain {a}, {k} hidden x 64","plain",a,64,k))
    print("== composite"); res["composite"]=run("composite",cfg)
    cfg_s=[(f"DGP-shaped erf, L={L}, M=16","matched","erf",16,L) for L in [1,2,3]]+[(f"plain {a}, 3 hidden x 64","plain",a,64,3) for a in ["relu","tanh"]]
    print("== step"); res["step"]=run("step",cfg_s)
    json.dump(res,open("mlp_baseline.json","w"),indent=1,default=float)
