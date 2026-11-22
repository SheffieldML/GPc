import numpy as np
def task(name, seed=0):
    r=np.random.default_rng(seed)
    if name=="step":    # 1-D step: classic case where depth helps
        X=r.uniform(-1,1,(300,1)); Xt=np.linspace(-1,1,500)[:,None]
        f=lambda x:(x[:,0]>0).astype(float); n=0.02
    elif name=="composite":  # 10-D: low-dim projection then a nonlinearity of the projection
        Bm=np.linalg.qr(r.normal(size=(10,2)))[0]
        X=r.normal(size=(2000,10)); Xt=r.normal(size=(1000,10))
        f=lambda x:np.sin(2*(x@Bm)[:,0]+(x@Bm)[:,1]**2); n=0.1
    y=f(X)+n*r.normal(size=len(X)); yt=f(Xt)+n*r.normal(size=len(Xt))
    mu,sd=y.mean(),y.std()
    return X,(y-mu)/sd,Xt,(yt-mu)/sd

