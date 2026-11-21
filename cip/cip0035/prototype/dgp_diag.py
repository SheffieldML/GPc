import numpy as np, jax, jax.numpy as jnp, json, time
from dgp import DeepGP, adam_train
from dgp_train import task
X,y,Xt,yt=task("composite"); out=[]
for L in [2,3]:
  for kind in ["ridge","hermite"]:
    for M in [64,128]:
      for resid in [True,False]:
        m=DeepGP(kind,X,D=5,L=L,M=M,seed=0,resid=resid); adam_train(m,X,y,steps=2500,lr=0.01)
        P=m.params; lpd,mu=m.predict_logdensity(P,jnp.asarray(Xt),jnp.asarray(yt),jax.random.PRNGKey(1))
        # fraction of each layer's prior variance captured by the windows, at trained hyperparameters, on propagated inputs
        h=jnp.asarray(X); caps=[]; resv=[]
        key=jax.random.PRNGKey(3)
        for ly,p in zip(m.layers,P["layers"]):
            phi,kd=ly.features(p,h); caps.append(float(jnp.mean(jnp.sum(phi**2,1)/kd))); resv.append(float(jnp.mean(kd-jnp.sum(phi**2,1))))
            key,k1,k2=jax.random.split(key,3)
            h=ly.sample(p,h,jax.random.normal(k1,(ly.M,ly.D_out)),jax.random.normal(k2,(h.shape[0],ly.D_out)))
        r=dict(L=L,kind=kind,M=M,resid=resid,rmse=float(np.sqrt(np.mean((np.asarray(mu)-yt)**2))),nll=float(-jnp.mean(lpd)),captured=caps,resid_var=resv)
        out.append(r); print(r,flush=True)
json.dump(out,open("dgp_diag.json","w"),indent=1)
