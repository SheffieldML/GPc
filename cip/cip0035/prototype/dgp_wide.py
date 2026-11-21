import numpy as np, jax, jax.numpy as jnp, json, time
from dgp import DeepGP, adam_train
from dgp_train import task
X,y,Xt,yt=task("composite"); out=[]
for D in [16,32]:
  for kind in ["ip","ridge_learn","hermite","rf"]:
    for M in [32,128]:
      m=DeepGP(kind,X,D=D,L=3,M=M,seed=0); t=time.time(); adam_train(m,X,y,steps=2500,lr=0.01); tt=time.time()-t
      P=m.params; lpd,mu=m.predict_logdensity(P,jnp.asarray(Xt),jnp.asarray(yt),jax.random.PRNGKey(1))
      el=np.mean([float(m.elbo(P,jnp.asarray(X),jnp.asarray(y),jax.random.PRNGKey(k),len(X))) for k in range(8)])/len(X)
      r=dict(D=D,kind=kind,M=M,rmse=float(np.sqrt(np.mean((np.asarray(mu)-yt)**2))),nll=float(-jnp.mean(lpd)),elbo_per_n=el,sec=tt)
      out.append(r); print(r,flush=True)
json.dump(out,open("dgp_wide.json","w"),indent=1)
