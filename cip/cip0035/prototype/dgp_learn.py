import numpy as np, jax, jax.numpy as jnp, json
from dgp import DeepGP, adam_train, ridge_kuu_j, ridge_kuu
from dgp_train import task
# check jnp K_uu against numpy
rng=np.random.default_rng(0); V=rng.normal(size=(20,6)); V/=np.linalg.norm(V,axis=1,keepdims=True); a=rng.uniform(0.5,2.5,20)
print("K_uu jnp vs numpy", np.abs(np.asarray(ridge_kuu_j(jnp.asarray(V),jnp.asarray(a),0.7))-ridge_kuu(V,a,0.7)).max())
X,y,Xt,yt=task("composite"); out=[]
for L in [2,3]:
  for M in [32,64,128]:
    m=DeepGP("ridge_learn",X,D=5,L=L,M=M,seed=0); adam_train(m,X,y,steps=2500,lr=0.01)
    P=m.params; lpd,mu=m.predict_logdensity(P,jnp.asarray(Xt),jnp.asarray(yt),jax.random.PRNGKey(1))
    el=np.mean([float(m.elbo(P,jnp.asarray(X),jnp.asarray(y),jax.random.PRNGKey(k),len(X))) for k in range(8)])/len(X)
    conds=[]
    for ly,p in zip(m.layers,P["layers"]):
        Vn=p["V"]/jnp.linalg.norm(p["V"],axis=1,keepdims=True); ev=np.linalg.eigvalsh(np.asarray(ridge_kuu_j(Vn,p["a"],jnp.exp(p["ls"])))); conds.append(f"{ev.max()/max(ev.min(),1e-300):.0e}, s={float(jnp.exp(p['ls'])):.2f}")
    r=dict(L=L,M=M,rmse=float(np.sqrt(np.mean((np.asarray(mu)-yt)**2))),nll=float(-jnp.mean(lpd)),elbo_per_n=el,kuu=conds)
    out.append(r); print(r,flush=True)
json.dump(out,open("dgp_learn.json","w"),indent=1)
