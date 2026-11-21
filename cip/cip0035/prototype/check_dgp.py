import numpy as np, jax, jax.numpy as jnp
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cip0028", "prototype", "deepws"))
from dgp import DeepGP, adam_train, xhat, kcross, softplus
from ridge import RidgeFeatures
rng=np.random.default_rng(0)
# 1) JAX ridge features vs numpy reference, and resid >= 0
X=rng.normal(size=(300,10)); y=np.sin(2*X[:,0])+0.1*rng.normal(size=300)
for kind in ["ridge","hermite","rf","ip"]:
    m=DeepGP(kind,X,D=5,L=1,M=48,seed=1)
    ly,p=m.layers[0],m.params["layers"][0]
    phi,kd=ly.features(p,jnp.asarray(X))
    print(kind,"M",ly.M,"min resid",float(jnp.min(kd-jnp.sum(phi**2,1))),"mean captured",float(jnp.mean(jnp.sum(phi**2,1)/kd)), "kappa",ly.lay.get("kappa"))
ly=DeepGP("ridge",X,5,1,48,seed=1).layers[0]; lay=ly.lay
sig2,w,b=1.0,1.0,0.1
f=RidgeFeatures(np.asarray(lay["V"]),np.asarray(lay["a"]),lay["s"],sig2,np.r_[np.full(10,w/10),b])
p=dict(ly.init); 
phi,_=ly.features(jax.tree_util.tree_map(jnp.asarray,p),jnp.asarray(X))
ref=f.Kfu(X)@np.asarray(lay["LinvT"])
print("ridge phi vs numpy ref max abs diff",np.abs(np.asarray(phi)-ref).max())
# 2) L=1: ELBO <= exact log marginal at the same hyperparameters
for kind in ["ridge","hermite","ip"]:
    m=DeepGP(kind,X,5,1,48,seed=2)
    adam_train(m,X,y,steps=1500,B=300,lr=0.02)
    P=m.params; ly=m.layers[0]; pl=P["layers"][0]
    e=float(m.elbo(P,jnp.asarray(X),jnp.asarray(y),jax.random.PRNGKey(0),len(X)))
    sig2,w,b=ly.hyp(pl); noise=softplus(P["noise"])
    xh=xhat(jnp.asarray(X),w,b); K=kcross(xh,xh,sig2)+noise*jnp.eye(len(X))
    Lc=jnp.linalg.cholesky(K); a=jax.scipy.linalg.solve_triangular(Lc,jnp.asarray(y),lower=True)
    lml=float(-0.5*a@a-jnp.sum(jnp.log(jnp.diag(Lc)))-0.5*len(X)*np.log(2*np.pi))
    print(f"{kind}: ELBO {e:.2f} <= exact log marginal {lml:.2f} ? {e<=lml}")
