"""Prior Jacobian spread of weight-space deep GP stacks (windowed part; residual r omitted)."""
import numpy as np, jax, jax.numpy as jnp, json
from dgp import Layer, inv_softplus
D=8; M=64; NS=200
rng=np.random.default_rng(0)
def spread(L, mean, chi, kind="ridge"):
    layers=[Layer(kind,D,D,M,jnp.eye(D) if mean else None,rng,chi=chi) for _ in range(L)]
    ps=[jax.tree_util.tree_map(jnp.asarray,ly.init) for ly in layers]
    def g(h,vs):
        for ly,p,v in zip(layers,ps,vs):
            phi,_=ly.features(p,h[None,:]); h=ly.mean_fn(h[None,:])[0] if mean else 0*h
            h=h+(phi@v)[0]
        return h
    J=jax.jit(jax.jacfwd(g))
    out=[];logdet=[]
    for i in range(NS):
        vs=[jnp.asarray(rng.normal(size=(ly.M,D))) for ly in layers]
        x=jnp.asarray(rng.normal(size=D))
        sv=np.linalg.svd(np.asarray(J(x,vs)),compute_uv=False)
        ls=np.log(np.maximum(sv,1e-300)); out.append(ls.max()-ls.min()); logdet.append(ls.mean())
    return float(np.median(out)), float(np.median(logdet))
Ls=[1,2,4,8,16,32]; res={}
for name,mean,chi_fn in [("zero mean, chi=1 (plain deep GP)",False,lambda L:1.0),
                         ("identity mean, chi=1",True,lambda L:1.0),
                         ("identity mean, chi=1/L",True,lambda L:1.0/L)]:
    r=[spread(L,mean,chi_fn(L)) for L in Ls]; res[name]=r
    sp=np.array([a for a,_ in r]); slope=np.polyfit(np.log(Ls[1:]),np.log(sp[1:]),1)[0]
    print(f"{name:34s} spread (max-min log sv): "+" ".join(f"{a:7.2f}" for a in sp)+f"   log-log slope {slope:.2f}")
    print(f"{'':34s} mean log sv           : "+" ".join(f"{b:7.2f}" for _,b in r))
json.dump(res,open("dgp_jacobian.json","w"),indent=1)
