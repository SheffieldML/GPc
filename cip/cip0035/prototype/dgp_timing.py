import time, numpy as np, jax, jax.numpy as jnp, optax
from jax.scipy.special import erf
from dgp import DeepGP
rng=np.random.default_rng(0); d=D=8; L=5; B=256; N=10000
X=rng.normal(size=(N,d)); y=np.sin(X[:,0]); Xb=jnp.asarray(X[:B]); yb=jnp.asarray(y[:B])
def timeit(f,*a,n=50):
    r=f(*a); jax.block_until_ready(r)
    t=time.perf_counter()
    for _ in range(n): r=f(*a)
    jax.block_until_ready(r); return (time.perf_counter()-t)/n*1e3
def gp_step(kind,M,mode,resid=True):
    m=DeepGP(kind,X[:2000],D=D,L=L,M=M,seed=0,resid=resid); opt=optax.adam(1e-3); st=opt.init(m.params)
    @jax.jit
    def step(p,st,key):
        g=jax.grad(lambda q:-m.elbo(q,Xb,yb,key,N,mode))(p); u,st=opt.update(g,st); return optax.apply_updates(p,u),st
    return timeit(step,m.params,st,jax.random.PRNGKey(0))
def mlp_step(M):
    ps=[dict(W1=jnp.asarray(rng.normal(size=(D+1,M))/3),W2=jnp.asarray(rng.normal(size=(M,D if l<L-1 else 1))/np.sqrt(M))) for l in range(L)]
    opt=optax.adam(1e-3); st=opt.init(ps)
    def f(ps):
        h=Xb
        for l,p in enumerate(ps):
            z=erf(jnp.concatenate([h,jnp.ones((B,1))],1)@p["W1"])@p["W2"]
            h=h+z if l<L-1 else z
        return jnp.mean((h[:,0]-yb)**2)
    @jax.jit
    def step(ps,st):
        g=jax.grad(f)(ps); u,st=opt.update(g,st); return optax.apply_updates(ps,u),st
    return timeit(step,ps,st)
print(f"per Adam step, L={L} layers, width {D}, batch {B} (ms)")
print(f"{'M':>6} {'MLP':>8} {'WS w, no r':>11} {'WS w + r':>9} {'WS local':>9} {'ind. pts':>9}")
for M in [64,256,1024]:
    print(f"{M:6d} {mlp_step(M):8.2f} {gp_step('ridge',M,'weights',False):11.2f} {gp_step('ridge',M,'weights'):9.2f} {gp_step('ridge',M,'local'):9.2f} {gp_step('ip',M,'local'):9.2f}",flush=True)
