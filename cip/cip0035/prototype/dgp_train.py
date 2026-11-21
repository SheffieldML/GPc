"""Train deep weight-space GPs on synthetic tasks: units per layer x depth x construction."""
import numpy as np, jax, jax.numpy as jnp, json, time, sys
from dgp import DeepGP, adam_train, xhat, kcross, softplus, inv_softplus
import optax

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

def exact_gp(X,y,Xt,yt,steps=400):
    """Single-layer exact GP with the same MLP kernel (hyperparameters by Adam)."""
    Xj,yj,Xtj,ytj=map(jnp.asarray,(X,y,Xt,yt))
    p=dict(s=inv_softplus(1.0),w=inv_softplus(1.0),b=inv_softplus(0.1),n=inv_softplus(0.1))
    p=jax.tree_util.tree_map(jnp.asarray,p)
    def parts(p):
        xh=xhat(Xj,softplus(p["w"]),softplus(p["b"])); K=kcross(xh,xh,softplus(p["s"]))
        return xh,K+softplus(p["n"])*jnp.eye(len(X))
    def nlml(p):
        _,K=parts(p); L=jnp.linalg.cholesky(K); a=jax.scipy.linalg.solve_triangular(L,yj,lower=True)
        return 0.5*a@a+jnp.sum(jnp.log(jnp.diag(L)))
    opt=optax.adam(0.02); st=opt.init(p); g=jax.jit(jax.value_and_grad(nlml))
    for _ in range(steps):
        v,gr=g(p); u,st=opt.update(gr,st); p=optax.apply_updates(p,u)
    xh,K=parts(p); xs=xhat(Xtj,softplus(p["w"]),softplus(p["b"]))
    Ks=kcross(xs,xh,softplus(p["s"])); L=jnp.linalg.cholesky(K)
    A=jax.scipy.linalg.solve_triangular(L,Ks.T,lower=True); al=jax.scipy.linalg.solve_triangular(L,yj,lower=True)
    mu=A.T@al; var=jnp.diag(kcross(xs,xs,softplus(p["s"])))-jnp.sum(A*A,0)+softplus(p["n"])
    nll=-jnp.mean(-0.5*jnp.log(2*jnp.pi*var)-0.5*(ytj-mu)**2/var)
    return float(jnp.sqrt(jnp.mean((mu-ytj)**2))),float(nll),float(-v)/len(X)

def run(tname,kinds,Ls,Ms,D,steps,seeds):
    X,y,Xt,yt=task(tname)
    rm,nl,lml=exact_gp(X,y,Xt,yt)
    print(f"== {tname}: exact 1-layer GP  RMSE {rm:.3f}  test NLL {nl:.3f}  (log marginal/N {lml:.3f})",flush=True)
    out=dict(exact=dict(rmse=rm,nll=nl,lml_per_n=lml),runs=[])
    for L in Ls:
        for kind in kinds:
            for M in Ms:
                rs=[]
                for sd in seeds:
                    m=DeepGP(kind,X,D=D,L=L,M=M,seed=sd,chi=1.0)
                    t0=time.time(); adam_train(m,X,y,steps=steps,B=min(256,len(X)),lr=0.01,seed=sd); tt=time.time()-t0
                    P=m.params
                    elbo=np.mean([float(m.elbo(P,jnp.asarray(X),jnp.asarray(y),jax.random.PRNGKey(k),len(X))) for k in range(8)])/len(X)
                    lpd,mu=m.predict_logdensity(P,jnp.asarray(Xt),jnp.asarray(yt),jax.random.PRNGKey(1))
                    rs.append((float(np.sqrt(np.mean((np.asarray(mu)-yt)**2))),float(-np.mean(lpd)),elbo,tt,[ly.M for ly in m.layers]))
                r=np.array([x[:4] for x in rs]).mean(0)
                out["runs"].append(dict(L=L,kind=kind,M=M,Meff=rs[0][4],rmse=r[0],nll=r[1],elbo_per_n=r[2],sec=r[3]))
                print(f"  L={L} {kind:7s} M={M:4d} (eff {rs[0][4]}): RMSE {r[0]:.3f}  NLL {r[1]:7.3f}  ELBO/N {r[2]:7.3f}  [{r[3]:.0f}s]",flush=True)
    return out

if __name__=="__main__":
    which=sys.argv[1]
    if which=="step":
        res=run("step",["ridge","hermite","rf","ip"],[1,2,3],[8,16],D=1,steps=3000,seeds=[0,1])
    else:
        res=run("composite",["ridge","hermite","rf","ip"],[1,2,3],[16,32,64,128],D=5,steps=2500,seeds=[0])
    json.dump(res,open(f"dgp_train_{which}.json","w"),indent=1)
