import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"]=42; plt.rcParams["ps.fonttype"]=42; plt.rcParams["font.family"]="DejaVu Sans"
D="../data/"; F="../fig/"; import os; os.makedirs(F,exist_ok=True)
BLUE,ORANGE,AQUA,VIOLET,YELLOW="#2a78d6","#eb6834","#1baf7a","#6250d6","#eda100"
INK,MUTED,GRID="#0b0b0b","#52514e","#e4e3df"
SEQ=["#a9c8ef","#6aa1e3","#2a78d6","#16457f"]          # one-hue sequential (blue)
plt.rcParams.update({"font.size":9,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,
  "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,
  "lines.linewidth":2,"legend.frameon":False,"figure.dpi":150,"savefig.bbox":"tight","axes.titlesize":9.5,"axes.titleweight":"bold"})
def tag(ax,s): ax.text(-0.16,1.06,s,transform=ax.transAxes,fontweight="bold",fontsize=11,color=INK)
def save(fig,name): fig.savefig(F+name+".pdf"); fig.savefig(F+name+".png",dpi=300); plt.close(fig)

# ---------- Fig 1: Mittag-Leffler and exponential sums
r=json.load(open(D+"r31.json")); c=np.load(D+"r31_curves.npz"); t=c["t"]
fig,ax=plt.subplots(1,3,figsize=(10,3))
from scipy.special import gamma as G
for i,g in enumerate([0.3,0.5,0.7,0.9]):
    ax[0].loglog(t,c[f"g{g}"],color=SEQ[i],label=f"γ = {g}")
    ax[0].loglog(t[t>10],t[t>10]**(-g)/G(1-g),color=SEQ[i],ls=":",lw=1.2)
    n,e=zip(*r["fits"][str(g)]); ax[1].semilogy(n,e,"o-",color=SEQ[i],ms=4,label=f"γ = {g}")
ax[0].set(xlabel=r"t  (units of $\lambda^{-1/\gamma}$)",ylabel="E$_γ$(−t$^γ$)",title="Mittag-Leffler relaxation"); ax[0].legend(fontsize=7.5)
ax[0].text(0.03,0.05,"dotted: t$^{-γ}$/Γ(1−γ) tail",transform=ax[0].transAxes,color=MUTED,fontsize=7.5)
ax[1].axhline(0.01,color=MUTED,lw=1,ls="--"); ax[1].set(xlabel="number of exponentials n",ylabel="max relative error",title="Exponential-sum fit, 6 decades"); ax[1].legend(fontsize=7.5)
dec=sorted(int(k) for k in r["ndec"]); nn=[r["ndec"][str(k)] for k in dec]
ax[2].plot(dec,nn,"o-",color=BLUE,ms=5); p=np.polyfit(dec,nn,1)
ax[2].plot(dec,np.polyval(p,dec),color=MUTED,lw=1,ls="--"); ax[2].text(2.2,16,f"n ≈ {p[1]:.1f} + {p[0]:.2f}·(decades)",color=MUTED,fontsize=8)
ax[2].set(xlabel="decades of memory",ylabel="n for 1% max error",title="Terms vs memory (γ = 0.5)")
for a,s in zip(ax,"abc"): tag(a,s)
save(fig,"fig1_ml_expsum")

# ---------- Fig 2: GL recurrence truncation + Chebotarev statistics
tr=np.load(D+"r32_trunc.npz"); ch=json.load(open(D+"r32b_cheb.json"))
fig,ax=plt.subplots(1,2,figsize=(8,3))
ax[0].semilogx(tr["t"][1:],tr["exact"][1:],color=INK,lw=2.5,label="exact E$_{1/2}$")
for i,L in enumerate([10,100,1000]): ax[0].semilogx(tr["t"][1:],tr[f"L{L}"][1:],color=SEQ[i+1],lw=1.6,label=f"memory L = {L}")
ax[0].set(xlabel="t",ylabel="ρ(t)",title="GL recurrence with memory L"); ax[0].legend(fontsize=7.5)
mk={"1/3":"o","2/3":"s","1/5":"^","3/5":"D"}
for i,gs in enumerate(["1/3","2/3","1/5","3/5"]):
    rows=[x for x in ch if x["gamma"]==gs]
    ax[1].plot([x["L"] for x in rows],[x["freq_Lcycle"] for x in rows],mk[gs],color=[BLUE,ORANGE,AQUA,VIOLET][i],ms=5,ls="none",label=f"γ = {gs}")
L=np.linspace(3.5,12.5,50); ax[1].plot(L,1/L,color=MUTED,lw=1.2,ls="--",label="1/L (Chebotarev, S$_L$)")
ax[1].set(xlabel="memory length L (polynomial degree)",ylabel="fraction of primes with an L-cycle",title="Galois group of the memory polynomial"); ax[1].legend(fontsize=7.5)
for a,s in zip(ax,"ab"): tag(a,s)
save(fig,"fig2_recurrence_galois")

# ---------- Fig 3: LFSR as doubling map
S=np.load(D+"r33_orbit_n8.npy"); th=S*360/256; r3=json.load(open(D+"r33.json"))
fig,ax=plt.subplots(1,2,figsize=(8,3.2))
x=np.linspace(0,360,400); ax[0].plot(x,(2*x)%360,".",color=GRID,ms=1.5)
nxt=np.roll(th,-1); ex=np.isclose(nxt,(2*th)%360)
ax[0].plot(th[ex],nxt[ex],"o",color=BLUE,ms=3.5,label="exact doubling (top bit 0)")
ax[0].plot(th[~ex],nxt[~ex],"^",color=ORANGE,ms=3.5,label="doubling + feedback kick")
ax[0].set(xlabel="θ$_k$ (deg)",ylabel="θ$_{k+1}$ (deg)",title="Return map, x⁸+x⁴+x³+x²+1",xlim=(0,360),ylim=(0,360)); ax[0].legend(fontsize=7.5,loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=2)
ns=[q["n"] for q in r3["rows"]]; ax[1].semilogy(ns,[q["max_kick_deg"] for q in r3["rows"]],"o-",color=ORANGE,ms=4,label="max kick")
ax[1].semilogy(ns,[q["mean_kick_deg"] for q in r3["rows"]],"s-",color=BLUE,ms=4,label="mean |θ$_{k+1}$−2θ$_k$|")
ax[1].semilogy(ns,[q["grid_deg"] for q in r3["rows"]],color=MUTED,lw=1,ls="--",label="grid spacing 360°/2ⁿ")
ax[1].set(xlabel="degree n",ylabel="degrees",title="Deviation from the doubling map",xticks=range(3,17)); ax[1].legend(fontsize=7.5)
for a,s in zip(ax,"ab"): tag(a,s)
save(fig,"fig3_lfsr_doubling")

# ---------- Fig 4: wrap-around discrepancy
dd=json.load(open(D+"r34_disc.json")); lat=json.load(open(D+"r37.json"))["lattice"]
fig,ax=plt.subplots(1,4,figsize=(11,2.9),sharex=True)
for a,d in zip(ax,[2,4,8,16]):
    rows=[q for q in dd if q["d"]==d]; N=[2**q["m"] for q in rows]
    a.loglog(N,[q["rand_WD"] for q in rows],"o-",color=ORANGE,ms=3.5,label="pseudo-random")
    fr=[q for q in rows if "symfrob_WD" in q]
    if fr: a.loglog([2**q["m"] for q in fr],[q["symfrob_WD"] for q in fr],"D:",color=VIOLET,ms=3.5,lw=1.5,label="Galois-symmetric (Frobenius)")
    a.loglog(N,[q["symshift_WD"] for q in rows],"^--",color=AQUA,ms=3.5,lw=1.5,label="one polynomial, shifted")
    a.loglog(N,[q["sobol_WD"] for q in rows],"s-",color=BLUE,ms=3.5,label="digital net (Sobol)")
    lt=[q for q in lat if q["d"]==d]; a.loglog([2**q["m"] for q in lt],[q["WD"] for q in lt],"v-",color=YELLOW,ms=3.5,lw=1.5,label="Korobov lattice")
    a.set(title=f"d = {d}",xlabel="N"); a.set_xscale("log",base=2)
ax[0].set_ylabel("wrap-around discrepancy"); ax[0].legend(fontsize=7,loc="lower left")
for a,s in zip(ax,"abcd"): tag(a,s)
save(fig,"fig4_discrepancy")

# ---------- Fig 5: Ramachandran coverage
from samplers import sobol, rand
r5=json.load(open(D+"r35.json")); rho=np.load(D+"r35_rama_rho_model.npy"); g=np.linspace(-180,180,721)[:-1]
P=sobol(10,20)*360-180; Rn=rand(10,20,1)*360-180
fig,ax=plt.subplots(1,4,figsize=(13,3.3),gridspec_kw={"wspace":0.38})
o=np.sort(rho.ravel())[::-1]; thr=[o[np.searchsorted(np.cumsum(o),q)] for q in (0.99,0.9)]
for a,(X,ttl) in zip(ax[:3],[(P[:,[0,1]],"digital net, pair t = 0"),(P[:,[14,15]],"digital net, pair t = 5"),(Rn[:,[0,1]],"pseudo-random")]):
    a.contourf(g,g,rho.T,levels=[thr[0],thr[1],rho.max()],colors=["#dbe8f8","#a9c8ef"])
    a.plot(X[:,0],X[:,1],".",color=INK,ms=1.6)
    a.set(xlim=(-180,180),ylim=(-180,180),xticks=[-180,-90,0,90,180],yticks=[-180,-90,0,90,180],xlabel="φ (deg)",title=ttl); a.set_aspect("equal"); a.grid(False)
ax[0].set_ylabel("ψ (deg)")
cov=r5["arc_coverage"]
for i,k in enumerate(["2","4","6","8"]):
    ax[3].semilogx(cov[k]["N"],cov[k]["sobol"],color=SEQ[i],label=f"{360/2**int(k):.3g}° arcs")
    ax[3].semilogx(cov[k]["N"],cov[k]["rand"],color=SEQ[i],ls=":",lw=1.4)
ax[3].set(xlabel="N",ylabel="fraction of arcs occupied",title="Coarse-to-fine coverage"); ax[3].set_xscale("log",base=2)
ax[3].legend(fontsize=6.5,loc="lower right")
for a,s in zip(ax,"abcd"): tag(a,s)
save(fig,"fig5_ramachandran")

# ---------- Fig 6: convergence on landscapes
r6=json.load(open(D+"r36.json")); ms=r6["ms"]; N=[2**m for m in ms]; res=r6["res"]
lands=[("diffusive_0.5","σ$_u$ = 0.5 k$_B$T"),("diffusive_1","σ$_u$ = 1 k$_B$T"),("diffusive_2","σ$_u$ = 2 k$_B$T"),("funnel_8","funnel (8 k$_B$T)")]
fig,ax=plt.subplots(3,4,figsize=(11,7.2),sharex=True)
for i,d in enumerate([4,8,16]):
    for j,(L,lab) in enumerate(lands):
        a=ax[i,j]; q=res[f"{L}|d{d}"]
        a.loglog(N,q["rand_lnZ"],"o-",color=ORANGE,ms=3,label="pseudo-random")
        a.loglog(N,q["scr_lnZ"],"s-",color=BLUE,ms=3,label="scrambled digital net")
        a.set_xscale("log",base=2)
        if i==0: a.set_title(lab)
        if j==0: a.set_ylabel(f"d = {d}\nRMSE of ln Z (k$_B$T)")
        if i==2: a.set_xlabel("N")
        a.text(0.04,0.06,f"ESS/N = {q['ESS_frac']:.1e}",transform=a.transAxes,ha="left",fontsize=7,color=MUTED)
ax[0,0].legend(fontsize=7,loc="upper right")
save(fig,"fig6_convergence")

fig,ax=plt.subplots(figsize=(4.6,3.3))
mkd={4:"o",8:"s",16:"^"}
for d in [4,8,16]:
    xs=[res[f"{L}|d{d}"]["ESS_frac"] for L,_ in lands+[("rugged_4","")]]; ys=[res[f"{L}|d{d}"]["gain_lnZ_N2^14"] for L,_ in lands+[("rugged_4","")]]
    ax.semilogx(xs,ys,mkd[d],color=BLUE,ms=5.5,mfc=["none","#a9c8ef",BLUE][[4,8,16].index(d)],label=f"d = {d}",ls="none")
ax.axhline(1,color=MUTED,lw=1,ls="--")
ax.set(xlabel="effective sample fraction ESS/N of the Boltzmann factor",ylabel="error ratio random / digital net\n(ln Z, N = 2$^{14}$)",title="Where the digital net pays off")
ax.legend(fontsize=7.5)
save(fig,"fig7_gain_vs_ess")

# ---------- Fig 8: quadrature vs Markov chains
r8=json.load(open(D+"r38.json")); ms8=r8["ms"]; N8=[2**m for m in ms8]; R8=r8["res"]; ci=json.load(open(D+"r41_ci.json"))
fig=plt.figure(figsize=(12,3.6)); gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.45],wspace=0.36)
meth=[("iid","independent random",ORANGE,"o-"),("net","scrambled digital net",BLUE,"s-"),("lattice","shifted Korobov lattice",YELLOW,"v-"),("rwm","random-walk Metropolis",AQUA,"^--"),("mala","MALA",VIOLET,"D--")]
for j,(key,ttl) in enumerate([("diffusive_0.5|d4","d = 4, σ$_u$ = 0.5 k$_B$T"),("diffusive_1|d8","d = 8, σ$_u$ = 1 k$_B$T")]):
    a=fig.add_subplot(gs[0,j])
    for k,lab,c,st in meth: a.loglog(N8,R8[key][k]["rmse_U"],st,color=c,ms=3,lw=1.6,label=lab)
    a.set_xscale("log",base=2); a.set(xlabel="energy evaluations N",title=ttl)
    if j==0: a.set_ylabel("RMSE of $\\langle U\\rangle$ (k$_B$T)"); a.legend(fontsize=6.8,loc="lower left")
    tag(a,"ab"[j])
a=fig.add_subplot(gs[0,2]); a.grid(False)
Ls=["diffusive_0.5","diffusive_1","diffusive_2","funnel_8"]; ds=[4,8,16]
M=np.array([[np.log10(ci[f"{L}_d{d}"]["U_mala/net"][0]) for L in Ls] for d in ds])
from matplotlib.colors import LinearSegmentedColormap
cm=LinearSegmentedColormap.from_list("div",["#eb6834","#f3f2ef","#2a78d6"])
im=a.imshow(M,cmap=cm,vmin=-1.5,vmax=1.5,aspect="auto")
for i,d in enumerate(ds):
    for j,L in enumerate(Ls):
        v,(lo,hi)=ci[f"{L}_d{d}"]["U_mala/net"]
        a.text(j,i,f"{v:.2g}\n[{lo:.2g}, {hi:.2g}]",ha="center",va="center",fontsize=5.6,color=INK)
a.set_xticks(range(4),["σ$_u$=0.5","σ$_u$=1","σ$_u$=2","funnel"]); a.set_yticks(range(3),[f"d = {d}" for d in ds])
a.set_title("MALA ÷ net error in $\\langle U\\rangle$, N = 2$^{14}$")
cb=fig.colorbar(im,ax=a,fraction=0.046,pad=0.03); cb.set_label("log$_{10}$ ratio  (>0: net better)",fontsize=7.5)
tag(a,"c"); save(fig,"fig8_quadrature_vs_mcmc")
print("ok")

# ---------- Fig 10: heavy-tailed trap landscapes
from trap import Trap
r42=json.load(open(D+"r42.json")); r43=json.load(open(D+"r43.json")); r44=json.load(open(D+"r44.json"))
fig=plt.figure(figsize=(8,7.4)); gs=fig.add_gridspec(2,3,height_ratios=[1,1.3],wspace=0.55,hspace=0.48)
a=fig.add_subplot(gs[0,0]); T=Trap(0.6); th=np.linspace(0,2*np.pi,20000)
a.plot(np.degrees(th),T.u(th),color=BLUE,lw=0.8); a.text(0.97,0.04,"γ = 0.6",transform=a.transAxes,ha="right",fontsize=8,color=INK); a.set(xlabel="θ (deg)",ylabel="u(θ) (k$_B$T)",title="Trap landscape",xlim=(0,360),xticks=[0,90,180,270,360]); tag(a,"a")
a=fig.add_subplot(gs[0,1])
for i,g in enumerate(["0.6","0.8","1.25"]):
    v=r42["acf"][g]; l=np.array(v["lags"]); c=np.array(v["acf"]); ok=c>1e-3
    a.semilogx(l[ok],c[ok],"o-",ms=2.5,lw=1.5,color=[SEQ[3],SEQ[2],SEQ[1]][i],label=f"{g} ({r44['tau_int_'+g]:.0f})")
a.set(xlabel="lag k (proposals)",ylabel="autocorrelation of U",title="Memory of the chain"); a.legend(fontsize=7,loc="upper right",title="γ (τ$_{int}$)",title_fontsize=7,frameon=False); tag(a,"b")
a=fig.add_subplot(gs[0,2])
for i,g in enumerate(["0.6","0.8","1.25"]):
    v=r43[g]; N=np.array(v["N"]); y=np.array(v["rmse"])
    a.loglog(N,y,"s-",ms=2.5,lw=1.5,color=[SEQ[3],SEQ[2],SEQ[1]][i],label=f"γ = {g}")
    t=r44["tau_int_"+g]; a.axvline(t,color=[SEQ[3],SEQ[2],SEQ[1]][i],lw=0.8,ls=":")
NN=np.array([2**8,2**16]); a.loglog(NN,3*NN**-0.5*np.sqrt(256),color=MUTED,lw=1,ls="--",label="∝ N$^{-1/2}$")
a.set_xscale("log",base=2); a.set(xlabel="chain length N",ylabel="RMSE of time average of U",title="Plateau, then N$^{-1/2}$"); a.legend(fontsize=7,loc="lower left"); tag(a,"c")
a=fig.add_subplot(gs[1,:]); a.grid(False)
meths=[("net","net"),("lattice","lattice"),("iid","iid"),("imh_prng","IMH"),("imh_net","IMH+net"),("rwm","local RWM")]
rows=[(g,d) for g in ["0.6","0.8","1.25"] for d in [1,2,4]]; i14=r42["ms"].index(14)
M=np.zeros((len(rows),len(meths))); V=np.zeros_like(M)
for ii,(g,d) in enumerate(rows):
    e=r42["err"][f"{g}|{d}"]
    for jj,(k,_) in enumerate(meths):
        V[ii,jj]=r44[f"{g}|{d}"]["rmse"][i14] if k=="imh_net" else e[k][i14]
    M[ii]=np.log10(V[ii]/V[ii].min())
from matplotlib.colors import LinearSegmentedColormap
cm=LinearSegmentedColormap.from_list("seq",["#2a78d6","#a9c8ef","#f3f2ef"])
im=a.imshow(np.minimum(M,3),cmap=cm,vmin=0,vmax=3,aspect="auto")
for ii in range(len(rows)):
    for jj in range(len(meths)):
        a.text(jj,ii,f"{V[ii,jj]:.2g}",ha="center",va="center",fontsize=7.5,color=INK,fontweight="bold" if M[ii,jj]==0 else "normal")
a.set_xticks(range(len(meths)),[m for _,m in meths],fontsize=8)
a.set_yticks(range(len(rows)),[f"γ={g}, d={d}" for g,d in rows],fontsize=8)
a.set_title("RMSE of ⟨U⟩ at N = 2$^{14}$ (k$_B$T)".replace("⟨U⟩","$\\langle U\\rangle$"))
cb=fig.colorbar(im,ax=a,fraction=0.025,pad=0.015); cb.set_label("log$_{10}$(error / best in row)",fontsize=7); cb.ax.tick_params(labelsize=7)
tag(a,"d"); save(fig,"fig10_heavy_tail")
print("fig10 ok")
