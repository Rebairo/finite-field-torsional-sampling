"""Builds all Supplementary Information tables (LaTeX) and figures from the stored results in ../data."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"]=42; plt.rcParams["ps.fonttype"]=42; plt.rcParams["font.family"]="DejaVu Sans"
D="../data/"; T="../si/tables/"; F="../si/figures/"
import os; os.makedirs(T,exist_ok=True); os.makedirs(F,exist_ok=True)
def sci(x,p=2):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "--"
    if x==0: return "0"
    e=int(np.floor(np.log10(abs(x)))); m=x/10**e
    if -2<=e<=2: return f"{x:.{max(p-1-e,0) if e<1 else p-1}f}" if abs(x)>=0.01 else f"${m:.{p-1}f}\\times10^{{{e}}}$"
    return f"${m:.{p-1}f}\\times10^{{{e}}}$"
def w(name,s): open(T+name,"w").write(s)
# ---------------- S1 exponential sums
r31=json.load(open(D+"r31.json")); gs=["0.3","0.5","0.7","0.9"]
ns=[n for n,_ in r31["fits"]["0.3"]]
rows="\n".join(f"{n} & "+" & ".join(sci(dict(r31['fits'][g])[n]) for g in gs)+r" \\" for n in ns)
rows+="\n\\midrule\n$E_\\gamma/\\text{tail}$ at $t=10^4$ & "+" & ".join(f"{r31['tail_ratio'][g]:.4f}" for g in gs)+r" \\"
w("S1_expsum.tex",rows)
w("S2_decades.tex","\n".join(f"{k} & {v} \\\\" for k,v in sorted(r31["ndec"].items(),key=lambda x:int(x[0]))))
r32=json.load(open(D+"r32.json"))
w("S2b_trunc.tex","\n".join(f"{L} & {v:.3f} \\\\" for L,v in sorted(r32["truncation"].items(),key=lambda x:int(x[0])))+f"\n\\midrule\nfull & {r32['full_memory_max_abs_err']:.3f} \\\\")
# ---------------- S3 Galois verdicts and mod-2 patterns
Ls=sorted({r["L"] for r in r32["rows"]})
combos=[(g,m) for g in ["1/3","2/3","1/5","3/5"] for m in ["1","1/3"]]
def cell(r): return r"\checkmark" if r["S_L"] else "u"
rows=[]
for g,m in combos:
    rr={r["L"]:r for r in r32["rows"] if r["gamma"]==g and r["mu"]==m}
    rows.append(f"${g}$ & ${m}$ & "+" & ".join(cell(rr[L]) for L in Ls)+r" \\")
w("S3_galois.tex","\n".join(rows))
rows=[]
for g,m in combos:
    rr={r["L"]:r for r in r32["rows"] if r["gamma"]==g and r["mu"]==m}
    def pat(r):
        p=r["Prec_mod2"]; return "bad" if p is None else "("+",".join(map(str,p))+")"
    rows.append(f"${g}$ & ${m}$ & "+" & ".join(pat(rr[L]) for L in Ls)+r" \\")
w("S3b_mod2.tex","\n".join(rows))
w("S3_Ls.tex"," & ".join(f"$L={L}$" for L in Ls))
# ---------------- S4 discrepancy, full
dd=json.load(open(D+"r34_disc.json")); lat=json.load(open(D+"r37.json"))["lattice"]
latd={(q["d"],q["m"]):q["WD"] for q in lat}
out=[]
for d in [2,4,8,16]:
    out.append(f"\\midrule\n\\multicolumn{{7}}{{@{{}}l}}{{$d={d}$}}\\\\")
    for q in [x for x in dd if x["d"]==d]:
        m=q["m"]; out.append(f"$2^{{{m}}}$ & {sci(q.get('symfrob_WD'))} & {sci(q['symshift_WD'])} & {sci(q['sobol_WD'])} & {sci(q['scr_WD'])} & {sci(latd[(d,m)])} & {sci(q['rand_WD'])} \\\\")
w("S4_wd.tex","\n".join(out))
out=[]
for d in [2,4,8,16]:
    out.append(f"\\midrule\n\\multicolumn{{6}}{{@{{}}l}}{{$d={d}$}}\\\\")
    for q in [x for x in dd if x["d"]==d]:
        m=q["m"]; out.append(f"$2^{{{m}}}$ & {sci(q.get('symfrob_L2-star'))} & {sci(q['symshift_L2-star'])} & {sci(q['sobol_L2-star'])} & {sci(q['scr_L2-star'])} & {sci(q['rand_L2-star'])} \\\\")
w("S5_l2.tex","\n".join(out))
# ---------------- S6 polynomials of the primitivity test
from polyutil import ppow
def poly_tex(b):
    f=int(b,2); n=f.bit_length()-1
    return "$"+"+".join(("x^{%d}"%i if i>1 else ("x" if i==1 else "1")) for i in range(n,-1,-1) if f>>i&1)+"$"
def order(b):
    f=int(b,2); n=f.bit_length()-1; fl=[(f>>i)&1 for i in range(n+1)]; M=2**n-1
    return min(k for k in range(1,M+1) if M%k==0 and ppow([0,1],k,fl,2)==[1])
r37=json.load(open(D+"r37.json"))
P=r37["irr"]["prim"]["polys"]; NP=r37["irr"]["nonprim"]["polys"]
w("S6_polys.tex","\n".join(f"{j+2} & {int(a,2).bit_length()-1} & {poly_tex(a)} & {order(a)} & {poly_tex(b)} & {order(b)} \\\\" for j,(a,b) in enumerate(zip(P,NP))))
# ---------------- S7 pair t-values and empty cells
r35=json.load(open(D+"r35.json"))
w("S7_pairs.tex","\n".join(f"({2*i},{2*i+1}) & {r35['t_sobol_pairs'][i]} & {r35['empty32_sobol'][i]:.2f} & {r35['t_scr_pairs'][i]} & {r35['empty32_scr'][i]:.2f} & {r35['t_sym_pairs'][i]} & {r35['empty32_sym'][i]:.2f} & {r35['empty32_rand'][i]:.2f} \\\\" for i in range(10)))
# ---------------- S8 landscapes: exact values, ESS, V proxy, MCMC step sizes, Korobov generators
r36=json.load(open(D+"r36.json"))["res"]; r38=json.load(open(D+"r38.json"))["res"]
lab={"diffusive_0.5":"$\\sigma_u=0.5$","diffusive_1":"$\\sigma_u=1$","diffusive_2":"$\\sigma_u=2$","rugged_4":"$\\sigma_u=4$","funnel_8":"funnel"}
rows=[]
for d in [4,8,16,24]:
    for L in lab:
        q=r36[f"{L}|d{d}"]; k=f"{L}|d{d}"; s=r38.get(k,{})
        ess=sci(q["ESS_frac"])+("$^{\\dagger}$" if q["ESS_frac"]<4e-5 else "")
        rows.append(f"{lab[L]} & {d} & {q['lnZ']:.3f} & {q['EU']:.3f} & {ess} & {q['Vproxy']:.1f} & "+((f"{s['rwm_step']:.3f}" if s['rwm_step']<2*np.pi else "$>2\\pi$")+f" & {s['mala_step']:.4f} & {s['korobov_z'].get('14',s['korobov_z'].get(14))[1]}" if s else "-- & -- & --")+r" \\")
w("S8_land.tex","\n".join(rows))
# ---------------- S9 <U> and TV errors at 2^14 for all methods
meth=["iid","net","lattice","rwm","mala"]; ms=json.load(open(D+"r38.json"))["ms"]; i14=ms.index(14)
rows=[]
for d in [4,8,16]:
    for L in ["diffusive_0.5","diffusive_1","diffusive_2","funnel_8"]:
        q=r38[f"{L}|d{d}"]
        best=min(q[k]["rmse_U"][i14] for k in meth)
        cells=[(("\\textbf{%s}" if q[k]["rmse_U"][i14]==best else "%s")%f"{q[k]['rmse_U'][i14]:.3g}") for k in meth]
        bestt=min(q[k]["mean_TV"][i14] for k in meth)
        cellt=[(("\\textbf{%s}" if q[k]["mean_TV"][i14]==bestt else "%s")%f"{q[k]['mean_TV'][i14]:.3f}") for k in meth]
        rows.append(f"{lab[L]} & {d} & "+" & ".join(cells)+" & "+" & ".join(cellt)+r" \\")
w("S9_errors.tex","\n".join(rows))
# ---------------- S10 bootstrap CIs
ci=json.load(open(D+"r41_ci.json"))
def c(t): return f"{t[0]:.2f} [{t[1][0]:.2f}, {t[1][1]:.2f}]"
rows=[]
for d in [4,8,16]:
    for L in ["diffusive_0.5","diffusive_1","diffusive_2","funnel_8"]:
        o=ci[f"{L}_d{d}"]
        rows.append(f"{lab[L]} & {d} & {c(o['lnZ_iid/net'])} & {c(o['lnZ_iid/lat'])} & {c(o['U_mala/net'])} & {c(o['U_mala/lat'])} & {c(o['TV_mala/net'])} \\\\")
w("S10_ci.tex","\n".join(rows))
# ---------------- S11 multistart MALA seeding
r39=json.load(open(D+"r39.json")); rows=[]
for k,v in r39.items():
    L,d=k.split("|"); 
    rows.append(f"{lab[L]} & {d[1:]} & "+" & ".join(f"{v[f'K{K}_net']['rmse_U']:.3f} / {v[f'K{K}_iid']['rmse_U']:.3f}" for K in [16,64,256])+" & "+" & ".join(f"{v[f'K{K}_net']['TV']:.3f} / {v[f'K{K}_iid']['TV']:.3f}" for K in [16,256])+r" \\")
w("S11_mala_seed.tex","\n".join(rows))
# ---------------- S12 multistart minimisation, all K
r40=json.load(open(D+"r40.json")); rows=[]
for k,v in r40.items():
    L,d=k.split("|")
    for K in [16,64,256,1024]:
        a=v[f"K{K}_net"]; b=v[f"K{K}_iid"]
        rows.append(f"{lab[L]} & {d[1:]} & {v['n_ref_minima'] if K==16 else ''} & {K} & {a['minima']:.1f} / {b['minima']:.1f} & {a['boltz_weight']:.3f} / {b['boltz_weight']:.3f} & {a['p_global']:.2f} / {b['p_global']:.2f} \\\\")
    rows.append("\\midrule")
w("S12_minima.tex","\n".join(rows[:-1]))
# ---------------- Figures S1 (<U>) and S2 (TV)
BLUE,ORANGE,AQUA,VIOLET,YELLOW="#2a78d6","#eb6834","#1baf7a","#6250d6","#eda100"; INK,MUTED,GRID="#0b0b0b","#52514e","#e4e3df"
plt.rcParams.update({"font.size":8.5,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"axes.grid":True,"grid.color":GRID,
 "grid.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"axes.titlesize":9,"axes.titleweight":"bold","savefig.bbox":"tight"})
M=[("iid","independent random",ORANGE,"o-"),("net","scrambled digital net",BLUE,"s-"),("lattice","shifted Korobov lattice",YELLOW,"v-"),("rwm","random-walk Metropolis",AQUA,"^--"),("mala","MALA",VIOLET,"D--")]
N=[2**m for m in ms]
for key,ylab,name in [("rmse_U","RMSE of $\\langle U\\rangle$ (k$_B$T)","figS1_U"),("mean_TV","TV error of torsion marginal","figS2_TV")]:
    fig,ax=plt.subplots(3,4,figsize=(11,7.4),sharex=True)
    for i,d in enumerate([4,8,16]):
        for j,L in enumerate(["diffusive_0.5","diffusive_1","diffusive_2","funnel_8"]):
            a=ax[i,j]; q=r38[f"{L}|d{d}"]
            for k,l,c_,st in M: a.loglog(N,q[k][key],st,color=c_,ms=2.5,lw=1.3,label=l)
            a.set_xscale("log",base=2)
            if i==0: a.set_title({"diffusive_0.5":"σ$_u$ = 0.5 k$_B$T","diffusive_1":"σ$_u$ = 1 k$_B$T","diffusive_2":"σ$_u$ = 2 k$_B$T","funnel_8":"funnel (8 k$_B$T)"}[L])
            if j==0: a.set_ylabel(f"d = {d}\n"+ylab)
            if i==2: a.set_xlabel("energy evaluations N")
    ax[0,0].legend(fontsize=6.5,loc="lower left")
    fig.savefig(F+name+".pdf"); fig.savefig(F+name+".png",dpi=200); plt.close(fig)
# ---------------- Figure S3: exponential-sum fits and GL truncation detail
c=np.load(D+"r31_curves.npz"); t=c["t"]
fig,ax=plt.subplots(1,2,figsize=(8,3))
for i,g in enumerate(["0.3","0.5","0.7","0.9"]):
    ax[0].semilogx(t,c[f"g{g}"]*t**float(g),color=["#a9c8ef","#6aa1e3","#2a78d6","#16457f"][i],lw=1.8,label=f"γ = {g}")
from scipy.special import gamma as G
for i,g in enumerate([0.3,0.5,0.7,0.9]): ax[0].axhline(1/G(1-g),color=["#a9c8ef","#6aa1e3","#2a78d6","#16457f"][i],lw=0.9,ls=":")
ax[0].set(xlabel="t",ylabel="t$^γ$ E$_γ$(−t$^γ$)",title="Approach to the power-law tail"); ax[0].legend(fontsize=7)
tr=np.load(D+"r32_trunc.npz")
for i,L in enumerate([10,100,1000]): ax[1].loglog(tr["t"][1:],np.abs(tr[f"L{L}"][1:]-tr["exact"][1:]),color=["#6aa1e3","#2a78d6","#16457f"][i],lw=1.4,label=f"L = {L}")
ax[1].loglog(tr["t"][1:],np.abs(tr["full"][1:]-tr["exact"][1:]),color=INK,lw=1.4,label="full memory")
ax[1].set(xlabel="t",ylabel="|y$_{GL}$ − E$_{1/2}$|",title="GL error with truncated memory"); ax[1].legend(fontsize=7)
fig.savefig(F+"figS3_tail_gl.pdf"); fig.savefig(F+"figS3_tail_gl.png",dpi=200); plt.close(fig)
print("SI tables and figures written")
