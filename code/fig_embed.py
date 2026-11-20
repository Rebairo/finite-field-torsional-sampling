import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"]=42; plt.rcParams["ps.fonttype"]=42; plt.rcParams["font.family"]="DejaVu Sans"
from matplotlib.patches import FancyArrowPatch
from rama_model import density
from samplers import sobol
INK,MUTED,GRID="#0b0b0b","#52514e","#e4e3df"; BLUE="#2a78d6"; SEQ=["#f3f7fd","#dbe8f8","#a9c8ef","#6aa1e3","#2a78d6","#16457f"]
plt.rcParams.update({"font.size":9,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,
  "axes.titlesize":9.5,"axes.titleweight":"bold","savefig.bbox":"tight"})
from matplotlib.colors import ListedColormap
fig=plt.figure(figsize=(13,4.3)); fig.subplots_adjust(wspace=0.45)
# (a) Ramachandran torsional map (model distribution)
a=fig.add_subplot(1,3,1)
g=np.linspace(-180,180,361); F,P=np.meshgrid(g,g,indexing="ij"); rho=density(F,P)
cf=a.contourf(g,g,rho.T,levels=np.linspace(0.02,rho.max(),12),cmap=ListedColormap(SEQ[1:]+["#0d2f5c"]),extend="neither")
a.contour(g,g,rho.T,levels=[0.05],colors=[MUTED],linewidths=0.8)
for txt,(x,y) in {"$\\beta$-sheet":(-120,120),"$\\alpha$-helix":(-60,-45),"left-handed\nhelix":(60,60)}.items():
    a.text(x,y,txt,ha="center",va="center",fontsize=7.5,color="white",fontweight="bold")
a.set(xlim=(-180,180),ylim=(-180,180),xticks=range(-180,181,90),yticks=range(-180,181,90),xlabel="$\\phi$ (deg)",ylabel="$\\psi$ (deg)",title="Torsional (Ramachandran) map")
for v in range(-180,181,45): a.axvline(v,color="white",lw=0.4,alpha=0.6); a.axhline(v,color="white",lw=0.4,alpha=0.6)
a.set_aspect("equal"); cb=fig.colorbar(cf,ax=a,fraction=0.046,pad=0.03,ticks=[0.2,0.4,0.6,0.8]); cb.set_label("relative probability",fontsize=7.5); cb.ax.tick_params(labelsize=7)
a.text(-0.14,1.04,"a",transform=a.transAxes,fontweight="bold",fontsize=11)
# (b) GF(2^3) BAM embedding, x^3+x^2+1
b=fig.add_subplot(1,3,2); b.set_aspect("equal"); b.axis("off")
t=np.linspace(0,2*np.pi,400); b.plot(np.cos(t),np.sin(t),color=GRID,lw=2)
states=[1,2,4,5,7,3,6]  # alpha^0..alpha^6
pos=lambda s:(np.cos(2*np.pi*s/8),np.sin(2*np.pi*s/8))
for s in range(8):
    x,y=pos(s); filled=s!=0
    b.plot(x,y,"o",ms=9,mfc=BLUE if filled else "white",mec=BLUE,mew=1.5,zorder=3)
    lab=f"{s:03b}\n{s*45}°"+("" if s==0 else f"\n$\\alpha^{states.index(s)}$")
    b.text(1.33*x,1.33*y,lab,ha="center",va="center",fontsize=7.5,color=INK)
for k in range(7):
    s0,s1=states[k],states[(k+1)%7]; p0=np.array(pos(s0))*0.9; p1=np.array(pos(s1))*0.9
    b.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=9,color="#6aa1e3",lw=1.1,connectionstyle="arc3,rad=0.15",zorder=2))
b.set_xlim(-1.6,1.6); b.set_ylim(-1.6,1.6)
b.set_title("BAM embedding of $\\mathbb{F}_{2^3}$, $x^3+x^2+1$")
b.text(0,0,"$\\theta=2\\pi s/2^3$",ha="center",va="center",fontsize=8,color=MUTED)
b.text(-0.05,1.04,"b",transform=b.transAxes,fontweight="bold",fontsize=11)
# (c) torus embedding of (phi, psi) with net points
c=fig.add_subplot(1,3,3,projection="3d")
R,r=2.2,0.85; u=np.linspace(0,2*np.pi,90); v=np.linspace(0,2*np.pi,45); U,V=np.meshgrid(u,v)
X=(R+r*np.cos(V))*np.cos(U); Y=(R+r*np.cos(V))*np.sin(U); Z=r*np.sin(V)
D=density(np.degrees(U)-180,np.degrees(V)-180)
c.plot_surface(X,Y,Z,facecolors=plt.cm.Blues(0.15+0.75*D/D.max()),rstride=1,cstride=1,linewidth=0,antialiased=False,alpha=0.55,shade=False)
pts=sobol(8,2)*2*np.pi; ph,ps=pts[:,0],pts[:,1]
c.scatter((R+r*np.cos(ps))*np.cos(ph),(R+r*np.cos(ps))*np.sin(ph),r*np.sin(ps),s=7,color=INK,depthshade=False)
c.set_box_aspect((1,1,0.42),zoom=1.35); c.set_axis_off(); c.view_init(elev=38,azim=-60)
c.set_title("Torus embedding of $(\\phi,\\psi)$")
c.text2D(-0.12,1.0,"c",transform=c.transAxes,fontweight="bold",fontsize=11)
import os; os.makedirs("../fig",exist_ok=True)
fig.savefig("../fig/fig5_torsion_embedding.pdf"); fig.savefig("../fig/fig5_torsion_embedding.png",dpi=300)
