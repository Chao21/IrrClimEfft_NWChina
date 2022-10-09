# --coding:utf-8--
# Zhang Chao 2022/5/24
# Plot the reault of Dorminance analysis and partial correlation analysis in the nighttime

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

Dir  = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root1 = Dir  + r'\DominanceAnalysis'
root2 = Dir  + r'\CorrelationAnalysis'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\FigS15.svg'

filef = root1+'\\daf.csv'
fileg = root1+'\\dag.csv'
fileu = root1+'\\dau.csv'

dff_da = pd.read_csv(filef)
dfg_da = pd.read_csv(fileg)
dfu_da = pd.read_csv(fileu)

filef = root2+'\\crf.csv'
fileg = root2+'\\crg.csv'
fileu = root2+'\\cru.csv'

dff_pr = pd.read_csv(filef)
dfg_pr = pd.read_csv(fileg)
dfu_pr = pd.read_csv(fileu)

def drawCircle(ax,a,b,r):
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = a + r * np.cos(theta)
    y = b + r * np.sin(theta)
    ax.plot(x,y)

def plot_textgrid(ax,df,var):#
    p = df[[var + '_BJp', var + '_NJp', var + '_HXp', var + '_NWCp']]
    # print(p)
    x1, y1, x2, y2= 0,0,0,0
    for i in range(0,4):
        for j in range(0,7):
            x1, y1 = (0.5 + i) / 0.5 * 1 / 8 - 0.02, (j + 0.5) / 0.5 * 1 / 14 - 0.05
            x2, y2 = (0.5 + i) / 0.5 * 1 / 8 - 0.031, (j + 0.5) / 0.5 * 1 / 14 - 0.025
            if p.iloc[j,i] < 0.05:
                ax.text(x1,y1,'*',transform=ax.transAxes)
            elif p.iloc[j,i] < 0.1:
                ax.text(x2, y2, '$\circ$', transform=ax.transAxes, fontsize=15)
                # circle = Circle(xy=((0.5+i) ,(j+0.5))
                #                 , radius=0.1, alpha=0.9, color='k')
                # ax.add_patch(circle)

            else:
                ax.text(x1,y1, '', transform=ax.transAxes)


def plot_meshgrid_da(ax,df,var,vmin,vmax,xlabflag,ylabflag,ytext):#
    plt.sca(ax)
    r = df[[var+'_BJ', var+'_NJ', var+'_HX', var+'_NWC']]*100

    lst2 = r
    lst3 = lst2.stack()
    lst4 = r#lst3.unstack(level=0)

    mymap = plt.cm.YlGn
    norm = Normalize(vmin=vmin, vmax=vmax)
    n_cmap = plt.cm.ScalarMappable(norm=norm, cmap=mymap)
    c = ax.pcolormesh(lst4, cmap=n_cmap.cmap,vmin=vmin, vmax=vmax, alpha=0.8)
    xticks = [0.5, 1.5, 2.5, 3.5]
    xticklabels = ['NX', 'SX', 'HX', 'NWC']
    yticks = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    yticklabels = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'GS', 'Yr']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation=0)
    if xlabflag == False:
        ax.set_xticklabels([])
    if ylabflag == False:
        ax.set_yticklabels([])
    return c

def plot_meshgrid_pr(ax,df,var,vmin,vmax,xlabflag,ylabflag,ytext):#
    plt.sca(ax)
    plt.sca(ax)
    r = df[[var + '_BJr', var + '_NJr', var + '_HXr', var + '_NWCr']]

    lst2 = r
    lst3 = lst2.stack()
    lst4 = r  # lst3.unstack(level=0)
    # combine two colors
    colors1 = plt.cm.PiYG(np.linspace(0, 0.5, 128))  # seismic
    colors2 = plt.cm.PiYG(np.linspace(0.5, 1, 128))  # 0.67
    # combine them and build a new colormap
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    # mymap = plt.cm.pink
    norm = Normalize(vmin=vmin, vmax=vmax)
    n_cmap = plt.cm.ScalarMappable(norm=norm, cmap=mymap)
    c = ax.pcolormesh(lst4, cmap=n_cmap.cmap, vmin=vmin, vmax=vmax, alpha=0.8)
    xticks = [0.5, 1.5, 2.5, 3.5]
    xticklabels = ['NX', 'SX', 'HX', 'NWC']
    yticks = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    yticklabels = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'GS', 'Yr']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation=0)
    # ax.set_ylabel('Month',labelpad=1)
    if xlabflag == False:
        ax.set_xticklabels([])
    if ylabflag == False:
        ax.set_yticklabels([])
    # plt.text(0.22, 1.05, xtext, transform=ax.transAxes, fontsize=12)  # , style='italic'
    ax.text(1.02, 0.25, ytext, transform=ax.transAxes, fontsize=12, rotation=270)  # , style='italic'
    plot_textgrid(ax, df, var)
    return c
def drawCaptionPlot(ax,title,pos):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc='#D9D9D9', edgecolor='k'))  # ,alpha=0.6)
    ax.text(pos[0],pos[1],title,transform=ax.transAxes,fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig = plt.figure(figsize=(10.0, 5))
fig.subplots_adjust(left=0.06, right=0.86,
                    bottom=0.06, top=0.92,
                    wspace=0.03, hspace=0.03)
plt.rc('font',size = 12)#, family='Times New Roman'


ax1  = fig.add_axes([0.040, 0.55,0.151,0.37])
ax2  = fig.add_axes([0.197, 0.55,0.151,0.37])
ax3  = fig.add_axes([0.354, 0.55,0.151,0.37])
ax4  = fig.add_axes([0.511, 0.55,0.151,0.37])
ax5  = fig.add_axes([0.668, 0.55,0.151,0.37])
ax6  = fig.add_axes([0.825, 0.55,0.151,0.37])
ax7  = fig.add_axes([0.040, 0.17,0.151,0.37])
ax8  = fig.add_axes([0.197, 0.17,0.151,0.37])
ax9  = fig.add_axes([0.354, 0.17,0.151,0.37])
ax10 = fig.add_axes([0.511, 0.17,0.151,0.37])
ax11 = fig.add_axes([0.668, 0.17,0.151,0.37])
ax12 = fig.add_axes([0.825, 0.17,0.151,0.37])

ax1_1= fig.add_axes([0.040,0.92,0.151,0.07])
ax2_1= fig.add_axes([0.197,0.92,0.151,0.07])
ax3_1= fig.add_axes([0.354,0.92,0.151,0.07])
ax4_1= fig.add_axes([0.511,0.92,0.151,0.07])
ax5_1= fig.add_axes([0.668,0.92,0.151,0.07])
ax6_1= fig.add_axes([0.825,0.92,0.151,0.07])
drawCaptionPlot(ax1_1,'forest2crop',[0.16,0.30])
drawCaptionPlot(ax2_1,'grass2crop', [0.16,0.30])
drawCaptionPlot(ax3_1,'unuse2crop', [0.16,0.30])
drawCaptionPlot(ax4_1,'forest2crop',[0.16,0.30])
drawCaptionPlot(ax5_1,'grass2crop', [0.16,0.30])
drawCaptionPlot(ax6_1,'unuse2crop', [0.16,0.30])


vmin, vmax = 0, 60

plot_meshgrid_da(  ax1,   dff_da, 'ETn', vmin, vmax,  False, True ,'')
plot_meshgrid_da(  ax2,   dfg_da, 'ETn', vmin, vmax,  False, False,'')
plot_meshgrid_da(  ax3,   dfu_da, 'ETn', vmin, vmax,  False, False,'ET vs. LST')
plot_meshgrid_da(  ax7,   dff_da, 'ALn', vmin, vmax,  True,  True ,'')
plot_meshgrid_da(  ax8,   dfg_da, 'ALn', vmin, vmax,  True,  False,'')
c=plot_meshgrid_da(ax9,   dfu_da, 'ALn', vmin, vmax,True,  False,'Albedo vs. LST')


cax1 = fig.add_axes([0.04,0.08,0.45,0.04])
fcb1 = mpl.colorbar.ColorbarBase(ax=cax1,
                                 norm = Normalize(vmin=vmin, vmax=vmax),
                                 cmap = c.cmap,# cmap = mpl.cm.get_cmap('bwr'),
                                 orientation='horizontal',
                                 ticks= np.arange(vmin,vmax+0.01,20),
                                )
fig.text(0.18,0.005,'Relative importance(%)')

# ####################################
vmin, vmax = -0.8,0.8
plot_meshgrid_pr(  ax4,   dff_pr, 'ETn', vmin, vmax,  False, False ,'')
plot_meshgrid_pr(  ax5,   dfg_pr, 'ETn', vmin, vmax,  False, False,'')
plot_meshgrid_pr(  ax6,   dfu_pr, 'ETn', vmin, vmax,  False, False,'ET vs. LST')
plot_meshgrid_pr(  ax10,  dff_pr, 'ALn', vmin, vmax,  True,  False ,'')
plot_meshgrid_pr(  ax11,  dfg_pr, 'ALn', vmin, vmax,  True,  False,'')
c=plot_meshgrid_pr(ax12,  dfu_pr, 'ALn', vmin, vmax,  True,  False,'Albedo vs. LST')
cax2 = fig.add_axes([0.53,0.08,0.45,0.04])
fcb2 = mpl.colorbar.ColorbarBase(ax=cax2,
                                 norm = Normalize(vmin=vmin, vmax=vmax),
                                 cmap = c.cmap,# cmap = mpl.cm.get_cmap('bwr'),
                                 orientation='horizontal',
                                 ticks= np.arange(vmin,vmax+0.01,0.2),
                                )

fig.text(0.632,0.005,'Partial correlation coefficient')

# plt.savefig(outname)

plt.show()