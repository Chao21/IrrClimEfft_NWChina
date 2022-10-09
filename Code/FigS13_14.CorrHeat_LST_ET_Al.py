# --coding:utf-8--
# Zhang Chao 2022/5/24
# Correlation heatmap for LST vs. ET and LST vs. Albedo


import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

Dir  = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root = Dir  + r'\CorrelationAnalysis'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname1 = outdir + r'\FigS13.svg'
outname2 = outdir + r'\FigS14.svg'

filef=root+'\\crf.csv'
fileg=root+'\\crg.csv'
fileu=root+'\\cru.csv'

dff = pd.read_csv(filef)
dfg = pd.read_csv(fileg)
dfu = pd.read_csv(fileu)

def drawCircle(ax,a,b,r):
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = a + r * np.cos(theta)
    y = b + r * np.sin(theta)
    ax.plot(x,y)

def plot_textgrid(ax,df,var):
    p = df[[var + '_BJp', var + '_NJp', var + '_HXp', var + '_NWCp']]
    for i in range(0,4):
        for j in range(0,7):
            x1, y1 = (0.5 + i) / 0.5 * 1 / 8 - 0.02, (j + 0.5) / 0.5 * 1 / 14 - 0.05
            x2, y2 = (0.5 + i) / 0.5 * 1 / 8 - 0.031, (j + 0.5) / 0.5 * 1 / 14 - 0.025
            if p.iloc[j,i] < 0.05:
                ax.text(x1,y1,'*',transform=ax.transAxes)
            elif p.iloc[j,i] < 0.1:
                ax.text(x2, y2, '$\circ$', transform=ax.transAxes, fontsize=15)
            else:
                ax.text(x1,y1, '', transform=ax.transAxes)


def plot_meshgrid(ax,df,var,vmin,vmax,xlabflag,ylabflag,ytext):#
    plt.sca(ax)
    r = df[[var+'_BJr', var+'_NJr', var+'_HXr', var+'_NWCr']]

    lst2 = r
    lst3 = lst2.stack()
    lst4 = r#lst3.unstack(level=0)
    # combine two colors
    colors1 = plt.cm.PiYG(np.linspace(0, 0.5, 128))#seismic
    colors2 = plt.cm.PiYG(np.linspace(0.5, 1, 128))#0.67
    # combine them and build a new colormap
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    # mymap = plt.cm.pink
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
    # ax.set_ylabel('Month',labelpad=1)
    if xlabflag == False:
        ax.set_xticklabels([])
    if ylabflag == False:
        ax.set_yticklabels([])
    ax.text(1.02, 0.30, ytext, transform=ax.transAxes,fontsize=12,rotation=270)  # , style='italic'
    plot_textgrid(ax, df, var)
    return c

def drawCaptionPlot(ax,title,pos):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc='#D9D9D9', edgecolor='k'))  # ,alpha=0.6)
    ax.text(pos[0],pos[1],title,transform=ax.transAxes,fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

fig = plt.figure(figsize=(8.0, 5.0))
fig.subplots_adjust(left=0.06, right=0.86,
                    bottom=0.06, top=0.92,
                    wspace=0.03, hspace=0.03)
plt.rc('font',size = 12)#, family='Times New Roman'


ax1 = plt.subplot(231)
ax2 = plt.subplot(232)
ax3 = plt.subplot(233)
ax4 = plt.subplot(234)
ax5 = plt.subplot(235)
ax6 = plt.subplot(236)
ax11= fig.add_axes([0.06 ,0.92,0.261,0.07])
ax21= fig.add_axes([0.329,0.92,0.262,0.07])
ax31= fig.add_axes([0.599 ,0.92,0.261,0.07])
drawCaptionPlot(ax11,'forest2crop',[0.26,0.30])
drawCaptionPlot(ax21,'grass2crop', [0.26,0.30])
drawCaptionPlot(ax31,'unuse2crop', [0.26,0.30])

vmin, vmax = -0.8,0.8

plot_meshgrid(ax1, dff, 'ETd', vmin, vmax,  False, True ,'')
plot_meshgrid(ax2, dfg, 'ETd', vmin, vmax,  False, False,'')
plot_meshgrid(ax3, dfu, 'ETd', vmin, vmax,  False, False,'Daytime')
plot_meshgrid(ax4, dff, 'ETn', vmin, vmax,  True,  True ,'')
plot_meshgrid(ax5, dfg, 'ETn', vmin, vmax,  True,  False,'')
c=plot_meshgrid(ax6, dfu, 'ETn', vmin, vmax,True,  False,'Nighttime')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
cax = fig.add_axes([0.92,0.06,0.02,0.86])
fcb = mpl.colorbar.ColorbarBase(ax=cax,
                                 norm = Normalize(vmin=vmin, vmax=vmax),
                                 cmap = c.cmap,# cmap = mpl.cm.get_cmap('bwr'),
                                 ticks= np.arange(vmin,vmax+0.01,0.2),
                                )
fig.text(0.92,0.95,'Cor')

# plt.savefig(outname1)

#############################################################################
#############################################################################
#############################################################################
#Albedo vs. LST
fig = plt.figure(figsize=(8.0, 5.0))
fig.subplots_adjust(left=0.06, right=0.86,
                    bottom=0.06, top=0.92,
                    wspace=0.03, hspace=0.03)
plt.rc('font',size = 12)#, family='Times New Roman'


ax1 = plt.subplot(231)
ax2 = plt.subplot(232)
ax3 = plt.subplot(233)
ax4 = plt.subplot(234)
ax5 = plt.subplot(235)
ax6 = plt.subplot(236)
ax11= fig.add_axes([0.06 ,0.92,0.261,0.07])
ax21= fig.add_axes([0.329,0.92,0.262,0.07])
ax31= fig.add_axes([0.599 ,0.92,0.261,0.07])
drawCaptionPlot(ax11,'forest2crop',[0.26,0.30])
drawCaptionPlot(ax21,'grass2crop', [0.26,0.30])
drawCaptionPlot(ax31,'unuse2crop', [0.26,0.30])

vmin, vmax = -0.8,0.8

plot_meshgrid(ax1, dff,   'ALd', vmin, vmax,  False, True ,'')
plot_meshgrid(ax2, dfg,   'ALd', vmin, vmax,  False, False,'')
plot_meshgrid(ax3, dfu,   'ALd', vmin, vmax,  False, False,'Daytime')
plot_meshgrid(ax4, dff,   'ALn', vmin, vmax,  True,  True ,'')
plot_meshgrid(ax5, dfg,   'ALn', vmin, vmax,  True,  False,'')
c=plot_meshgrid(ax6, dfu, 'ALn', vmin, vmax,True,  False,'Nighttime')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
cax = fig.add_axes([0.92,0.06,0.02,0.86])
fcb = mpl.colorbar.ColorbarBase(ax=cax,
                                 norm = Normalize(vmin=vmin, vmax=vmax),
                                 cmap = c.cmap,# cmap = mpl.cm.get_cmap('bwr'),
                                 ticks= np.arange(vmin,vmax+0.01,0.2),
                                )
fig.text(0.92,0.95,'Cor')

# plt.savefig(outname2)

plt.show()