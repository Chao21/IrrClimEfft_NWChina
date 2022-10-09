# Zhang Chao 2022/9/22
# Energy decomposition for the growing season
# scatter plot + bar plot

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

Dir     = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root = Dir + r'\EnergyEq_DTM'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\Fig8.svg'

infilef = root + r'\Eneq_f2c_daily.csv'
infileg = root + r'\Eneq_g2c_daily.csv'
infileu = root + r'\Eneq_u2c_daily.csv'

dff = pd.read_csv(infilef)
dfg = pd.read_csv(infileg)
dfu = pd.read_csv(infileu)


def getSigFlag(p):
    strSig = ''
    if p<0.05:
        strSig = '**'
    elif p<0.1:
        strSig = '*'
    else:
        strSig = ''
    return strSig

def getSE(data):
    count = data.shape[0]
    return data.std()/math.sqrt(count)*1.96

def drawCaptionPlot(ax,title,pos):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc='#D9D9D9', edgecolor='k'))
    ax.text(pos[0],pos[1],title,transform=ax.transAxes,fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

def LinearRegressionFunc(X, Y):
    X_train = np.array(X).reshape((len(X), 1))
    Y_train = np.array(Y).reshape((len(Y), 1))

    lineModel = LinearRegression()
    lineModel.fit(X_train, Y_train)


    Y_predict = lineModel.predict(X_train)

    a = lineModel.coef_[0][0]
    b = lineModel.intercept_[0]
    R2 = lineModel.score(X_train, Y_train)
    return a, b, Y_predict,R2

def plotScatter(ax,df,ylabel,No):
    plt.sca(ax)
    x,y = df['dTc_GS'],df['dTo_GS']
    newx = np.arange(x.min(),x.max()+0.3)

    rmse = sqrt(mean_squared_error(df['dTc_GS'],df['dTo_GS']))
    # 计算概率密度
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    #
    im = plt.scatter(x,y,c=z,marker='o',s=10,cmap='Spectral_r')
    parameter = np.polyfit(x,y,1)
    f = np.poly1d(parameter)
    r, sig = stats.pearsonr(x,y)
    a, b, Yp, R2 = LinearRegressionFunc(x,y)
    plt.axhline(y=0.0, c="k", alpha=0.4, ls="--", lw=1)
    plt.axvline(x=0.0, c="k", alpha=0.4, ls="--", lw=1)
    plt.plot(newx,f(newx),ls='--', lw = 1, c='k')
    plt.text(0.02, 0.89, "y = %.2fx" % (a),transform=ax.transAxes, c='k',fontsize=11)
    plt.text(0.02,0.74,"R\u00b2 = %.3f$^{%s}$" % (r * r, getSigFlag(sig)),
             transform=ax.transAxes,fontsize=11)
    plt.text(0.02, 0.61, "RMSE = %.3f" % (rmse), transform=ax.transAxes,fontsize=11)
    plt.xlabel('$\Delta$T_cal (K)')
    plt.ylabel(ylabel)

    plt.rcParams['xtick.direction'] = 'in'
    axins = ax.inset_axes((0.38, 0.13, 0.55, 0.06))
    plt.colorbar(im, cax=axins, orientation='horizontal') #, label='Density'
    plt.text(0.25,1.5,'Density',transform=axins.transAxes,fontsize=10)
    plt.text(-0.15,1.01,No,transform=ax.transAxes)#,weight='bold'


def plotBars(ax,df,ylabel,No):
    plt.sca(ax)
    x = [1,3,5,7,9]
    data = df[['dTo_GS','dTc_GS','dT_Al_GS','dT_LE_GS','dT_H_GS']]
    count = data.shape[0]
    err_attri = dict(elinewidth=.8, ecolor='k', capsize=1)

    colors = ['#A9A9A9', '#696969', '#E36F9A', '#4D78EA', '#B9AF33']
    # '#4D78EA' 潜热, '#B9AF33' 感热, '#E36F9A' 反照率,
    plt.bar(x, data.mean(), alpha=1.0, width=1, yerr = getSE(data), error_kw = err_attri,color=colors)
    plt.xticks(x,['$\Delta$T_obs','$\Delta$T_cal','$\Delta$T(a)',
                  '$\Delta$T(LE)','$\Delta$T(H)'],rotation=90)
    plt.axhline(y=0.0, c="k",   alpha = 0.4,ls="-", lw=1)
    plt.ylim(-1.4,1.2)
    plt.ylabel(ylabel)
    plt.text(-0.15, 1.02, No, transform=ax.transAxes)


####################################################################
#Figure
plt.rc('font',size=12)
fig = plt.figure(figsize=(8,5))
fig.subplots_adjust(left = 0.08, right = 0.99,
                    bottom = 0.14, top = 0.92,
                    wspace = 0.20, hspace=0.45)

ax1 = plt.subplot(231)
ax2 = plt.subplot(232)
ax3 = plt.subplot(233)
ax4 = plt.subplot(234)
ax5 = plt.subplot(235)
ax6 = plt.subplot(236)


h = 0.07
ax_c1= fig.add_axes([ax1.get_position().x0,
                     ax1.get_position().y1,
                     ax1.get_position().x1-ax1.get_position().x0,
                     h])
ax_c2= fig.add_axes([ax2.get_position().x0,
                     ax2.get_position().y1,
                     ax2.get_position().x1-ax2.get_position().x0,
                     h])
ax_c3= fig.add_axes([ax3.get_position().x0,
                     ax3.get_position().y1,
                     ax3.get_position().x1-ax3.get_position().x0,
                     h])
drawCaptionPlot(ax_c1,'forest2crop', [0.25,0.30])
drawCaptionPlot(ax_c2,'grass2crop',  [0.25,0.30])
drawCaptionPlot(ax_c3,'unuse2crop',  [0.25,0.30])

plotScatter(ax1,dff,'$\Delta$T_obs (K)','(a)')
plotScatter(ax2,dfg,'','(b)')
plotScatter(ax3,dfu,'','(c)')
plotBars(ax4,dff,'$\Delta$T (K)','(d)')
plotBars(ax5,dfg,'','(e)')
plotBars(ax6,dfu,'','(f)')

# plt.savefig(outname)
plt.show()