import numpy as np
import matplotlib.pyplot as plt


# =============================================================
# showmat
# =============================================================
def showmat(X, xlabel=None, ylabel=None, fontsize=14, crange=None, figsize=None):

    # Prepare plot data ---------------------------------------
    if figsize is None:
        figsize = [1, 1]
    X = X.copy()
    if len(X.shape) > 2:
        print("X has to be matrix or vector")
        return

    if X.shape[0]==1 or X.shape[1]==1:
        Nsize = X.size
        X = X.reshape(np.sqrt(Nsize), np.sqrt(Nsize))

    # Plot ----------------------------------------------------
    fig = plt.figure(figsize=(8*figsize[0], 6*figsize[1]))

    plt.imshow(X,interpolation='none',aspect='auto')
    plt.colorbar()

    # Color range
    if not(crange is None):
        if len(crange)==2:
            plt.clim(crange[0], crange[1])

        elif crange == "maxabs":
            xmaxabs = np.absolute(X).max()
            plt.clim(-xmaxabs, xmaxabs)

    if not(xlabel is None):
        plt.xlabel(xlabel)
    if not(ylabel is None):
        plt.ylabel(ylabel)

    plt.rcParams["font.size"] = fontsize

    plt.ion()
    plt.show()
    plt.pause(0.001)


# =============================================================
# showtimedata
# =============================================================
def showtimedata(X, xlabel="Time", ylabel="Channel", fontsize=14, linewidth=1.5,
                 intervalstd=10, figsize=None):

    # Prepare plot data ---------------------------------------
    if figsize is None:
        figsize = [2, 1]
    X = X.copy()
    X = X.reshape([X.shape[0],-1])

    if X.shape[1]==1:
        X = X.reshape([1,-1])

    Nch = X.shape[0]
    Nt = X.shape[1]

    vInterval = X.std(axis=1).max() * intervalstd
    vPos = vInterval * (np.arange(Nch,0,-1) - 1)
    vPos = vPos.reshape([1, -1]).T  # convert to column vector
    X = X + vPos

    # Plot ----------------------------------------------------
    fig = plt.figure(figsize=(8*figsize[0], 6*figsize[1]))

    for i in range(Nch):
        plt.plot(list(range(Nt)), X[i,:], linewidth=linewidth)

    plt.xlim(0, Nt-1)
    plt.ylim(X.min(),X.max())

    ylabels = [str(num) for num in range(Nch)]
    plt.yticks(vPos,ylabels)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.rcParams["font.size"] = fontsize

    plt.ion()
    plt.show()
    plt.pause(0.001)


