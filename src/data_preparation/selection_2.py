def selection_2():

    # Library import
    import numpy
    import matplotlib
    import matplotlib.pyplot   as plt
    import matplotlib.gridspec as gridspec

    # Library version
    matplotlib_version = matplotlib.__version__
    numpy_version      = numpy.__version__

    # Histo binning
    xBinning = numpy.linspace(0.0,550.0,51,endpoint=True)

    # Creating data sequence: middle of each bin
    xData = numpy.array([5.5,16.5,27.5,38.5,49.5,60.5,71.5,82.5,93.5,104.5,115.5,126.5,137.5,148.5,159.5,170.5,181.5,192.5,203.5,214.5,225.5,236.5,247.5,258.5,269.5,280.5,291.5,302.5,313.5,324.5,335.5,346.5,357.5,368.5,379.5,390.5,401.5,412.5,423.5,434.5,445.5,456.5,467.5,478.5,489.5,500.5,511.5,522.5,533.5,544.5])

    # Creating weights for histo: y3_MET_0
    y3_MET_0_weights = numpy.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1288973854329619,0.12332346843846351,0.10015677419196202,0.08639609816978777,0.07176451426527874,0.06497127888104241,0.05452011792413554,0.040062703757476156,0.03954014718945798,0.03396620651772495,0.028392265845991936,0.023689250814519665,0.020379723910634973,0.020031354838392405,0.020902280478653154,0.016896009510974672,0.011496256335017264,0.015676708879162702,0.012192994479502398,0.010102768207429681,0.011844625407259833,0.0071416103757875595,0.00766416694380574,0.006096497239751199,0.005922315663284243,0.005399756135611736,0.004354641815713646,0.0036578989357815856,0.0029611566477803905,0.001393485167933255,0.004528827535696661,0.0036578989357815856,0.0033095274958155553,0.0017418566078992851,0.0029611566477803905,0.0033095274958155553,0.0006967428799320602,0.0,0.0,0.0,0.0])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y3_MET_0_weights,\
             label="$run\_29$", histtype="step", rwidth=1.0,\
             color=None, edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$\slash{E}_{T}$ $(GeV)$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathrm{scaled}\ \mathrm{to}\ \mathrm{one})$",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=0.8
    ymin=0.0001 # linear scale
    ymin=0.0001 # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    plt.show()
    # Saving the image
    #plt.savefig('../../HTML/MadAnalysis5job_0/selection_2.png')
    #plt.savefig('../../PDF/MadAnalysis5job_0/selection_2.png')
    #plt.savefig('../../DVI/MadAnalysis5job_0/selection_2.eps')

# Running!
if __name__ == '__main__':
    selection_2()
