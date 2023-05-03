def selection_1():

    # Library import
    import numpy
    import matplotlib
    import matplotlib.pyplot   as plt
    import matplotlib.gridspec as gridspec

    # Library version
    matplotlib_version = matplotlib.__version__
    numpy_version      = numpy.__version__

    # Histo binning
    xBinning = numpy.linspace(0.0,1000.0,51,endpoint=True)

    # Creating data sequence: middle of each bin
    xData = numpy.array([10.0,30.0,50.0,70.0,90.0,110.0,130.0,150.0,170.0,190.0,210.0,230.0,250.0,270.0,290.0,310.0,330.0,350.0,370.0,390.0,410.0,430.0,450.0,470.0,490.0,510.0,530.0,550.0,570.0,590.0,610.0,630.0,650.0,670.0,690.0,710.0,730.0,750.0,770.0,790.0,810.0,830.0,850.0,870.0,890.0,910.0,930.0,950.0,970.0,990.0])

    # Creating weights for histo: y2_PT_0
    y2_PT_0_weights = numpy.array([0.0,0.44143332079883085,0.19126669926512505,0.10336665685529951,0.06856666674094047,0.046433327673031634,0.03350000265754735,0.024933336449682927,0.01859999393508269,0.013499998703994185,0.0107666664162036,0.007699999596428785,0.007099999024718853,0.005733332880823561,0.004866666269387925,0.0037666661652183352,0.0033333328595005177,0.0024333337010731338,0.0021333328488389964,0.0014333335033954757,0.0009999999711259896,0.001133333783919681,0.0012333336904116127,0.000966666631203401,0.0007999999315904579,0.0004999999855629948,0.0006333332319775149,0.0004333333057178176,0.00039999996579522895,0.0004333333057178176,0.00026666660610487455,0.00023333326618228594,0.00013333335969035442,9.999999711259895e-05,0.00013333335969035442,3.3333328595005177e-05,9.999999711259895e-05,6.666665719001035e-05,9.999999711259895e-05,3.3333328595005177e-05,3.3333328595005177e-05,0.0,3.3333328595005177e-05,3.3333328595005177e-05,3.3333328595005177e-05,6.666665719001035e-05,9.999999711259895e-05,0.0,3.3333328595005177e-05,3.3333328595005177e-05])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y2_PT_0_weights,\
             label="$run\_29$", histtype="step", rwidth=1.0,\
             color=None, edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$p_{T}$ $[ j_{1} ]$ $(GeV/c)$ ",\
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
    #plt.savefig('../../HTML/MadAnalysis5job_0/selection_1.png')
    #plt.savefig('../../PDF/MadAnalysis5job_0/selection_1.png')
    #plt.savefig('../../DVI/MadAnalysis5job_0/selection_1.eps')

# Running!
if __name__ == '__main__':
    selection_1()
