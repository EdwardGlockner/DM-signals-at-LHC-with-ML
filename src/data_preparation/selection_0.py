def selection_0():

    # Library import
    import numpy
    import matplotlib
    import matplotlib.pyplot   as plt
    import matplotlib.gridspec as gridspec

    # Library version
    matplotlib_version = matplotlib.__version__
    numpy_version      = numpy.__version__

    # Histo binning
    xBinning = numpy.linspace(-6.0,6.0,51,endpoint=True)

    # Creating data sequence: middle of each bin
    xData = numpy.array([-5.88,-5.64,-5.4,-5.16,-4.92,-4.68,-4.4399999999999995,-4.2,-3.96,-3.72,-3.48,-3.24,-3.0,-2.7600000000000002,-2.52,-2.2800000000000002,-2.04,-1.7999999999999998,-1.5600000000000005,-1.3200000000000003,-1.08,-0.8399999999999999,-0.6000000000000005,-0.3600000000000003,-0.1200000000000001,0.1200000000000001,0.35999999999999943,0.5999999999999996,0.8399999999999999,1.08,1.3199999999999994,1.5599999999999996,1.7999999999999998,2.039999999999999,2.2799999999999994,2.5199999999999996,2.76,3.0,3.24,3.4800000000000004,3.719999999999999,3.959999999999999,4.199999999999999,4.4399999999999995,4.68,4.92,5.16,5.4,5.639999999999999,5.879999999999999])

    # Creating weights for histo: y1_ETA_0
    y1_ETA_0_weights = numpy.array([0.0,0.0,0.0,0.0,0.0002999999662777819,0.0006333332747281887,0.0017666670218732762,0.002500000096567656,0.004333333349682814,0.005100000106377344,0.007800000029429065,0.009399999774060008,0.013733336522018077,0.01659999692576162,0.020566667702751708,0.02466666414567505,0.027999998362937536,0.03520000344394808,0.03753333852879024,0.042733334017375946,0.04359999615627883,0.04906666563189941,0.050200001757837176,0.05193333736322714,0.05496667184076353,0.054666660773453245,0.051133338057290874,0.0499666648510777,0.049433328204592134,0.044099995722488995,0.039833336533356994,0.039066662980111945,0.033866667491526246,0.028866671829424612,0.02366666501325472,0.021566666835172034,0.016933333745763123,0.012766663142288968,0.010699999778964853,0.007600000202944999,0.00493333282913501,0.0038000001014724997,0.0031999994892618847,0.0016999996578729748,0.0010000000386270623,0.0005666667036587808,0.0,0.0,0.0,0.0])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y1_ETA_0_weights,\
             label="$run\_29$", histtype="step", rwidth=1.0,\
             color=None, edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$\eta$ $[ j_{1} ]$ ",\
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
    # plt.savefig('../../HTML/MadAnalysis5job_0/selection_0.png')
    # plt.savefig('../../PDF/MadAnalysis5job_0/selection_0.png')
    # plt.savefig('../../DVI/MadAnalysis5job_0/selection_0.eps')

# Running!
if __name__ == '__main__':
    selection_0()
