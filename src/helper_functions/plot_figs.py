import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Monojet dicts
neutralino_jet = pd.DataFrame(columns=['mass', 'eta', 'pt', 'tet'])
sneutrino_jet  = pd.DataFrame(columns=['mass', 'eta', 'pt', 'tet'])

# Monoz dicts
neutralino_z   = pd.DataFrame(columns=['mass', 'eta', 'pt', 'mt_met', 'tet'])
sneutrino_z    = pd.DataFrame(columns=['mass', 'eta', 'pt', 'mt_met', 'tet'])

filename = "samples.csv"

with open(filename, 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        model = row[1]
        mass = float(row[0])

        if model in ["0", "1"]:
            eta = row[2:47]
            pt = row[47:92]
            tet = row[92:]
            new_row = {'mass': mass, 'eta': eta, 'pt': pt, 'tet': tet}

            if model == "0": # MSSM Neutralino jet
                neutralino_jet.loc[len(neutralino_jet)] = new_row
            else: # MSSM Sneutrino jet
                sneutrino_jet.loc[len(sneutrino_jet)] = new_row

        else: # model == "2", "3"
            eta = row[2:47]
            pt = row[47:92]
            mt_met = row[92:137]
            tet = row[137:182]

            new_row = {'mass': mass, 'eta': eta, 'pt': pt, "mt_met": mt_met, 'tet': tet}

            if model == "2": # MSSM Neutralino z
                neutralino_z.loc[len(neutralino_z)] = new_row
            else: # MSSM Sneutrino z
                sneutrino_z.loc[len(sneutrino_z)] = new_row


# Plot frame
frame = gridspec.GridSpec(1,1) 

mass_neutralino_jet = neutralino_jet["mass"]
mass_sneutrino_jet = sneutrino_jet["mass"]

mass_neutralino_z = neutralino_z["mass"]
mass_sneutrino_z = sneutrino_z["mass"]

colors = ["green", "blue", "red", "black", "orange", "yellow", "purple"]

# Mono-jet data
eta_xData_jet = np.array([-4.888888888888889,-4.666666666666667,-4.444444444444445,-4.222222222222222,-4.0,-3.7777777777777777,-3.5555555555555554,-3.3333333333333335,-3.111111111111111,-2.888888888888889,-2.666666666666667,-2.4444444444444446,-2.2222222222222223,-2.0,-1.7777777777777781,-1.5555555555555558,-1.3333333333333335,-1.1111111111111112,-0.8888888888888893,-0.666666666666667,-0.44444444444444464,-0.22222222222222232,0.0,0.22222222222222232,0.44444444444444375,0.6666666666666661,0.8888888888888884,1.1111111111111107,1.333333333333333,1.5555555555555554,1.7777777777777777,2.0,2.2222222222222214,2.4444444444444438,2.666666666666666,2.8888888888888884,3.1111111111111107,3.333333333333332,3.5555555555555554,3.777777777777777,4.0,4.222222222222221,4.444444444444445,4.666666666666666,4.8888888888888875])
tet_xData_jet = np.array([144.11111111111111,152.33333333333334,160.55555555555554,168.77777777777777,177.0,185.22222222222223,193.44444444444446,201.66666666666666,209.88888888888889,218.1111111111111,226.33333333333331,234.55555555555554,242.77777777777777,251.0,259.22222222222223,267.44444444444446,275.66666666666663,283.8888888888889,292.1111111111111,300.3333333333333,308.55555555555554,316.77777777777777,325.0,333.2222222222222,341.44444444444446,349.66666666666663,357.88888888888886,366.1111111111111,374.3333333333333,382.55555555555554,390.7777777777777,399.0,407.2222222222222,415.4444444444444,423.66666666666663,431.88888888888886,440.1111111111111,448.3333333333333,456.55555555555554,464.77777777777777,472.99999999999994,481.2222222222222,489.4444444444444,497.66666666666663,505.88888888888886])
pt_xData_jet  = np.array([5.555555555555555,16.666666666666664,27.77777777777778,38.888888888888886,50.0,61.11111111111111,72.22222222222221,83.33333333333333,94.44444444444444,105.55555555555556,116.66666666666666,127.77777777777777,138.88888888888889,150.0,161.11111111111111,172.22222222222223,183.33333333333331,194.44444444444443,205.55555555555554,216.66666666666666,227.77777777777777,238.88888888888889,250.0,261.1111111111111,272.22222222222223,283.3333333333333,294.44444444444446,305.55555555555554,316.66666666666663,327.77777777777777,338.88888888888886,350.0,361.1111111111111,372.22222222222223,383.3333333333333,394.44444444444446,405.55555555555554,416.66666666666663,427.77777777777777,438.88888888888886,450.0,461.1111111111111,472.22222222222223,483.3333333333333,494.4444444444444])

eta_xBinning_jet = np.linspace(-5.0,5.0,46,endpoint=True)
pt_xBinning_jet = np.linspace(0,500.0,46,endpoint=True)
tet_xBinning_jet = np.linspace(140.0,510.0,46,endpoint=True)

eta_neutralino_jet = [[float(i) for i in x] for x in neutralino_jet["eta"]]
pt_neutralino_jet = [[float(i) for i in x] for x in neutralino_jet["pt"]]
tet_neutralino_jet = [[float(i) for i in x] for x in neutralino_jet["tet"]]

eta_sneutrino_jet = [[float(i) for i in x] for x in sneutrino_jet["eta"]]
pt_sneutrino_jet = [[float(i) for i in x] for x in sneutrino_jet["pt"]]
tet_sneutrino_jet = [[float(i) for i in x] for x in sneutrino_jet["tet"]]

#Mono-z data
eta_xData_z = np.array([-2.4444444444444446,-2.3333333333333335,-2.2222222222222223,-2.111111111111111,-2.0,-1.8888888888888888,-1.7777777777777777,-1.6666666666666667,-1.5555555555555556,-1.4444444444444444,-1.3333333333333335,-1.2222222222222223,-1.1111111111111112,-1.0,-0.8888888888888891,-0.7777777777777779,-0.6666666666666667,-0.5555555555555556,-0.44444444444444464,-0.3333333333333335,-0.22222222222222232,-0.11111111111111116,0.0,0.11111111111111116,0.22222222222222188,0.33333333333333304,0.4444444444444442,0.5555555555555554,0.6666666666666665,0.7777777777777777,0.8888888888888888,1.0,1.1111111111111107,1.2222222222222219,1.333333333333333,1.4444444444444442,1.5555555555555554,1.666666666666666,1.7777777777777777,1.8888888888888884,2.0,2.1111111111111107,2.2222222222222223,2.333333333333333,2.4444444444444438])
pt_xData_z = np.array([5.555555555555555,16.666666666666664,27.77777777777778,38.888888888888886,50.0,61.11111111111111,72.22222222222221,83.33333333333333,94.44444444444444,105.55555555555556,116.66666666666666,127.77777777777777,138.88888888888889,150.0,161.11111111111111,172.22222222222223,183.33333333333331,194.44444444444443,205.55555555555554,216.66666666666666,227.77777777777777,238.88888888888889,250.0,261.1111111111111,272.22222222222223,283.3333333333333,294.44444444444446,305.55555555555554,316.66666666666663,327.77777777777777,338.88888888888886,350.0,361.1111111111111,372.22222222222223,383.3333333333333,394.44444444444446,405.55555555555554,416.66666666666663,427.77777777777777,438.88888888888886,450.0,461.1111111111111,472.22222222222223,483.3333333333333,494.4444444444444])
mt_met_xData_z = np.array([10.0,30.0,50.0,70.0,90.0,110.0,130.0,150.0,170.0,190.0,210.0,230.0,250.0,270.0,290.0,310.0,330.0,350.0,370.0,390.0,410.0,430.0,450.0,470.0,490.0,510.0,530.0,550.0,570.0,590.0,610.0,630.0,650.0,670.0,690.0,710.0,730.0,750.0,770.0,790.0,810.0,830.0,850.0,870.0,890.0])
tet_xData_z = np.array([153.88888888888889,161.66666666666666,169.44444444444446,177.22222222222223,185.0,192.77777777777777,200.55555555555554,208.33333333333334,216.11111111111111,223.88888888888889,231.66666666666669,239.44444444444446,247.22222222222223,255.0,262.77777777777777,270.55555555555554,278.33333333333337,286.1111111111111,293.8888888888889,301.66666666666663,309.44444444444446,317.22222222222223,325.0,332.77777777777777,340.55555555555554,348.33333333333337,356.1111111111111,363.8888888888889,371.66666666666663,379.44444444444446,387.22222222222223,395.0,402.77777777777777,410.55555555555554,418.3333333333333,426.1111111111111,433.88888888888886,441.6666666666667,449.44444444444446,457.22222222222223,465.0,472.77777777777777,480.55555555555554,488.3333333333333,496.1111111111111])

eta_xBinning_z = np.linspace(-2.5,2.5,46,endpoint=True)
pt_xBinning_z = np.linspace(0.0,500.0,46,endpoint=True)
mt_met_xBinning_z = np.linspace(0.0,900.0,46,endpoint=True)
tet_xBinning_z = np.linspace(150.0,500.0,46,endpoint=True)


eta_neutralino_z = [[float(i) for i in x] for x in neutralino_z["eta"]]
pt_neutralino_z = [[float(i) for i in x] for x in neutralino_z["pt"]]
mt_met_neutralino_z = [[float(i) for i in x] for x in neutralino_z["mt_met"]]
tet_neutralino_z = [[float(i) for i in x] for x in neutralino_z["tet"]]

eta_sneutrino_z = [[float(i) for i in x] for x in sneutrino_z["eta"]]
pt_sneutrino_z = [[float(i) for i in x] for x in sneutrino_z["pt"]]
mt_met_sneutrino_z = [[float(i) for i in x] for x in sneutrino_z["mt_met"]]
tet_sneutrino_z = [[float(i) for i in x] for x in sneutrino_z["tet"]]


def set_configurations(quantity):
    plt.rc('text',usetex=False)
    plt.legend()  
    plt.ylabel(r"$\mathrm{Events}$ $(\mathrm{scaled}\ \mathrm{to}\ \mathrm{one})$", fontsize=16,color="black")

    # log scale
    ymax=0.8
    ymin=0.0001

    if quantity == "tet":
        # Boundary of y-axis for TET
        ymin=10**(-3) 
        ymax=2*10**(-1)
    elif quantity == "eta":
        # Boundary of y-axis for ETA
        ymax=2*10**(-1)

    plt.gca().set_ylim(ymin,ymax)
    plt.gca().set_xscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

def gen_fig():
    return plt.figure(figsize=(8.75,6.25),dpi=80)

def set_data(data, binning, weight, color, pad_num, quantity, signature, i):
    if signature == "neutralino_jet":
        mass = mass_neutralino_jet
    elif signature == "sneutrino_jet":
        mass = mass_sneutrino_jet
    elif signature == "neutralino_z":
        mass = mass_neutralino_z
    else:
        mass = mass_sneutrino_z

    pad_num.hist(x=data, bins=binning, weights=weight,\
				 label=f'$m_{{DM}} = ${mass[i]} GeV', histtype="step", rwidth=1.0,\
				 color=None, edgecolor=color, linewidth=1, linestyle="solid",\
				 bottom=None, cumulative=False, align="mid", orientation="vertical")

    set_configurations(quantity)

def neutralino_jet_plots(runs):
    #-------------------MONOJET Neutralino---------------------
    # Neutralino jet ETA plots
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(eta_xData_jet, eta_xBinning_jet, eta_neutralino_jet[i], colors[i], pad, "eta", "neutralino_jet", i)
        plt.xlabel(r"Mono-jet Neutralino $\eta$ $[ j_{1} ]$ ", fontsize=16,color="black")

    # Neutralino jet PT plots
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(pt_xData_jet, pt_xBinning_jet, pt_neutralino_jet[i], colors[i], pad, "pt", "neutralino_jet", i)
        plt.xlabel(r"Mono-jet Neutralino $p_{T}$ $[ j_{1} ]$ $(GeV/c)$ ", fontsize=16,color="black")
	
    # Neutralino jet TET plots
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(tet_xData_jet, tet_xBinning_jet, tet_neutralino_jet[i], colors[i], pad, "tet", "neutralino_jet", i)
        plt.xlabel(r"Mono-jet Neutralino $E_{T}$ $(GeV)$ ", fontsize=16,color="black")

def sneutrino_jet_plots(runs):
    #--------------------MONOJET Snutrino---------------------
    # Snutrino jet ETA plots
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(eta_xData_jet, eta_xBinning_jet, eta_sneutrino_jet[i], colors[i], pad, "eta", "sneutrino_jet", i)
        plt.xlabel(r"Mono-jet Sneutrino $E_{T}$ $(GeV)$ ", fontsize=16,color="black")

    # Snutrino jet PT plots
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(pt_xData_jet, pt_xBinning_jet, pt_sneutrino_jet[i], colors[i], pad, "pt", "sneutrino_jet", i)
        plt.xlabel(r"Mono-jet Sneutrino $p_{T}$ $[ j_{1} ]$ $(GeV/c)$ ", fontsize=16,color="black")
	
    # Snutrino jet TET plots
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(tet_xData_jet, tet_xBinning_jet, tet_sneutrino_jet[i], colors[i], pad, "tet", "sneutrino_jet", i)
        plt.xlabel(r"Mono-jet Sneutrino $E_{T}$ $(GeV)$ ", fontsize=16,color="black")

#--------------------MONOZ-------------------------
def neutralino_z_plots(runs):
    #-------------------Sneutrino---------------------
    # Neutralino z PT
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(eta_xData_z, eta_xBinning_z, eta_neutralino_z[i], colors[i], pad, "eta", "neutralino_z", i)
        plt.xlabel(r"Mono-z Neutralino $\eta$ $[ l+_{1} ]$ ", fontsize=16,color="black")

    # Neutralino z PT plots
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(pt_xData_z, pt_xBinning_z, pt_neutralino_z[i], colors[i], pad, "pt", "neutralino_z", i)
        plt.xlabel(r"Mono-z Neutralino $p_{T}$ $[ l+_{1} ]$ $(GeV/c)$ ", fontsize=16,color="black")
	
    # Neutralino z TET plots
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(tet_xData_z, tet_xBinning_z, tet_neutralino_z[i], colors[i], pad, "tet", "neutralino_z", i)
        plt.xlabel(r"Mono-z Neutralino $E_{T}$ $(GeV)$ ", fontsize=16,color="black")

    # Neutralino z MT_TET plots
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(mt_met_xData_z, mt_met_xBinning_z, mt_met_neutralino_z[i], colors[i], pad, "mt_met", "neutralino_z", i)
        plt.xlabel(r"Mono-z Neutralino $M_{T}$ $[ l+_{1} ]$ $(GeV/c^{2})$ ", fontsize=16,color="black")

def sneutrino_z_plots(runs):
    #-------------------Sneutrino---------------------
    # Neutralino z PT
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(eta_xData_z, eta_xBinning_z, eta_sneutrino_z[i], colors[i], pad, "eta", "sneutrino_z", i)
        plt.xlabel(r"Mono-z Sneutrino $\eta$ $[ l+_{1} ]$ ", fontsize=16,color="black")

    # Neutralino z PT plots
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(pt_xData_z, pt_xBinning_z, pt_sneutrino_z[i], colors[i], pad, "pt", "sneutrino_z", i)
        plt.xlabel(r"Mono-z Sneutrino $p_{T}$ $[ l+_{1} ]$ $(GeV/c)$ ", fontsize=16,color="black")
	
    # Neutralino z TET plots
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(tet_xData_z, tet_xBinning_z, tet_sneutrino_z[i], colors[i], pad, "tet", "sneutrino_z", i)
        plt.xlabel(r"Mono-z Sneutrino $E_{T}$ $(GeV)$ ", fontsize=16,color="black")

    # Neutralino z MT_TET plots
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    for i in range(0, runs):
        set_data(mt_met_xData_z, mt_met_xBinning_z, mt_met_sneutrino_z[i], colors[i], pad, "mt_met", "sneutrino_z", i)
        plt.xlabel(r"Mono-z Sneutrino $M_{T}$ $[ l+_{1} ]$ $(GeV/c^{2})$ ", fontsize=16,color="black")

def monojet_comparing_plot(i):
    # MONOJET ETA
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    set_data(eta_xData_jet, eta_xBinning_jet, eta_neutralino_jet[i], "red", pad, "eta", "neutralino_jet", i)
    set_data(eta_xData_jet, eta_xBinning_jet, eta_sneutrino_jet[i], "blue", pad, "eta", "sneutrino_jet", i)

    plt.text(-5.2, 0.10,  r"Mono-jet Sneutrino",fontsize=12,color="blue")
    plt.text(-5.2, 0.14,  r"Mono-jet Neutralino",fontsize=12,color="red")
    plt.xlabel(r"$E_{T}$ $(GeV)$", fontsize=16,color="black")

    # MONJET PT
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])


    set_data(pt_xData_jet, pt_xBinning_jet, pt_neutralino_jet[i], "red", pad, "pt", "neutralino_jet", i)
    set_data(pt_xData_jet, pt_xBinning_jet, pt_sneutrino_jet[i], "blue", pad, "pt", "sneutrino_jet", i)

    plt.text(-5.2, 0.37,  r"Mono-jet Sneutrino",fontsize=12,color="blue")
    plt.text(-5.2, 0.54,  r"Mono-jet Neutralino",fontsize=12,color="red")
    plt.xlabel(r"$p_{T}$ $[ j_{1} ]$ $(GeV/c)$ ", fontsize=16,color="black")

    # MONJET TET
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])
	
    set_data(tet_xData_jet, tet_xBinning_jet, tet_neutralino_jet[i], "red", pad, "tet", "neutralino_jet", i)
    set_data(tet_xData_jet, tet_xBinning_jet, tet_sneutrino_jet[i], "blue", pad, "tet", "sneutrino_jet", i)

    plt.text(130, 0.13,  r"Mono-jet Sneutrino",fontsize=12,color="blue")
    plt.text(130, 0.16,  r"Mono-jet Neutralino",fontsize=12,color="red")
    plt.xlabel(r"$E_{T}$ $(GeV)$ ", fontsize=16,color="black")

def monoz_comparing_plot(i):
    # MONOZ ETA
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])

    set_data(eta_xData_z, eta_xBinning_z, eta_neutralino_z[i], "red", pad, "eta", "neutralino_z", i)
    set_data(eta_xData_z, eta_xBinning_z, eta_sneutrino_z[i], "blue", pad, "eta", "sneutrino_z", i)

    plt.text(-2.65, 0.105,  r"Mono-z Sneutrino",fontsize=12,color="blue")
    plt.text(-2.65, 0.140,  r"Mono-z Neutralino",fontsize=12,color="red")
    plt.xlabel(r"$E_{T}$ $(GeV)$", fontsize=16,color="black")

    # MONOZ PT
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])


    set_data(pt_xData_z, pt_xBinning_z, pt_neutralino_z[i], "red", pad, "pt", "neutralino_z", i)
    set_data(pt_xData_z, pt_xBinning_z, pt_sneutrino_z[i], "blue", pad, "pt", "sneutrino_z", i)

    plt.text(-15, 0.382,  r"Mono-z Sneutrino",fontsize=12,color="blue")
    plt.text(-15, 0.552,  r"Mono-z Neutralino",fontsize=12,color="red")
    plt.xlabel(r"$p_{T}$ $[ j_{1} ]$ $(GeV/c)$ ", fontsize=16,color="black")

    # MONZ TET
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])
	
    set_data(tet_xData_z, tet_xBinning_z, tet_neutralino_z[i], "red", pad, "tet", "neutralino_z", i)
    set_data(tet_xData_z, tet_xBinning_z, tet_sneutrino_z[i], "blue", pad, "tet", "sneutrino_z", i)

    plt.text(140, 0.13,  r"Mono-z Sneutrino",fontsize=12,color="blue")
    plt.text(140, 0.16,  r"Mono-z Neutralino",fontsize=12,color="red")
    plt.xlabel(r"$E_{T}$ $(GeV)$ ", fontsize=16,color="black")

    # MONOZ MT_MET
    fig = gen_fig()
    pad = fig.add_subplot(frame[0])
	
    set_data(tet_xData_z, tet_xBinning_z, mt_met_neutralino_z[i], "red", pad, "tet", "neutralino_z", i)
    set_data(tet_xData_z, tet_xBinning_z, mt_met_sneutrino_z[i], "blue", pad, "tet", "sneutrino_z", i)

    plt.text(140, 0.13,  r"Mono-z Sneutrino",fontsize=12,color="blue")
    plt.text(140, 0.16,  r"Mono-z Neutralino",fontsize=12,color="red")
    plt.xlabel(r"Mono-z Sneutrino $M_{T}$ $[ l+_{1} ]$ $(GeV/c^{2})$ ", fontsize=16,color="black")
def main():
	
    # Number of runs plotted
    runs = 4

    neutralino_jet_plots(runs)
    sneutrino_jet_plots(runs)
    neutralino_z_plots(runs)
    sneutrino_z_plots(runs)

    # mass_index chooses which mass to compare with, 0 lowerst
    mass_index = 3
    # monojet_comparing_plot(mass_index)
    # monoz_comparing_plot(mass_index)


    plt.show()
if __name__ == "__main__":
    main()