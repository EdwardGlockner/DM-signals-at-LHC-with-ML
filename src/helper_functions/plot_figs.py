import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Create dataframes with columns mass, eta, pt, and tet
neutralino_jet = pd.DataFrame(columns=['mass', 'eta', 'pt', 'tet'])
sneutrino_jet  = pd.DataFrame(columns=['mass', 'eta', 'pt', 'tet'])

# Needs to be added once we have the data
neutralino_z   = pd.DataFrame(columns=['mass', 'eta', 'pt', 'met', 'mt_met'])
sneutrino_z    = pd.DataFrame(columns=['mass', 'eta', 'pt', 'met', 'mt_met'])

# Number of runs plotted
runs = 7

filename = "test.csv"

with open(filename, 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        mass = float(row[0])
        eta = row[2:47]
        pt = row[47:92]
        tet = row[92:]

        new_row = {'mass': mass, 'eta': eta, 'pt': pt, 'tet': tet}
        model = row[1]

        if model == "0": # MSSM Neutralino jet
          neutralino_jet.loc[len(neutralino_jet)] = new_row
        if model == "1": # MSSM Sneutrino jet
          sneutrino_jet.loc[len(sneutrino_jet)] = new_row
        if model == "2": # MSSM Neutralino z
          neutralino_z.loc[len(neutralino_z)]
        if model == "3": # MSSM Sneutrino z
          sneutrino_z.loc[len(sneutrino_z)]

# Plot frame
frame = gridspec.GridSpec(1,1) 


masses = neutralino_jet["mass"]
colors = ["green", "blue", "red", "black", "orange", "yellow", "purple"]

eta_xData = np.array([-4.888888888888889,-4.666666666666667,-4.444444444444445,-4.222222222222222,-4.0,-3.7777777777777777,-3.5555555555555554,-3.3333333333333335,-3.111111111111111,-2.888888888888889,-2.666666666666667,-2.4444444444444446,-2.2222222222222223,-2.0,-1.7777777777777781,-1.5555555555555558,-1.3333333333333335,-1.1111111111111112,-0.8888888888888893,-0.666666666666667,-0.44444444444444464,-0.22222222222222232,0.0,0.22222222222222232,0.44444444444444375,0.6666666666666661,0.8888888888888884,1.1111111111111107,1.333333333333333,1.5555555555555554,1.7777777777777777,2.0,2.2222222222222214,2.4444444444444438,2.666666666666666,2.8888888888888884,3.1111111111111107,3.333333333333332,3.5555555555555554,3.777777777777777,4.0,4.222222222222221,4.444444444444445,4.666666666666666,4.8888888888888875])
tet_xData = np.array([144.11111111111111,152.33333333333334,160.55555555555554,168.77777777777777,177.0,185.22222222222223,193.44444444444446,201.66666666666666,209.88888888888889,218.1111111111111,226.33333333333331,234.55555555555554,242.77777777777777,251.0,259.22222222222223,267.44444444444446,275.66666666666663,283.8888888888889,292.1111111111111,300.3333333333333,308.55555555555554,316.77777777777777,325.0,333.2222222222222,341.44444444444446,349.66666666666663,357.88888888888886,366.1111111111111,374.3333333333333,382.55555555555554,390.7777777777777,399.0,407.2222222222222,415.4444444444444,423.66666666666663,431.88888888888886,440.1111111111111,448.3333333333333,456.55555555555554,464.77777777777777,472.99999999999994,481.2222222222222,489.4444444444444,497.66666666666663,505.88888888888886])
pt_xData  = np.array([5.555555555555555,16.666666666666664,27.77777777777778,38.888888888888886,50.0,61.11111111111111,72.22222222222221,83.33333333333333,94.44444444444444,105.55555555555556,116.66666666666666,127.77777777777777,138.88888888888889,150.0,161.11111111111111,172.22222222222223,183.33333333333331,194.44444444444443,205.55555555555554,216.66666666666666,227.77777777777777,238.88888888888889,250.0,261.1111111111111,272.22222222222223,283.3333333333333,294.44444444444446,305.55555555555554,316.66666666666663,327.77777777777777,338.88888888888886,350.0,361.1111111111111,372.22222222222223,383.3333333333333,394.44444444444446,405.55555555555554,416.66666666666663,427.77777777777777,438.88888888888886,450.0,461.1111111111111,472.22222222222223,483.3333333333333,494.4444444444444])

eta_xBinning = np.linspace(-5.0,5.0,46,endpoint=True)
pt_xBinning = np.linspace(0,500.0,46,endpoint=True)
tet_xBinning = np.linspace(140.0,510.0,46,endpoint=True)

def set_configurations():
  plt.rc('text',usetex=False)
  plt.legend()  
  plt.ylabel(r"$\mathrm{Events}$ $(\mathrm{scaled}\ \mathrm{to}\ \mathrm{one})$", fontsize=16,color="black")

  # Boundary of y-axis
  ymax=0.8
  ymin=0.0001 # linear scale
  ymin=0.0001 # log scale
  plt.gca().set_ylim(ymin,ymax)

  # Log/Linear
  plt.gca().set_xscale("linear")
  plt.gca().set_yscale("log",nonpositive="clip")

def gen_fig():
  return plt.figure(figsize=(8.75,6.25),dpi=80)

def set_data(data, binning, weight, color, pad_num):
  pad_num.hist(x=data, bins=binning, weights=weight,\
         label=f'$m_{{DM}} = ${masses[i]} GeV', histtype="step", rwidth=1.0,\
         color=None, edgecolor=color, linewidth=1, linestyle="solid",\
         bottom=None, cumulative=False, align="mid", orientation="vertical")
#---------------------MONOJET------------------------
#------------------Neutralino jet---------------------
# Neutralino jet ETA plots
fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
pad1   = fig.add_subplot(frame[0])

for i in range(0, runs):
  eta = [float(x) for x in neutralino_jet.iloc[i]["eta"]]

  set_data(eta_xData, eta_xBinning, eta, colors[i], pad1)
  plt.xlabel(r"Monojet Neutralino $\eta$ $[ j_{1} ]$ ", fontsize=16,color="black")
  set_configurations()

# Neutralino jet PT plots
fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
pad2   = fig.add_subplot(frame[0])

for i in range(0, runs):
  pt = [float(x) for x in neutralino_jet.iloc[i]["pt"]]

  set_data(pt_xData, pt_xBinning, pt, colors[i], pad2)
  plt.xlabel(r"Monojet Neutralino $p_{T}$ $[ j_{1} ]$ $(GeV/c)$ ", fontsize=16,color="black")
  set_configurations()
  
# Neutralino jet TET plots
fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
pad3   = fig.add_subplot(frame[0])

for i in range(0, runs):
  tet = [float(x) for x in neutralino_jet.iloc[i]["tet"]]

  set_data(tet_xData, tet_xBinning, tet, colors[i], pad3)
  plt.xlabel(r"Monojet Neutralino $E_{T}$ $(GeV)$ ", fontsize=16,color="black")
  set_configurations()

#--------------------Snutrino jet---------------------
# Snutrino jet ETA plots
fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
pad4   = fig.add_subplot(frame[0])

for i in range(0, runs):
  eta = [float(x) for x in sneutrino_jet.iloc[i]["eta"]]

  set_data(eta_xData, eta_xBinning, eta, colors[i], pad4)
  plt.xlabel(r"Monojet Snutrino $E_{T}$ $(GeV)$ ", fontsize=16,color="black")
  set_configurations()

# Snutrino jet PT plots
fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
pad5   = fig.add_subplot(frame[0])

for i in range(0, runs):
  pt = [float(x) for x in sneutrino_jet.iloc[i]["pt"]]

  set_data(pt_xData, pt_xBinning, pt, colors[i], pad5)
  plt.xlabel(r"Monojet Snutrino $p_{T}$ $[ j_{1} ]$ $(GeV/c)$ ", fontsize=16,color="black")
  set_configurations()
  
# Snutrino jet TET plots
fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
pad6   = fig.add_subplot(frame[0])

for i in range(0, runs):
  tet = [float(x) for x in sneutrino_jet.iloc[i]["tet"]]

  set_data(tet_xData, tet_xBinning, tet, colors[i], pad6)
  plt.xlabel(r"Monojet Snutrino $E_{T}$ $(GeV)$ ", fontsize=16,color="black")
  set_configurations()

plt.show()