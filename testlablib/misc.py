import numpy as np
import os
import matplotlib.pyplot as plt
import testlablib as lab



def fuel2gas(FuelFlow):
    # Testlab Motor
    # Exhaustgas Flow as a function of Fuel Flow
    return (FuelFlow < 122) * (26.5 * FuelFlow + 750) + (FuelFlow >= 122) * (27.5 * FuelFlow + 100)


def get_path():
    path = os.path.dirname(os.path.abspath(__file__))
    path = path.split("PYSCRIPTS", 1)[0]
    path = path + "PYSCRIPTS\\"
    return path


class Validate():

    def __init__(self):
        pass

    def profile_absorber_5_6m(self, figurenumber, index, absorber, df):
        plt.figure(figurenumber)
        plt.figure(figurenumber).patch.set_facecolor('#E0E0E0')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.2)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3)

        graph01 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
        graph02 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1, sharey=graph01)
        graph03 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1, sharey=graph01)

        graph01.scatter([df["TT-100 [C]"].values[index]], [-0.4], c="grey")
        graph01.scatter([df["TT-101 [C]"].values[index]], [-0.2], c="grey")
        graph01.scatter([df["TT-102 [C]"].values[index]], [0], c="grey")
        graph01.scatter([df["TT-112 [C]"].values[index]], [0], c="orange")

        graph01.scatter([df["TT-103 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-104 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-105 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-106 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-107 [C]"].values[index]], [1.7], c="black", marker="x")
        graph01.scatter([df["TT-108 [C]"].values[index]], [2.55], c="black", marker="x")
        graph01.scatter([df["TT-109 [C]"].values[index]], [3.4], c="black", marker="x")
        graph01.scatter([df["TT-110A [C]"].values[index]], [4.25], c="black", marker="x")
        graph01.scatter([df["TT-110B [C]"].values[index]], [4.25], c="black", marker="x")
        graph01.scatter([df["TT-110C [C]"].values[index]], [4.25], c="black", marker="x")
        graph01.scatter([df["TT-110D [C]"].values[index]], [4.25], c="black", marker="x")
        graph01.scatter([df["TT-114 [C]"].values[index]], [5.1], c="black", marker="x")
        graph01.scatter([df["TT-115 [C]"].values[index]], [5.6], c="grey")
        graph01.scatter([df["TT-111 [C]"].values[index]], [5.6], c="orange")

        graph01.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph01.hlines(5.6, 0, 100, colors="black", linestyles='dotted')

        graph01.plot(absorber.GasStream.temp_K[:, index] - 273.15, absorber.position_m, c="gray", label="Exhaust Gas")
        graph01.plot(absorber.LiquidStream.temp_K[:, index] - 273.15, absorber.position_m, c="orange", label="Solvent")

        graph01.grid(True)
        graph01.set_xlabel("Temp [C]")
        graph01.set_xlim([0, 75])
        graph01.set_ylabel("Height [m]")
        #graph01.legend()

        # ------------------------------------------------------------------------------------------------------

        graph02.scatter([df["Marsic CO2 [%]"].values[index]], [0.0], c="grey")
        graph02.scatter([df["ABB CEMCaptain CO2 [%]"].values[index]], [5.6], c="grey")
        graph02.scatter([df["NorskAnalyseCO2 [mol%]"].values[index]], [5.6], c="grey")
        graph02.scatter([df["Marsic CO2 Sample Point 2 [%]"].values[index]], [5.6], c="grey")

        graph02.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph02.hlines(5.6, 0, 100, colors="black", linestyles='dotted')

        graph02.plot(100 * absorber.GasStream.get_specie_molar_fraction(id="CO2")[:, index] / (1 - absorber.GasStream.get_specie_molar_fraction(id="H2O")[:, index]), absorber.position_m, c="gray")

        graph02.grid(True)
        graph02.set_xlabel("CO2 [% dry]")
        graph02.set_xlim([0, 8])

        # ------------------------------------------------------------------------------------------------------

        graph03.scatter([df["CO2 Load Rich"].values[index]], [0.0], c="orange")
        graph03.scatter([df["CO2 Load Lean"].values[index]], [5.6], c="orange")

        graph03.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph03.hlines(5.6, 0, 100, colors="black", linestyles='dotted')

        graph03.plot(absorber.LiquidStream.CO2Load(absorber.LiquidStream)[:, index], absorber.position_m, c="orange")

        graph03.grid(True)
        graph03.set_xlabel("CO2 Load")
        graph03.set_xlim([0, 1])

    def profile_absorber_5_6m_with_bicarb(self, figurenumber, index, absorber, df):
        plt.figure(figurenumber)
        plt.figure(figurenumber).patch.set_facecolor('#E0E0E0')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.2)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3)

        graph01 = plt.subplot2grid((1, 4), (0, 0), rowspan=1, colspan=1)
        graph02 = plt.subplot2grid((1, 4), (0, 1), rowspan=1, colspan=1, sharey=graph01)
        graph03 = plt.subplot2grid((1, 4), (0, 2), rowspan=1, colspan=1, sharey=graph01)
        graph04 = plt.subplot2grid((1, 4), (0, 3), rowspan=1, colspan=1, sharey=graph01)

        graph01.scatter([df["TT-100 [C]"].values[index]], [-0.4], c="grey")
        graph01.scatter([df["TT-101 [C]"].values[index]], [-0.2], c="grey")
        graph01.scatter([df["TT-102 [C]"].values[index]], [0], c="grey")
        graph01.scatter([df["TT-112 [C]"].values[index]], [0], c="orange")

        graph01.scatter([df["TT-103 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-104 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-105 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-106 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-107 [C]"].values[index]], [1.7], c="black", marker="x")
        graph01.scatter([df["TT-108 [C]"].values[index]], [2.55], c="black", marker="x")
        graph01.scatter([df["TT-109 [C]"].values[index]], [3.4], c="black", marker="x")
        graph01.scatter([df["TT-110A [C]"].values[index]], [4.25], c="black", marker="x")
        graph01.scatter([df["TT-110B [C]"].values[index]], [4.25], c="black", marker="x")
        graph01.scatter([df["TT-110C [C]"].values[index]], [4.25], c="black", marker="x")
        graph01.scatter([df["TT-110D [C]"].values[index]], [4.25], c="black", marker="x")
        graph01.scatter([df["TT-114 [C]"].values[index]], [5.1], c="black", marker="x")
        graph01.scatter([df["TT-115 [C]"].values[index]], [5.6], c="grey")
        graph01.scatter([df["TT-111 [C]"].values[index]], [5.6], c="orange")

        graph01.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph01.hlines(5.6, 0, 100, colors="black", linestyles='dotted')

        graph01.plot(absorber.GasStream.temp_K[:, index] - 273.15, absorber.position_m, c="gray", label="Exhaust Gas")
        graph01.plot(absorber.LiquidStream.temp_K[:, index] - 273.15, absorber.position_m, c="orange", label="Solvent")

        graph01.grid(True)
        graph01.set_xlabel("Temp [C]")
        graph01.set_xlim([0, 75])
        graph01.set_ylabel("Height [m]")
        #graph01.legend()

        # ------------------------------------------------------------------------------------------------------

        graph02.scatter([df["Marsic CO2 [%]"].values[index]], [0.0], c="grey")
        graph02.scatter([df["ABB CEMCaptain CO2 [%]"].values[index]], [5.6], c="grey")
        graph02.scatter([df["NorskAnalyseCO2 [mol%]"].values[index]], [5.6], c="grey")
        graph02.scatter([df["Marsic CO2 Sample Point 2 [%]"].values[index]], [5.6], c="grey")

        graph02.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph02.hlines(5.6, 0, 100, colors="black", linestyles='dotted')

        graph02.plot(100 * absorber.GasStream.get_specie_molar_fraction(id="CO2")[:, index] / (1 - absorber.GasStream.get_specie_molar_fraction(id="H2O")[:, index]), absorber.position_m, c="gray")

        graph02.grid(True)
        graph02.set_xlabel("CO2 [% dry]")
        graph02.set_xlim([0, 8.0])

        # ------------------------------------------------------------------------------------------------------

        graph03.scatter([df["CO2 Load Rich"].values[index]], [0.0], c="orange")
        graph03.scatter([df["CO2 Load Lean"].values[index]], [5.6], c="orange")

        graph03.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph03.hlines(5.6, 0, 100, colors="black", linestyles='dotted')

        graph03.plot(absorber.LiquidStream.CO2Load(absorber.LiquidStream)[:, index], absorber.position_m, c="orange")

        graph03.grid(True)
        graph03.set_xlabel("CO2 Load")
        graph03.set_xlim([0.1, 0.65])

        # ------------------------------------------------------------------------------------------------------

        graph04.hlines(0, 0, 6, colors="black", linestyles='dotted')
        graph04.hlines(5.6, 0, 6, colors="black", linestyles='dotted')

        graph04.plot(absorber.LiquidStream.get_specie_molality_mol_kg(id="HCO3-")[:, index], absorber.position_m, c="red", label="HCO3-")
        graph04.plot(absorber.LiquidStream.get_specie_molality_mol_kg(id="MEACOO-")[:, index], absorber.position_m, c="black", label="MEACOO-")
        graph04.plot(absorber.LiquidStream.get_specie_molality_mol_kg(id="MEA")[:, index], absorber.position_m, c="blue", label="MEA", linestyle="dotted")
        graph04.legend()

        graph04.grid(True)
        graph04.set_xlabel("Molality")
        #graph04.set_xlim([-0.1, 6])

    def profile_absorber_3_4m(self, figurenumber, index, absorber, df):

        plt.figure(figurenumber)
        plt.figure(figurenumber).patch.set_facecolor('#E0E0E0')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.2)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3)

        graph01 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
        graph02 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1, sharey=graph01)
        graph03 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1, sharey=graph01)

        graph01.scatter([df["TT-100 [C]"].values[index]], [-0.4], c="grey")
        graph01.scatter([df["TT-101 [C]"].values[index]], [-0.2], c="grey")
        graph01.scatter([df["TT-102 [C]"].values[index]], [0], c="grey")
        graph01.scatter([df["TT-112 [C]"].values[index]], [0], c="orange")

        graph01.scatter([df["TT-103 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-104 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-105 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-106 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-107 [C]"].values[index]], [1.7], c="black", marker="x")
        graph01.scatter([df["TT-108 [C]"].values[index]], [2.55], c="black", marker="x")
        graph01.scatter([df["TT-109 [C]"].values[index]], [3.4], c="grey", marker="o")
        #graph01.scatter([df["TT-110A [C]"].values[index]], [4.25], c="black", marker="x")
        #graph01.scatter([df["TT-110B [C]"].values[index]], [4.25], c="black", marker="x")
        #graph01.scatter([df["TT-110C [C]"].values[index]], [4.25], c="black", marker="x")
        #graph01.scatter([df["TT-110D [C]"].values[index]], [4.25], c="black", marker="x")
        #graph01.scatter([df["TT-114 [C]"].values[index]], [5.1], c="black", marker="x")
        #graph01.scatter([df["TT-115 [C]"].values[index]], [5.6], c="grey")
        graph01.scatter([df["TT-111 [C]"].values[index]], [3.4], c="orange")

        graph01.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph01.hlines(3.4, 0, 100, colors="black", linestyles='dotted')

        graph01.plot(absorber.GasStream.temp_K[:, index] - 273.15, absorber.position_m, c="gray", label="Exhaust Gas")
        graph01.plot(absorber.LiquidStream.temp_K[:, index] - 273.15, absorber.position_m, c="orange", label="Solvent")

        graph01.grid(True)
        graph01.set_xlabel("Temp [C]")
        graph01.set_xlim([0, 75])
        #graph01.legend()

        # ------------------------------------------------------------------------------------------------------

        graph02.scatter([df["Marsic CO2 [%]"].values[index]], [0.0], c="grey")
        graph02.scatter([df["ABB CEMCaptain CO2 [%]"].values[index]], [3.4], c="grey")
        graph02.scatter([df["NorskAnalyseCO2 [mol%]"].values[index]], [3.4], c="grey")
        graph02.scatter([df["Marsic CO2 Sample Point 2 [%]"].values[index]], [3.4], c="grey")

        graph02.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph02.hlines(3.4, 0, 100, colors="black", linestyles='dotted')

        graph02.plot(100 * absorber.GasStream.get_specie_molar_fraction(id="CO2")[:, index] / (1 - absorber.GasStream.get_specie_molar_fraction(id="H2O")[:, index]), absorber.position_m, c="gray")

        graph02.grid(True)
        graph02.set_xlabel("CO2 [% dry]")
        graph02.set_xlim([0, 8])

        # ------------------------------------------------------------------------------------------------------

        graph03.scatter([df["CO2 Load Rich"].values[index]], [0.0], c="orange")
        graph03.scatter([df["CO2 Load Lean"].values[index]], [3.4], c="orange")

        graph03.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph03.hlines(3.4, 0, 100, colors="black", linestyles='dotted')

        graph03.plot(absorber.LiquidStream.CO2Load(absorber.LiquidStream)[:, index], absorber.position_m, c="orange")

        graph03.grid(True)
        graph03.set_xlabel("CO2 Load")
        graph03.set_xlim([0, 1])

    def profile_absorber_5_6m_washwater_1_6m(self, figurenumber, index, absorber, washwatersection, df):
        plt.figure(figurenumber)
        plt.figure(figurenumber).patch.set_facecolor('#E0E0E0')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.2)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3)

        graph01 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
        graph02 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1, sharey=graph01)
        graph03 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1, sharey=graph01)

        graph01.scatter([df["TT-100 [C]"].values[index]], [-0.4], c="grey")
        graph01.scatter([df["TT-101 [C]"].values[index]], [-0.2], c="grey")
        graph01.scatter([df["TT-102 [C]"].values[index]], [0], c="grey")
        graph01.scatter([df["TT-112 [C]"].values[index]], [0], c="orange")

        graph01.scatter([df["TT-103 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-104 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-105 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-106 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-107 [C]"].values[index]], [1.7], c="black", marker="x")
        graph01.scatter([df["TT-108 [C]"].values[index]], [2.55], c="black", marker="x")
        graph01.scatter([df["TT-109 [C]"].values[index]], [3.4], c="black", marker="x")
        graph01.scatter([df["TT-110A [C]"].values[index]], [4.25], c="black", marker="x")
        graph01.scatter([df["TT-110B [C]"].values[index]], [4.25], c="black", marker="x")
        graph01.scatter([df["TT-110C [C]"].values[index]], [4.25], c="black", marker="x")
        graph01.scatter([df["TT-110D [C]"].values[index]], [4.25], c="black", marker="x")
        graph01.scatter([df["TT-114 [C]"].values[index]], [5.1], c="black", marker="x")
        graph01.scatter([df["TT-115 [C]"].values[index]], [5.6], c="grey")
        graph01.scatter([df["TT-111 [C]"].values[index]], [5.6], c="orange")
        graph01.scatter([df["TT-116 [C]"].values[index]], [5.6 + 0.4], c="black", marker="x")
        graph01.scatter([df["TT-117 [C]"].values[index]], [5.6 + 0.8], c="black", marker="x")
        graph01.scatter([df["TT-118 [C]"].values[index]], [5.6 + 1.2], c="black", marker="x")
        graph01.scatter([df["TT-119 [C]"].values[index]], [5.6 + 1.6], c="grey")
        graph01.scatter([df["TT-113 [C]"].values[index]], [5.6 + 1.6], c="blue")
        graph01.scatter([df["Temp 8  [Deg C] "].values[index]], [5.6 + 1.6 + 0.2], c="grey")

        graph01.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph01.hlines(5.6, 0, 100, colors="black", linestyles='dotted')
        graph01.hlines(5.6 + 1.6, 0, 100, colors="black", linestyles='dotted')

        graph01.plot(absorber.GasStream.temp_K[:, index] - 273.15, absorber.position_m, c="gray", label="Exhaust Gas")
        graph01.plot(absorber.LiquidStream.temp_K[:, index] - 273.15, absorber.position_m, c="orange", label="Solvent")

        graph01.plot(washwatersection.GasStream.temp_K[:, index] - 273.15, washwatersection.position_m + 5.6, c="gray")
        graph01.plot(washwatersection.LiquidStream.temp_K[:, index] - 273.15, washwatersection.position_m + 5.6, c="blue", label="Water")

        graph01.grid(True)
        graph01.set_xlabel("Temp [C]")
        graph01.set_xlim([0, 75])
        #graph01.legend()

        # ------------------------------------------------------------------------------------------------------

        graph02.scatter([df["Marsic CO2 [%]"].values[index]], [0.0], c="grey")
        graph02.scatter([df["ABB CEMCaptain CO2 [%]"].values[index]], [5.6], c="grey")
        graph02.scatter([df["NorskAnalyseCO2 [mol%]"].values[index]], [5.6], c="grey")
        graph02.scatter([df["Marsic CO2 Sample Point 2 [%]"].values[index]], [5.6], c="grey")

        graph02.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph02.hlines(5.6, 0, 100, colors="black", linestyles='dotted')
        graph02.hlines(5.6 + 1.6, 0, 100, colors="black", linestyles='dotted')

        graph02.plot(100 * absorber.GasStream.get_specie_molar_fraction(id="CO2")[:, index] / (1 - absorber.GasStream.get_specie_molar_fraction(id="H2O")[:, index]), absorber.position_m, c="gray")

        graph02.grid(True)
        graph02.set_xlabel("CO2 [% dry]")
        graph02.set_xlim([0, 8])

        # ------------------------------------------------------------------------------------------------------

        graph03.scatter([df["CO2 Load Rich"].values[index]], [0.0], c="orange")
        graph03.scatter([df["CO2 Load Lean"].values[index]], [5.6], c="orange")

        graph03.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph03.hlines(5.6, 0, 100, colors="black", linestyles='dotted')
        graph03.hlines(5.6 + 1.6, 0, 100, colors="black", linestyles='dotted')

        graph03.plot(absorber.LiquidStream.CO2Load(absorber.LiquidStream)[:, index], absorber.position_m, c="orange")

        graph03.grid(True)
        graph03.set_xlabel("CO2 Load")
        graph03.set_xlim([0, 1])

    def profile_absorber_3_4m_washwater_0_8m(self, figurenumber, index, absorber, washwatersection, df):
        plt.figure(figurenumber)
        plt.figure(figurenumber).patch.set_facecolor('#E0E0E0')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.2)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3)

        graph01 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
        graph02 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1, sharey=graph01)
        graph03 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1, sharey=graph01)

        graph01.scatter([df["TT-100 [C]"].values[index]], [-0.4], c="grey")
        graph01.scatter([df["TT-101 [C]"].values[index]], [-0.2], c="grey")
        graph01.scatter([df["TT-102 [C]"].values[index]], [0], c="grey")
        graph01.scatter([df["TT-112 [C]"].values[index]], [0], c="orange")
        graph01.scatter([df["TT-103 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-104 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-105 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-106 [C]"].values[index]], [0.85], c="black", marker="x")
        graph01.scatter([df["TT-107 [C]"].values[index]], [1.7], c="black", marker="x")
        graph01.scatter([df["TT-108 [C]"].values[index]], [2.55], c="black", marker="x")
        graph01.scatter([df["TT-109 [C]"].values[index]], [3.4], c="grey", marker="o")
        graph01.scatter([df["TT-111 [C]"].values[index]], [3.4], c="orange")
        graph01.scatter([df["TT-113 [C]"].values[index]], [3.4 + 0.8], c="blue")
        graph01.scatter([df["Temp 8  [Deg C] "].values[index]], [3.4 + 0.8], c="grey")

        graph01.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph01.hlines(3.4, 0, 100, colors="black", linestyles='dotted')
        graph01.hlines(3.4 + 0.8, 0, 100, colors="black", linestyles='dotted')

        graph01.plot(absorber.GasStream.temp_K[:, index] - 273.15, absorber.position_m, c="gray", label="Exhaust Gas")
        graph01.plot(absorber.LiquidStream.temp_K[:, index] - 273.15, absorber.position_m, c="orange", label="Solvent")

        graph01.plot(washwatersection.GasStream.temp_K[:, index] - 273.15, washwatersection.position_m + 3.4, c="gray")
        graph01.plot(washwatersection.LiquidStream.temp_K[:, index] - 273.15, washwatersection.position_m + 3.4, c="blue",
                     label="Water")

        graph01.grid(True)
        graph01.set_xlabel("Temp [C]")
        graph01.set_xlim([0, 75])
        # graph01.legend()

        # ------------------------------------------------------------------------------------------------------

        graph02.scatter([df["Marsic CO2 [%]"].values[index]], [0.0], c="grey")
        graph02.scatter([df["ABB CEMCaptain CO2 [%]"].values[index]], [3.4], c="grey")
        graph02.scatter([df["NorskAnalyseCO2 [mol%]"].values[index]], [3.4], c="grey")
        graph02.scatter([df["Marsic CO2 Sample Point 2 [%]"].values[index]], [3.4], c="grey")

        graph02.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph02.hlines(3.4, 0, 100, colors="black", linestyles='dotted')
        graph02.hlines(3.4 + 0.8, 0, 100, colors="black", linestyles='dotted')
        graph02.plot(100 * absorber.GasStream.get_specie_molar_fraction(id="CO2")[:, index] / (1 - absorber.GasStream.get_specie_molar_fraction(id="H2O")[:, index]), absorber.position_m, c="gray")

        graph02.grid(True)
        graph02.set_xlabel("CO2 [% dry]")
        graph02.set_xlim([0, 8])

        # ------------------------------------------------------------------------------------------------------

        graph03.scatter([df["CO2 Load Rich"].values[index]], [0.0], c="orange")
        graph03.scatter([df["CO2 Load Lean"].values[index]], [3.4], c="orange")

        graph03.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph03.hlines(3.4, 0, 100, colors="black", linestyles='dotted')
        graph03.hlines(3.4 + 0.8, 0, 100, colors="black", linestyles='dotted')

        graph03.plot(absorber.LiquidStream.CO2Load(absorber.LiquidStream)[:, index], absorber.position_m, c="orange")

        graph03.grid(True)
        graph03.set_xlabel("CO2 Load")
        graph03.set_xlim([0, 1])

    def profile_washwater_1_6m(self, figurenumber, index, washwatersection, df):

        plt.figure(figurenumber)
        plt.figure(figurenumber).patch.set_facecolor('#E0E0E0')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.2)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3)

        graph01 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

        graph01.scatter([df["TT-113 [C]"].values[index]], [1.6], c="blue", s=75, label="TT-113")
        graph01.scatter([df["Temp 8  [Deg C] "].values[index]], [1.6], c="grey", label="TT-8", marker="*", s=75)
        graph01.scatter([df["TT-119 [C]"].values[index]], [1.6], c="grey", label="TT-119", marker="d", s=75)
        graph01.scatter([df["TT-118 [C]"].values[index]], [1.2], c="blue", marker="x", label="TT-118", s=75)
        graph01.scatter([df["TT-117 [C]"].values[index]], [0.8], c="blue", marker="x", label="TT-117", s=75)
        graph01.scatter([df["TT-116 [C]"].values[index]], [0.4], c="blue", marker="x", label="TT-116", s=75)
        graph01.scatter([df["TT-115 [C]"].values[index]], [0], c="grey", label="TT-115", s=75)


        graph01.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph01.hlines(1.6, 0, 100, colors="black", linestyles='dotted')


        graph01.plot(washwatersection.GasStream.temp_K[:, index] - 273.15, washwatersection.position_m, c="gray", label="Exhaust Gas")
        graph01.plot(washwatersection.LiquidStream.temp_K[:, index] - 273.15, washwatersection.position_m, c="blue", label="Water")

        graph01.grid(True)
        graph01.set_xlabel("Temp [C]")
        graph01.set_ylabel("Height [m]")
        graph01.set_xlim([0, 75])
        graph01.legend()

    def capture_rate(self, figurenumber, absorber, df):
        CO2_out = 100 * absorber.GasStreamOut.get_specie_molar_fraction(id="CO2") / (1 - absorber.GasStreamOut.get_specie_molar_fraction(id="H2O"))
        CO2_in = 100 * absorber.GasStreamIn.get_specie_molar_fraction(id="CO2") / (1 - absorber.GasStreamIn.get_specie_molar_fraction(id="H2O"))
        CO2_cap = 100 * (1 - CO2_out / CO2_in)

        plt.figure(figurenumber)
        plt.scatter(100 * (1 - df["ABB CEMCaptain CO2 [%]"].values / df["Marsic CO2 [%]"].values), CO2_cap, label="ABB CEMCaptain", marker="x")
        plt.scatter(100 * (1 - df["NorskAnalyseCO2 [mol%]"].values / df["Marsic CO2 [%]"].values), CO2_cap, label="Norsk Analyse", marker="x")
        plt.plot([0, 100], [0, 100])
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.title("CO2 Cap [%]")
        plt.xlabel("Measured")
        plt.ylabel("Estimated")

    def heater_mapping(self, figurenumber, system, df, gas_in, heat_kW, title):

        cap = 100 * (1 - system.gas_out.get_specie_flow_kg_h(id="CO2") / gas_in.get_specie_flow_kg_h(id="CO2"))
        cap2 = 100 * system.condenser_vapor.get_specie_flow_kg_h(id="CO2") / gas_in.get_specie_flow_kg_h(id="CO2")
        MJ_kg_CO2 = (heat_kW / 1000) / (system.condenser_vapor.get_specie_flow_kg_h(id="CO2") / 3600)

        if df is not None:
            df["CO2 Captured [%]_2"] = ((1 - (df["Marsic CO2 Sample Point 2 [%]"] / df["Marsic CO2 [%]"])) * 100)

        plt.figure(figurenumber, figsize=(10, 6))
        plt.subplots_adjust(top=0.93)
        plt.suptitle(title)

        graph00 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1)
        graph10 = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1, sharex=graph00)
        graph20 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1, sharex=graph00)

        graph01 = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1, sharex=graph00)
        graph11 = plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1, sharex=graph00)
        graph21 = plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1, sharex=graph00)

        # -----------------------------------------------------------------------

        if df is not None:
            df1 = df[df["FT-201 Flow [kg/h]"] < 130]
            df2 = df[df["FT-201 Flow [kg/h]"] >= 130]

            graph00.scatter(df["H-400 Power [kW]"].values, df["CO2 Captured [%]_2"].values, label="Marsic", marker="d", c="red")
            graph10.scatter(df1["H-400 Power [kW]"].values, df1["CCS MJ per kg CO2"].values, label="FT-200", marker="x", c="black")
            graph01.scatter(df["H-400 Power [kW]"].values, df["CO2 Load Lean"].values, label=None)
            graph01.scatter(df["H-400 Power [kW]"].values, df["CO2 Load Rich"].values, label=None)
            graph11.scatter(df["H-400 Power [kW]"].values, df["TT-200 [C]"].values, label=None)
            graph11.scatter(df["H-400 Power [kW]"].values, df["TT-205 [C]"].values, label=None)
            graph11.scatter(df["H-400 Power [kW]"].values, df["TT-202 [C]"].values, label=None)
            graph20.scatter(df1["H-400 Power [kW]"].values, df1["FT-200 Flow [kg/h]"].values, label="FT-200", marker="x", c="black")
            graph21.scatter(df1["H-400 Power [kW]"].values, df1["FT-201 Flow [kg/h]"].values, label="FT-201")
            graph21.scatter(df2["H-400 Power [kW]"].values, df2["FT-201 Flow [kg/h]"].values, label=None, c="red")

        # -----------------------------------------------------------------------

        graph00.plot(heat_kW, cap, label=None)
        graph00.plot(heat_kW, cap2, label=None)
        graph00.grid(True)
        graph00.tick_params(labelbottom=False)
        graph00.legend(prop={'size': 9}, loc="upper left")
        graph00.set_ylabel("CO2 Cap [%]")
        graph00.set_ylim([0, 100])

        graph10.plot(heat_kW, MJ_kg_CO2, label=None)
        graph10.grid(True)
        graph10.tick_params(labelbottom=False)
        graph10.legend(prop={'size': 9})
        graph10.set_ylabel("MJ per kg CO2")
        graph10.set_ylim([3, 7])

        graph20.plot(heat_kW, system.condenser_vapor.get_specie_flow_kg_h(id="CO2"), label=None)
        graph20.grid(True)
        graph20.legend(prop={'size': 9})
        graph20.set_ylabel("CO2 Cap [kg/h]")
        graph20.set_xlabel("Heater [kW]")
        graph20.set_ylim([50, 400])
        graph20.set_xlim([75, 500])

        # -----------------------------------------------------------------------

        graph01.plot(heat_kW, system.lean_cold.CO2Load(system.lean_cold), label="Lean")
        graph01.plot(heat_kW, system.rich_cold.CO2Load(system.rich_cold), label="Rich")
        graph01.grid(True)
        graph01.tick_params(labelbottom=False)
        graph01.legend(prop={'size': 9})
        graph01.set_ylabel("CO2 Load")
        graph01.set_ylim([0, 0.55])

        graph11.plot(heat_kW, system.rich_hot.get_solution_temp_K() - 273.15, label="Feed")
        graph11.plot(heat_kW, system.lean_hot.get_solution_temp_K() - 273.15, label="Reboiler")
        graph11.plot(heat_kW, system.lean_cold.get_solution_temp_K() - 273.15, label="Absorber")

        try:
            graph11.plot(heat_kW, system.stripper.GasStreams[int(system.stripper.num_of_stages / 2)].get_gas_temp_K() - 273.15, label="Stripper")
        except:
            pass

        try:
            graph11.plot(heat_kW, system.stripper.LiquidStream.get_solution_temp_K()[50, :] - 273.15, label="Stripper")
        except:
            pass



        graph11.grid(True)
        graph11.tick_params(labelbottom=False)
        graph11.legend(prop={'size': 9})
        graph11.set_ylabel("Temperature")
        graph11.set_ylim([80, 130])

        graph21.plot(heat_kW, system.reflux.get_specie_flow_kg_h(id="H2O"), label=None)
        graph21.grid(True)
        graph21.legend(prop={'size': 9})
        graph21.set_ylabel("Reflux [kg/h]")
        graph21.set_xlabel("Heater [kW]")
        graph21.set_ylim([0, 200])

    def heater_mapping_lvc(self, figurenumber, system, df, gas_in, heat_kW, title):

        cap = 100 * (1 - system.gas_out.get_specie_flow_kg_h(id="CO2") / gas_in.get_specie_flow_kg_h(id="CO2"))
        cap2 = 100 * system.condenser_vapor.get_specie_flow_kg_h(id="CO2") / gas_in.get_specie_flow_kg_h(id="CO2")
        MJ_kg_CO2 = (heat_kW / 1000) / (system.condenser_vapor.get_specie_flow_kg_h(id="CO2") / 3600)

        if df is not None:
            df["CO2 Captured [%]_2"] = ((1 - (df["Marsic CO2 Sample Point 2 [%]"] / df["Marsic CO2 [%]"])) * 100)

        plt.figure(figurenumber, figsize=(10, 6))
        plt.subplots_adjust(top=0.93)
        plt.suptitle(title)

        graph00 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1)
        graph10 = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1, sharex=graph00)
        graph20 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1, sharex=graph00)

        graph01 = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1, sharex=graph00)
        graph11 = plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1, sharex=graph00)
        graph21 = plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1, sharex=graph00)

        # -----------------------------------------------------------------------

        if df is not None:
            df1 = df[df["FT-201 Flow [kg/h]"] < 130]
            df2 = df[df["FT-201 Flow [kg/h]"] >= 130]

            graph00.scatter(df["H-400 Power [kW]"].values, df["CO2 Captured [%]_2"].values, label="Marsic", marker="d", c="red")
            graph10.scatter(df1["H-400 Power [kW]"].values, df1["CCS MJ per kg CO2"].values, label="FT-200", marker="x", c="black")
            graph01.scatter(df["H-400 Power [kW]"].values, df["CO2 Load Lean"].values, label=None)
            graph01.scatter(df["H-400 Power [kW]"].values, df["CO2 Load Rich"].values, label=None)
            graph11.scatter(df["H-400 Power [kW]"].values, df["TT-200 [C]"].values, label=None)
            graph11.scatter(df["H-400 Power [kW]"].values, df["TT-205 [C]"].values, label=None)
            graph11.scatter(df["H-400 Power [kW]"].values, df["TT-202 [C]"].values, label=None)
            graph20.scatter(df1["H-400 Power [kW]"].values, df1["FT-200 Flow [kg/h]"].values, label="FT-200", marker="x", c="black")
            graph21.scatter(df1["H-400 Power [kW]"].values, df1["FT-201 Flow [kg/h]"].values, label="FT-201")
            graph21.scatter(df2["H-400 Power [kW]"].values, df2["FT-201 Flow [kg/h]"].values, label=None, c="red")

        # -----------------------------------------------------------------------

        graph00.plot(heat_kW, cap, label=None)
        graph00.plot(heat_kW, cap2, label=None)
        graph00.grid(True)
        graph00.tick_params(labelbottom=False)
        graph00.legend(prop={'size': 9}, loc="upper left")
        graph00.set_ylabel("CO2 Cap [%]")
        graph00.set_ylim([0, 100])

        graph10.plot(heat_kW, MJ_kg_CO2, label=None)
        graph10.grid(True)
        graph10.tick_params(labelbottom=False)
        graph10.legend(prop={'size': 9})
        graph10.set_ylabel("MJ per kg CO2")
        graph10.set_ylim([3, 7])

        graph20.plot(heat_kW, system.condenser_vapor.get_specie_flow_kg_h(id="CO2"), label=None)
        graph20.grid(True)
        graph20.legend(prop={'size': 9})
        graph20.set_ylabel("CO2 Cap [kg/h]")
        graph20.set_xlabel("Heater [kW]")
        graph20.set_ylim([50, 400])
        graph20.set_xlim([75, 500])

        # -----------------------------------------------------------------------

        graph01.plot(heat_kW, system.lean_cold.CO2Load(system.lean_cold), label="Lean")
        graph01.plot(heat_kW, system.rich_cold.CO2Load(system.rich_cold), label="Rich")
        graph01.grid(True)
        graph01.tick_params(labelbottom=False)
        graph01.legend(prop={'size': 9})
        graph01.set_ylabel("CO2 Load")
        graph01.set_ylim([0, 0.55])

        graph11.plot(heat_kW, system.rich_hot.get_solution_temp_K() - 273.15, label="Feed")
        graph11.plot(heat_kW, system.reboiler_condense.get_solution_temp_K() - 273.15, label="Reboiler")
        graph11.plot(heat_kW, system.lean_hot.get_solution_temp_K() - 273.15, label="Flash Tank")
        graph11.plot(heat_kW, system.stripper.GasStreams[int(system.stripper.num_of_stages / 2)].get_gas_temp_K() - 273.15, label="Stripper")

        graph11.grid(True)
        graph11.tick_params(labelbottom=False)
        graph11.legend(prop={'size': 9})
        graph11.set_ylabel("Temperature")
        graph11.set_ylim([80, 130])

        graph21.plot(heat_kW, system.reflux.get_specie_flow_kg_h(id="H2O"), label=None)
        graph21.grid(True)
        graph21.legend(prop={'size': 9})
        graph21.set_ylabel("Reflux [kg/h]")
        graph21.set_xlabel("Heater [kW]")
        graph21.set_ylim([0, 200])




    def heater_mapping_jarles_washwater(self, figurenumber, system, df, gas_in, heat_kW, title):

        cap = 100 * (1 - system.gas_out.get_specie_flow_kg_h(id="CO2") / gas_in.get_specie_flow_kg_h(id="CO2"))
        cap2 = 100 * system.condenser_vapor.get_specie_flow_kg_h(id="CO2") / gas_in.get_specie_flow_kg_h(id="CO2")
        MJ_kg_CO2 = (heat_kW / 1000) / (system.condenser_vapor.get_specie_flow_kg_h(id="CO2") / 3600)

        df["CO2 Captured [%]_2"] = ((1 - (df["Marsic CO2 Sample Point 2 [%]"] / df["Marsic CO2 [%]"])) * 100)

        plt.figure(figurenumber, figsize=(10, 6))
        plt.subplots_adjust(top=0.93)
        plt.suptitle(title)

        graph00 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1)
        graph10 = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1, sharex=graph00)
        graph20 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1, sharex=graph00)

        graph01 = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1, sharex=graph00)
        graph11 = plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1, sharex=graph00)
        graph21 = plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1, sharex=graph00)

        # -----------------------------------------------------------------------

        df1 = df[df["FT-201 Flow [kg/h]"] < 130]
        df2 = df[df["FT-201 Flow [kg/h]"] >= 130]

        graph00.scatter(df["H-400 Power [kW]"].values, df["CO2 Captured [%]_2"].values, label="Marsic", marker="d", c="red")
        graph10.scatter(df1["H-400 Power [kW]"].values, df1["MJ per kg CO2"].values, label="FT-200", marker="x", c="black")

        graph01.scatter(df["H-400 Power [kW]"].values, df["TT-111 [C]"].values, label=None, c="Orange")

        graph11.scatter(df["H-400 Power [kW]"].values, df["TT-200 [C]"].values, label=None)
        graph11.scatter(df["H-400 Power [kW]"].values, df["TT-205 [C]"].values, label=None)
        graph11.scatter(df["H-400 Power [kW]"].values, df["TT-202 [C]"].values, label=None)
        graph20.scatter(df1["H-400 Power [kW]"].values, df1["FT-200 Flow [kg/h]"].values, label="FT-200", marker="x", c="black")
        graph21.scatter(df["H-400 Power [kW]"].values, df["TT-111 [C]"].values, label="TT-111 [C]")


        # -----------------------------------------------------------------------

        graph00.plot(heat_kW, cap, label=None)
        graph00.plot(heat_kW, cap2, label=None)
        graph00.grid(True)
        graph00.tick_params(labelbottom=False)
        graph00.legend(prop={'size': 9}, loc="upper left")
        graph00.set_ylabel("CO2 Cap [%]")
        graph00.set_ylim([0, 100])

        graph10.plot(heat_kW, MJ_kg_CO2, label=None)
        graph10.grid(True)
        graph10.tick_params(labelbottom=False)
        graph10.legend(prop={'size': 9})
        graph10.set_ylabel("MJ per kg CO2")
        graph10.set_ylim([3, 7])

        graph20.plot(heat_kW, system.condenser_vapor.get_specie_flow_kg_h(id="CO2"), label=None)
        graph20.grid(True)
        graph20.legend(prop={'size': 9})
        graph20.set_ylabel("CO2 Cap [kg/h]")
        graph20.set_xlabel("Heater [kW]")
        graph20.set_ylim([50, 400])
        graph20.set_xlim([75, 500])

        # -----------------------------------------------------------------------

        graph01.plot(heat_kW, system.ww_hot.temp_K - 273.15, label="Wash Water to HX")
        graph01.plot(heat_kW, system.absorber.LiquidStreamOut.temp_K - 273.15, label="Rich Cold from Absorber")
        graph01.plot(heat_kW, system.rich_cold.temp_K - 273.15, label="Rich Cold (Preheated)")
        graph01.grid(True)
        graph01.tick_params(labelbottom=False)
        graph01.legend(prop={'size': 9})
        graph01.set_ylabel("Temperature")
        graph01.set_ylim([0, 80])

        try:
            graph11.plot(heat_kW, system.rich_hot.get_solution_temp_K() - 273.15, label="Feed")
            graph11.plot(heat_kW, system.lean_hot.get_solution_temp_K() - 273.15, label="Reboiler")
            graph11.plot(heat_kW, system.stripper.GasStream.get_gas_temp_K()[int(system.stripper.num_of_heights / 2), :] - 273.15, label="Stripper")
            graph11.grid(True)
            graph11.tick_params(labelbottom=False)
            graph11.legend(prop={'size': 9})
            graph11.set_ylabel("Temperature")
            graph11.set_ylim([80, 130])
        except:
            pass

        graph21.plot(heat_kW, system.lean_cold.get_solution_temp_K() - 273.15, label="Lean Cold to Absorber")
        graph21.grid(True)
        graph21.legend(prop={'size': 9})
        graph21.set_ylabel("Temperature")
        graph21.set_xlabel("Heater [kW]")
        graph21.set_ylim([0, 80])


class Profile():

    def __init__(self):
        pass

    def profile_absorber(self, figurenumber, index, absorber):

        plt.figure(figurenumber)
        plt.figure(figurenumber).patch.set_facecolor('#E0E0E0')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.2)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3)

        graph01 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
        graph02 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1, sharey=graph01)
        graph03 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1, sharey=graph01)

        graph01.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph01.hlines(5.6, 0, 100, colors="black", linestyles='dotted')

        graph01.plot(absorber.GasStream.temp_K[:, index] - 273.15, absorber.position_m, c="gray", label="Exhaust Gas")
        graph01.plot(absorber.LiquidStream.temp_K[:, index] - 273.15, absorber.position_m, c="orange", label="Solvent")

        graph01.grid(True)
        graph01.set_xlabel("Temp [C]")
        graph01.set_xlim([0, 75])

        # ------------------------------------------------------------------------------------------------------
        graph02.plot(100 * absorber.GasStream.get_specie_pressure_bara(id="CO2")[:, index], absorber.position_m, c="gray", label="Gas")
        graph02.plot(100 * absorber.LiquidStream.get_specie_vapor_pressure_bara(gas_id="CO2")[:, index], absorber.position_m, c="orange", linestyle="--", label="Solvent")
        graph02.grid(True)
        graph02.legend()
        graph02.set_xlabel("CO2 Press [kPa]")
        graph02.set_xlim([0, 8])

        # ------------------------------------------------------------------------------------------------------

        graph03.plot(absorber.LiquidStream.CO2Load(absorber.LiquidStream)[:, index], absorber.position_m, c="orange")
        graph03.grid(True)
        graph03.set_xlabel("CO2 Load")
        graph03.set_xlim([0, 1.0])

    def profile_stripper(self, figurenumber, index, stripper):

        plt.figure(figurenumber)
        plt.figure(figurenumber).patch.set_facecolor('#E0E0E0')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.2)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3)

        graph01 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
        graph02 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1, sharey=graph01)
        graph03 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1, sharey=graph01)

        graph01.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph01.hlines(5.6, 0, 100, colors="black", linestyles='dotted')

        graph01.plot(stripper.GasStream.temp_K[:, index] - 273.15, stripper.position_m, c="gray", label="Vapor")
        graph01.plot(stripper.LiquidStream.temp_K[:, index] - 273.15, stripper.position_m, c="orange", label="Solvent")

        graph01.grid(True)
        graph01.set_xlabel("Temp [C]")
        graph01.set_xlim([70, 140])

        # ------------------------------------------------------------------------------------------------------

        #graph02.plot(100 * absorber.GasStream.get_specie_molar_fraction(id="CO2")[:, index] / (1 - absorber.GasStream.get_specie_molar_fraction(id="H2O")[:, index]), absorber.position_m, c="gray")
        #graph02.grid(True)
        #graph02.set_xlabel("CO2 [% dry]")
        #graph02.set_xlim([0, 8])

        # ------------------------------------------------------------------------------------------------------

        graph03.plot(stripper.LiquidStream.CO2Load(stripper.LiquidStream)[:, index], stripper.position_m, c="orange")
        graph03.grid(True)
        graph03.set_xlabel("CO2 Load")
        graph03.set_xlim([0, 1.0])

    def profile_rectifier(self, figurenumber, index, rectifier):

        plt.figure(figurenumber)
        plt.figure(figurenumber).patch.set_facecolor('#E0E0E0')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.2)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.3)

        graph01 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
        graph02 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1, sharey=graph01)
        graph03 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1, sharey=graph01)

        graph01.hlines(0, 0, 100, colors="black", linestyles='dotted')
        graph01.hlines(0.8, 0, 100, colors="black", linestyles='dotted')

        graph01.plot(rectifier.GasStream.temp_K[:, index] - 273.15, rectifier.position_m, c="gray", label="Vapor")
        graph01.plot(rectifier.LiquidStream.temp_K[:, index] - 273.15, rectifier.position_m, c="orange", label="Solvent")

        graph01.grid(True)
        graph01.set_xlabel("Temp [C]")
        graph01.set_xlim([0, 120])

        # ------------------------------------------------------------------------------------------------------

        graph02.plot(3600 * rectifier.get_superficial_liquid_velocity_m_s()[:, index], rectifier.position_m, c="gray")
        graph02.grid(True)
        graph02.set_xlabel("Liquid Load [m/h]")
        #graph02.set_xlim([0, 8])


        # ------------------------------------------------------------------------------------------------------

        graph03.plot(rectifier.get_superficial_gas_velocity_m_s()[:, index], rectifier.position_m, c="gray")
        graph03.grid(True)
        graph03.set_xlabel("Superficial Gas Velocity [m/s]")
        #graph03.set_xlim([0, 1.0])




