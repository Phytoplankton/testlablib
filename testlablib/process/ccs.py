import numpy as np
import testlablib as lab
import time
from copy import deepcopy



# --- Finished ---------------------------------------------------------------------------------

class ExhaustGas(lab.GasStream):

    def __init__(self, flow_Nm3_h_dry, pressure_bara, temp_K, CO2_pct_dry, H2O_pct):
        super().__init__()

        self.load_heat_capacity_kJ_kmolK(function=self.__heat_capacity_kJ_kmolK__)
        self.load_viscosity_Pas(function=self.__viscosity_Pas__)
        self.load_diffusivity_m2_s(function=self.__diffusivity_m2_s__)
        self.load_thermal_conductivity_kW_mK(function=self.__thermal_conductivity_kW_mK__)

        self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
        self.add_specie(id="O2", molar_mass_kg_kmol=32, charge=0)
        self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
        self.add_specie(id="N2", molar_mass_kg_kmol=28, charge=0)

        self.set_gas_temp_K(value=temp_K)
        self.set_gas_pressure_bara(value=pressure_bara)
        shape = np.ones(shape=temp_K.shape)
        for id in self.specie.keys():
            self.set_specie_molar_fraction(id=id, value=0 * shape)
        y_CO2_dry = CO2_pct_dry / 100
        y_O2_dry = 12.3 * shape / 100       # 12.3
        y_N2_dry = 1 - y_CO2_dry - y_O2_dry

        if H2O_pct is None:
            p_H2O = self.__H2O_vapor_pressure_bara__(temp_K=temp_K)
            y_H2O = p_H2O / pressure_bara
        else:
            y_H2O = H2O_pct / 100

        y_CO2_wet = y_CO2_dry * (1 - y_H2O)
        y_O2_wet = y_O2_dry * (1 - y_H2O)
        y_N2_wet = y_N2_dry * (1 - y_H2O)

        self.set_specie_molar_fraction(id="CO2", value=y_CO2_wet)
        self.set_specie_molar_fraction(id="O2", value=y_O2_wet)
        self.set_specie_molar_fraction(id="N2", value=y_N2_wet)
        self.set_specie_molar_fraction(id="H2O", value=y_H2O)

        flow_kmol_h_dry = flow_Nm3_h_dry / (0.08314 * 273.15)
        flow_kmol_h_H2O = flow_kmol_h_dry * y_H2O
        flow_kmol_h_wet = flow_kmol_h_dry + flow_kmol_h_H2O
        self.set_gas_flow_kmol_h(value=flow_kmol_h_wet)

        self.normalize_molar_fractions()

    def __heat_capacity_kJ_kmolK__(self, GasStream, id):

        T = GasStream.get_gas_temp_K()
        if id == "O2":
            A, B, C, D, E = 29.103, 10.040, 2526.5, 9.356, 1153.8
            Cp = A + B * ((C / T) / (np.sinh(C / T))) ** 2 + D * ((E / T) / (np.cosh(E / T))) ** 2
        elif id == "N2":
            A, B, C, D, E = 29.105, 8.6149, 1701.6, 0.10347, 909.79
            Cp = A + B * ((C / T) / (np.sinh(C / T))) ** 2 + D * ((E / T) / (np.cosh(E / T))) ** 2
        elif id == "H2O":
            A, B, C, D, E = 33.363, 26.790, 2610.5, 8.896, 1169
            Cp = A + B * ((C / T) / (np.sinh(C / T))) ** 2 + D * ((E / T) / (np.cosh(E / T))) ** 2
        elif id == "CO2":
            A, B, C, D, E = 29.370, 34.540, 1428, 26.4, 588
            Cp = A + B * ((C / T) / (np.sinh(C / T))) ** 2 + D * ((E / T) / (np.cosh(E / T))) ** 2
        else:
            Cp = 30 * np.ones(shape=T.shape)
        return Cp

    def __H2O_vapor_pressure_bara__(self, temp_K):
        T = np.minimum(temp_K, 273.15 + 150)
        pc = 220.64
        Tc = 647.096
        tau = 1 - T / Tc
        a1 = -7.85951783
        a2 = 1.84408259
        a3 = -11.7866497
        a4 = 22.6807411
        a5 = -15.9618719
        a6 = 1.80122502
        p = pc * np.exp((Tc / T) * (a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 3 + a4 * tau ** 3.5 + a5 * tau ** 4 + a6 * tau ** 7.5))
        return p

    def __viscosity_Pas__(self, GasStream):
        T = GasStream.get_gas_temp_K()
        t = T - 273.15
        mu = (17.36589187 + 4.22323528 * (t / 100) - 0.11854257 * (t / 100) ** 2) * 10 ** (-6)
        return mu

    def __thermal_conductivity_kW_mK__(self, GasStream):
        T = GasStream.get_gas_temp_K()
        t = T - 273.15
        k = (23.84429941 + 7.19127866 * (t / 100) - 0.14985262 * (t / 100) ** 2) * 10 ** (-6)
        return k

    def __diffusivity_m2_s__(self, GasStream, id):
        T = GasStream.get_gas_temp_K()
        t = T - 273.15
        D_H2O = (20.18437495 + 19.74335108 * (t / 100) + 0.90921174 * (t / 100) ** 2) * 10 ** (-6)
        M = GasStream.get_specie_molar_mass_kg_kmol(id)
        D = D_H2O * np.sqrt(18 / M)
        return D


class ReboilerVapor(lab.GasStream):

    def __init__(self, flow_kmol_h, pressure_bara, temp_K, CO2_molar_fraction, H2O_molar_fraction):
        super().__init__()
        self.load_heat_capacity_kJ_kmolK(function=self.__heat_capacity_kJ_kmolK__)
        self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
        self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
        self.set_gas_flow_kmol_h(value=flow_kmol_h)
        self.set_gas_temp_K(value=temp_K)
        self.set_gas_pressure_bara(value=pressure_bara)
        self.set_specie_molar_fraction(id="CO2", value=CO2_molar_fraction)
        self.set_specie_molar_fraction(id="H2O", value=H2O_molar_fraction)

    def __heat_capacity_kJ_kmolK__(self, GasStream, id):

        T = GasStream.get_gas_temp_K()
        if id == "O2":
            A, B, C, D, E = 29.103, 10.040, 2526.5, 9.356, 1153.8
            Cp = A + B * ((C / T) / (np.sinh(C / T))) ** 2 + D * ((E / T) / (np.cosh(E / T))) ** 2
        elif id == "N2":
            A, B, C, D, E = 29.105, 8.6149, 1701.6, 0.10347, 909.79
            Cp = A + B * ((C / T) / (np.sinh(C / T))) ** 2 + D * ((E / T) / (np.cosh(E / T))) ** 2
        elif id == "H2O":
            A, B, C, D, E = 33.363, 26.790, 2610.5, 8.896, 1169
            Cp = A + B * ((C / T) / (np.sinh(C / T))) ** 2 + D * ((E / T) / (np.cosh(E / T))) ** 2
        elif id == "CO2":
            A, B, C, D, E = 29.370, 34.540, 1428, 26.4, 588
            Cp = A + B * ((C / T) / (np.sinh(C / T))) ** 2 + D * ((E / T) / (np.cosh(E / T))) ** 2
        else:
            Cp = 30 * np.ones(shape=T.shape)
        return Cp


class LiquidStream_KOH_8N(lab.LiquidStream):

    def __init__(self, temp_K, flow_kg_h):
        super().__init__(stream_id="", solvent_id="H2O")
        self.load_density_kg_m3(function=self.__density_kg_m3__)
        shape = np.ones(shape=temp_K.shape)
        self.set_solution_temp_K(value=temp_K)
        self.set_solution_flow_kg_h(value=flow_kg_h)
        self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
        self.add_specie(id="K+", molar_mass_kg_kmol=39, charge=1)
        self.add_specie(id="OH-", molar_mass_kg_kmol=17, charge=-1)
        w_OH = shape * 17 * 8 / 1326
        w_K = shape * 39 * 8 / 1326
        w_H2O = 1 - w_OH - w_K
        self.set_specie_mass_fraction(id="H2O", value=w_H2O)
        self.set_specie_mass_fraction(id="OH-", value=w_OH)
        self.set_specie_mass_fraction(id="K+", value=w_K)

    def __density_kg_m3__(self, LiquidStream):
        return 1326 * np.ones(shape=LiquidStream.temp_K.shape)



# --- Potassium Carbonate ----------------------------------------------------------------------

class LiquidStream_K2CO3_20(lab.LiquidStream):

    def __init__(self, temp_K, flow_kg_h, CO2Load):

        super().__init__(stream_id="", solvent_id="H2O")
        z = np.zeros(shape=temp_K.shape)
        shape = np.ones(shape=temp_K.shape)

        self.load_heat_capacity_kJ_kgK(function=self.__heat_capacity_kJ_kgK__)
        self.load_density_kg_m3(function=self.__density_kg_m3__)
        self.load_activity_coefficient(function=self.__activity_coefficient__)

        self.add_function(key="CO2 Load", function=self.CO2Load)

        self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
        self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
        self.add_specie(id="HCO3-", molar_mass_kg_kmol=61, charge=-1)
        self.add_specie(id="CO3-2", molar_mass_kg_kmol=60, charge=-2)
        self.add_specie(id="H+", molar_mass_kg_kmol=1, charge=1)
        self.add_specie(id="OH-", molar_mass_kg_kmol=17, charge=-1)
        self.add_specie(id="K+", molar_mass_kg_kmol=39, charge=1)

        self.add_vapor_pressure_bara_henry(id="CO2(g) = CO2(aq)",
                                           gas_id="CO2",
                                           liq_id="CO2",
                                           liq_unit="m",
                                           henrys_coefficient=self.__CO2_henrys_constant__)

        self.add_vapor_pressure_bara_raoult(id="H2O(g) = H2O(l)",
                                            gas_id="H2O",
                                            liq_id="H2O",
                                            pure_vapor_pressure_bara=self.__H2O_vapor_pressure_bara__)

        self.add_rxn_insta(id="H2O = H+ + OH-",
                           stoch={"H2O": -1, "H+": 1, "OH-": 1},
                           unit={"H2O": "x", "H+": "m", "OH-": "m"},
                           equilibrium_constant=self.__water_autoprotolysis_eq_constant__)

        self.add_rxn_insta(id="CO2 + H2O = HCO3- + H+",
                           stoch={"H2O": -1, "CO2": -1, "H+": 1, "HCO3-": 1},
                           unit={"H2O": "x", "CO2": "m", "H+": "m", "HCO3-": "m"},
                           equilibrium_constant=self.__carbonic_acid_rxn_1_eq_const__)

        self.add_rxn_insta(id="HCO3- = CO3-2 + H+",
                           stoch={"HCO3-": -1, "CO3-2": 1, "H+": 1},
                           unit={"HCO3-": "m", "CO3-2": "m", "H+": "m"},
                           equilibrium_constant=self.__carbonic_acid_rxn_2_eq_const__)

        self.set_solution_temp_K(value=temp_K)
        self.set_solution_flow_kg_h(value=flow_kg_h)

        # Mass Fractions
        K2CO3_mass_fraction = 0.2 *shape
        w_K2CO3 = K2CO3_mass_fraction
        w_H2O = 1.0 - K2CO3_mass_fraction
        w_CO2 = K2CO3_mass_fraction * (44 / 138) * CO2Load

        # Normalizing
        w = w_K2CO3 + w_H2O + w_CO2
        w_K2CO3 = w_K2CO3 / w
        w_H2O = w_H2O / w
        w_CO2 = w_CO2 / w

        self.set_specie_mass_fraction(id="H2O", value=w_H2O)
        self.set_specie_mass_fraction(id="CO2", value=w_CO2)
        self.set_specie_mass_fraction(id="HCO3-", value=z)
        self.set_specie_mass_fraction(id="CO3-2", value=K2CO3_mass_fraction * 60 / 138)
        self.set_specie_mass_fraction(id="H+", value=z)
        self.set_specie_mass_fraction(id="OH-", value=z)
        self.set_specie_mass_fraction(id="K+", value=K2CO3_mass_fraction * 2 * 39 / 138)

    def CO2Load(self, LiquidStream):
        x_CO2 = 0
        x_Amine = 0
        C = {"CO2": 1, "HCO3-": 1, "CO3-2": 1, "K+":-0.5}
        A = {"K+": 0.5}
        for c in C.keys():
            x_CO2 = x_CO2 + C[c] * LiquidStream.get_specie_molar_fraction(id=c)
        for a in A.keys():
            x_Amine = x_Amine + A[a] * LiquidStream.get_specie_molar_fraction(id=a)
        alpha = np.maximum(x_CO2 / np.maximum(x_Amine, 10 ** (-9)), 10**(-9))
        return alpha

    def __heat_capacity_kJ_kgK__(self, LiquidStream):
        T = LiquidStream.temp_K
        cp = (4.2 + 0.41 * 0.2) / (1.0 + 1.05 * 0.2) + (0.3/0.5) * (T - 298)
        return cp

    def __density_kg_m3__(self, LiquidStream):
        return 1050

    def __activity_coefficient__(self, LiquidStream, id):
        I = LiquidStream.get_solution_ionic_strength_mol_kg()
        z = LiquidStream.get_specie_charge(id)
        log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))
        gamma = 10 ** log10_gamma
        return gamma

    def __CO2_henrys_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        H_CO2 = 1.153 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)
        return H_CO2

    def __H2O_vapor_pressure_bara__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        T = np.minimum(T, 273.15 + 150)
        pc = 220.64
        Tc = 647.096
        tau = 1 - T / Tc
        a1 = -7.85951783
        a2 = 1.84408259
        a3 = -11.7866497
        a4 = 22.6807411
        a5 = -15.9618719
        a6 = 1.80122502
        p = pc * np.exp((Tc / T) * (a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 3 + a4 * tau ** 3.5 + a5 * tau ** 4 + a6 * tau ** 7.5))
        return p

    def __water_autoprotolysis_eq_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        Kw = 10 ** (-14) * np.exp(-13445.9 * (1 / T - 1 / 298) - 22.48 * np.log(T / 298))
        return Kw

    def __carbonic_acid_rxn_1_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-6.32) * np.exp(5139 * (1 / T - 1 / 298) + 14.5258479 * np.log(T / 298))
        return K

    def __carbonic_acid_rxn_2_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-10.33) * np.exp(22062 * (1 / T - 1 / 298) + 67.264072 * np.log(T / 298))
        return K


class LiquidStream_K2CO3_30(lab.LiquidStream):

    def __init__(self, temp_K, flow_kg_h, CO2Load):

        super().__init__(stream_id="", solvent_id="H2O")
        z = np.zeros(shape=temp_K.shape)
        shape = np.ones(shape=temp_K.shape)

        self.load_heat_capacity_kJ_kgK(function=self.__heat_capacity_kJ_kgK__)
        self.load_density_kg_m3(function=self.__density_kg_m3__)
        self.load_activity_coefficient(function=self.__activity_coefficient__)

        self.add_function(key="CO2 Load", function=self.CO2Load)

        self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
        self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
        self.add_specie(id="HCO3-", molar_mass_kg_kmol=61, charge=-1)
        self.add_specie(id="CO3-2", molar_mass_kg_kmol=60, charge=-2)
        self.add_specie(id="H+", molar_mass_kg_kmol=1, charge=1)
        self.add_specie(id="OH-", molar_mass_kg_kmol=17, charge=-1)
        self.add_specie(id="K+", molar_mass_kg_kmol=39, charge=1)

        self.add_vapor_pressure_bara_henry(id="CO2(g) = CO2(aq)",
                                           gas_id="CO2",
                                           liq_id="CO2",
                                           liq_unit="m",
                                           henrys_coefficient=self.__CO2_henrys_constant__)

        self.add_vapor_pressure_bara_raoult(id="H2O(g) = H2O(l)",
                                            gas_id="H2O",
                                            liq_id="H2O",
                                            pure_vapor_pressure_bara=self.__H2O_vapor_pressure_bara__)

        self.add_rxn_insta(id="H2O = H+ + OH-",
                           stoch={"H2O": -1, "H+": 1, "OH-": 1},
                           unit={"H2O": "x", "H+": "m", "OH-": "m"},
                           equilibrium_constant=self.__water_autoprotolysis_eq_constant__)

        self.add_rxn_insta(id="CO2 + H2O = HCO3- + H+",
                           stoch={"H2O": -1, "CO2": -1, "H+": 1, "HCO3-": 1},
                           unit={"H2O": "x", "CO2": "m", "H+": "m", "HCO3-": "m"},
                           equilibrium_constant=self.__carbonic_acid_rxn_1_eq_const__)

        self.add_rxn_insta(id="HCO3- = CO3-2 + H+",
                           stoch={"HCO3-": -1, "CO3-2": 1, "H+": 1},
                           unit={"HCO3-": "m", "CO3-2": "m", "H+": "m"},
                           equilibrium_constant=self.__carbonic_acid_rxn_2_eq_const__)

        self.set_solution_temp_K(value=temp_K)
        self.set_solution_flow_kg_h(value=flow_kg_h)

        # Mass Fractions
        K2CO3_mass_fraction = 0.3 *shape
        w_K2CO3 = K2CO3_mass_fraction
        w_H2O = 1.0 - K2CO3_mass_fraction
        w_CO2 = K2CO3_mass_fraction * (44 / 138) * CO2Load

        # Normalizing
        w = w_K2CO3 + w_H2O + w_CO2
        w_K2CO3 = w_K2CO3 / w
        w_H2O = w_H2O / w
        w_CO2 = w_CO2 / w

        self.set_specie_mass_fraction(id="H2O", value=w_H2O)
        self.set_specie_mass_fraction(id="CO2", value=w_CO2)
        self.set_specie_mass_fraction(id="HCO3-", value=z)
        self.set_specie_mass_fraction(id="CO3-2", value=K2CO3_mass_fraction * 60 / 138)
        self.set_specie_mass_fraction(id="H+", value=z)
        self.set_specie_mass_fraction(id="OH-", value=z)
        self.set_specie_mass_fraction(id="K+", value=K2CO3_mass_fraction * 2 * 39 / 138)

    def CO2Load(self, LiquidStream):
        x_CO2 = 0
        x_Amine = 0
        C = {"CO2": 1, "HCO3-": 1, "CO3-2": 1, "K+":-0.5}
        A = {"K+": 0.5}
        for c in C.keys():
            x_CO2 = x_CO2 + C[c] * LiquidStream.get_specie_molar_fraction(id=c)
        for a in A.keys():
            x_Amine = x_Amine + A[a] * LiquidStream.get_specie_molar_fraction(id=a)
        alpha = np.maximum(x_CO2 / np.maximum(x_Amine, 10 ** (-9)), 10**(-9))
        return alpha

    def __heat_capacity_kJ_kgK__(self, LiquidStream):
        T = LiquidStream.temp_K
        cp = (4.2 + 0.41 * 0.3) / (1.0 + 1.05 * 0.3) + (0.3/0.5) * (T - 298)
        return cp

    def __density_kg_m3__(self, LiquidStream):
        return 1050

    def __activity_coefficient__(self, LiquidStream, id):
        I = LiquidStream.get_solution_ionic_strength_mol_kg()
        z = LiquidStream.get_specie_charge(id)
        log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))
        gamma = 10 ** log10_gamma
        return gamma

    def __CO2_henrys_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        H_CO2 = 1.153 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)
        return H_CO2

    def __H2O_vapor_pressure_bara__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        T = np.minimum(T, 273.15 + 150)
        pc = 220.64
        Tc = 647.096
        tau = 1 - T / Tc
        a1 = -7.85951783
        a2 = 1.84408259
        a3 = -11.7866497
        a4 = 22.6807411
        a5 = -15.9618719
        a6 = 1.80122502
        p = pc * np.exp((Tc / T) * (a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 3 + a4 * tau ** 3.5 + a5 * tau ** 4 + a6 * tau ** 7.5))
        return p

    def __water_autoprotolysis_eq_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        Kw = 10 ** (-14) * np.exp(-2578 * (1 / T - 1 / 298) - 33.02 * np.log(T / 298))
        return Kw

    def __carbonic_acid_rxn_1_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-6.32) * np.exp(3504 * (1 / T - 1 / 298) + 18.33 * np.log(T / 298))
        return K

    def __carbonic_acid_rxn_2_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-10.33) * np.exp(23307 * (1 / T - 1 / 298) + 78.84 * np.log(T / 298))
        return K


class LiquidStream_K2CO3_40(lab.LiquidStream):

    def __init__(self, temp_K, flow_kg_h, CO2Load):

        super().__init__(stream_id="", solvent_id="H2O")
        z = np.zeros(shape=temp_K.shape)
        shape = np.ones(shape=temp_K.shape)

        self.load_heat_capacity_kJ_kgK(function=self.__heat_capacity_kJ_kgK__)
        self.load_density_kg_m3(function=self.__density_kg_m3__)
        self.load_activity_coefficient(function=self.__activity_coefficient__)

        self.add_function(key="CO2 Load", function=self.CO2Load)

        self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
        self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
        self.add_specie(id="HCO3-", molar_mass_kg_kmol=61, charge=-1)
        self.add_specie(id="CO3-2", molar_mass_kg_kmol=60, charge=-2)
        self.add_specie(id="H+", molar_mass_kg_kmol=1, charge=1)
        self.add_specie(id="OH-", molar_mass_kg_kmol=17, charge=-1)
        self.add_specie(id="K+", molar_mass_kg_kmol=39, charge=1)

        self.add_vapor_pressure_bara_henry(id="CO2(g) = CO2(aq)",
                                           gas_id="CO2",
                                           liq_id="CO2",
                                           liq_unit="m",
                                           henrys_coefficient=self.__CO2_henrys_constant__)

        self.add_vapor_pressure_bara_raoult(id="H2O(g) = H2O(l)",
                                            gas_id="H2O",
                                            liq_id="H2O",
                                            pure_vapor_pressure_bara=self.__H2O_vapor_pressure_bara__)

        self.add_rxn_insta(id="H2O = H+ + OH-",
                           stoch={"H2O": -1, "H+": 1, "OH-": 1},
                           unit={"H2O": "x", "H+": "m", "OH-": "m"},
                           equilibrium_constant=self.__water_autoprotolysis_eq_constant__)

        self.add_rxn_insta(id="CO2 + H2O = HCO3- + H+",
                           stoch={"H2O": -1, "CO2": -1, "H+": 1, "HCO3-": 1},
                           unit={"H2O": "x", "CO2": "m", "H+": "m", "HCO3-": "m"},
                           equilibrium_constant=self.__carbonic_acid_rxn_1_eq_const__)

        self.add_rxn_insta(id="HCO3- = CO3-2 + H+",
                           stoch={"HCO3-": -1, "CO3-2": 1, "H+": 1},
                           unit={"HCO3-": "m", "CO3-2": "m", "H+": "m"},
                           equilibrium_constant=self.__carbonic_acid_rxn_2_eq_const__)

        self.set_solution_temp_K(value=temp_K)
        self.set_solution_flow_kg_h(value=flow_kg_h)

        # Mass Fractions
        K2CO3_mass_fraction = 0.4 *shape
        w_K2CO3 = K2CO3_mass_fraction
        w_H2O = 1.0 - K2CO3_mass_fraction
        w_CO2 = K2CO3_mass_fraction * (44 / 138) * CO2Load

        # Normalizing
        w = w_K2CO3 + w_H2O + w_CO2
        w_K2CO3 = w_K2CO3 / w
        w_H2O = w_H2O / w
        w_CO2 = w_CO2 / w

        self.set_specie_mass_fraction(id="H2O", value=w_H2O)
        self.set_specie_mass_fraction(id="CO2", value=w_CO2)
        self.set_specie_mass_fraction(id="HCO3-", value=z)
        self.set_specie_mass_fraction(id="CO3-2", value=K2CO3_mass_fraction * 60 / 138)
        self.set_specie_mass_fraction(id="H+", value=z)
        self.set_specie_mass_fraction(id="OH-", value=z)
        self.set_specie_mass_fraction(id="K+", value=K2CO3_mass_fraction * 2 * 39 / 138)

    def CO2Load(self, LiquidStream):
        x_CO2 = 0
        x_Amine = 0
        C = {"CO2": 1, "HCO3-": 1, "CO3-2": 1, "K+":-0.5}
        A = {"K+": 0.5}
        for c in C.keys():
            x_CO2 = x_CO2 + C[c] * LiquidStream.get_specie_molar_fraction(id=c)
        for a in A.keys():
            x_Amine = x_Amine + A[a] * LiquidStream.get_specie_molar_fraction(id=a)
        alpha = np.maximum(x_CO2 / np.maximum(x_Amine, 10 ** (-9)), 10**(-9))
        return alpha

    def __heat_capacity_kJ_kgK__(self, LiquidStream):
        T = LiquidStream.temp_K
        cp = (4.2 + 0.41 * 0.4) / (1.0 + 1.05 * 0.4) + (0.3/0.5) * (T - 298)
        return cp

    def __density_kg_m3__(self, LiquidStream):
        return 1050

    def __activity_coefficient__(self, LiquidStream, id):
        I = LiquidStream.get_solution_ionic_strength_mol_kg()
        z = LiquidStream.get_specie_charge(id)
        log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))
        gamma = 10 ** log10_gamma
        return gamma

    def __CO2_henrys_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        H_CO2 = 1.153 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)
        return H_CO2

    def __H2O_vapor_pressure_bara__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        T = np.minimum(T, 273.15 + 150)
        pc = 220.64
        Tc = 647.096
        tau = 1 - T / Tc
        a1 = -7.85951783
        a2 = 1.84408259
        a3 = -11.7866497
        a4 = 22.6807411
        a5 = -15.9618719
        a6 = 1.80122502
        p = pc * np.exp((Tc / T) * (a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 3 + a4 * tau ** 3.5 + a5 * tau ** 4 + a6 * tau ** 7.5))
        return p

    def __water_autoprotolysis_eq_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        Kw = 10 ** (-14) * np.exp(-4280 * (1 / T - 1 / 298) - 31.08 * np.log(T / 298))
        return Kw

    def __carbonic_acid_rxn_1_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-6.32) * np.exp(2939 * (1 / T - 1 / 298) + 15.68 * np.log(T / 298))
        return K

    def __carbonic_acid_rxn_2_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-10.33) * np.exp(38279 * (1 / T - 1 / 298) + 121 * np.log(T / 298))
        return K


# --- Amino Acids -------------------------------------------------------------------------------


class LiquidStream_MegaMix(lab.LiquidStream):

    def __init__(self, stream_id):
        super().__init__(stream_id=stream_id, solvent_id="H2O")

    def init_with_mass_frac(self, temp_K, flow_kg_h, CO2Load, NH3_mass_fraction=None, MEA_mass_fraction=None, MDEA_mass_fraction=None, PZ_mass_fraction=None, _1MPZ_mass_fraction=None, DMPZ_mass_fraction=None, EDA_mass_fraction=None, DETA_mass_fraction=None, AMP_mass_fraction=None, KLys_mass_fraction=None, KSar_mass_fraction=None, KPro_mass_fraction=None, Arg_mass_fraction=None, carboxylGroup=False):

        z = np.zeros(shape=temp_K.shape)

        self.set_solution_temp_K(value=temp_K)
        self.set_solution_flow_kg_h(value=flow_kg_h)

        w_NH3 = z if NH3_mass_fraction is None else NH3_mass_fraction
        w_MEA = z if MEA_mass_fraction is None else MEA_mass_fraction
        w_MDEA = z if MDEA_mass_fraction is None else MDEA_mass_fraction
        w_PZ = z if PZ_mass_fraction is None else PZ_mass_fraction
        w_1MPZ = z if _1MPZ_mass_fraction is None else _1MPZ_mass_fraction
        w_DMPZ = z if DMPZ_mass_fraction is None else DMPZ_mass_fraction
        w_EDA = z if EDA_mass_fraction is None else EDA_mass_fraction
        w_DETA = z if DETA_mass_fraction is None else DETA_mass_fraction
        w_AMP = z if AMP_mass_fraction is None else AMP_mass_fraction

        w_KLys = KLys_mass_fraction if KLys_mass_fraction is not None else z
        w_KSar = KSar_mass_fraction if KSar_mass_fraction is not None else z
        w_KPro = KPro_mass_fraction if KPro_mass_fraction is not None else z
        w_Arg = Arg_mass_fraction if Arg_mass_fraction is not None else z

        self.add_info(key="NH3 Mass Fraction", value=w_NH3)
        self.add_info(key="MEA Mass Fraction", value=w_MEA)
        self.add_info(key="MDEA Mass Fraction", value=w_MDEA)
        self.add_info(key="PZ Mass Fraction", value=w_PZ)
        self.add_info(key="1-MPZ Mass Fraction", value=w_1MPZ)
        self.add_info(key="DMPZ Mass Fraction", value=w_DMPZ)
        self.add_info(key="EDA Mass Fraction", value=w_EDA)
        self.add_info(key="DETA Mass Fraction", value=w_DETA)
        self.add_info(key="AMP Mass Fraction", value=w_AMP)

        self.add_info(key="KLys Mass Fraction", value=w_KLys)
        self.add_info(key="KSar Mass Fraction", value=w_KSar)
        self.add_info(key="KPro Mass Fraction", value=w_KPro)
        self.add_info(key="Arg Mass Fraction", value=w_Arg)

        self.add_info(key="Amino Acid Mass Fraction", value=w_KLys + w_KSar + w_KPro + w_Arg)
        self.add_info(key="Amine Mass Fraction", value=w_NH3 + w_MEA + w_MDEA + w_AMP + w_PZ + w_1MPZ + w_DMPZ + w_EDA + w_DETA)

        self.load_heat_capacity_kJ_kgK(function=self.__heat_capacity_kJ_kgK__)
        self.load_density_kg_m3(function=self.__density_kg_m3__)
        self.load_activity_coefficient(function=self.__activity_coefficient__)

        self.add_function(key="CO2 Load", function=self.CO2Load)

        self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
        self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
        self.add_specie(id="HCO3-", molar_mass_kg_kmol=61, charge=-1)
        self.add_specie(id="CO3-2", molar_mass_kg_kmol=60, charge=-2)
        self.add_specie(id="H+", molar_mass_kg_kmol=1, charge=1)
        self.add_specie(id="OH-", molar_mass_kg_kmol=17, charge=-1)
        self.add_specie(id="K+", molar_mass_kg_kmol=39, charge=1)

        if KLys_mass_fraction is not None:

            if carboxylGroup:
                self.add_specie(id="Lys+2", molar_mass_kg_kmol=148, charge=2)
                self.add_rxn_insta(id="Lys+2 = Lys+ + H+",
                                   stoch={"Lys+2": -1, "Lys+": 1, "H+": 1},
                                   unit={"Lys+2": "m", "Lys+": "m", "H+": "m"},
                                   equilibrium_constant=self.__lysine_dissociation_constant_K1__)

            self.add_specie(id="Lys+", molar_mass_kg_kmol=147, charge=1)
            self.add_specie(id="Lys", molar_mass_kg_kmol=146, charge=0)
            self.add_specie(id="Lys-", molar_mass_kg_kmol=145, charge=-1)
            self.add_specie(id="LysCOO-", molar_mass_kg_kmol=189, charge=-1)
            self.add_specie(id="LysCOO-2", molar_mass_kg_kmol=188, charge=-2)

            self.add_rxn_insta(id="Lys+ = Lys + H+",
                               stoch={"Lys+": -1, "Lys": 1, "H+": 1},
                               unit={"Lys+": "m", "Lys": "m", "H+": "m"},
                               equilibrium_constant=self.__lysine_dissociation_constant_K2__)

            self.add_rxn_insta(id="Lys = Lys- + H+",
                               stoch={"Lys": -1, "Lys-": 1, "H+": 1},
                               unit={"Lys": "m", "Lys-": "m", "H+": "m"},
                               equilibrium_constant=self.__lysine_dissociation_constant_K3__)

            self.add_rxn_insta(id="CO2 + Lys- = LysCOO-",
                               stoch={"CO2": -1, "Lys-": -1, "LysCOO-": 1},
                               unit={"CO2": "m", "Lys-": "m", "LysCOO-": "m"},
                               equilibrium_constant=self.__lysine_stability_carbamate__)

        if KSar_mass_fraction is not None:
            self.add_specie(id="Sar", molar_mass_kg_kmol=89, charge=0)
            self.add_specie(id="Sar-", molar_mass_kg_kmol=88, charge=-1)
            self.add_specie(id="SarCOO-", molar_mass_kg_kmol=88 + 44, charge=-1)
            self.add_specie(id="SarCOO-2", molar_mass_kg_kmol=88 + 44 - 1, charge=-2)

            self.add_rxn_insta(id="Sar = Sar- + H+",
                               stoch={"Sar": -1, "Sar-": 1, "H+": 1},
                               unit={"Sar": "m", "Sar-": "m", "H+": "m"},
                               equilibrium_constant=self.__sarcosine_dissociation_constant_K2__)

            self.add_rxn_insta(id="CO2 + Sar- = H+ + SarCOO-2",
                               stoch={"CO2": -1, "Sar-": -1, "H+": 1, "SarCOO-2": 1},
                               unit={"CO2": "m", "Sar-": "m", "H+": "m", "SarCOO-2": "m"},
                               equilibrium_constant=self.__sarcosine_stability_carbamate__)

            #self.add_rxn_insta(id="SarCOO- = SarCOO-2 + H+",
            #                   stoch={"SarCOO-": -1, "SarCOO-2": 1, "H+": 1},
            #                   unit={"SarCOO-": "m", "SarCOO-2": "m", "H+": "m"},
            #                   equilibrium_constant=self.__sarcosine_carbamate_dissociation_constant__)

        if KPro_mass_fraction is not None:
            self.add_specie(id="Pro", molar_mass_kg_kmol=115, charge=0)
            self.add_specie(id="Pro-", molar_mass_kg_kmol=114, charge=-1)
            self.add_specie(id="ProCOO-", molar_mass_kg_kmol=158, charge=-1)
            self.add_specie(id="ProCOO-2", molar_mass_kg_kmol=157, charge=-2)

            self.add_rxn_insta(id="Pro = Pro- + H+",
                               stoch={"Pro": -1, "Pro-": 1, "H+": 1},
                               unit={"Pro": "m", "Pro-": "m", "H+": "m"},
                               equilibrium_constant=self.__proline_dissociation_constant_K2__)

            self.add_rxn_insta(id="CO2 + Pro- = H+ + ProCOO-2",
                               stoch={"CO2": -1, "Pro-": -1, "H+": 1, "ProCOO-2": 1},
                               unit={"CO2": "m", "Pro-": "m", "H+": "m", "ProCOO-2": "m"},
                               equilibrium_constant=self.__proline_carbamate_stability__)

        if Arg_mass_fraction is not None:
            self.add_specie(id="Arg+2", molar_mass_kg_kmol=176, charge=2)
            self.add_specie(id="Arg+", molar_mass_kg_kmol=175, charge=1)
            self.add_specie(id="Arg", molar_mass_kg_kmol=174, charge=0)
            self.add_specie(id="ArgCOO-", molar_mass_kg_kmol=174 + 44 - 1, charge=-1)
            self.add_specie(id="ArgCOO", molar_mass_kg_kmol=174 + 44, charge=0)

            self.add_rxn_insta(id="Arg+ = Arg + H+",
                               stoch={"Arg+": -1, "Arg": 1, "H+": 1},
                               unit={"Arg+": "m", "Arg": "m", "H+": "m"},
                               equilibrium_constant=self.__arginine_dissociation_constant_K2__)

        if NH3_mass_fraction is not None:

            self.add_specie(id="NH3", molar_mass_kg_kmol=17, charge=0)
            self.add_specie(id="NH4+", molar_mass_kg_kmol=18, charge=1)
            self.add_specie(id="NH2COO-", molar_mass_kg_kmol=16+44, charge=-1)

            self.add_rxn_insta(id="NH4+ = NH3 + H+",
                               stoch={"NH4+": -1, "NH3": 1, "H+": 1},
                               unit={"NH4+": "m", "NH3": "m", "H+": "m"},
                               equilibrium_constant=self.__ammonia_dissociation_constant__)

            self.add_rxn_insta(id="CO2 + NH3 = NH2COO- + H+",
                               stoch={"CO2": -1, "NH3": -1, "NH2COO-": 1, "H+": 1},
                               unit={"CO2": "m", "NH3": "m", "NH2COO-": "m", "H+": "m"},
                               equilibrium_constant=self.__ammonia_stability_carbamate__)

        if MEA_mass_fraction is not None:

            self.add_specie(id="MEA", molar_mass_kg_kmol=61, charge=0)
            self.add_specie(id="MEA+", molar_mass_kg_kmol=62, charge=1)
            self.add_specie(id="MEACOO-", molar_mass_kg_kmol=104, charge=-1)
            self.add_specie(id="MEACOO", molar_mass_kg_kmol=105, charge=0)

            self.add_rxn_insta(id="MEA+ = MEA + H+",
                               stoch={"MEA+": -1, "MEA": 1, "H+": 1},
                               unit={"MEA+": "m", "MEA": "m", "H+": "m"},
                               equilibrium_constant=self.__mea_dissociation_constant__)

            self.add_rxn_insta(id="MEACOO = MEACOO- + H+",
                               stoch={"MEACOO": -1, "MEACOO-": 1, "H+": 1},
                               unit={"MEACOO": "m", "MEACOO-": "m", "H+": "m"},
                               equilibrium_constant=self.__mea_carbamate_dissociation_constant__)

            self.add_rxn_insta(id="CO2 + MEA = MEACOO- + H+",
                               stoch={"CO2": -1, "MEA": -1, "MEACOO-": 1, "H+": 1},
                               unit={"CO2": "m", "MEA": "m", "MEACOO-": "m", "H+": "m"},
                               equilibrium_constant=self.__mea_stability_carbamate__)

        if MDEA_mass_fraction is not None:

            self.add_specie(id="MDEA", molar_mass_kg_kmol=119, charge=0)
            self.add_specie(id="MDEA+", molar_mass_kg_kmol=120, charge=1)

            self.add_rxn_insta(id="MDEA+ = MDEA + H+",
                               stoch={"MDEA+": -1, "H+": 1, "MDEA": 1},
                               unit={"MDEA+": "m", "H+": "m", "MDEA": "m"},
                               equilibrium_constant=self.__mdea_dissociation_constant__)

        if PZ_mass_fraction is not None:

            self.add_specie(id="PZ", molar_mass_kg_kmol=86, charge=0)
            self.add_specie(id="PZ+", molar_mass_kg_kmol=87, charge=1)
            self.add_specie(id="PZ+2", molar_mass_kg_kmol=88, charge=2,)
            self.add_specie(id="PZCOO", molar_mass_kg_kmol=130, charge=0)
            self.add_specie(id="PZCOO-", molar_mass_kg_kmol=129, charge=-1)
            self.add_specie(id="PZ(COO)2-2", molar_mass_kg_kmol=172, charge=-2)
            self.add_specie(id="PZ(COO)2-", molar_mass_kg_kmol=173, charge=-1)

            #self.add_rxn_insta(id="PZ+2 = PZ+ + H+",
            #                   stoch={"PZ+2": -1, "PZ+": 1, "H+": 1},
            #                   unit={"PZ+2": "m", "PZ+": "m", "H+": "m"},
            #                   equilibrium_constant=self.__pz_dissociation_constant_K1__)

            self.add_rxn_insta(id="PZ+ = PZ + H+",
                               stoch={"PZ+": -1, "PZ": 1, "H+": 1},
                               unit={"PZ+": "m", "PZ": "m", "H+": "m"},
                               equilibrium_constant=self.__pz_dissociation_constant_K2__)

            self.add_rxn_insta(id="PZCOO = PZCOO- + H+",
                               stoch={"PZCOO": -1, "PZCOO-": 1, "H+": 1},
                               unit={"PZCOO": "m", "PZCOO-": "m", "H+": "m"},
                               equilibrium_constant=self.__pz_carbamate_dissociation_constant__)

            self.add_rxn_insta(id="PZ(COO)2- = PZ(COO)2-2 + H+",
                               stoch={"PZ(COO)2-": -1, "PZ(COO)2-2": 1, "H+": 1},
                               unit={"PZ(COO)2-": "m", "PZ(COO)2-2": "m", "H+": "m"},
                               equilibrium_constant=self.__pz_dicarbamate_dissociation_constant__)

            self.add_rxn_insta(id="CO2 + PZ = PZCOO- + H+",
                               stoch={"CO2": -1, "PZ": -1, "PZCOO-": 1, "H+": 1},
                               unit={"CO2": "m", "PZ": "m", "PZCOO-": "m", "H+": "m"},
                               equilibrium_constant=self.__pz_carbamate_stability__)

            self.add_rxn_insta(id="CO2 + PZCOO- = PZ(COO)2-2 + H+",
                               stoch={"CO2": -1, "PZCOO-": -1, "PZ(COO)2-2": 1, "H+": 1},
                               unit={"CO2": "m", "PZCOO-": "m", "PZ(COO)2-2": "m", "H+": "m"},
                               equilibrium_constant=self.__pz_dicarbamate_stability__)

        if _1MPZ_mass_fraction is not None:

            self.add_specie(id="1MPZ", molar_mass_kg_kmol=100, charge=0)
            self.add_specie(id="1MPZ+", molar_mass_kg_kmol=101, charge=1)
            self.add_specie(id="1MPZ+2", molar_mass_kg_kmol=102, charge=2,)
            self.add_specie(id="1MPZCOO", molar_mass_kg_kmol=100 + 44, charge=0)
            self.add_specie(id="1MPZCOO-", molar_mass_kg_kmol=100 + 44 - 1, charge=0)


            self.add_rxn_insta(id="1MPZ+2 = 1MPZ+ + H+",
                               stoch={"1MPZ+2": -1, "1MPZ+": 1, "H+": 1},
                               unit={"1MPZ+2": "m", "1MPZ+": "m", "H+": "m"},
                               equilibrium_constant=self.__1mpz_dissociation_constant_K1__)

            self.add_rxn_insta(id="1MPZ+ = 1MPZ + H+",
                               stoch={"1MPZ+": -1, "1MPZ": 1, "H+": 1},
                               unit={"1MPZ+": "m", "1MPZ": "m", "H+": "m"},
                               equilibrium_constant=self.__1mpz_dissociation_constant_K2__)

            self.add_rxn_insta(id="CO2 + 1MPZ = 1MPZCOO",
                               stoch={"CO2": -1, "1MPZ": -1, "1MPZCOO": 1},
                               unit={"CO2": "m", "1MPZ": "m", "1MPZCOO": "m"},
                               equilibrium_constant=self.__1mpz_stability_carbamate__)

            self.add_rxn_insta(id="CO2 + 1MPZ = 1MPZCOO- + H+",
                               stoch={"CO2": -1, "1MPZ": -1, "1MPZCOO-": 1, "H+":1},
                               unit={"CO2": "m", "1MPZ": "m", "1MPZCOO-": "m", "H+":"m"},
                               equilibrium_constant=self.__1mpz_stability_carbamate_2__)

        if DMPZ_mass_fraction is not None:

            self.add_specie(id="DMPZ", molar_mass_kg_kmol=114, charge=0)
            self.add_specie(id="DMPZ+", molar_mass_kg_kmol=115, charge=1)
            self.add_specie(id="DMPZ+2", molar_mass_kg_kmol=116, charge=2,)

            self.add_rxn_insta(id="DMPZ+2 = DMPZ+ + H+",
                               stoch={"DMPZ+2": -1, "DMPZ+": 1, "H+": 1},
                               unit={"DMPZ+2": "m", "DMPZ+": "m", "H+": "m"},
                               equilibrium_constant=self.__dmpz_dissociation_constant_K1__)

            self.add_rxn_insta(id="DMPZ+ = DMPZ + H+",
                               stoch={"DMPZ+": -1, "DMPZ": 1, "H+": 1},
                               unit={"DMPZ+": "m", "DMPZ": "m", "H+": "m"},
                               equilibrium_constant=self.__dmpz_dissociation_constant_K2__)

        if EDA_mass_fraction is not None:

            self.add_specie(id="EDA", molar_mass_kg_kmol=60, charge=0)
            self.add_specie(id="EDA+", molar_mass_kg_kmol=60 + 1, charge=1)
            self.add_specie(id="EDA+2", molar_mass_kg_kmol=60 + 2, charge=2,)
            self.add_specie(id="EDACOO-", molar_mass_kg_kmol=60 + 44 - 1, charge=-1)

            self.add_rxn_insta(id="EDA+2 = EDA+ + H+",
                               stoch={"EDA+2": -1, "EDA+": 1, "H+": 1},
                               unit={"EDA+2": "m", "EDA+": "m", "H+": "m"},
                               equilibrium_constant=self.__eda_dissociation_constant_K2__)

            self.add_rxn_insta(id="EDA+ = EDA + H+",
                               stoch={"EDA+": -1, "EDA": 1, "H+": 1},
                               unit={"EDA+": "m", "EDA": "m", "H+": "m"},
                               equilibrium_constant=self.__eda_dissociation_constant_K3__)

            self.add_rxn_insta(id="CO2 + EDA = EDACOO- + H+",
                               stoch={"CO2": -1, "EDA": -1, "EDACOO-": 1, "H+": 1},
                               unit={"CO2": "m", "EDA": "m", "EDACOO-": "m", "H+": "m"},
                               equilibrium_constant=self.__eda_carbamate_stability__)

        if DETA_mass_fraction is not None:

            self.add_specie(id="DETA", molar_mass_kg_kmol=103, charge=0)
            self.add_specie(id="DETA+", molar_mass_kg_kmol=104, charge=1)
            self.add_specie(id="DETA+2", molar_mass_kg_kmol=105, charge=2)
            self.add_specie(id="DETA+3", molar_mass_kg_kmol=106, charge=3)
            self.add_specie(id="DETACOO", molar_mass_kg_kmol=103+44, charge=0)
            self.add_specie(id="DETACOO-", molar_mass_kg_kmol=103+44-1, charge=-1)
            self.add_specie(id="DETA(COO)2-2", molar_mass_kg_kmol=103+2*44-2, charge=-2)
            self.add_specie(id="DETA(COO)2-", molar_mass_kg_kmol=103+2*44-1, charge=-1)

            self.add_rxn_insta(id="DETA+3 = DETA+2 + H+",
                               stoch={"DETA+3": -1, "DETA+2": 1, "H+": 1},
                               unit={"DETA+3": "m", "DETA+2": "m", "H+": "m"},
                               equilibrium_constant=self.__deta_dissociation_constant_K3__)

            self.add_rxn_insta(id="DETA+2 = DETA+ + H+",
                               stoch={"DETA+2": -1, "DETA+": 1, "H+": 1},
                               unit={"DETA+2": "m", "DETA+": "m", "H+": "m"},
                               equilibrium_constant=self.__deta_dissociation_constant_K2__)

            self.add_rxn_insta(id="DETA+ = DETA + H+",
                               stoch={"DETA+": -1, "DETA": 1, "H+": 1},
                               unit={"DETA+": "m", "DETA": "m", "H+": "m"},
                               equilibrium_constant=self.__deta_dissociation_constant_K1__)

            self.add_rxn_insta(id="DETACOO = DETACOO- + H+",
                               stoch={"DETACOO": -1, "DETACOO-": 1, "H+": 1},
                               unit={"DETACOO": "m", "DETACOO-": "m", "H+": "m"},
                               equilibrium_constant=self.__deta_carbamate_dissociation_constant__)

            #self.add_rxn_insta(id="DETA(COO)2- = DETA(COO)2-2 + H+",
            #                   stoch={"DETA(COO)2-": -1, "DETA(COO)2-2": 1, "H+": 1},
            #                   unit={"DETA(COO)2-": "m", "DETA(COO)2-2": "m", "H+": "m"},
            #                   equilibrium_constant=self.__deta_dicarbamate_dissociation_constant__)

            self.add_rxn_insta(id="CO2 + DETA = DETACOO- + H+",
                               stoch={"CO2": -1, "DETA": -1, "DETACOO-": 1, "H+": 1},
                               unit={"CO2": "m", "DETA": "m", "DETACOO-": "m", "H+": "m"},
                               equilibrium_constant=self.__deta_carbamate_stability__)

            #self.add_rxn_insta(id="CO2 + DETACOO- = DETA(COO)2-2 + H+",
            #                   stoch={"CO2": -1, "DETACOO-": -1, "DETA(COO)2-2": 1, "H+": 1},
            #                   unit={"CO2": "m", "DETACOO-": "m", "DETA(COO)2-2": "m", "H+": "m"},
            #                   equilibrium_constant=self.__deta_dicarbamate_stability__)

        if AMP_mass_fraction is not None:
            self.add_specie(id="AMP", molar_mass_kg_kmol=89, charge=0)
            self.add_specie(id="AMP+", molar_mass_kg_kmol=90, charge=1)
            self.add_rxn_insta(id="AMP+ = AMP + H+",
                               stoch={"AMP+": -1, "AMP": 1, "H+": 1},
                               unit={"AMP+": "m", "AMP": "m", "H+": "m"},
                               equilibrium_constant=self.__amp_dissociation_constant__)


        self.add_rxn_insta(id="H2O = H+ + OH-",
                           stoch={"H2O": -1, "H+": 1, "OH-": 1},
                           unit={"H2O": "x", "H+": "m", "OH-": "m"},
                           equilibrium_constant=self.__water_autoprotolysis_eq_constant__)

        self.add_rxn_insta(id="CO2 + H2O = HCO3- + H+",
                           stoch={"H2O": -1, "CO2": -1, "H+": 1, "HCO3-": 1},
                           unit={"H2O": "x", "CO2": "m", "H+": "m", "HCO3-": "m"},
                           equilibrium_constant=self.__carbonic_acid_rxn_1_eq_const__)

        self.add_rxn_insta(id="HCO3- = CO3-2 + H+",
                           stoch={"HCO3-": -1, "CO3-2": 1, "H+": 1},
                           unit={"HCO3-": "m", "CO3-2": "m", "H+": "m"},
                           equilibrium_constant=self.__carbonic_acid_rxn_2_eq_const__)

        self.add_vapor_pressure_bara_henry(id="CO2(g) = CO2(aq)",
                                           gas_id="CO2",
                                           liq_id="CO2",
                                           liq_unit="m",
                                           henrys_coefficient=self.__CO2_henrys_constant__)

        self.add_vapor_pressure_bara_raoult(id="H2O(g) = H2O(l)",
                                            gas_id="H2O",
                                            liq_id="H2O",
                                            pure_vapor_pressure_bara=self.__H2O_vapor_pressure_bara__)

        # Number of moles in 1 kg of Unloaded Solution
        n_KLys = w_KLys / 184
        n_KSar = w_KSar / 127
        n_KPro = w_KPro / 153
        n_Arg = w_Arg / 174
        n_MDEA = w_MDEA / 119
        n_PZ = w_PZ / 86
        n_1MPZ = w_1MPZ / 100
        n_DMPZ = w_DMPZ / 114

        n_EDA = w_EDA / 60
        n_DETA = w_DETA / 103
        n_MEA = w_MEA / 61
        n_AMP = w_AMP / 89
        n_NH3 = w_NH3 / 17

        # Number of Moles CO2 in 1 kg Unloaded Solution, After Loading
        n_CO2 = CO2Load * (n_MDEA + n_MEA + n_PZ + n_1MPZ + n_DMPZ + n_EDA + n_DETA + n_AMP + n_KLys + n_KSar + n_KPro + n_Arg + n_NH3)

        CO2_mass_fraction = 44 * n_CO2

        # Concentrations
        H2O_mass_fraction = 1 - self.get_info(id="Amino Acid Mass Fraction") - self.get_info(id="Amine Mass Fraction")

        # Setting all mass fractions to zero initially
        zero = 0 * np.ones(shape=H2O_mass_fraction.shape)
        for id in self.specie.keys():
            self.set_specie_mass_fraction(id=id, value=zero)

        # Mass Fractions
        self.set_specie_mass_fraction(id="CO2", value=CO2_mass_fraction)
        self.set_specie_mass_fraction(id="H2O", value=H2O_mass_fraction)

        if "MDEA" in self.specie.keys():
            self.set_specie_mass_fraction(id="MDEA", value=w_MDEA)

        if "PZ" in self.specie.keys():
            self.set_specie_mass_fraction(id="PZ", value=w_PZ)

        if "1MPZ" in self.specie.keys():
            self.set_specie_mass_fraction(id="1MPZ", value=w_1MPZ)

        if "DMPZ" in self.specie.keys():
            self.set_specie_mass_fraction(id="DMPZ", value=w_DMPZ)

        if "EDA" in self.specie.keys():
            self.set_specie_mass_fraction(id="EDA", value=w_EDA)

        if "DETA" in self.specie.keys():
            self.set_specie_mass_fraction(id="DETA", value=w_DETA)

        if "NH3" in self.specie.keys():
            self.set_specie_mass_fraction(id="NH3", value=w_NH3)

        if "MEA" in self.specie.keys():
            self.set_specie_mass_fraction(id="MEA", value=w_MEA)

        if "AMP" in self.specie.keys():
            self.set_specie_mass_fraction(id="AMP", value=w_AMP)

        if "Lys-" in self.specie.keys():
            self.set_specie_mass_fraction(id="Lys-", value=w_KLys * 145 / 184)

        if "Sar-" in self.specie.keys():
            self.set_specie_mass_fraction(id="Sar-", value=w_KSar * 88 / 127)

        if "Pro-" in self.specie.keys():
            self.set_specie_mass_fraction(id="Pro-", value=w_KPro * 114 / 153)

        if "Arg" in self.specie.keys():
            self.set_specie_mass_fraction(id="Arg", value=w_Arg)

        self.set_specie_mass_fraction(id="K+", value=39 * (w_KLys / 184 + w_KSar / 127 + w_KPro / 153))
        self.normalize_mass_fractions()

    def init_with_molality(self, temp_K, flow_kg_h, Lys_molality=None, Sar_molality=None, Pro_molality=None, Arg_molality=None, Cys_molality=None, KOH_molality=None, HCl_molality=None, CO2_molality=None):

        z = np.zeros(shape=temp_K.shape)

        self.set_solution_temp_K(value=temp_K)
        self.set_solution_flow_kg_h(value=flow_kg_h)

        m_Lys = Lys_molality if Lys_molality is not None else z
        m_Sar = Sar_molality if Sar_molality is not None else z
        m_Pro = Pro_molality if Pro_molality is not None else z
        m_Arg = Arg_molality if Arg_molality is not None else z
        m_Cys = Cys_molality if Cys_molality is not None else z
        m_KOH = KOH_molality if KOH_molality is not None else z
        m_HCl = HCl_molality if HCl_molality is not None else z
        m_CO2 = CO2_molality if CO2_molality is not None else z

        self.load_activity_coefficient(function=self.__activity_coefficient__)
        self.load_density_kg_m3(function=self.__density_kg_m3__)

        self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
        self.add_specie(id="H+", molar_mass_kg_kmol=1, charge=1)
        self.add_specie(id="OH-", molar_mass_kg_kmol=17, charge=-1)
        self.add_specie(id="K+", molar_mass_kg_kmol=39, charge=1)

        self.add_rxn_insta(id="H2O = H+ + OH-",
                           stoch={"H2O": -1, "H+": 1, "OH-": 1},
                           unit={"H2O": "x", "H+": "m", "OH-": "m"},
                           equilibrium_constant=self.__water_autoprotolysis_eq_constant__)

        if Lys_molality is not None:
            self.add_specie(id="Lys+", molar_mass_kg_kmol=147, charge=1)
            self.add_specie(id="Lys", molar_mass_kg_kmol=146, charge=0)
            self.add_specie(id="Lys-", molar_mass_kg_kmol=145, charge=-1)
            self.add_rxn_insta(id="Lys+ = Lys + H+",
                               stoch={"Lys+": -1, "Lys": 1, "H+": 1},
                               unit={"Lys+": "m", "Lys": "m", "H+": "m"},
                               equilibrium_constant=self.__lysine_dissociation_constant_K2__)
            self.add_rxn_insta(id="Lys = Lys- + H+",
                               stoch={"Lys": -1, "Lys-": 1, "H+": 1},
                               unit={"Lys": "m", "Lys-": "m", "H+": "m"},
                               equilibrium_constant=self.__lysine_dissociation_constant_K3__)

        if Lys_molality is not None and CO2_molality is not None:
            self.add_specie(id="LysCOO-", molar_mass_kg_kmol=189, charge=-1)
            self.add_specie(id="LysCOO-2", molar_mass_kg_kmol=188, charge=-2)
            self.add_rxn_insta(id="CO2 + Lys- = H+ + LysCOO-2",
                               stoch={"CO2": -1, "Lys-": -1, "H+": 1, "LysCOO-2": 1},
                               unit={"CO2": "m", "Lys-": "m", "H+": "m", "LysCOO-2": "m"},
                               equilibrium_constant=self.__lysine_stability_carbamate__)
            self.add_rxn_insta(id="LysCOO- = LysCOO-2 + H+",
                               stoch={"LysCOO-": -1, "LysCOO-2": 1, "H+": 1},
                               unit={"LysCOO-": "m", "LysCOO-2": "m", "H+": "m"},
                               equilibrium_constant=self.__lysine_carbamate_dissociation_constant__)

        if Sar_molality is not None:
            self.add_specie(id="Sar+", molar_mass_kg_kmol=90, charge=1)
            self.add_specie(id="Sar", molar_mass_kg_kmol=89, charge=0)
            self.add_specie(id="Sar-", molar_mass_kg_kmol=88, charge=-1)
            self.add_rxn_insta(id="Sar+ = Sar + H+",
                               stoch={"Sar+": -1, "Sar": 1, "H+": 1},
                               unit={"Sar+": "m", "Sar": "m", "H+": "m"},
                               equilibrium_constant=self.__sarcosine_dissociation_constant_K1__)
            self.add_rxn_insta(id="Sar = Sar- + H+",
                               stoch={"Sar": -1, "Sar-": 1, "H+": 1},
                               unit={"Sar": "m", "Sar-": "m", "H+": "m"},
                               equilibrium_constant=self.__sarcosine_dissociation_constant_K2__)

        if Sar_molality is not None and CO2_molality is not None:
            self.add_specie(id="SarCOO-", molar_mass_kg_kmol=88 + 44, charge=-1)
            self.add_specie(id="SarCOO-2", molar_mass_kg_kmol=88 + 44 - 1, charge=-2)
            self.add_rxn_insta(id="CO2 + Sar- = H+ + SarCOO-2",
                               stoch={"CO2": -1, "Sar-": -1, "H+": 1, "SarCOO-2": 1},
                               unit={"CO2": "m", "Sar-": "m", "H+": "m", "SarCOO-2": "m"},
                               equilibrium_constant=self.__sarcosine_stability_carbamate__)
            self.add_rxn_insta(id="SarCOO- = SarCOO-2 + H+",
                               stoch={"SarCOO-": -1, "SarCOO-2": 1, "H+": 1},
                               unit={"SarCOO-": "m", "SarCOO-2": "m", "H+": "m"},
                               equilibrium_constant=self.__sarcosine_carbamate_dissociation_constant__)

        if Pro_molality is not None:
            self.add_specie(id="Pro+", molar_mass_kg_kmol=116, charge=1)
            self.add_specie(id="Pro", molar_mass_kg_kmol=115, charge=0)
            self.add_specie(id="Pro-", molar_mass_kg_kmol=114, charge=-1)
            self.add_rxn_insta(id="Pro+ = Pro + H+",
                               stoch={"Pro+": -1, "Pro": 1, "H+": 1},
                               unit={"Pro+": "m", "Pro": "m", "H+": "m"},
                               equilibrium_constant=self.__proline_dissociation_constant_K1__)
            self.add_rxn_insta(id="Pro = Pro- + H+",
                               stoch={"Pro": -1, "Pro-": 1, "H+": 1},
                               unit={"Pro": "m", "Pro-": "m", "H+": "m"},
                               equilibrium_constant=self.__proline_dissociation_constant_K2__)

        if Pro_molality is not None and CO2_molality is not None:
            self.add_specie(id="ProCOO-", molar_mass_kg_kmol=158, charge=-1)
            self.add_specie(id="ProCOO-2", molar_mass_kg_kmol=157, charge=-2)
            self.add_rxn_insta(id="CO2 + Pro- = H+ + ProCOO-2",
                               stoch={"CO2": -1, "Pro-": -1, "H+": 1, "ProCOO-2": 1},
                               unit={"CO2": "m", "Pro-": "m", "H+": "m", "ProCOO-2": "m"},
                               equilibrium_constant=self.__proline_carbamate_stability__)
            self.add_rxn_insta(id="ProCOO- = ProCOO-2 + H+",
                               stoch={"ProCOO-": -1, "ProCOO-2": 1, "H+": 1},
                               unit={"ProCOO-": "m", "ProCOO-2": "m", "H+": "m"},
                               equilibrium_constant=self.__proline_carbamate_dissociation_constant__)

        if Arg_molality is not None:
            self.add_specie(id="Arg+2", molar_mass_kg_kmol=176, charge=2)
            self.add_specie(id="Arg+", molar_mass_kg_kmol=175, charge=1)
            self.add_specie(id="Arg", molar_mass_kg_kmol=174, charge=0)
            self.add_rxn_insta(id="Arg+ = Arg + H+",
                               stoch={"Arg+": -1, "Arg": 1, "H+": 1},
                               unit={"Arg+": "m", "Arg": "m", "H+": "m"},
                               equilibrium_constant=self.__arginine_dissociation_constant_K2__)

        if Arg_molality is not None and CO2_molality is not None:
            self.add_specie(id="ArgCOO-", molar_mass_kg_kmol=174 + 44 - 1, charge=-1)
            self.add_specie(id="ArgCOO", molar_mass_kg_kmol=174 + 44, charge=0)

        if Cys_molality is not None:
            pass

        if Cys_molality is not None and CO2_molality is not None:
            pass

        if HCl_molality is not None:
            self.add_specie(id="Cl-", molar_mass_kg_kmol=35.5, charge=-1)

        if CO2_molality is not None:
            self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
            self.add_specie(id="HCO3-", molar_mass_kg_kmol=61, charge=-1)
            self.add_specie(id="CO3-2", molar_mass_kg_kmol=60, charge=-2)
            self.add_rxn_insta(id="CO2 + H2O = HCO3- + H+",
                               stoch={"H2O": -1, "CO2": -1, "H+": 1, "HCO3-": 1},
                               unit={"H2O": "x", "CO2": "m", "H+": "m", "HCO3-": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_1_eq_const__)
            self.add_rxn_insta(id="HCO3- = CO3-2 + H+",
                               stoch={"HCO3-": -1, "CO3-2": 1, "H+": 1},
                               unit={"HCO3-": "m", "CO3-2": "m", "H+": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_2_eq_const__)

        # Setting all molalities to zero initially
        m = {}
        for id in self.specie.keys():
            if id != "H2O":
                m[id] = z

        m["K+"] = m_KOH
        m["OH-"] = m_KOH
        m["H+"] = m_HCl

        if HCl_molality is not None:
            m["Cl-"] = m_HCl

        if "Lys" in self.specie.keys():
            m["Lys"] = m_Lys

        if "Sar" in self.specie.keys():
            m["Sar"] = m_Sar

        if "Pro" in self.specie.keys():
            m["Pro"] = m_Pro

        if "Arg" in self.specie.keys():
            m["Arg"] = m_Arg

        self.set_species_molality(solutes_molality_mol_kg=m)

    def CO2Load(self, LiquidStream):
        x_CO2 = 0
        x_Amine = 0

        C = {"CO2": 1, "HCO3-": 1, "CO3-2": 1,
             "LysCOO-2": 1, "LysCOO-": 1, "LysCOO": 1, "Lys(COO)2-3": 2, "Lys(COO)2-2": 2, "Lys(COO)2-": 2,
             "ProCOO-2":1, "ProCOO-":1,
             "SarCOO-2":1, "SarCOO-":1,
             "ArgCOO-":1, "ArgCOO":1,
             "MEACOO-": 1, "MEACOO": 1,
             "PZCOO": 1, "PZCOO-": 1,
             "PZ(COO)2-2": 2, "PZ(COO)2-": 2,
             "EDACOO": 1, "EDACOO-": 1,
             "EDA(COO)2-2": 2, "EDA(COO)2-": 2,
             "DETACOO": 1, "DETACOO-": 1,
             "DETA(COO)2-2": 2, "DETA(COO)2-": 2,
             "AMPCOO-": 1,
             "NH2COO-":1,
             "1MPZCOO":1, "1MPZCOO-":1}

        A = {"Lys+2": 1, "Lys+": 1, "Lys": 1, "Lys-": 1, "LysCOO": 1, "LysCOO-": 1, "LysCOO-2": 1, "Lys(COO)2-": 1, "Lys(COO)2-2":1, "Lys(COO)2-3":1,
             "Pro+":1, "Pro":1, "Pro-":1, "ProCOO-":1, "ProCOO-2":1,
             "Sar+":1, "Sar":1, "Sar-":1, "SarCOO-":1, "SarCOO-2":1,
             "Arg+2":1, "Arg+":1, "Arg":1, "ArgCOO-":1, "ArgCOO":1,
             "MEA": 1, "MEACOO": 1, "MEACOO-": 1, "MEA+": 1,
             "MDEA": 1, "MDEA+": 1,
             "PZ": 1, "PZ+": 1, "PZ+2": 1, "PZCOO": 1, "PZCOO-": 1, "PZ(COO)2-2": 1, "PZ(COO)2-": 1,
             "EDA": 1, "EDA+": 1, "EDA+2": 1, "EDACOO": 1, "EDACOO-": 1, "EDA(COO)2-2": 1, "EDA(COO)2-": 1,
             "DETA": 1, "DETA+": 1, "DETA+2": 1, "DETACOO": 1, "DETACOO-": 1, "DETA(COO)2-2": 1, "DETA(COO)2-": 1,
             "AMP": 1, "AMPCOO-": 1, "AMP+": 1,
             "NH3": 1, "NH4+":1, "NH2COO-":1,
             "DMPZ":1, "DMPZ+":1, "DMPZ+2":1,
             "1MPZ": 1, "1MPZ+": 1, "1MPZ+2": 1, "1MPZCOO":1, "1MPZCOO-":1}

        for c in C.keys():
            if c in LiquidStream.specie.keys():
                x_CO2 = x_CO2 + C[c] * LiquidStream.get_specie_molar_fraction(id=c)

        for a in A.keys():
            if a in LiquidStream.specie.keys():
                x_Amine = x_Amine + A[a] * LiquidStream.get_specie_molar_fraction(id=a)

        alpha = np.maximum(x_CO2 / np.maximum(x_Amine, 10 ** (-9)), 10**(-9))
        return alpha

    def __heat_capacity_kJ_kgK__(self, LiquidStream):
        T = LiquidStream.temp_K
        w_Amine = self.get_info(id="Amine Mass Fraction")
        w_AA = self.get_info(id="Amino Acid Mass Fraction")
        w_H2O = 1 - w_Amine - w_AA
        alpha = LiquidStream.CO2Load(LiquidStream)
        cp_AA = ((4.2 + 0.41 * (w_Amine + w_AA)) / (1 + 1.05 * (w_Amine + w_AA))) + (0.3 / 75) * (T - 298.15)
        cp_Amine = 4.2 * w_H2O + 2.8 * (w_Amine + w_AA) + (0.3 / 75) * (T - 298.15) - (0.3 / 0.5) * alpha
        cp = cp_AA * w_AA / (w_AA + w_Amine) + cp_Amine * w_Amine / (w_Amine + w_AA)
        return cp

    def __density_kg_m3__(self, LiquidStream):
        T = LiquidStream.temp_K
        return 1050 * np.ones(shape=T.shape)

    def __activity_coefficient__(self, LiquidStream, id):
        I = LiquidStream.get_solution_ionic_strength_mol_kg()
        T = LiquidStream.temp_K
        z = LiquidStream.get_specie_charge(id)
        log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))
        gamma = 10 ** log10_gamma
        return gamma

    # --------------------------------------------------------------------------------------------

    def __CO2_henrys_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        H_CO2 = 1.153 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)
        return H_CO2

    def __H2O_vapor_pressure_bara__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        T = np.minimum(T, 273.15 + 150)
        pc = 220.64
        Tc = 647.096
        tau = 1 - T / Tc
        a1 = -7.85951783
        a2 = 1.84408259
        a3 = -11.7866497
        a4 = 22.6807411
        a5 = -15.9618719
        a6 = 1.80122502
        p = pc * np.exp(
            (Tc / T) * (a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 3 + a4 * tau ** 3.5 + a5 * tau ** 4 + a6 * tau ** 7.5))
        return p

    # --------------------------------------------------------------------------------------------

    def __water_autoprotolysis_eq_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        Kw = 10 ** (-14) * np.exp(-13445.9 * (1 / T - 1 / 298) - 22.48 * np.log(T / 298))
        return Kw

    # --------------------------------------------------------------------------------------------

    def __carbonic_acid_rxn_1_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-6.32) * np.exp(5139 * (1 / T - 1 / 298) + 14.5258479 * np.log(T / 298))
        return K

    def __carbonic_acid_rxn_2_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-10.33) * np.exp(22062 * (1 / T - 1 / 298) + 67.264072 * np.log(T / 298))
        return K

    # --------------------------------------------------------------------------------------------

    def __lysine_dissociation_constant_K2__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-9.16) * np.exp(-4900 * (1 / T - 1 / 298.15))
        return K

    def __lysine_dissociation_constant_K3__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-10.9) * np.exp(-5700 * (1 / T - 1 / 298.15))
        return K

    def __lysine_stability_carbamate__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 2.78 * 10 ** (5) * np.exp(10294 * (1 / T - 1 / 298))
        return K



    # --------------------------------------------------------------------------------------------

    def __proline_dissociation_constant_K1__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-1.8) * np.exp(0 * (1 / T - 1 / 298.15))
        return K

    def __proline_dissociation_constant_K2__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-9.83) * np.exp(-4900 * (1 / T - 1 / 298.15))
        return K

    def __proline_carbamate_stability__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 5.65 * 10**(-6) * np.exp(3018 * (1 / T - 1 / 298))
        return K

    # --------------------------------------------------------------------------------------------

    def __sarcosine_dissociation_constant_K1__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-1.8) * np.exp(0 * (1 / T - 1 / 298.15))
        return K

    def __sarcosine_dissociation_constant_K2__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        I = LiquidStream.get_solution_ionic_strength_mol_kg()
        #K = 10 ** (-11.64) * np.exp(-4900 * (1 / T - 1 / 298.15))          # Literature
        #K = 10 ** (-10.25) * np.exp(-4900 * (1 / T - 1 / 298.15))          # Lab Container. pH titration. Dilute Solution.
        #K = 10 ** (-11.15) * np.exp(-2900 * (1 / T - 1 / 298.15))           # Lab Container. pH titration. 30% KSar Solution
        K = 10 ** (-10.8) * np.exp(-4900 * (1 / T - 1 / 298.15))
        #K = K * 10 ** (-0.52 * I)                                          # Guessimate
        #K = K * 10 ** (-0.75 * np.sqrt(I))                                  # Guessimate II
        #K = 10 ** (-10.9) * np.exp(-4900 * (1 / T - 1 / 298.15))           # Literature, CO2 Vapor Press
        return K

    def __sarcosine_stability_carbamate__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 9.25 * 10 ** (-7) * np.exp(3018 * (1 / T - 1 / 298))
        return K

    # --------------------------------------------------------------------------------------------

    def __arginine_dissociation_constant_K2__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-9.09) * np.exp(-4900 * (1 / T - 1 / 298.15))
        return K

    # --------------------------------------------------------------------------------------------

    def __ammonia_dissociation_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-10.34) * np.exp(-7988 * (1 / T - 1 / 298))       # 9.25
        return K

    def __ammonia_stability_carbamate__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 1.056e-08 * np.exp(-3844 * (1 / T - 1 / 298))
        return K

    # --------------------------------------------------------------------------------------------

    def __mea_dissociation_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-9.72) * np.exp(-6375 * (1 / T - 1 / 298))
        return K

    def __mea_carbamate_dissociation_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-7.19) * np.exp(-2766 * (1 / T - 1 / 298))
        return K

    def __mea_stability_carbamate__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 2.15 * 10 ** (-5) * np.exp(3018 * (1 / T - 1 / 298))
        return K

    # --------------------------------------------------------------------------------------------

    def __mdea_dissociation_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-8.75) * np.exp(-4900 * (1 / T - 1 / 298.15))        # 8.35
        return K

    # --------------------------------------------------------------------------------------------

    def __pz_dissociation_constant_K1__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-5.34) * np.exp(-3850 * (1 / T - 1 / 298))
        return K

    def __pz_dissociation_constant_K2__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-9.62) * np.exp(-4570 * (1 / T - 1 / 298))
        return K

    def __pz_carbamate_dissociation_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-9.05) * np.exp(-2165 * (1 / T - 1 / 298))
        return K

    def __pz_dicarbamate_dissociation_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-8.94) * np.exp(-7096 * (1 / T - 1 / 298))
        return K

    def __pz_carbamate_stability__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 1.326 * 10**(-4) * np.exp(7230 * (1 / T - 1 / 298))
        return K

    def __pz_dicarbamate_stability__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 6.835 * 10**(-7) * np.exp(1513 * (1 / T - 1 / 298))
        return K

    # --------------------------------------------------------------------------------------------

    def __1mpz_dissociation_constant_K1__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-4.63) * np.exp(-4400 * (1 / T - 1 / 298))
        return K

    def __1mpz_dissociation_constant_K2__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-9.14) * np.exp(-5000 * (1 / T - 1 / 298))
        return K

    def __1mpz_stability_carbamate__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 93441 * np.exp(29639 * (1 / T - 1 / 298))
        return K

    def __1mpz_stability_carbamate_2__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 5.788e-06 * np.exp(6682 * (1 / T - 1 / 298))
        return K


    # --------------------------------------------------------------------------------------------

    def __dmpz_dissociation_constant_K1__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-3.81) * np.exp(-3000 * (1 / T - 1 / 298))
        return K

    def __dmpz_dissociation_constant_K2__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-8.38) * np.exp(-4900 * (1 / T - 1 / 298))
        return K

    # --------------------------------------------------------------------------------------------

    def __eda_dissociation_constant_K2__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-6.86) * np.exp(-63668 * (1 / T - 1 / 298.15))
        return K

    def __eda_dissociation_constant_K3__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-9.82) * np.exp(-7329 * (1 / T - 1 / 298.15))
        return K

    def __eda_carbamate_stability__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 2.727e-05 * np.exp(338 * (1 / T - 1 / 298))
        return K

    # --------------------------------------------------------------------------------------------

    def __deta_dissociation_constant_K1__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-10.12) * np.exp(-5510 * (1 / T - 1 / 298))
        return K

    def __deta_dissociation_constant_K2__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-9.64) * np.exp(-5710 * (1 / T - 1 / 298))
        return K

    def __deta_dissociation_constant_K3__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-5.08) * np.exp(-3509 * (1 / T - 1 / 298))
        return K

    def __deta_carbamate_dissociation_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-9.49) * np.exp(-4119 * (1 / T - 1 / 298))
        return K

    def __deta_carbamate_stability__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K =4.171 * 10**(-3) * np.exp(7320 * (1 / T - 1 / 298))
        return K

    # --------------------------------------------------------------------------------------------

    def __amp_dissociation_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-9.84) * np.exp(-6682 * (1 / T - 1 / 298))
        return K

    # --------------------------------------------------------------------------------------------


class LiquidStream_KPro30(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="30% KPro")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, CO2Load=CO2Load, KPro_mass_fraction=0.3 * np.ones(shape=temp_K.shape))


class LiquidStream_KPro33(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="33% KPro")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, CO2Load=CO2Load, KPro_mass_fraction=0.33 * np.ones(shape=temp_K.shape))


class LiquidStream_KPro55(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="55% KPro")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, CO2Load=CO2Load, KPro_mass_fraction=0.55 * np.ones(shape=temp_K.shape))


class LiquidStream_KSar30(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="30% KSar")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, CO2Load=CO2Load, KSar_mass_fraction=0.3 * np.ones(shape=temp_K.shape))


class LiquidStream_KLys30(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="30% KLys")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, CO2Load=CO2Load, KLys_mass_fraction=0.3 * np.ones(shape=temp_K.shape))


class LiquidStream_Ammonia30(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="30% NH3")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, CO2Load=CO2Load, NH3_mass_fraction=0.3 * np.ones(shape=temp_K.shape))


class LiquidStream_MEA30(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="30% MEA")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, CO2Load=CO2Load, MEA_mass_fraction=0.3 * np.ones(shape=temp_K.shape))


class LiquidStream_MDEA30(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="30% MDEA")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, CO2Load=CO2Load, MDEA_mass_fraction=0.3 * np.ones(shape=temp_K.shape))


class LiquidStream_DETA30(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="30% DETA")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, CO2Load=CO2Load, DETA_mass_fraction=0.3 * np.ones(shape=temp_K.shape))


class LiquidStream_EDA30(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="30% DETA")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, CO2Load=CO2Load, EDA_mass_fraction=0.3 * np.ones(shape=temp_K.shape))


class LiquidStream_PZ30(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="30% PZ")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, PZ_mass_fraction=0.30 * np.ones(shape=temp_K.shape), CO2Load=CO2Load)


class LiquidStream_1MPZ30(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="30% 1MPZ")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, _1MPZ_mass_fraction=0.30 * np.ones(shape=temp_K.shape), CO2Load=CO2Load)


class LiquidStream_DMPZ30(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="30% DMPZ")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, DMPZ_mass_fraction=0.30 * np.ones(shape=temp_K.shape), CO2Load=CO2Load)


class LiquidStream_AMP30(LiquidStream_MegaMix):

    def __init__(self, temp_K, flow_kg_h, CO2Load):
        super().__init__(stream_id="30% AMP")
        self.init_with_mass_frac(temp_K=temp_K, flow_kg_h=flow_kg_h, AMP_mass_fraction=0.30 * np.ones(shape=temp_K.shape), CO2Load=CO2Load)





# --- Various -----------------------------------------------------------------------------------


class LiquidStream_CO2(lab.LiquidStream):

    def __init__(self, temp_K, flow_kg_h):
        super().__init__(stream_id="CO2",solvent_id="CO2")
        self.add_specie(id="CO2", library=library)
        shape = np.ones(shape=temp_K.shape)
        self.set_solution_flow_kg_h(value=flow_kg_h)
        self.set_solution_temp_K(value=temp_K)
        self.set_specie_mass_fraction(id="CO2", value=1.0 * shape)


class LiquidStream_Water(lab.LiquidStream):

    def __init__(self, temp_K, flow_kg_h):
        super().__init__(stream_id="Water",solvent_id="H2O")

        self.load_density_kg_m3(function=self.__density_kg_m3__)
        self.load_diffusivity(function=self.__diffusivity_m2_s__)
        self.load_viscosity_Pas(function=self.__viscosity_Pas__)
        self.load_heat_capacity_kJ_kgK(function=self.__heat_capacity_kJ_kgK__)
        self.load_activity_coefficient(function=self.__activity__)

        self.add_info(key="Amine Mass Fraction", value=0.0* np.ones(shape=temp_K.shape))

        self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
        self.add_vapor_pressure_bara_raoult(id="H2O(g) = H2O(l)",
                                            gas_id="H2O",
                                            liq_id="H2O",
                                            pure_vapor_pressure_bara=self.__H2O_vapor_pressure_bara__)

        shape = np.ones(shape=temp_K.shape)
        self.set_solution_flow_kg_h(value=flow_kg_h)
        self.set_solution_temp_K(value=temp_K)
        self.set_specie_mass_fraction(id="H2O", value=1.0 * shape)

    def __activity__(self, LiquidStream, id):
        return 1

    def __heat_capacity_kJ_kgK__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        t = T - 273.15
        t = np.maximum(t, 0.0)
        a = 4.2174356
        b = -0.0056181625
        c = 0.0012992528
        d = -0.00011535353
        e = 4.14964 * 10 ** (-6)
        cp = a + b * t + c * t ** 1.5 + d * t ** 2 + e * t ** 2.5
        return cp

    def __density_kg_m3__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        Tc = 647.096  # Critical Temp [K]
        rhoc = 322  # Critical Density [kg/m3]
        tau = 1 - T / Tc
        b1 = 1.99274064
        b2 = 1.09965342
        b3 = -0.510839303
        b4 = -1.75493479
        b5 = -45.5170352
        b6 = -6.74694450 * 10 ** 5
        rho = rhoc * (1 + b1 * tau ** (1 / 3) + b2 * tau ** (2 / 3) + b3 * tau ** (5 / 3) + b4 * tau ** (16 / 3) + b5 * tau ** (43 / 3) + b6 * tau ** (110 / 3))
        return rho

    def __diffusivity_m2_s__(self, LiquidStream, id):
        D_in_H2O_at_298K = {}
        D_in_H2O_at_298K["H2O"] = 2.27 * 10 ** (-9)
        D_in_H2O_at_298K["H+"] = 9.311 * 10 ** (-9)
        D_in_H2O_at_298K["OH-"] = 4.56 * 10 ** (-9)
        D_in_H2O_at_298K["SO2"] = 1.83 * 10 ** (-9)
        D_in_H2O_at_298K["HSO3-"] = 1.33 * 10 ** (-9)
        D_in_H2O_at_298K["SO3-2"] = 0.959 * 10 ** (-9)
        D_in_H2O_at_298K["HSO4-"] = 1.33 * 10 ** (-9)
        D_in_H2O_at_298K["SO4-2"] = 1.07 * 10 ** (-9)
        D_in_H2O_at_298K["CO2"] = 1.92 * 10 ** (-9)
        D_in_H2O_at_298K["HCO3-"] = 1.19 * 10 ** (-9)
        D_in_H2O_at_298K["CO3-2"] = 0.92 * 10 ** (-9)
        D_in_H2O_at_298K["O2"] = 2.01 * 10 ** (-9)
        D_in_H2O_at_298K["Ca+2"] = 0.792 * 10 ** (-9)
        D_in_H2O_at_298K["Mg+2"] = 0.706 * 10 ** (-9)
        D_in_H2O_at_298K["Na+"] = 1.334 * 10 ** (-9)
        D_in_H2O_at_298K["NH4+"] = 1.957 * 10 ** (-9)
        D_in_H2O_at_298K["Cl-"] = 2.03 * 10 ** (-9)
        D_in_H2O_at_298K["MEA"] = 1.078025911 * 10 ** (-9)
        D_in_H2O_at_298K["MEAH+"] = 0.75 * 10 ** (-9)
        D_in_H2O_at_298K["MEACOO-"] = 0.75 * 10 ** (-9)

        # Temperature Compensation
        T = LiquidStream.get_solution_temp_K()
        tempComp = 10 ** (-8.1764) * 10 ** (712.5 / T - 2.591 * 10 ** 5 / T ** 2) / (1.92 * 10 ** (-9))
        D_in_H2O = D_in_H2O_at_298K[id] * tempComp if id in D_in_H2O_at_298K.keys() else 1.0 * 10 ** (-9) * tempComp

        return D_in_H2O

    def __viscosity_Pas__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        t = T - 273.15
        t = np.maximum(t, 1.0)
        a = 557.82468
        b = 19.408782
        c = 0.1360459
        d = -3.1160832 * 10 ** (-4)
        mu = 1 / (a + b * t + c * t ** 2 + d * t ** 3)
        return mu

    def __H2O_vapor_pressure_bara__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        T = np.minimum(T, 273.15 + 150)
        pc = 220.64
        Tc = 647.096
        tau = 1 - T / Tc
        a1 = -7.85951783
        a2 = 1.84408259
        a3 = -11.7866497
        a4 = 22.6807411
        a5 = -15.9618719
        a6 = 1.80122502
        p = pc * np.exp((Tc / T) * (a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 3 + a4 * tau ** 3.5 + a5 * tau ** 4 + a6 * tau ** 7.5))
        return p

    def __surface_tension_H2O_N_m__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        t = T - 273.15
        t = np.maximum(t, 0.0)
        a = 0.075652711
        b = -0.00013936956
        c = -3.0842103 * 10 ** (-7)
        d = 2.7588435 * 10 ** (-10)
        sigma = a + b * t + c * t ** 2 + d * t ** 3
        return sigma


class LiquidStream_ScrubbingWater_OpenLoop(lab.LiquidStream):

    def __init__(self, temp_K, flow_kg_h):
        super().__init__(stream_id="", solvent_id="H2O")
        z = np.zeros(shape=temp_K.shape)

        self.add_rxn_insta(id="SO2 + H2O = HSO3- + H+",
                           stoch={"SO2": -1, "H2O": -1, "HSO3-": 1, "H+": 1},
                           unit={"SO2": "c", "H2O": "x", "HSO3-": "c", "H+": "c"},
                           equilibrium_constant=self.__sulfurous_acid_rxn_10_eq_const__)

        self.add_rxn_insta(id="HSO3- = SO3-2 + H+",
                           stoch={"HSO3-": -1, "SO3-2": 1, "H+": 1},
                           unit={"HSO3-": "c", "SO3-2": "c", "H+": "c"},
                           equilibrium_constant=self.__sulfurous_acid_rxn_11_eq_const__)

        self.add_rxn_insta(id="HSO3- = SO2 + OH-",
                           stoch={"HSO3-": -1, "SO2": 1, "OH-": 1},
                           unit={"HSO3-": "c", "SO2": "c", "OH-": "c"},
                           equilibrium_constant=self.__sulfurous_acid_rxn_12_eq_const__)

        self.add_rxn_insta(id="SO3-2 + H2O = HSO3- + OH-",
                           stoch={"SO3-2": -1, "H2O": -1, "HSO3-": 1, "OH-": 1},
                           unit={"SO3-2": "c", "H2O": "x", "HSO3-": "c", "OH-": "c"},
                           equilibrium_constant=self.__sulfurous_acid_rxn_13_eq_const__)

        self.add_rxn_insta(id="HSO4- = SO4-2 + H+",
                           stoch={"HSO4-": -1, "SO4-2": 1, "H+": 1},
                           unit={"HSO4-": "c", "SO4-2": "c", "H+": 1},
                           equilibrium_constant=self.__sulfuric_acid_rxn_1_eq_const__)

    def __activity_coefficient__(self, LiquidStream, id):
        I = LiquidStream.get_solution_ionic_strength_mol_kg()
        z = LiquidStream.get_specie_charge(id)
        log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))
        gamma = 10 ** log10_gamma
        return gamma

    def __sulfurous_acid_rxn_10_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-1.86) * np.exp(2140 * (1 / T - 1 / 298.15))
        return K

    def __sulfurous_acid_rxn_11_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-7.17) * np.exp(439 * (1 / T - 1 / 298.15))
        return K

    def __sulfurous_acid_rxn_12_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-12.13) * np.exp(-8863 * (1 / T - 1 / 298.15))
        return K

    def __sulfurous_acid_rxn_13_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-6.815) * np.exp(-7161 * (1 / T - 1 / 298.15))
        return K

    def __sulfuric_acid_rxn_1_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-1.97689) * np.exp(2637 * (1 / T - 1 / 298.15))
        return K


class LiquidStream_CitricAcid(lab.LiquidStream):

    def __init__(self, temp_K, flow_kg_h):
        super().__init__(stream_id="", solvent_id="H2O")
        z = np.zeros(shape=temp_K.shape)

        self.add_specie(id="Ci", molar_mass_kg_kmol=192, charge=0)
        self.add_specie(id="Ci-", molar_mass_kg_kmol=191, charge=-1)
        self.add_specie(id="Ci-2", molar_mass_kg_kmol=190, charge=-2)
        self.add_specie(id="Ci-3", molar_mass_kg_kmol=189, charge=-3)

        self.add_rxn_insta(id="Ci = Ci- + H+",
                           stoch={"Ci": -1, "Ci-": 1, "H+": 1},
                           unit={"Ci": "c", "Ci-": "c", "H+": "c"},
                           equilibrium_constant=self.__citric_acid_rxn_1_eq_const__)

        self.add_rxn_insta(id="Ci- = Ci-2 + H+",
                           stoch={"Ci-": -1, "Ci-2": 1, "H+": 1},
                           unit={"Ci-": "c", "Ci-2": "c", "H+": "c"},
                           equilibrium_constant=self.__citric_acid_rxn_2_eq_const__)

        self.add_rxn_insta(id="Ci-2 = Ci-3 + H+",
                           stoch={"Ci-2": -1, "Ci-3": 1, "H+": 1},
                           unit={"Ci-2": "c", "Ci-3": "c", "H+": "c"},
                           equilibrium_constant=self.__citric_acid_rxn_3_eq_const__)

    def __activity_coefficient__(self, LiquidStream, id):
        I = LiquidStream.get_solution_ionic_strength_mol_kg()
        z = LiquidStream.get_specie_charge(id)
        log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))
        gamma = 10 ** log10_gamma
        return gamma

    def __water_autoprotolysis_eq_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        Kw = 10 ** (-14) * np.exp(-13445.9 * (1 / T - 1 / 298) - 22.48 * np.log(T / 298))
        return Kw

    def __citric_acid_rxn_1_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        ln_K = 100.446 - 52.159 * (100 / T) - 15.823 * np.log(T)
        return np.exp(ln_K)

    def __citric_acid_rxn_2_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        ln_K = 134.23 - 67.107 * (100 / T) - 21.532 * np.log(T)
        return np.exp(ln_K)

    def __citric_acid_rxn_3_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        ln_K = 188.717 - 87.047 * (100 / T) - 30.583 * np.log(T)
        return np.exp(ln_K)


class LiquidStream_AscorbicAcid(lab.LiquidStream):

    def __init__(self, temp_K, flow_kg_h):
        super().__init__(stream_id="", solvent_id="H2O")
        z = np.zeros(shape=temp_K.shape)

        self.add_rxn_insta(id="Asc = Asc- + H+",
                           stoch={"Asc": -1, "Asc-": 1, "H+": 1},
                           unit={"Asc": "m", "Asc-": "m", "H+": "m"},
                           equilibrium_constant=self.__ascorbic_acid_rxn_1_eq_const__)

        self.add_rxn_insta(id="Asc- = Asc-2 + H+",
                           stoch={"Asc-": -1, "Asc-2": 1, "H+": 1},
                           unit={"Asc-": "m", "Asc-2": "m", "H+": "m"},
                           equilibrium_constant=self.__ascorbic_acid_rxn_2_eq_const__)

    def __activity_coefficient__(self, LiquidStream, id):
        I = LiquidStream.get_solution_ionic_strength_mol_kg()
        z = LiquidStream.get_specie_charge(id)
        log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))
        gamma = 10 ** log10_gamma
        return gamma

    def __water_autoprotolysis_eq_constant__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        Kw = 10 ** (-14) * np.exp(-13445.9 * (1 / T - 1 / 298) - 22.48 * np.log(T / 298))
        return Kw

    def __ascorbic_acid_rxn_1_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-4.1) * np.exp(0 * (1 / T - 1 / 298.15))
        return K

    def __ascorbic_acid_rxn_2_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        K = 10 ** (-11.8) * np.exp(0 * (1 / T - 1 / 298.15))
        return K


class LiquidStream_Vanadium(lab.LiquidStream):

    def __init__(self, temp_K, flow_kg_h):

        super().__init__(stream_id="", solvent_id="H2O")
        z = np.zeros(shape=temp_K.shape)

        self.add_rxn_insta(id="H2VO4- = HVO4-2 + H+",
                           stoch={"H2VO4-": -1, "HVO4-2": 1, "H+": 1},
                           unit={"H2VO4-": "c", "HVO4-2": "c", "H+": "c"},
                           equilibrium_constant=self.__vanadium_rxn_1_eq_const__)

        self.add_rxn_insta(id="HVO4-2 = VO4-3 + H+",
                           stoch={"HVO4-2": -1, "VO4-3": 1, "H+": 1},
                           unit={"HVO4-2": "c", "VO4-3": "c", "H+": "c"},
                           equilibrium_constant=self.__vanadium_rxn_2_eq_const__)

        self.add_rxn_insta(id="HV2O7-3 = V2O7-4 + H+",
                           stoch={"HV2O7-3": -1, "V2O7-4": 1, "H+": 1},
                           unit={"HV2O7-3": "c", "V2O7-4": "c", "H+": "c"},
                           equilibrium_constant=self.__vanadium_rxn_3_eq_const__)

        self.add_rxn_insta(id="2HVO4-2 = V2O7-4 + H2O",
                           stoch={"HVO4-2": -2, "V2O7-4": 1, "H2O": 1},
                           unit={"HVO4-2": "c", "V2O7-4": "c", "H2O": "x"},
                           equilibrium_constant=self.__vanadium_rxn_4_eq_const__)

        self.add_rxn_insta(id="HV2O7-3 + HVO4-2 = V3O10-5 + H2O",
                           stoch={"HV2O7-3": -1, "HVO4-2": -1, "V3O10-5": 1, "H2O": 1},
                           unit={"HV2O7-3": "c", "HVO4-2": "c", "V3O10-5": "c", "H2O": "x"},
                           equilibrium_constant=self.__vanadium_rxn_5_eq_const__)

    def __vanadium_rxn_1_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        return 10 ** (-7.92) * np.exp(-1082 * (1 / T - 1 / 298.15))

    def __vanadium_rxn_2_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        return 10 ** (-13.26) * np.exp(-3247 * (1 / T - 1 / 298.15))

    def __vanadium_rxn_3_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        return 10 ** (-9.79) * np.exp(-841 * (1 / T - 1 / 298.15))

    def __vanadium_rxn_4_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        return 2.75 * np.exp(37747 * (1 / T - 1 / 298.15))

    def __vanadium_rxn_5_eq_const__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        return 1.325 * 10 ** 43 * np.exp(36784 * (1 / T - 1 / 298.15))


# ---------------------------------------------------------------------------------


class Absorber(lab.Column_StructuredPacking_CounterCurrent):

    def __init__(self, height_m=5.6, num_of_heights=90, cross_sectional_area_m2=0.5, void_fraction_m3_m3=0.98, packing_area_m2_m3=350, corrugation_angle_degree=60):

        super().__init__(height_m=height_m,
                         num_of_heights=num_of_heights,
                         cross_sectional_area_m2=cross_sectional_area_m2,
                         void_fraction_m3_m3=void_fraction_m3_m3,
                         packing_area_m2_m3=packing_area_m2_m3,
                         corrugation_angle_degree=corrugation_angle_degree)

        self.add_mass_transfer_kmol_m3s(id="CO2(g) -> CO2(aq)",
                                        stoch_gas={"CO2": -1},
                                        stoch_liq={"CO2": 1},
                                        rate_kmol_m3s=self.Mass_Transfer_CO2_kmol_m3s,
                                        exothermic_heat_kJ_kmol=self.Mass_Transfer_CO2_kJ_kmol)


        self.add_mass_transfer_kmol_m3s(id="H2O(g) -> H2O(aq)",
                                        stoch_gas={"H2O": -1},
                                        stoch_liq={"H2O": 1},
                                        rate_kmol_m3s=self.Mass_Transfer_H2O_kmol_m3s,
                                        exothermic_heat_kJ_kmol=self.Mass_Transfer_H2O_kJ_kmol)

        self.add_heat_transfer_kW_m3(heat_transfer_kW_m3=self.Heat_Transfer_kW_m3)
        self.add_liquid_holdup_m3_m3(liquid_holdup_m3_m3=self.Liquid_Holdup_m3_m3)
        self.add_pressure_drop_Pa_m(pressure_drop_Pa_m=None)

    def Mass_Transfer_CO2_kmol_m3s(self, Column):

        # Features
        ap = Column.get_packing_area_m2_m3()
        theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
        M = (3 * ap ** 3 * np.sin(theta) * np.cos(theta)) / (16 * (np.sin(theta) ** 2 + 1) ** (3 / 2))
        Mi = M / ap ** 3
        v_gas = Column.get_superficial_gas_velocity_m_s()
        v_liq = Column.get_superficial_liquid_velocity_m_s()
        m = Column.LiquidStream.get_solution_flow_kg_h()
        A = Column.get_cross_sectional_area_m2()
        T = Column.LiquidStreamIn.temp_K
        wA = Column.LiquidStream.get_info(id="Amine Mass Fraction")
        alpha = Column.LiquidStream.CO2Load(Column.LiquidStream)

        # Driving Force
        p_CO2 = Column.GasStream.get_specie_pressure_bara(id="CO2")
        p_CO2_vap = Column.LiquidStream.get_specie_vapor_pressure_bara(gas_id="CO2")

        # Henry's Law Coefficient
        H_CO2 = 1.153 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)


        # Overall Volumetric Mass Transfer Coefficient
        if Column.LiquidStream.id == "30% MEA":
            w_MEA = Column.LiquidStream.get_specie_mass_fraction(id="MEA")
            KGa = 1.16              # 1.16
            KGa = KGa * (ap / 350) ** 0.96
            KGa = KGa * (Mi / 0.035) ** 0.42
            KGa = KGa * (v_gas / 2.4) ** 0.27
            KGa = KGa * ((m / A) * (0.5 / 6000)) ** 0.19
            KGa = KGa * (w_MEA / 0.3) ** 0.75
            KGa = KGa * np.sqrt(np.exp(-5000 * (1 / T - 1 / 298)))      # 7000
            KGa = KGa * H_CO2

        if Column.LiquidStream.id == "30% KPro":
            w_Pro = Column.LiquidStream.get_specie_mass_fraction(id="Pro-")
            KGa = 1.16              # 1.16
            KGa = KGa * (ap / 350) ** 0.96
            KGa = KGa * (Mi / 0.035) ** 0.42
            KGa = KGa * (v_gas / 2.4) ** 0.27
            KGa = KGa * ((m / A) * (0.5 / 6000)) ** 0.19
            KGa = KGa * (w_Pro / 0.3) ** 0.75
            KGa = KGa * np.sqrt(np.exp(-5000 * (1 / T - 1 / 298)))      # 7000
            KGa = KGa * H_CO2

        if Column.LiquidStream.id == "45% KPro":
            w_Pro = Column.LiquidStream.get_specie_mass_fraction(id="Pro-")
            KGa = 1.16              # 1.16
            KGa = KGa * (ap / 350) ** 0.96
            KGa = KGa * (Mi / 0.035) ** 0.42
            KGa = KGa * (v_gas / 2.4) ** 0.27
            KGa = KGa * ((m / A) * (0.5 / 6000)) ** 0.19
            KGa = KGa * (w_Pro / 0.45) ** 0.75
            KGa = KGa * np.sqrt(np.exp(-5000 * (1 / T - 1 / 298)))      # 7000
            KGa = KGa * H_CO2

        if Column.LiquidStream.id == "60% KPro":
            w_Pro = Column.LiquidStream.get_specie_mass_fraction(id="Pro-")
            KGa = 2*1.16              # 1.16
            KGa = KGa * (ap / 350) ** 0.96
            KGa = KGa * (Mi / 0.035) ** 0.42
            KGa = KGa * (v_gas / 2.4) ** 0.27
            KGa = KGa * ((m / A) * (0.5 / 6000)) ** 0.19
            KGa = KGa * (w_Pro / 0.6) ** 0.75
            KGa = KGa * np.sqrt(np.exp(-5000 * (1 / T - 1 / 298)))      # 7000
            KGa = KGa * H_CO2

        if Column.LiquidStream.id == "30% KPro + 30% KSar":
            w_Pro = Column.LiquidStream.get_specie_mass_fraction(id="Pro-")
            w_Sar = Column.LiquidStream.get_specie_mass_fraction(id="Sar-")
            KGa = 2 * 1.16              # 1.16
            KGa = KGa * (ap / 350) ** 0.96
            KGa = KGa * (Mi / 0.035) ** 0.42
            KGa = KGa * (v_gas / 2.4) ** 0.27
            KGa = KGa * ((m / A) * (0.5 / 6000)) ** 0.19
            KGa = KGa * (w_Pro / 0.3 + w_Sar/0.3) ** 0.75
            KGa = KGa * np.sqrt(np.exp(-5000 * (1 / T - 1 / 298)))      # 7000
            KGa = KGa * H_CO2

        if Column.LiquidStream.id == "45% MEA":
            w_MEA = Column.LiquidStream.get_specie_mass_fraction(id="MEA")
            KGa = 0.92
            KGa = KGa * (ap / 350) ** 0.96
            KGa = KGa * (Mi / 0.035) ** 0.42
            KGa = KGa * (v_gas / 2.4) ** 0.27
            KGa = KGa * ((m / A) * (0.5 / 6000)) ** 0.19
            KGa = KGa * (w_MEA / 0.45) ** 0.75
            KGa = KGa * np.sqrt(np.exp(-5000 * (1 / T - 1 / 298)))
            KGa = KGa * H_CO2

        if Column.LiquidStream.id == "28% AMP":
            w_AMP = Column.LiquidStream.get_specie_mass_fraction(id="AMP")
            KGa = 0.22
            KGa = KGa * (ap / 350) ** 0.96
            KGa = KGa * (Mi / 0.035) ** 0.42
            KGa = KGa * (v_gas / 2.4) ** 0.27
            KGa = KGa * ((m / A) * (0.5 / 6000)) ** 0.19
            KGa = KGa * (w_AMP / 0.28) ** 0.75
            KGa = KGa * np.sqrt(np.exp(-5000 * (1 / T - 1 / 298)))
            KGa = KGa * H_CO2

        if Column.LiquidStream.id == "15% PZ":
            w_PZ = Column.LiquidStream.get_specie_mass_fraction(id="PZ")
            KGa = 10.5
            KGa = KGa * (ap / 350) ** 0.96
            KGa = KGa * (Mi / 0.035) ** 0.42
            KGa = KGa * (v_gas / 2.4) ** 0.27
            KGa = KGa * ((m / A) * (0.5 / 6000)) ** 0.19
            KGa = KGa * (w_PZ / 0.15) ** 1.0
            KGa = KGa * np.sqrt(np.exp(-5000 * (1 / T - 1 / 298)))
            KGa = KGa * H_CO2

        if Column.LiquidStream.id == "34% MDEA + 10% PZ":
            w_PZ = Column.LiquidStream.get_specie_mass_fraction(id="PZ")
            w_PZCOO = Column.LiquidStream.get_specie_mass_fraction(id="PZCOO-")
            KGa = 5.5
            KGa = KGa * (ap / 350) ** 0.96
            KGa = KGa * (Mi / 0.035) ** 0.42
            KGa = KGa * (v_gas / 2.4) ** 0.27
            KGa = KGa * ((m / A) * (0.5 / 6000)) ** 0.19
            KGa = KGa * ((w_PZ  + 0.1 * w_PZCOO) / 0.10) ** 1.0
            KGa = KGa * np.sqrt(np.exp(-5000 * (1 / T - 1 / 298)))
            KGa = KGa * H_CO2

        if Column.LiquidStream.id == "22% AMP + 13% PZ":
            w_PZ = Column.LiquidStream.get_specie_mass_fraction(id="PZ")
            w_PZCOO = Column.LiquidStream.get_specie_mass_fraction(id="PZCOO-")
            KGa = 5.5
            KGa = KGa * (ap / 350) ** 0.96
            KGa = KGa * (Mi / 0.035) ** 0.42
            KGa = KGa * (v_gas / 2.4) ** 0.27
            KGa = KGa * ((m / A) * (0.5 / 6000)) ** 0.19
            KGa = KGa * ((w_PZ + 0.0 * w_PZCOO) / 0.13) ** 1.0
            KGa = KGa * np.sqrt(np.exp(-5000 * (1 / T - 1 / 298)))
            KGa = KGa * H_CO2


        # Absorption Rate
        r = KGa * (p_CO2 - p_CO2_vap)
        return r

    def Mass_Transfer_CO2_kJ_kmol(self, Column):
        T0 = Column.LiquidStream.temp_K
        H0 = Column.LiquidStream.vapor_pressure_bara["CO2(g) = CO2(aq)"]["H"](Column.LiquidStream)
        Column.LiquidStream.temp_K = Column.LiquidStream.temp_K + 0.05
        T1 = Column.LiquidStream.temp_K
        H1 = Column.LiquidStream.vapor_pressure_bara["CO2(g) = CO2(aq)"]["H"](Column.LiquidStream)
        Column.LiquidStream.temp_K = Column.LiquidStream.temp_K - 0.05
        q = 8.314 * (np.log(H1) - np.log(H0)) / ((1 / T1) - (1 / T0))
        return q

    def Mass_Transfer_H2O_kmol_m3s(self, Column):

        # Features
        ap = Column.get_packing_area_m2_m3()
        theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
        M = (3 * ap ** 3 * np.sin(theta) * np.cos(theta)) / (16 * (np.sin(theta) ** 2 + 1) ** (3 / 2))
        Mi = M / ap ** 3
        v_gas = Column.get_superficial_gas_velocity_m_s()
        v_liq = Column.get_superficial_liquid_velocity_m_s()
        m = Column.LiquidStream.get_solution_flow_kg_h()
        A = Column.get_cross_sectional_area_m2()

        # Driving Force
        p_H2O = Column.GasStream.get_specie_pressure_bara(id="H2O")
        p_H2O_vap = Column.LiquidStream.get_specie_vapor_pressure_bara(gas_id="H2O")

        # Overall Volumetric Mass Transfer Coefficient
        KGa = 0.325 * ((m/6000)*(0.5/A))**0.15 * (v_gas/2.4)**0.54 * (Mi/0.035)**0.29 * (ap/350)**1.22

        r = KGa * (p_H2O - p_H2O_vap)
        return r

    def Mass_Transfer_H2O_kJ_kmol(self, Column):
        T0 = Column.LiquidStream.temp_K
        p0 = Column.LiquidStream.vapor_pressure_bara["H2O(g) = H2O(l)"]["p0"](Column.LiquidStream)
        Column.LiquidStream.temp_K = Column.LiquidStream.temp_K + 0.05
        T1 = Column.LiquidStream.temp_K
        p1 = Column.LiquidStream.vapor_pressure_bara["H2O(g) = H2O(l)"]["p0"](Column.LiquidStream)
        Column.LiquidStream.temp_K = Column.LiquidStream.temp_K - 0.05
        h = 8.314 * (np.log(1 / p1) - np.log(1 / p0)) / ((1 / T1) - (1 / T0))
        return h

    def Heat_Transfer_kW_m3(self, Column):

        # Features
        ap = Column.get_packing_area_m2_m3()
        theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
        M = (3 * ap ** 3 * np.sin(theta) * np.cos(theta)) / (16 * (np.sin(theta) ** 2 + 1) ** (3 / 2))
        Mi = M / ap ** 3
        m = Column.LiquidStream.get_solution_flow_kg_h()
        A = Column.get_cross_sectional_area_m2()
        T_gas = Column.GasStream.get_gas_temp_K()
        T_liq = Column.LiquidStream.get_solution_temp_K()
        v_gas = Column.get_superficial_gas_velocity_m_s()

        # Heat Transfer Coefficient
        kHa = 11.14 * ((m/6000)*(0.5/A))**0.15 * (v_gas/2.4)**0.54 * (Mi/0.035)**0.29 * (ap/350)**1.22

        # Heat Transfer [kW/m3]
        q = kHa * (T_gas - T_liq)
        return q

    def Liquid_Holdup_m3_m3(self, Column):
        theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
        ap = Column.get_packing_area_m2_m3()
        mu = 0.89 * 10 ** (-3)
        rho = Column.LiquidStream.get_solution_density_kg_m3()
        v_LS = Column.get_superficial_liquid_velocity_m_s()
        g = 9.81
        enu = 3 * np.abs(v_LS) * mu * ap ** 2
        den = rho * g * np.sin(theta) ** 2
        h_liq = (enu / den) ** (1 / 3)
        return h_liq


class WashWaterSection(lab.Column_StructuredPacking_CounterCurrent):

    def __init__(self, height_m=1.6, num_of_heights=90, cross_sectional_area_m2=0.5, void_fraction_m3_m3=0.98, packing_area_m2_m3=350, corrugation_angle_degree=60):

        super().__init__(height_m=height_m,
                         num_of_heights=num_of_heights,
                         cross_sectional_area_m2=cross_sectional_area_m2,
                         void_fraction_m3_m3=void_fraction_m3_m3,
                         packing_area_m2_m3=packing_area_m2_m3,
                         corrugation_angle_degree=corrugation_angle_degree)

        self.add_mass_transfer_kmol_m3s(id="H2O(g) -> H2O(aq)",
                                        stoch_gas={"H2O": -1},
                                        stoch_liq={"H2O": 1},
                                        rate_kmol_m3s=self.Mass_Transfer_H2O_kmol_m3s,
                                        exothermic_heat_kJ_kmol=self.Mass_Transfer_H2O_kJ_kmol)

        self.add_heat_transfer_kW_m3(heat_transfer_kW_m3=self.Heat_Transfer_kW_m3)
        self.add_liquid_holdup_m3_m3(liquid_holdup_m3_m3=self.Liquid_Holdup_m3_m3)
        self.add_pressure_drop_Pa_m(pressure_drop_Pa_m=None)

    def Mass_Transfer_H2O_kmol_m3s(self, Column):

        # Features
        ap = Column.get_packing_area_m2_m3()
        theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
        M = (3 * ap ** 3 * np.sin(theta) * np.cos(theta)) / (16 * (np.sin(theta) ** 2 + 1) ** (3 / 2))
        Mi = M / ap ** 3
        v_gas = Column.get_superficial_gas_velocity_m_s()
        v_liq = Column.get_superficial_liquid_velocity_m_s()
        m = Column.LiquidStream.get_solution_flow_kg_h()
        A = Column.get_cross_sectional_area_m2()

        # Driving Force
        p_H2O = Column.GasStream.get_specie_pressure_bara(id="H2O")
        p_H2O_vap = Column.LiquidStream.get_specie_vapor_pressure_bara(gas_id="H2O")

        # Overall Volumetric Mass Transfer Coefficient
        KGa = 0.325 * ((m/6000)*(0.5/A))**0.15 * (v_gas/2.4)**0.54 * (Mi/0.035)**0.29 * (ap/350)**1.22

        r = KGa * (p_H2O - p_H2O_vap)
        return r

    def Mass_Transfer_H2O_kJ_kmol(self, Column):
        T0 = Column.LiquidStream.temp_K
        p0 = Column.LiquidStream.vapor_pressure_bara["H2O(g) = H2O(l)"]["p0"](Column.LiquidStream)
        Column.LiquidStream.temp_K = Column.LiquidStream.temp_K + 0.05
        T1 = Column.LiquidStream.temp_K
        p1 = Column.LiquidStream.vapor_pressure_bara["H2O(g) = H2O(l)"]["p0"](Column.LiquidStream)
        Column.LiquidStream.temp_K = Column.LiquidStream.temp_K - 0.05
        h = 8.314 * (np.log(1 / p1) - np.log(1 / p0)) / ((1 / T1) - (1 / T0))
        return h

    def Heat_Transfer_kW_m3(self, Column):

        # Features
        ap = Column.get_packing_area_m2_m3()
        theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
        M = (3 * ap ** 3 * np.sin(theta) * np.cos(theta)) / (16 * (np.sin(theta) ** 2 + 1) ** (3 / 2))
        Mi = M / ap ** 3
        m = Column.LiquidStream.get_solution_flow_kg_h()
        A = Column.get_cross_sectional_area_m2()
        T_gas = Column.GasStream.get_gas_temp_K()
        T_liq = Column.LiquidStream.get_solution_temp_K()
        v_gas = Column.get_superficial_gas_velocity_m_s()

        # Heat Transfer Coefficient
        kHa = 11.14 * ((m/6000)*(0.5/A))**0.15 * (v_gas/2.4)**0.54 * (Mi/0.035)**0.29 * (ap/350)**1.22

        # Heat Transfer [kW/m3]
        q = kHa * (T_gas - T_liq)
        return q

    def Liquid_Holdup_m3_m3(self, Column):
        theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
        ap = Column.get_packing_area_m2_m3()
        mu = 0.89 * 10 ** (-3)
        rho = Column.LiquidStream.get_solution_density_kg_m3()
        v_LS = Column.get_superficial_liquid_velocity_m_s()
        g = 9.81
        enu = 3 * np.abs(v_LS) * mu * ap ** 2
        den = rho * g * np.sin(theta) ** 2
        h_liq = (enu / den) ** (1 / 3)
        return h_liq


class HeatExchanger(lab.LiquidHeatExchanger_CounterCurrent):

    def __init__(self, interface_area_m2=18.52):
        super().__init__(area_m2=interface_area_m2)
        self.load_heat_transfer_coefficient_kW_m2K(function=self.Heat_Transfer_Coefficient_kW_m2K)

    def Heat_Transfer_Coefficient_kW_m2K(self, HX):
        A = HX.get_interface_area_m2()
        m1 = np.abs(HX.LiquidStream1.get_solution_flow_kg_h())
        m2 = np.abs(HX.LiquidStream2.get_solution_flow_kg_h())
        w1 = HX.LiquidStream1.get_info(id="Amine Mass Fraction")
        w2 = HX.LiquidStream2.get_info(id="Amine Mass Fraction")
        k1 = 4.1 * np.exp(-4.65 * w1**2.7) * (18.52 / A) ** 0.8 * (m1 / 5000) ** 0.8
        k2 = 4.1 * np.exp(-4.65 * w1**2.7) * (18.52 / A) ** 0.8 * (m2 / 5000) ** 0.8
        k = (1 / k1 + 1 / k2) ** (-1)
        return k


class QPFlash_StenMartin(lab.Serializer):

    def __init__(self):
        self.flash = lab.LiquidEquilibrium_QPFlash()

    def react(self, LiquidStreamIn, ratio_to_hx, pressure_bara, flash_always_occur, lr):
        L2HX = deepcopy(LiquidStreamIn)
        L2FL = deepcopy(LiquidStreamIn)
        L2HX.flow_kg_h = LiquidStreamIn.flow_kg_h * ratio_to_hx
        L2FL.flow_kg_h = LiquidStreamIn.flow_kg_h * (1 - ratio_to_hx)

        shape = np.ones(shape=LiquidStreamIn.temp_K.shape)
        heat_kW = 0.0 * shape

        flash_vapor, flash_condense = self.flash.react(L2FL, heat_kW=heat_kW, pressure_bara=pressure_bara, lr=lr, flash_always_occur=flash_always_occur)

        # Heat Transfer
        T = (flash_condense.flow_kg_h * flash_condense.temp_K + L2HX.flow_kg_h * L2HX.temp_K) / (flash_condense.flow_kg_h + L2HX.flow_kg_h)
        flash_condense.temp_K = 1.0 * T
        L2HX.temp_K = 1.0 * T
        flash_vapor_2, flash_condense = self.flash.react(L2FL, heat_kW=heat_kW, pressure_bara=pressure_bara, lr=lr, flash_always_occur=flash_always_occur)

        # Heat Transfer
        T = (flash_condense.flow_kg_h * flash_condense.temp_K + L2HX.flow_kg_h * L2HX.temp_K) / (flash_condense.flow_kg_h + L2HX.flow_kg_h)
        flash_condense.temp_K = 1.0 * T
        L2HX.temp_K = 1.0 * T
        flash_vapor_3, flash_condense = self.flash.react(L2FL, heat_kW=heat_kW, pressure_bara=pressure_bara, lr=lr, flash_always_occur=flash_always_occur)

        # Heat Transfer
        T = (flash_condense.flow_kg_h * flash_condense.temp_K + L2HX.flow_kg_h * L2HX.temp_K) / (flash_condense.flow_kg_h + L2HX.flow_kg_h)
        flash_condense.temp_K = 1.0 * T
        L2HX.temp_K = 1.0 * T
        flash_vapor_4, flash_condense = self.flash.react(L2FL, heat_kW=heat_kW, pressure_bara=pressure_bara, lr=lr, flash_always_occur=flash_always_occur)

        # Outlet...
        flash_vapor = lab.GasStreamSum(streams=[flash_vapor, flash_vapor_2, flash_vapor_3, flash_vapor_4])
        liq_out = lab.LiquidStreamSum(streams=[L2HX, flash_condense])

        return flash_vapor, liq_out



# ---------------------------------------------------------------------------------


class CCS_equilibrium(lab.Serializer):

    def __init__(self, absorber, stripper, hx, reboiler, solvent):
        self.firstscan = True
        self.absorber = absorber
        self.stripper = stripper
        self.hx = hx
        self.reboiler = reboiler
        self.solvent = solvent
        self.equilibrium = lab.LiquidEquilibrium_Isothermal()
        self.nozzle = lab.LiquidEquilibrium_QPFlash()

    def react(self, gas_in, lean_cold_CO2Load_init, lean_cold_flow_kg_h, lean_cold_temp_init_C, heat_kW, make_up_water_temp_C, reflux_temp_C, reboiler_temp_init_C, stripper_pressure_bara, epochs, lr):

        num_of_samples = make_up_water_temp_C.shape[0]
        shape = np.ones(shape=(num_of_samples,))

        self.gas_in = gas_in
        self.heat_kW = heat_kW
        self.heat_dissipated_pct = 5

        if self.firstscan:

            self.lean_cold = self.solvent(temp_K=273.15 + lean_cold_temp_init_C, flow_kg_h=lean_cold_flow_kg_h, CO2Load=lean_cold_CO2Load_init)

            print("Init")
            print("- Lean Cold Processing")
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber Init")
            self.gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in,
                                                               LiquidStreamIn=self.lean_cold)

            make_up_water = LiquidStream_Water(temp_K=make_up_water_temp_C + 273.15, flow_kg_h=np.maximum(self.gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0))
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, make_up_water])

            print("- Heat Exchanger")
            self.lean_hot = deepcopy(self.lean_cold)
            self.lean_hot.temp_K = reboiler_temp_init_C + 273.15
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                                temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Reboiler Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.nozzle_condens, self.heat_kW * (100 - self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Stripper Init")
            self.stripper_vapor, self.stripper_condense = self.stripper.react(GasStreamIn=self.reboiler_vapor,
                                                                              LiquidStreamIn=self.nozzle_condens)

            print("- Condenser Init")
            self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor])
            self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape,
                                      flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Reboiler Re-Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_condense, self.heat_kW * (100 - self.heat_dissipated_pct) / 100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold,
                                                          LiquidStreamIn2=self.lean_hot)

        self.firstscan = False


        for epoch in range(epochs):

            print("Epoch " + str(epoch+1) + " of " + str(epochs))

            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=self.lean_cold.temp_K, flow_kg_h=lean_cold_flow_kg_h, CO2Load=self.lean_cold.CO2Load(self.lean_cold))
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber")
            self.gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in,
                                                               LiquidStreamIn=self.lean_cold)

            self.make_up_water = LiquidStream_Water(temp_K=make_up_water_temp_C + 273.15, flow_kg_h=np.maximum(self.gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0))
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, self.make_up_water])

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                              pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                              temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                              CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                              H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            for _ in range(3):

                print("- Stripper")
                stripper_liquid_inlet = lab.LiquidStreamSum(streams=[self.nozzle_condens, self.reflux])
                stripper_liquid_inlet = self.equilibrium.react(stripper_liquid_inlet, lr=lr)
                self.stripper_vapor, self.stripper_condense = self.stripper.react(self.reboiler_vapor, stripper_liquid_inlet)

                print("- Reboiler")
                self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_condense,
                                                                        self.heat_kW * (100 - self.heat_dissipated_pct) / 100,
                                                                        stripper_pressure_bara,
                                                                        lr=lr,
                                                                        flash_always_occur=True)

                self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                    pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                    temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                    CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                    H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

                print("- Condenser")
                self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor])
                self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape, flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.lean_hot)


class CCS_baseline(lab.Serializer):

    def __init__(self, absorber, stripper, rectifier, hx, reboiler, solvent):
        self.firstscan = True
        self.absorber = absorber
        self.stripper = stripper
        self.rectifier = rectifier
        self.hx = hx
        self.reboiler = reboiler
        self.solvent = solvent
        self.lean_cold_cstr = lab.LiquidEquilibrium_Isothermal()
        self.nozzle = lab.LiquidEquilibrium_QPFlash()

    def react(self, gas_in, lean_cold_CO2Load_init, lean_cold_flow_kg_h, lean_cold_temp_init_C, heat_kW, make_up_water_temp_C, reflux_temp_C, reboiler_temp_init_C, stripper_pressure_bara, heat_dissipated_kW, epochs, lr):

        num_of_samples = make_up_water_temp_C.shape[0]
        shape = np.ones(shape=(num_of_samples,))

        self.gas_in = gas_in
        self.heat_kW = heat_kW
        self.heat_dissipated_pct = 100 * heat_dissipated_kW / self.heat_kW

        if self.firstscan:

            self.lean_cold = self.solvent(temp_K=273.15 + lean_cold_temp_init_C, flow_kg_h=lean_cold_flow_kg_h, CO2Load=lean_cold_CO2Load_init)

            print("Init")
            print("- Lean Cold Processing")
            self.lean_cold = self.lean_cold_cstr.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber Init")
            self.gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2 * self.absorber.num_of_heights, lr=lr)
            make_up_water = LiquidStream_Water(temp_K=make_up_water_temp_C + 273.15, flow_kg_h=np.maximum(self.gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0))
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, make_up_water])

            print("- Heat Exchanger")
            self.lean_hot = deepcopy(self.lean_cold)
            self.lean_hot.temp_K = reboiler_temp_init_C + 273.15
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = lab.LiquidEquilibrium_Isothermal().react(self.rich_hot, lr=lr)

            print("- Nozzle Flash")
            nozzle_vapor_ref, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=nozzle_vapor_ref.get_gas_flow_kmol_h(),
                                                pressure_bara=nozzle_vapor_ref.get_gas_pressure_bara(),
                                                temp_K=nozzle_vapor_ref.get_gas_temp_K(),
                                                CO2_molar_fraction=nozzle_vapor_ref.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=nozzle_vapor_ref.get_specie_molar_fraction(id="H2O"))



            print("- Reboiler Init")
            reboiler_vapor_ref, self.lean_hot = self.reboiler.react(self.nozzle_condens, self.heat_kW * (100 - self.heat_dissipated_pct) / 100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=reboiler_vapor_ref.get_gas_flow_kmol_h(),
                                                pressure_bara=reboiler_vapor_ref.get_gas_pressure_bara(),
                                                temp_K=reboiler_vapor_ref.get_gas_temp_K(),
                                                CO2_molar_fraction=reboiler_vapor_ref.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=reboiler_vapor_ref.get_specie_molar_fraction(id="H2O"))

            print("- Stripper Init")
            #self.stripper_vapor, self.stripper_condense = self.stripper.react(GasStreamIn=self.reboiler_vapor,
            #                                                                  LiquidStreamIn=self.nozzle_condens,
            #                                                                  epochs=2 * self.stripper.num_of_heights,
            #                                                                  lr=lr)

            self.stripper_vapor, self.stripper_condense = self.stripper.react(GasStreamIn=self.reboiler_vapor,
                                                                              LiquidStreamIn=self.nozzle_condens)


            print("- Condenser Init")
            self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor])
            self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape,
                                      flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Rectifier Init")
            self.rectifier_vapor, self.rectifier_condense = self.rectifier.react(
                GasStreamIn=lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor]),
                LiquidStreamIn=self.reflux)


            print("- Reboiler Re-Init")
            reboiler_vapor_ref, self.lean_hot = self.reboiler.react(self.stripper_condense, self.heat_kW * (100 - self.heat_dissipated_pct) / 100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=reboiler_vapor_ref.get_gas_flow_kmol_h(),
                                                pressure_bara=reboiler_vapor_ref.get_gas_pressure_bara(),
                                                temp_K=reboiler_vapor_ref.get_gas_temp_K(),
                                                CO2_molar_fraction=reboiler_vapor_ref.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=reboiler_vapor_ref.get_specie_molar_fraction(id="H2O"))

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold,
                                                          LiquidStreamIn2=self.lean_hot)

        self.firstscan = False


        for epoch in range(epochs):

            print("Epoch " + str(epoch+1) + " of " + str(epochs))

            self.lean_cold = self.solvent(temp_K=self.lean_cold.temp_K, flow_kg_h=lean_cold_flow_kg_h, CO2Load=self.lean_cold.CO2Load(self.lean_cold))

            print("- Lean Cold Processing")
            self.lean_cold = self.lean_cold_cstr.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber")
            self.gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2*self.absorber.num_of_heights, lr=lr)
            self.make_up_water = LiquidStream_Water(temp_K=make_up_water_temp_C + 273.15, flow_kg_h=np.maximum(self.gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0))
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, self.make_up_water])

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.lean_hot)

            print("- Nozzle Flash")
            nozzle_vapor_ref, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=nozzle_vapor_ref.get_gas_flow_kmol_h(),
                                              pressure_bara=nozzle_vapor_ref.get_gas_pressure_bara(),
                                              temp_K=nozzle_vapor_ref.get_gas_temp_K(),
                                              CO2_molar_fraction=nozzle_vapor_ref.get_specie_molar_fraction(id="CO2"),
                                              H2O_molar_fraction=nozzle_vapor_ref.get_specie_molar_fraction(id="H2O"))

            for _ in range(3):

                print("- Stripper")
                #self.stripper_vapor, self.stripper_condense = self.stripper.react(GasStreamIn=self.reboiler_vapor,
                #                                                                  LiquidStreamIn=lab.LiquidStreamSum(streams=[self.nozzle_condens,self.rectifier_condense]),
                #                                                                  epochs=2 * self.stripper.num_of_heights,
                #                                                                  lr=lr)

                self.stripper_vapor, self.stripper_condense = self.stripper.react(GasStreamIn=self.reboiler_vapor,
                                                                                  LiquidStreamIn=lab.LiquidStreamSum(
                                                                                      streams=[self.nozzle_condens,
                                                                                               self.rectifier_condense]))

                print("- Reboiler")
                reboiler_vapor_ref, self.lean_hot = self.reboiler.react(self.stripper_condense,
                                                                        self.heat_kW * (100 - self.heat_dissipated_pct) / 100,
                                                                        stripper_pressure_bara,
                                                                        lr=lr,
                                                                        flash_always_occur=True)
                self.reboiler_vapor = ReboilerVapor(flow_kmol_h=reboiler_vapor_ref.get_gas_flow_kmol_h(),
                                                    pressure_bara=reboiler_vapor_ref.get_gas_pressure_bara(),
                                                    temp_K=reboiler_vapor_ref.get_gas_temp_K(),
                                                    CO2_molar_fraction=reboiler_vapor_ref.get_specie_molar_fraction(id="CO2"),
                                                    H2O_molar_fraction=reboiler_vapor_ref.get_specie_molar_fraction(id="H2O"))

                print("- Rectifier")
                self.rectifier_vapor, self.rectifier_condense = self.rectifier.react(
                    GasStreamIn=lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor]),
                    LiquidStreamIn=self.reflux)

                print("- Condenser")
                self.condenser_vapor = deepcopy(self.rectifier_vapor)
                self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape,
                                          flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.lean_hot)


class CCS_no_rectifier(lab.Serializer):

    def __init__(self, absorber, stripper, hx, reboiler, solvent):
        self.firstscan = True
        self.absorber = absorber
        self.stripper = stripper
        self.hx = hx
        self.reboiler = reboiler
        self.solvent = solvent
        self.equilibrium = lab.LiquidEquilibrium_Isothermal()
        self.nozzle = lab.LiquidEquilibrium_QPFlash()

    def react(self, gas_in, lean_cold_CO2Load_init, lean_cold_flow_kg_h, lean_cold_temp_init_C, heat_kW, make_up_water_temp_C, reflux_temp_C, reboiler_temp_init_C, stripper_pressure_bara, epochs, lr):

        num_of_samples = make_up_water_temp_C.shape[0]
        shape = np.ones(shape=(num_of_samples,))

        self.gas_in = gas_in
        self.heat_kW = heat_kW
        self.heat_dissipated_pct = 5

        if self.firstscan:

            self.lean_cold = self.solvent(temp_K=273.15 + lean_cold_temp_init_C, flow_kg_h=lean_cold_flow_kg_h, CO2Load=lean_cold_CO2Load_init)

            print("Init")
            print("- Lean Cold Processing")
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber Init")
            self.gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2 * self.absorber.num_of_heights, lr=lr)
            make_up_water = LiquidStream_Water(temp_K=make_up_water_temp_C + 273.15, flow_kg_h=np.maximum(self.gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0))
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, make_up_water])

            print("- Heat Exchanger")
            self.lean_hot = deepcopy(self.lean_cold)
            self.lean_hot.temp_K = reboiler_temp_init_C + 273.15
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                                temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Reboiler Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.nozzle_condens, self.heat_kW * (100 - self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Stripper Init")
            self.stripper_vapor, self.stripper_condense = self.stripper.react(GasStreamIn=self.reboiler_vapor,
                                                                              LiquidStreamIn=self.nozzle_condens)

            print("- Condenser Init")
            self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor])
            self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape,
                                      flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Reboiler Re-Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_condense, self.heat_kW * (100 - self.heat_dissipated_pct) / 100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold,
                                                          LiquidStreamIn2=self.lean_hot)

        self.firstscan = False


        for epoch in range(epochs):

            print("Epoch " + str(epoch+1) + " of " + str(epochs))

            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=self.lean_cold.temp_K, flow_kg_h=lean_cold_flow_kg_h, CO2Load=self.lean_cold.CO2Load(self.lean_cold))
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber")
            self.gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2*self.absorber.num_of_heights, lr=lr)
            self.make_up_water = LiquidStream_Water(temp_K=make_up_water_temp_C + 273.15, flow_kg_h=np.maximum(self.gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0))
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, self.make_up_water])

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                              pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                              temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                              CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                              H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            for _ in range(3):

                print("- Stripper")
                stripper_liquid_inlet = lab.LiquidStreamSum(streams=[self.nozzle_condens, self.reflux])
                stripper_liquid_inlet = self.equilibrium.react(stripper_liquid_inlet, lr=lr)
                self.stripper_vapor, self.stripper_condense = self.stripper.react(self.reboiler_vapor, stripper_liquid_inlet)

                print("- Reboiler")
                self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_condense,
                                                                        self.heat_kW * (100 - self.heat_dissipated_pct) / 100,
                                                                        stripper_pressure_bara,
                                                                        lr=lr,
                                                                        flash_always_occur=True)

                self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                    pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                    temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                    CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                    H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

                print("- Condenser")
                self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor])
                self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape, flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.lean_hot)


class CCS_rich_preheat(lab.Serializer):

    def __init__(self, absorber, stripper, rectifier, washwater, hx, hx_ww, reboiler, solvent):
        self.firstscan = True
        self.absorber = absorber
        self.stripper = stripper
        self.rectifier = rectifier
        self.washwater = washwater
        self.hx = hx
        self.hx_ww = hx_ww
        self.reboiler = reboiler
        self.solvent = solvent
        self.equilibrium = lab.LiquidEquilibrium_Isothermal()
        self.nozzle = lab.LiquidEquilibrium_QPFlash()

    def react(self, gas_in, ww_flow_kg_h, ww_temp_C, lean_cold_CO2Load_init, lean_cold_flow_kg_h, lean_cold_temp_init_C, heat_kW, make_up_water_temp_C, reflux_temp_C, reboiler_temp_init_C, stripper_pressure_bara, heat_dissipated_kW, epochs, lr):

        num_of_samples = make_up_water_temp_C.shape[0]
        shape = np.ones(shape=(num_of_samples,))

        self.gas_in = gas_in
        self.heat_kW = heat_kW
        self.heat_dissipated_pct = 100 * heat_dissipated_kW / self.heat_kW

        if self.firstscan:

            self.lean_cold = self.solvent(temp_K=273.15 + lean_cold_temp_init_C, flow_kg_h=lean_cold_flow_kg_h, CO2Load=lean_cold_CO2Load_init)

            print("Init")
            print("- Lean Cold Processing")
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber Init")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2 * self.absorber.num_of_heights, lr=lr)

            print("- Wash Water Init")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2*self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water Init")
            water_loss = np.maximum(self.gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0)
            make_up_water = LiquidStream_Water(temp_K=make_up_water_temp_C + 273.15, flow_kg_h=water_loss)
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, make_up_water])

            print("- Heat Exchanger (1)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)

            print("- Heat Exchanger (2)")
            self.lean_hot = deepcopy(self.lean_cold)
            self.lean_hot.temp_K = reboiler_temp_init_C + 273.15
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                                temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Reboiler Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.nozzle_condens, self.heat_kW * (100 - self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Stripper Init")
            self.stripper_vapor, self.stripper_condense = self.stripper.react(GasStreamIn=self.reboiler_vapor,
                                                                              LiquidStreamIn=self.nozzle_condens)

            print("- Condenser Init")
            self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor])
            self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape,
                                      flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Rectifier Init")
            self.rectifier_vapor, self.rectifier_condense = self.rectifier.react(GasStreamIn=lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor]), LiquidStreamIn=self.reflux)


            print("- Reboiler Re-Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_condense, self.heat_kW * (100 - self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_hot)

        self.firstscan = False


        for epoch in range(epochs):

            print("Epoch " + str(epoch+1) + " of " + str(epochs))

            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=self.lean_cold.temp_K, flow_kg_h=lean_cold_flow_kg_h, CO2Load=self.lean_cold.CO2Load(self.lean_cold))
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2*self.absorber.num_of_heights, lr=lr)

            print("- Wash Water")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2 * self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water")
            self.make_up_water = LiquidStream_Water(temp_K=make_up_water_temp_C + 273.15, flow_kg_h=np.maximum(self.gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0))
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, self.make_up_water])

            print("- Heat Exchanger (Wash Water)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                              pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                              temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                              CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                              H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            for _ in range(3):

                print("- Stripper")
                self.stripper_vapor, self.stripper_condense = self.stripper.react(self.reboiler_vapor,
                                                                                  lab.LiquidStreamSum(
                                                                                      streams=[self.nozzle_condens,
                                                                                               self.rectifier_condense]))

                print("- Reboiler")
                reboiler_vapor_ref, self.lean_hot = self.reboiler.react(self.stripper_condense,
                                                                        self.heat_kW * (
                                                                                    100 - self.heat_dissipated_pct) / 100,
                                                                        stripper_pressure_bara,
                                                                        lr=lr,
                                                                        flash_always_occur=True)
                self.reboiler_vapor = ReboilerVapor(flow_kmol_h=reboiler_vapor_ref.get_gas_flow_kmol_h(),
                                                    pressure_bara=reboiler_vapor_ref.get_gas_pressure_bara(),
                                                    temp_K=reboiler_vapor_ref.get_gas_temp_K(),
                                                    CO2_molar_fraction=reboiler_vapor_ref.get_specie_molar_fraction(
                                                        id="CO2"),
                                                    H2O_molar_fraction=reboiler_vapor_ref.get_specie_molar_fraction(
                                                        id="H2O"))

                print("- Rectifier")
                self.rectifier_vapor, self.rectifier_condense = self.rectifier.react(
                    GasStreamIn=lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor]),
                    LiquidStreamIn=self.reflux)

                print("- Condenser")
                self.condenser_vapor = deepcopy(self.rectifier_vapor)
                self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape,
                                                 flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_hot)


class CCS_lean_vapor_recompression(lab.Serializer):

    def __init__(self, absorber, stripper, washwater, hx, hx_ww, reboiler, solvent):
        self.firstscan = True
        self.absorber = absorber
        self.stripper = stripper
        self.washwater = washwater
        self.hx = hx
        self.hx_ww = hx_ww
        self.reboiler = reboiler
        self.solvent = solvent
        self.equilibrium = lab.LiquidEquilibrium_Isothermal()
        self.nozzle = lab.LiquidEquilibrium_QPFlash()
        self.flashtank = lab.LiquidEquilibrium_QPFlash()
        self.compressor = lab.GasCompressor_Isentropic()

    def react(self, gas_in, ww_flow_kg_h, ww_temp_C, lean_cold_CO2Load_init, lean_cold_flow_kg_h, lean_cold_temp_init_C, heat_kW, make_up_water_temp_C, reflux_temp_C, reboiler_temp_init_C, stripper_pressure_bara, flash_tank_pressure_bara, heat_dissipated_kW, epochs, lr):

        num_of_samples = make_up_water_temp_C.shape[0]
        shape = np.ones(shape=(num_of_samples,))

        self.gas_in = gas_in
        self.heat_kW = heat_kW
        self.heat_dissipated_pct = 100 * heat_dissipated_kW / self.heat_kW

        if self.firstscan:

            self.lean_cold = self.solvent(temp_K=273.15 + lean_cold_temp_init_C, flow_kg_h=lean_cold_flow_kg_h, CO2Load=lean_cold_CO2Load_init)

            print("Init")
            print("- Lean Cold Processing")
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber Init")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2 * self.absorber.num_of_heights, lr=lr)

            print("- Wash Water Init")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2*self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water Init")
            water_loss = np.maximum(self.gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0)
            make_up_water = LiquidStream_Water(temp_K=make_up_water_temp_C + 273.15, flow_kg_h=water_loss)
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, make_up_water])

            print("- Heat Exchanger (Wash Water)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)
            self.rich_semihot = self.equilibrium.react(self.rich_semihot, lr=lr)

            print("- Heat Exchanger (Lean/Rich)")
            self.lean_hot = deepcopy(self.lean_cold)
            self.lean_hot.temp_K = reboiler_temp_init_C + 273.15
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                                temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Reboiler Init")
            self.reboiler_vapor, self.reboiler_condense = self.reboiler.react(self.nozzle_condens, self.heat_kW * (100 - self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Flash Tank Init")
            self.flashtank_vapor, self.lean_hot = self.flashtank.react(self.reboiler_condense, 0 * heat_kW, flash_tank_pressure_bara, lr=lr, flash_always_occur=False)
            self.flashtank_vapor = ReboilerVapor(flow_kmol_h=self.flashtank_vapor.get_gas_flow_kmol_h(),
                                                 pressure_bara=self.flashtank_vapor.get_gas_pressure_bara(),
                                                 temp_K=self.flashtank_vapor.get_gas_temp_K(),
                                                 CO2_molar_fraction=self.flashtank_vapor.get_specie_molar_fraction(id="CO2"),
                                                 H2O_molar_fraction=self.flashtank_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Compressor Init")
            self.compressor_vapor = self.compressor.react(GasStreamIn=self.flashtank_vapor, pressure_out_bara=stripper_pressure_bara)

            print("- Stripper Init")
            self.stripper_vapor, self.stripper_condense = self.stripper.react(GasStreamIn=lab.GasStreamSum(streams=[self.reboiler_vapor, self.compressor_vapor]),
                                                                              LiquidStreamIn=self.nozzle_condens)

            print("- Condenser Init")
            self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor])
            self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape, flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Reboiler Re-Init")
            self.reboiler_vapor, self.reboiler_condense = self.reboiler.react(self.stripper_condense, self.heat_kW * (100 - self.heat_dissipated_pct) / 100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Flash Tank Re-Init")
            self.flashtank_vapor, self.lean_hot = self.flashtank.react(self.reboiler_condense, 0 * heat_kW,
                                                                       flash_tank_pressure_bara, lr=lr,
                                                                       flash_always_occur=False)

            self.flashtank_vapor = ReboilerVapor(flow_kmol_h=self.flashtank_vapor.get_gas_flow_kmol_h(),
                                                 pressure_bara=self.flashtank_vapor.get_gas_pressure_bara(),
                                                 temp_K=self.flashtank_vapor.get_gas_temp_K(),
                                                 CO2_molar_fraction=self.flashtank_vapor.get_specie_molar_fraction(id="CO2"),
                                                 H2O_molar_fraction=self.flashtank_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Compressor Re-Init")
            self.compressor_vapor = self.compressor.react(GasStreamIn=self.flashtank_vapor,
                                                          pressure_out_bara=stripper_pressure_bara)

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)


        self.firstscan = False


        for epoch in range(epochs):

            print("Epoch " + str(epoch+1) + " of " + str(epochs))

            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=self.lean_cold.temp_K, flow_kg_h=lean_cold_flow_kg_h, CO2Load=self.lean_cold.CO2Load(self.lean_cold))
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2*self.absorber.num_of_heights, lr=lr)

            print("- Wash Water")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2 * self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water")
            self.make_up_water = LiquidStream_Water(temp_K=make_up_water_temp_C + 273.15, flow_kg_h=np.maximum(self.gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0))
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, self.make_up_water])

            print("- Heat Exchanger (Wash Water)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)

            print("- Heat Exchanger (Lean/rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                              pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                              temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                              CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                              H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            for _ in range(3):

                print("- Stripper")
                stripper_liquid_inlet = lab.LiquidStreamSum(streams=[self.nozzle_condens, self.reflux])
                stripper_liquid_inlet = self.equilibrium.react(stripper_liquid_inlet, lr=lr)
                self.stripper_vapor, self.stripper_condense = self.stripper.react(lab.GasStreamSum(streams=[self.reboiler_vapor, self.compressor_vapor]), stripper_liquid_inlet)

                print("- Reboiler")
                self.reboiler_vapor, self.reboiler_condense = self.reboiler.react(self.stripper_condense, self.heat_kW * (100 - self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
                self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                    pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                    temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                    CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                    H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

                print("- Flash Tank")
                self.flashtank_vapor, self.lean_hot = self.flashtank.react(self.reboiler_condense, 0 * heat_kW, flash_tank_pressure_bara, lr=lr, flash_always_occur=False)
                self.flashtank_vapor = ReboilerVapor(flow_kmol_h=self.flashtank_vapor.get_gas_flow_kmol_h(),
                                                     pressure_bara=self.flashtank_vapor.get_gas_pressure_bara(),
                                                     temp_K=self.flashtank_vapor.get_gas_temp_K(),
                                                     CO2_molar_fraction=self.flashtank_vapor.get_specie_molar_fraction( id="CO2"),
                                                     H2O_molar_fraction=self.flashtank_vapor.get_specie_molar_fraction(id="H2O"))

                print("- Compressor Init")
                self.compressor_vapor = self.compressor.react(GasStreamIn=self.flashtank_vapor,
                                                              pressure_out_bara=stripper_pressure_bara)

                print("- Condenser")
                self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor])
                self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape, flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)


class CCS_lean_vapor_recompression_SM(lab.Serializer):

    def __init__(self, absorber, stripper, washwater, hx, hx_ww, reboiler, solvent):
        self.firstscan = True
        self.absorber = absorber
        self.stripper = stripper
        self.washwater = washwater
        self.hx = hx
        self.hx_ww = hx_ww
        self.reboiler = reboiler
        self.solvent = solvent
        self.equilibrium = lab.LiquidEquilibrium_Isothermal()
        self.nozzle = lab.LiquidEquilibrium_QPFlash()
        self.flashtank = SMFlash()
        self.compressor = lab.GasCompressor_Isentropic()

    def react(self, gas_in, ww_flow_kg_h, ww_temp_C, lean_cold_CO2Load_init, lean_cold_flow_kg_h, lean_cold_temp_init_C, heat_kW, make_up_water_temp_C, reflux_temp_C, reboiler_temp_init_C, stripper_pressure_bara, flash_tank_pressure_bara, heat_dissipated_kW, ratio_to_hx, epochs, lr):

        num_of_samples = make_up_water_temp_C.shape[0]
        shape = np.ones(shape=(num_of_samples,))

        self.gas_in = gas_in
        self.heat_kW = heat_kW
        self.heat_dissipated_pct = 100 * heat_dissipated_kW / self.heat_kW

        flash_always_occur=False

        if self.firstscan:

            self.lean_cold = self.solvent(temp_K=273.15 + lean_cold_temp_init_C, flow_kg_h=lean_cold_flow_kg_h, CO2Load=lean_cold_CO2Load_init)

            print("Init")
            print("- Lean Cold Processing")
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber Init")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2 * self.absorber.num_of_heights, lr=lr)

            print("- Wash Water Init")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2*self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water Init")
            water_loss = np.maximum(self.gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0)
            make_up_water = LiquidStream_Water(temp_K=make_up_water_temp_C + 273.15, flow_kg_h=water_loss)
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, make_up_water])

            print("- Heat Exchanger (Wash Water)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)
            self.rich_semihot = self.equilibrium.react(self.rich_semihot, lr=lr)

            print("- Heat Exchanger (Lean/Rich)")
            self.lean_hot = deepcopy(self.lean_cold)
            self.lean_hot.temp_K = reboiler_temp_init_C + 273.15
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                                temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Reboiler Init")
            self.reboiler_vapor, self.reboiler_condense = self.reboiler.react(self.nozzle_condens, self.heat_kW * (100 - self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Flash Tank Init")


            self.flashtank_vapor, self.lean_hot = self.flashtank.react(self.reboiler_condense, ratio_to_hx, flash_tank_pressure_bara, flash_always_occur, lr)
            self.flashtank_vapor = ReboilerVapor(flow_kmol_h=self.flashtank_vapor.get_gas_flow_kmol_h(),
                                                 pressure_bara=self.flashtank_vapor.get_gas_pressure_bara(),
                                                 temp_K=self.flashtank_vapor.get_gas_temp_K(),
                                                 CO2_molar_fraction=self.flashtank_vapor.get_specie_molar_fraction(id="CO2"),
                                                 H2O_molar_fraction=self.flashtank_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Compressor Init")
            self.compressor_vapor = self.compressor.react(GasStreamIn=self.flashtank_vapor, pressure_out_bara=stripper_pressure_bara)

            print("- Stripper Init")
            self.stripper_vapor, self.stripper_condense = self.stripper.react(GasStreamIn=lab.GasStreamSum(streams=[self.reboiler_vapor, self.compressor_vapor]),
                                                                              LiquidStreamIn=self.nozzle_condens)

            print("- Condenser Init")
            self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor])
            self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape, flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Reboiler Re-Init")
            self.reboiler_vapor, self.reboiler_condense = self.reboiler.react(self.stripper_condense, self.heat_kW * (100 - self.heat_dissipated_pct) / 100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Flash Tank Re-Init")
            self.flashtank_vapor, self.lean_hot = self.flashtank.react(self.reboiler_condense, ratio_to_hx, flash_tank_pressure_bara, flash_always_occur, lr)

            self.flashtank_vapor = ReboilerVapor(flow_kmol_h=self.flashtank_vapor.get_gas_flow_kmol_h(),
                                                 pressure_bara=self.flashtank_vapor.get_gas_pressure_bara(),
                                                 temp_K=self.flashtank_vapor.get_gas_temp_K(),
                                                 CO2_molar_fraction=self.flashtank_vapor.get_specie_molar_fraction(id="CO2"),
                                                 H2O_molar_fraction=self.flashtank_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Compressor Re-Init")
            self.compressor_vapor = self.compressor.react(GasStreamIn=self.flashtank_vapor,
                                                          pressure_out_bara=stripper_pressure_bara)

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)


        self.firstscan = False


        for epoch in range(epochs):

            print("Epoch " + str(epoch+1) + " of " + str(epochs))

            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=self.lean_cold.temp_K, flow_kg_h=lean_cold_flow_kg_h, CO2Load=self.lean_cold.CO2Load(self.lean_cold))
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2*self.absorber.num_of_heights, lr=lr)

            print("- Wash Water")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2 * self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water")
            self.make_up_water = LiquidStream_Water(temp_K=make_up_water_temp_C + 273.15, flow_kg_h=np.maximum(self.gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0))
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, self.make_up_water])

            print("- Heat Exchanger (Wash Water)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)

            print("- Heat Exchanger (Lean/rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                              pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                              temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                              CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                              H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            for _ in range(3):

                print("- Stripper")
                stripper_liquid_inlet = lab.LiquidStreamSum(streams=[self.nozzle_condens, self.reflux])
                stripper_liquid_inlet = self.equilibrium.react(stripper_liquid_inlet, lr=lr)
                self.stripper_vapor, self.stripper_condense = self.stripper.react(lab.GasStreamSum(streams=[self.reboiler_vapor, self.compressor_vapor]), stripper_liquid_inlet)

                print("- Reboiler")
                self.reboiler_vapor, self.reboiler_condense = self.reboiler.react(self.stripper_condense, self.heat_kW * (100 - self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
                self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                    pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                    temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                    CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                    H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

                print("- Flash Tank")
                self.flashtank_vapor, self.lean_hot = self.flashtank.react(self.reboiler_condense, ratio_to_hx, flash_tank_pressure_bara, flash_always_occur, lr)
                self.flashtank_vapor = ReboilerVapor(flow_kmol_h=self.flashtank_vapor.get_gas_flow_kmol_h(),
                                                     pressure_bara=self.flashtank_vapor.get_gas_pressure_bara(),
                                                     temp_K=self.flashtank_vapor.get_gas_temp_K(),
                                                     CO2_molar_fraction=self.flashtank_vapor.get_specie_molar_fraction( id="CO2"),
                                                     H2O_molar_fraction=self.flashtank_vapor.get_specie_molar_fraction(id="H2O"))

                print("- Compressor Init")
                self.compressor_vapor = self.compressor.react(GasStreamIn=self.flashtank_vapor,
                                                              pressure_out_bara=stripper_pressure_bara)

                print("- Condenser")
                self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_vapor])
                self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape, flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Heat Exchanger")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)


class CCS_interheated_column(lab.Serializer):

    def __init__(self, absorber, stripper_top, stripper_bottom, washwater, hx, hx_ww, hx_interheater, reboiler, solvent):
        self.firstscan = True
        self.absorber = absorber
        self.stripper_top = stripper_top
        self.stripper_bottom = stripper_bottom
        self.washwater = washwater
        self.hx = hx
        self.hx_ww = hx_ww
        self.hx_interheater = hx_interheater
        self.reboiler = reboiler
        self.solvent = solvent
        self.equilibrium = lab.LiquidEquilibrium_Isothermal()
        self.nozzle = lab.LiquidEquilibrium_QPFlash()

    def react(self, gas_in, ww_flow_kg_h, ww_temp_C, lean_cold_CO2Load_init, lean_cold_flow_kg_h, lean_cold_temp_init_C, heat_kW, make_up_water_temp_C, reflux_temp_C, reboiler_temp_init_C, stripper_pressure_bara, heat_dissipated_kW, epochs, lr):

        num_of_samples = make_up_water_temp_C.shape[0]
        shape = np.ones(shape=(num_of_samples,))

        self.gas_in = gas_in
        self.heat_kW = heat_kW
        self.heat_dissipated_pct = 100 * heat_dissipated_kW / self.heat_kW

        if self.firstscan:

            print("Init")
            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=273.15 + lean_cold_temp_init_C, flow_kg_h=lean_cold_flow_kg_h, CO2Load=lean_cold_CO2Load_init)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber Init")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2 * self.absorber.num_of_heights, lr=lr)

            print("- Wash Water Init")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2*self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water Init")
            water_loss = np.maximum(self.absorber_gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0)
            self.ww_hot.flow_kg_h = self.ww_hot.flow_kg_h - water_loss
            make_up_water = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=water_loss)
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, make_up_water])

            print("- Heat Exchanger (WashWater)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)
            self.rich_semihot = self.equilibrium.react(self.rich_semihot, lr=lr)

            print("- Lean Hot Init")
            self.lean_hot = deepcopy(self.lean_cold)
            self.lean_hot.temp_K = reboiler_temp_init_C + 273.15
            self.lean_hot = self.equilibrium.react(self.lean_hot, lr=lr)

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                                temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Reboiler Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.nozzle_condens, self.heat_kW * (100-self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Stripper Top Init")
            self.stripper_top_vapor, self.stripper_top_condense = self.stripper_top.react(GasStreamIn=self.reboiler_vapor,
                                                                                          LiquidStreamIn=self.nozzle_condens)

            print("- Stripper Bottom Init")
            self.stripper_bottom_vapor, self.stripper_bottom_condense = self.stripper_bottom.react(GasStreamIn=self.reboiler_vapor,
                                                                                                   LiquidStreamIn=self.nozzle_condens)

            print("- Condenser Init")
            self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_top_vapor])
            self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape,
                                      flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Reboiler Re-Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_bottom_condense, self.heat_kW * (100-self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Heat Exhanger (InterHeater)")
            self.lean_intercooled, self.semilean_interheated = self.hx_interheater.react(LiquidStreamIn1=self.lean_hot, LiquidStreamIn2=self.stripper_top_condense)
            self.semilean_interheated = self.equilibrium.react(self.semilean_interheated, lr=lr)
            self.lean_intercooled = self.equilibrium.react(self.lean_intercooled, lr=lr)

            self.semilean_interheated_vapor, self.semilean_interheated_condense = self.nozzle.react(self.semilean_interheated, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)


            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)

        self.firstscan = False


        for epoch in range(epochs):

            print("Epoch " + str(epoch+1) + " of " + str(epochs))

            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=self.lean_cold.temp_K,
                                          flow_kg_h=lean_cold_flow_kg_h,
                                          CO2Load=self.lean_cold.CO2Load(self.lean_cold))
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2*self.absorber.num_of_heights, lr=lr)

            print("- Wash Water")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2 * self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water")
            water_loss = np.maximum(self.absorber_gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0)
            self.ww_hot.flow_kg_h = self.ww_hot.flow_kg_h - water_loss
            self.make_up_water = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=water_loss)
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, self.make_up_water])
            self.rich_cold = self.equilibrium.react(self.rich_cold, lr=lr)

            print("- Heat Exchanger (Wash Water)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)
            self.rich_semihot = self.equilibrium.react(self.rich_semihot, lr=lr)

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                              pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                              temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                              CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                              H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            for _ in range(3):

                print("- Stripper Top")
                stripper_top_liquid_inlet = lab.LiquidStreamSum(streams=[self.nozzle_condens, self.reflux])
                stripper_top_liquid_inlet = self.equilibrium.react(stripper_top_liquid_inlet, lr=lr)
                stripper_top_vapor_inlet = lab.GasStreamSum(streams=[self.stripper_bottom_vapor, self.semilean_interheated_vapor])
                self.stripper_top_vapor, self.stripper_top_condense = self.stripper_top.react(stripper_top_vapor_inlet, stripper_top_liquid_inlet)

                print("- Heat Exchanger (Interheater)")
                self.lean_intercooled, self.semilean_interheated = self.hx_interheater.react(LiquidStreamIn1=self.lean_hot, LiquidStreamIn2=self.stripper_top_condense)
                self.semilean_interheated = self.equilibrium.react(self.semilean_interheated, lr=lr)
                self.lean_intercooled = self.equilibrium.react(self.lean_intercooled, lr=lr)

                self.semilean_interheated_vapor, self.semilean_interheated_condense = self.nozzle.react(self.semilean_interheated, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)

                print("- Stripper Bottom")
                self.stripper_bottom_vapor, self.stripper_bottom_condense = self.stripper_bottom.react(self.reboiler_vapor, self.semilean_interheated_condense)

                print("- Reboiler")
                self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_bottom_condense,
                                                                        self.heat_kW * (100-self.heat_dissipated_pct)/100,
                                                                        stripper_pressure_bara,
                                                                        lr=lr,
                                                                        flash_always_occur=True)

                self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                    pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                    temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                    CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                    H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

                print("- Condenser")
                self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_top_vapor])
                self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape, flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)


class CCS_double_interheated_column(lab.Serializer):

    def __init__(self, absorber, stripper_top, stripper_middle, stripper_bottom, washwater, hx, hx_ww, hx_interheater, solvent):
        self.firstscan = True
        self.absorber = absorber
        self.stripper_top = stripper_top
        self.stripper_middle = stripper_middle
        self.stripper_bottom = stripper_bottom
        self.washwater = washwater
        self.hx = hx
        self.hx_ww = hx_ww
        self.hx_interheater = hx_interheater
        self.reboiler = lab.LiquidEquilibrium_QPFlash()
        self.solvent = solvent
        self.equilibrium = lab.LiquidEquilibrium_Isothermal()
        self.nozzle = lab.LiquidEquilibrium_QPFlash()

    def react(self, gas_in, ww_flow_kg_h, ww_temp_C, lean_cold_CO2Load_init, lean_cold_flow_kg_h, lean_cold_temp_init_C, heat_kW, make_up_water_temp_C, reflux_temp_C, reboiler_temp_init_C, stripper_pressure_bara, epochs, lr):

        num_of_samples = make_up_water_temp_C.shape[0]
        shape = np.ones(shape=(num_of_samples,))

        self.gas_in = gas_in
        self.heat_kW = heat_kW
        self.heat_dissipated_pct = 5

        if self.firstscan:

            print("Init")
            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=273.15 + lean_cold_temp_init_C, flow_kg_h=lean_cold_flow_kg_h, CO2Load=lean_cold_CO2Load_init)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber Init")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2 * self.absorber.num_of_heights, lr=lr)

            print("- Wash Water Init")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2*self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water Init")
            water_loss = np.maximum(self.absorber_gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0)
            self.ww_hot.flow_kg_h = self.ww_hot.flow_kg_h - water_loss
            make_up_water = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=water_loss)
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, make_up_water])

            print("- Heat Exchanger (WashWater)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)
            self.rich_semihot = self.equilibrium.react(self.rich_semihot, lr=lr)

            print("- Lean Hot Init")
            self.lean_hot = deepcopy(self.lean_cold)
            self.lean_hot.temp_K = reboiler_temp_init_C + 273.15
            self.lean_hot = self.equilibrium.react(self.lean_hot, lr=lr)

            print("- Lean Intercooled Init")
            self.lean_intercooled = deepcopy(self.lean_hot)

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                                temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Reboiler Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.nozzle_condens, self.heat_kW * (100-self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Stripper Top Init")
            self.stripper_top_vapor, self.stripper_top_condense = self.stripper_top.react(GasStreamIn=self.reboiler_vapor,
                                                                                          LiquidStreamIn=self.nozzle_condens)

            print("- Stripper Middle Init")
            self.stripper_middle_vapor, self.stripper_middle_condense = self.stripper_middle.react(GasStreamIn=self.reboiler_vapor,
                                                                                                   LiquidStreamIn=self.nozzle_condens)

            print("- Stripper Bottom Init")
            self.stripper_bottom_vapor, self.stripper_bottom_condense = self.stripper_bottom.react(GasStreamIn=self.reboiler_vapor,
                                                                                                   LiquidStreamIn=self.nozzle_condens)

            print("- Condenser Init")
            self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_top_vapor])
            self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape,
                                      flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Reboiler Re-Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_bottom_condense, self.heat_kW*(100-self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Heat Exhanger (InterHeaters) Init")
            self.s5, self.s4 = self.hx_interheater.react(LiquidStreamIn1=self.lean_hot, LiquidStreamIn2=self.stripper_middle_condense)
            self.s5 = self.equilibrium.react(self.s5, lr=lr)
            self.s4 = self.equilibrium.react(self.s4, lr=lr)

            self.lean_intercooled, self.s2 = self.hx_interheater.react(LiquidStreamIn1=self.s5, LiquidStreamIn2=self.stripper_top_condense)
            self.lean_intercooled = self.equilibrium.react(self.lean_intercooled, lr=lr)
            self.s2 = self.equilibrium.react(self.s2, lr=lr)

            self.s2_vapor, self.s2_condense = self.nozzle.react(self.s2, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.s2_vapor = ReboilerVapor(flow_kmol_h=self.s2_vapor.get_gas_flow_kmol_h(),
                                          pressure_bara=self.s2_vapor.get_gas_pressure_bara(),
                                          temp_K=self.s2_vapor.get_gas_temp_K(),
                                          CO2_molar_fraction=self.s2_vapor.get_specie_molar_fraction(id="CO2"),
                                          H2O_molar_fraction=self.s2_vapor.get_specie_molar_fraction(id="H2O"))

            self.s4_vapor, self.s4_condense = self.nozzle.react(self.s4, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.s4_vapor = ReboilerVapor(flow_kmol_h=self.s4_vapor.get_gas_flow_kmol_h(),
                                          pressure_bara=self.s4_vapor.get_gas_pressure_bara(),
                                          temp_K=self.s4_vapor.get_gas_temp_K(),
                                          CO2_molar_fraction=self.s4_vapor.get_specie_molar_fraction(id="CO2"),
                                          H2O_molar_fraction=self.s4_vapor.get_specie_molar_fraction(id="H2O"))


            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)

        self.firstscan = False


        for epoch in range(epochs):

            print("Epoch " + str(epoch+1) + " of " + str(epochs))

            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=self.lean_cold.temp_K,
                                          flow_kg_h=lean_cold_flow_kg_h,
                                          CO2Load=self.lean_cold.CO2Load(self.lean_cold))
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2*self.absorber.num_of_heights, lr=lr)

            print("- Wash Water")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2 * self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water")
            water_loss = np.maximum(self.absorber_gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0)
            self.ww_hot.flow_kg_h = self.ww_hot.flow_kg_h - water_loss
            self.make_up_water = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=water_loss)
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, self.make_up_water])
            self.rich_cold = self.equilibrium.react(self.rich_cold, lr=lr)

            print("- Heat Exchanger (Wash Water)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)
            self.rich_semihot = self.equilibrium.react(self.rich_semihot, lr=lr)

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                              pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                              temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                              CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                              H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            for _ in range(3):

                print("- Stripper Top")
                stripper_top_liquid_inlet = lab.LiquidStreamSum(streams=[self.nozzle_condens, self.reflux])
                stripper_top_liquid_inlet = self.equilibrium.react(stripper_top_liquid_inlet, lr=lr)
                stripper_top_vapor_inlet = lab.GasStreamSum(streams=[self.stripper_middle_vapor, self.s2_vapor])
                self.stripper_top_vapor, self.stripper_top_condense = self.stripper_top.react(stripper_top_vapor_inlet, stripper_top_liquid_inlet)

                print("- Heat Exchanger (Interheater 2)")
                self.lean_intercooled, self.s2 = self.hx_interheater.react(LiquidStreamIn1=self.s5, LiquidStreamIn2=self.stripper_top_condense)
                self.s2 = self.equilibrium.react(self.s2, lr=lr)
                self.lean_intercooled = self.equilibrium.react(self.lean_intercooled, lr=lr)

                print("- Nozzle Flash Below Stripper Top")
                self.s2_vapor, self.s2_condense = self.nozzle.react(self.s2, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)

                print("- Stripper Middle")
                stripper_middle_vapor_inlet = lab.GasStreamSum(streams=[self.stripper_bottom_vapor, self.s4_vapor])
                self.stripper_middle_vapor, self.stripper_middle_condense = self.stripper_middle.react(stripper_middle_vapor_inlet, self.s2_condense)

                print("- Heat Exchanger (Interheater 1)")
                self.s5, self.s4 = self.hx_interheater.react(LiquidStreamIn1=self.lean_hot, LiquidStreamIn2=self.stripper_middle_condense)
                self.s5 = self.equilibrium.react(self.s5, lr=lr)
                self.s4 = self.equilibrium.react(self.s4, lr=lr)

                print("- Nozzle Flash Below Stripper Middle")
                self.s4_vapor, self.s4_condense = self.nozzle.react(self.s4, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)

                print("- Stripper Bottom")
                self.stripper_bottom_vapor, self.stripper_bottom_condense = self.stripper_bottom.react(self.reboiler_vapor, self.s4_condense)

                print("- Reboiler")
                self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_bottom_condense,
                                                                        self.heat_kW * (100-self.heat_dissipated_pct)/100,
                                                                        stripper_pressure_bara,
                                                                        lr=lr,
                                                                        flash_always_occur=True)

                self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                    pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                    temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                    CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                    H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

                print("- Condenser")
                self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_top_vapor])
                self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape, flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)


class CCS_double_interheated_column_stripper_split_feed(lab.Serializer):

    def __init__(self, absorber, rectifier, stripper_top, stripper_middle, stripper_bottom, washwater, hx, hx_ww, hx_interheater, solvent):
        self.firstscan = True
        self.absorber = absorber
        self.stripper_top = stripper_top
        self.stripper_middle = stripper_middle
        self.stripper_bottom = stripper_bottom
        self.washwater = washwater
        self.rectifier = rectifier
        self.hx = hx
        self.hx_ww = hx_ww
        self.hx_interheater = hx_interheater
        self.reboiler = lab.LiquidEquilibrium_QPFlash()
        self.solvent = solvent
        self.equilibrium = lab.LiquidEquilibrium_Isothermal()
        self.nozzle = lab.LiquidEquilibrium_QPFlash()

    def react(self, ratio_to_stripper_top, gas_in, ww_flow_kg_h, ww_temp_C, lean_cold_CO2Load_init, lean_cold_flow_kg_h, lean_cold_temp_init_C, heat_kW, make_up_water_temp_C, reflux_temp_C, reboiler_temp_init_C, stripper_pressure_bara, epochs, lr):

        num_of_samples = make_up_water_temp_C.shape[0]
        shape = np.ones(shape=(num_of_samples,))

        self.gas_in = gas_in
        self.heat_kW = heat_kW
        self.heat_dissipated_pct = 5

        if self.firstscan:

            print("Init")
            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=273.15 + lean_cold_temp_init_C, flow_kg_h=lean_cold_flow_kg_h, CO2Load=lean_cold_CO2Load_init)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber Init")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2 * self.absorber.num_of_heights, lr=lr)

            print("- Wash Water Init")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2*self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water Init")
            water_loss = np.maximum(self.absorber_gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0)
            self.ww_hot.flow_kg_h = self.ww_hot.flow_kg_h - water_loss
            make_up_water = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=water_loss)
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, make_up_water])

            print("- Heat Exchanger (WashWater)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)
            self.rich_semihot = self.equilibrium.react(self.rich_semihot, lr=lr)

            print("- Rich Solvent Split")
            self.rich_semihot_to_top = deepcopy(self.rich_semihot)
            self.rich_semihot_to_top.flow_kg_h = self.rich_semihot.flow_kg_h * ratio_to_stripper_top
            self.rich_semihot.flow_kg_h = self.rich_semihot.flow_kg_h - self.rich_semihot_to_top.flow_kg_h

            print("- Lean Hot Init")
            self.lean_hot = deepcopy(self.lean_cold)
            self.lean_hot.temp_K = reboiler_temp_init_C + 273.15
            self.lean_hot = self.equilibrium.react(self.lean_hot, lr=lr)

            print("- Lean Intercooled Init")
            self.lean_intercooled = deepcopy(self.lean_hot)

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                                temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Reboiler Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.nozzle_condens, self.heat_kW * (100-self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Stripper Top Init")
            self.stripper_top_vapor, self.stripper_top_condense = self.stripper_top.react(GasStreamIn=self.reboiler_vapor,
                                                                                          LiquidStreamIn=self.nozzle_condens)

            print("- Stripper Middle Init")
            self.stripper_middle_vapor, self.stripper_middle_condense = self.stripper_middle.react(GasStreamIn=self.reboiler_vapor,
                                                                                                   LiquidStreamIn=self.nozzle_condens)

            print("- Stripper Bottom Init")
            self.stripper_bottom_vapor, self.stripper_bottom_condense = self.stripper_bottom.react(GasStreamIn=self.reboiler_vapor,
                                                                                                   LiquidStreamIn=self.nozzle_condens)

            print("- Condenser Init")
            self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_top_vapor])
            self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape,
                                      flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Rectifier Init")
            rectifier_liquid_inlet = lab.LiquidStreamSum(streams=[self.rich_semihot_to_top, self.reflux])
            rectifier_liquid_inlet = self.equilibrium.react(rectifier_liquid_inlet, lr=lr)
            self.rectifier_vapor, self.rectifier_condense = self.rectifier.react(GasStreamIn=self.reboiler_vapor, LiquidStreamIn=rectifier_liquid_inlet)

            print("- Reboiler Re-Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_bottom_condense, self.heat_kW * (100-self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Heat Exhanger (InterHeaters) Init")
            self.s5, self.s4 = self.hx_interheater.react(LiquidStreamIn1=self.lean_hot, LiquidStreamIn2=self.stripper_middle_condense)
            self.s5 = self.equilibrium.react(self.s5, lr=lr)
            self.s4 = self.equilibrium.react(self.s4, lr=lr)

            self.lean_intercooled, self.s2 = self.hx_interheater.react(LiquidStreamIn1=self.s5, LiquidStreamIn2=self.stripper_top_condense)
            self.lean_intercooled = self.equilibrium.react(self.lean_intercooled, lr=lr)
            self.s2 = self.equilibrium.react(self.s2, lr=lr)

            self.s2_vapor, self.s2_condense = self.nozzle.react(self.s2, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.s2_vapor = ReboilerVapor(flow_kmol_h=self.s2_vapor.get_gas_flow_kmol_h(),
                                          pressure_bara=self.s2_vapor.get_gas_pressure_bara(),
                                          temp_K=self.s2_vapor.get_gas_temp_K(),
                                          CO2_molar_fraction=self.s2_vapor.get_specie_molar_fraction(id="CO2"),
                                          H2O_molar_fraction=self.s2_vapor.get_specie_molar_fraction(id="H2O"))

            self.s4_vapor, self.s4_condense = self.nozzle.react(self.s4, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.s4_vapor = ReboilerVapor(flow_kmol_h=self.s4_vapor.get_gas_flow_kmol_h(),
                                          pressure_bara=self.s4_vapor.get_gas_pressure_bara(),
                                          temp_K=self.s4_vapor.get_gas_temp_K(),
                                          CO2_molar_fraction=self.s4_vapor.get_specie_molar_fraction(id="CO2"),
                                          H2O_molar_fraction=self.s4_vapor.get_specie_molar_fraction(id="H2O"))


            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)

        self.firstscan = False


        for epoch in range(epochs):

            print("Epoch " + str(epoch+1) + " of " + str(epochs))

            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=self.lean_cold.temp_K,
                                          flow_kg_h=lean_cold_flow_kg_h,
                                          CO2Load=self.lean_cold.CO2Load(self.lean_cold))
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold, epochs=2*self.absorber.num_of_heights, lr=lr)

            print("- Wash Water")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2 * self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water")
            water_loss = np.maximum(self.absorber_gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0)
            self.ww_hot.flow_kg_h = self.ww_hot.flow_kg_h - water_loss
            self.make_up_water = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=water_loss)
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, self.make_up_water])
            self.rich_cold = self.equilibrium.react(self.rich_cold, lr=lr)

            print("- Heat Exchanger (Wash Water)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)
            self.rich_semihot = self.equilibrium.react(self.rich_semihot, lr=lr)

            print("- Rich Solvent Split")
            self.rich_semihot_to_top = deepcopy(self.rich_semihot)
            self.rich_semihot_to_top.flow_kg_h = self.rich_semihot.flow_kg_h * ratio_to_stripper_top
            self.rich_semihot.flow_kg_h = self.rich_semihot.flow_kg_h - self.rich_semihot_to_top.flow_kg_h

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                              pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                              temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                              CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                              H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            for _ in range(3):

                print("- Stripper Top")
                stripper_top_liquid_inlet = lab.LiquidStreamSum(streams=[self.nozzle_condens, self.rectifier_condense])
                stripper_top_liquid_inlet = self.equilibrium.react(stripper_top_liquid_inlet, lr=lr)
                stripper_top_vapor_inlet = lab.GasStreamSum(streams=[self.stripper_middle_vapor, self.s2_vapor])
                self.stripper_top_vapor, self.stripper_top_condense = self.stripper_top.react(stripper_top_vapor_inlet, stripper_top_liquid_inlet)

                print("- Heat Exchanger (Interheater 2)")
                self.lean_intercooled, self.s2 = self.hx_interheater.react(LiquidStreamIn1=self.s5, LiquidStreamIn2=self.stripper_top_condense)
                self.s2 = self.equilibrium.react(self.s2, lr=lr)
                self.lean_intercooled = self.equilibrium.react(self.lean_intercooled, lr=lr)

                print("- Nozzle Flash Below Stripper Top")
                self.s2_vapor, self.s2_condense = self.nozzle.react(self.s2, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)

                print("- Stripper Middle")
                stripper_middle_vapor_inlet = lab.GasStreamSum(streams=[self.stripper_bottom_vapor, self.s4_vapor])
                self.stripper_middle_vapor, self.stripper_middle_condense = self.stripper_middle.react(stripper_middle_vapor_inlet, self.s2_condense)

                print("- Heat Exchanger (Interheater 1)")
                self.s5, self.s4 = self.hx_interheater.react(LiquidStreamIn1=self.lean_hot, LiquidStreamIn2=self.stripper_middle_condense)
                self.s5 = self.equilibrium.react(self.s5, lr=lr)
                self.s4 = self.equilibrium.react(self.s4, lr=lr)

                print("- Nozzle Flash Below Stripper Middle")
                self.s4_vapor, self.s4_condense = self.nozzle.react(self.s4, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)

                print("- Stripper Bottom")
                self.stripper_bottom_vapor, self.stripper_bottom_condense = self.stripper_bottom.react(self.reboiler_vapor, self.s4_condense)

                print("- Reboiler")
                self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_bottom_condense,
                                                                        self.heat_kW * (100-self.heat_dissipated_pct)/100,
                                                                        stripper_pressure_bara,
                                                                        lr=lr,
                                                                        flash_always_occur=True)

                self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                    pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                    temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                    CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                    H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

                print("- Rectifier")
                rectifier_vapor_inlet = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_top_vapor])
                rectifier_liquid_inlet = lab.LiquidStreamSum(streams=[self.rich_semihot_to_top, self.reflux])
                rectifier_liquid_inlet = self.equilibrium.react(rectifier_liquid_inlet, lr=lr)
                self.rectifier_vapor, self.rectifier_condense = self.rectifier.react(rectifier_vapor_inlet, rectifier_liquid_inlet)

                print("- Condenser")
                self.condenser_vapor = deepcopy(self.rectifier_vapor)
                self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape, flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)


class CCS_interheated_column_equilibrium_stage_absorber(lab.Serializer):

    def __init__(self, absorber, stripper_top, stripper_bottom, washwater, hx, hx_ww, hx_interheater, reboiler, solvent):
        self.firstscan = True
        self.absorber = absorber
        self.stripper_top = stripper_top
        self.stripper_bottom = stripper_bottom
        self.washwater = washwater
        self.hx = hx
        self.hx_ww = hx_ww
        self.hx_interheater = hx_interheater
        self.reboiler = reboiler
        self.solvent = solvent
        self.equilibrium = lab.LiquidEquilibrium_Isothermal()
        self.nozzle = lab.LiquidEquilibrium_QPFlash()

    def react(self, gas_in, ww_flow_kg_h, ww_temp_C, lean_cold_CO2Load_init, lean_cold_flow_kg_h, lean_cold_temp_init_C, heat_kW, make_up_water_temp_C, reflux_temp_C, reboiler_temp_init_C, stripper_pressure_bara, epochs, lr):

        num_of_samples = make_up_water_temp_C.shape[0]
        shape = np.ones(shape=(num_of_samples,))

        self.gas_in = gas_in
        self.heat_kW = heat_kW
        self.heat_dissipated_pct

        if self.firstscan:

            print("Init")
            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=273.15 + lean_cold_temp_init_C, flow_kg_h=lean_cold_flow_kg_h, CO2Load=lean_cold_CO2Load_init)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber Init")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold)

            print("- Wash Water Init")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2*self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water Init")
            water_loss = np.maximum(self.absorber_gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0)
            self.ww_hot.flow_kg_h = self.ww_hot.flow_kg_h - water_loss
            make_up_water = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=water_loss)
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, make_up_water])

            print("- Heat Exchanger (WashWater)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)
            self.rich_semihot = self.equilibrium.react(self.rich_semihot, lr=lr)

            print("- Lean Hot Init")
            self.lean_hot = deepcopy(self.lean_cold)
            self.lean_hot.temp_K = reboiler_temp_init_C + 273.15
            self.lean_hot = self.equilibrium.react(self.lean_hot, lr=lr)

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_hot)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                                temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Reboiler Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.nozzle_condens, self.heat_kW * (100-self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Stripper Top Init")
            self.stripper_top_vapor, self.stripper_top_condense = self.stripper_top.react(GasStreamIn=self.reboiler_vapor,
                                                                                          LiquidStreamIn=self.nozzle_condens)

            print("- Stripper Bottom Init")
            self.stripper_bottom_vapor, self.stripper_bottom_condense = self.stripper_bottom.react(GasStreamIn=self.reboiler_vapor,
                                                                                                   LiquidStreamIn=self.nozzle_condens)

            print("- Condenser Init")
            self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_top_vapor])
            self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape,
                                      flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Reboiler Re-Init")
            self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_bottom_condense, self.heat_kW * (100-self.heat_dissipated_pct)/100, stripper_pressure_bara, lr=lr, flash_always_occur=True)
            self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

            print("- Heat Exhanger (InterHeater)")
            self.lean_intercooled, self.semilean_interheated = self.hx_interheater.react(LiquidStreamIn1=self.lean_hot, LiquidStreamIn2=self.stripper_top_condense)
            self.semilean_interheated = self.equilibrium.react(self.semilean_interheated, lr=lr)
            self.lean_intercooled = self.equilibrium.react(self.lean_intercooled, lr=lr)

            self.semilean_interheated_vapor, self.semilean_interheated_condense = self.nozzle.react(self.semilean_interheated, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)


            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(self.lean_cold, lr=lr)

        self.firstscan = False


        for epoch in range(epochs):

            print("Epoch " + str(epoch+1) + " of " + str(epochs))

            print("- Lean Cold Processing")
            self.lean_cold = self.solvent(temp_K=self.lean_cold.temp_K,
                                          flow_kg_h=lean_cold_flow_kg_h,
                                          CO2Load=self.lean_cold.CO2Load(self.lean_cold))
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Absorber")
            self.absorber_gas_out, self.rich_cold = self.absorber.react(GasStreamIn=gas_in, LiquidStreamIn=self.lean_cold)

            print("- Wash Water")
            self.ww_cold = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=ww_flow_kg_h)
            self.gas_out, self.ww_hot = self.washwater.react(GasStreamIn=self.absorber_gas_out, LiquidStreamIn=self.ww_cold, epochs=2 * self.washwater.num_of_heights, lr=lr)

            print("- Make Up Water")
            water_loss = np.maximum(self.absorber_gas_out.get_specie_flow_kg_h(id="H2O") - gas_in.get_specie_flow_kg_h(id="H2O"), 0.0)
            self.ww_hot.flow_kg_h = self.ww_hot.flow_kg_h - water_loss
            self.make_up_water = LiquidStream_Water(temp_K=ww_temp_C + 273.15, flow_kg_h=water_loss)
            self.rich_cold = lab.LiquidStreamSum(streams=[self.rich_cold, self.make_up_water])
            self.rich_cold = self.equilibrium.react(self.rich_cold, lr=lr)

            print("- Heat Exchanger (Wash Water)")
            self.rich_semihot, _ = self.hx_ww.react(LiquidStreamIn1=self.rich_cold, LiquidStreamIn2=self.ww_hot)
            self.rich_semihot = self.equilibrium.react(self.rich_semihot, lr=lr)

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)

            print("- Nozzle Flash")
            self.nozzle_vapor, self.nozzle_condens = self.nozzle.react(self.rich_hot, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)
            self.nozzle_vapor = ReboilerVapor(flow_kmol_h=self.nozzle_vapor.get_gas_flow_kmol_h(),
                                              pressure_bara=self.nozzle_vapor.get_gas_pressure_bara(),
                                              temp_K=self.nozzle_vapor.get_gas_temp_K(),
                                              CO2_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="CO2"),
                                              H2O_molar_fraction=self.nozzle_vapor.get_specie_molar_fraction(id="H2O"))

            for _ in range(3):

                print("- Stripper Top")
                stripper_top_liquid_inlet = lab.LiquidStreamSum(streams=[self.nozzle_condens, self.reflux])
                stripper_top_liquid_inlet = self.equilibrium.react(stripper_top_liquid_inlet, lr=lr)
                stripper_top_vapor_inlet = lab.GasStreamSum(streams=[self.stripper_bottom_vapor, self.semilean_interheated_vapor])
                self.stripper_top_vapor, self.stripper_top_condense = self.stripper_top.react(stripper_top_vapor_inlet, stripper_top_liquid_inlet)

                print("- Heat Exchanger (Interheater)")
                self.lean_intercooled, self.semilean_interheated = self.hx_interheater.react(LiquidStreamIn1=self.lean_hot, LiquidStreamIn2=self.stripper_top_condense)
                self.semilean_interheated = self.equilibrium.react(self.semilean_interheated, lr=lr)
                self.lean_intercooled = self.equilibrium.react(self.lean_intercooled, lr=lr)

                self.semilean_interheated_vapor, self.semilean_interheated_condense = self.nozzle.react(self.semilean_interheated, 0 * heat_kW, stripper_pressure_bara, lr=lr, flash_always_occur=False)

                print("- Stripper Bottom")
                self.stripper_bottom_vapor, self.stripper_bottom_condense = self.stripper_bottom.react(self.reboiler_vapor, self.semilean_interheated_condense)

                print("- Reboiler")
                self.reboiler_vapor, self.lean_hot = self.reboiler.react(self.stripper_bottom_condense,
                                                                        self.heat_kW * (100-self.heat_dissipated_pct)/100,
                                                                        stripper_pressure_bara,
                                                                        lr=lr,
                                                                        flash_always_occur=True)

                self.reboiler_vapor = ReboilerVapor(flow_kmol_h=self.reboiler_vapor.get_gas_flow_kmol_h(),
                                                    pressure_bara=self.reboiler_vapor.get_gas_pressure_bara(),
                                                    temp_K=self.reboiler_vapor.get_gas_temp_K(),
                                                    CO2_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="CO2"),
                                                    H2O_molar_fraction=self.reboiler_vapor.get_specie_molar_fraction(id="H2O"))

                print("- Condenser")
                self.condenser_vapor = lab.GasStreamSum(streams=[self.nozzle_vapor, self.stripper_top_vapor])
                self.reflux = LiquidStream_Water(temp_K=273.15 + 25 * shape, flow_kg_h=self.condenser_vapor.get_specie_flow_kg_h(id="H2O"))

            print("- Heat Exchanger (Lean/Rich)")
            self.rich_hot, self.lean_cold = self.hx.react(LiquidStreamIn1=self.rich_semihot, LiquidStreamIn2=self.lean_intercooled)
            self.rich_hot = self.equilibrium.react(LiquidStreamIn=self.rich_hot, lr=lr)
            self.lean_cold = self.equilibrium.react(LiquidStreamIn=self.lean_cold, lr=lr)



# ---------------------------------------------------------------------------------


def __RecycleBin__():
    class LiquidStream_Amine(lab.LiquidStream):

        def __init__(self, stream_id, temp_K, flow_kg_h, CO2Load, MEA_mass_fraction=None, MDEA_mass_fraction=None,
                     PZ_mass_fraction=None, AMP_mass_fraction=None):

            super().__init__(stream_id=stream_id, solvent_id="H2O")

            z = np.zeros(shape=temp_K.shape)

            MEA_mass_fraction = z if MEA_mass_fraction is None else MEA_mass_fraction
            MDEA_mass_fraction = z if MDEA_mass_fraction is None else MDEA_mass_fraction
            PZ_mass_fraction = z if PZ_mass_fraction is None else PZ_mass_fraction
            AMP_mass_fraction = z if AMP_mass_fraction is None else AMP_mass_fraction

            self.add_info(key="MEA Mass Fraction", value=MEA_mass_fraction)
            self.add_info(key="MDEA Mass Fraction", value=MDEA_mass_fraction)
            self.add_info(key="PZ Mass Fraction", value=PZ_mass_fraction)
            self.add_info(key="AMP Mass Fraction", value=AMP_mass_fraction)
            self.add_info(key="Amine Mass Fraction",
                          value=MEA_mass_fraction + MDEA_mass_fraction + PZ_mass_fraction + AMP_mass_fraction)

            self.load_activity_coefficient(function=self.__activity_coefficient__)
            self.add_function(key="CO2 Load", function=self.CO2Load)
            self.load_heat_capacity_kJ_kgK(function=self.__heat_capacity_kJ_kgK__)
            self.load_density_kg_m3(function=self.__density_kg_m3__)

            if np.max(MEA_mass_fraction) > 0:
                self.add_specie(id="MEA", molar_mass_kg_kmol=61, charge=0)
                self.add_specie(id="MEA+", molar_mass_kg_kmol=62, charge=1)
                self.add_specie(id="MEACOO-", molar_mass_kg_kmol=104, charge=-1)
                self.add_specie(id="MEACOO", molar_mass_kg_kmol=105, charge=0)

                self.add_rxn_insta(id="MEA+ = MEA + H+",
                                   stoch={"MEA+": -1, "MEA": 1, "H+": 1},
                                   unit={"MEA+": "m", "MEA": "m", "H+": "m"},
                                   equilibrium_constant=self.__mea_dissociation_constant__)

                self.add_rxn_insta(id="MEACOO = MEACOO- + H+",
                                   stoch={"MEACOO": -1, "MEACOO-": 1, "H+": 1},
                                   unit={"MEACOO": "m", "MEACOO-": "m", "H+": "m"},
                                   equilibrium_constant=self.__mea_carbamate_dissociation_constant__)

                self.add_rxn_insta(id="CO2 + MEA = MEACOO- + H+",
                                   stoch={"CO2": -1, "MEA": -1, "MEACOO-": 1, "H+": 1},
                                   unit={"CO2": "m", "MEA": "m", "MEACOO-": "m", "H+": "m"},
                                   equilibrium_constant=self.__mea_stability_carbamate__)

            if np.max(MDEA_mass_fraction) > 0:
                self.add_specie(id="MDEA", molar_mass_kg_kmol=119, charge=0)
                self.add_specie(id="MDEA+", molar_mass_kg_kmol=120, charge=1)

                self.add_rxn_insta(id="MDEA+ = MDEA + H+",
                                   stoch={"MDEA+": -1, "H+": 1, "MDEA": 1},
                                   unit={"MDEA+": "m", "H+": "m", "MDEA": "m"},
                                   equilibrium_constant=self.__mdea_dissociation_constant__)

            if np.max(PZ_mass_fraction) > 0:
                self.add_specie(id="PZ", molar_mass_kg_kmol=86, charge=0)
                self.add_specie(id="PZ+", molar_mass_kg_kmol=87, charge=1)
                self.add_specie(id="PZ+2", molar_mass_kg_kmol=88, charge=2, )
                self.add_specie(id="PZCOO", molar_mass_kg_kmol=130, charge=0)
                self.add_specie(id="PZCOO-", molar_mass_kg_kmol=129, charge=-1)
                self.add_specie(id="PZ(COO)2-2", molar_mass_kg_kmol=172, charge=-2)
                self.add_specie(id="PZ(COO)2-", molar_mass_kg_kmol=173, charge=-1)

                # self.add_rxn_insta(id="PZ+2 = PZ+ + H+",
                #                   stoch={"PZ+2": -1, "PZ+": 1, "H+": 1},
                #                   unit={"PZ+2": "m", "PZ+": "m", "H+": "m"},
                #                   equilibrium_constant=self.__pz_dissociation_constant_K1__)

                self.add_rxn_insta(id="PZ+ = PZ + H+",
                                   stoch={"PZ+": -1, "PZ": 1, "H+": 1},
                                   unit={"PZ+": "m", "PZ": "m", "H+": "m"},
                                   equilibrium_constant=self.__pz_dissociation_constant_K2__)

                self.add_rxn_insta(id="PZCOO = PZCOO- + H+",
                                   stoch={"PZCOO": -1, "PZCOO-": 1, "H+": 1},
                                   unit={"PZCOO": "m", "PZCOO-": "m", "H+": "m"},
                                   equilibrium_constant=self.__pz_carbamate_dissociation_constant__)

                self.add_rxn_insta(id="PZ(COO)2- = PZ(COO)2-2 + H+",
                                   stoch={"PZ(COO)2-": -1, "PZ(COO)2-2": 1, "H+": 1},
                                   unit={"PZ(COO)2-": "m", "PZ(COO)2-2": "m", "H+": "m"},
                                   equilibrium_constant=self.__pz_dicarbamate_dissociation_constant__)

                self.add_rxn_insta(id="CO2 + PZ = PZCOO- + H+",
                                   stoch={"CO2": -1, "PZ": -1, "PZCOO-": 1, "H+": 1},
                                   unit={"CO2": "m", "PZ": "m", "PZCOO-": "m", "H+": "m"},
                                   equilibrium_constant=self.__pz_carbamate_stability__)

                self.add_rxn_insta(id="CO2 + PZCOO- = PZ(COO)2-2 + H+",
                                   stoch={"CO2": -1, "PZCOO-": -1, "PZ(COO)2-2": 1, "H+": 1},
                                   unit={"CO2": "m", "PZCOO-": "m", "PZ(COO)2-2": "m", "H+": "m"},
                                   equilibrium_constant=self.__pz_dicarbamate_stability__)

            if np.max(AMP_mass_fraction) > 0:
                self.add_specie(id="AMP", molar_mass_kg_kmol=89, charge=0)
                self.add_specie(id="AMP+", molar_mass_kg_kmol=90, charge=1)
                # self.add_specie(id="AMPCOO-", molar_mass_kg_kmol=89 + 44 - 1, charge=-1)

                self.add_rxn_insta(id="AMP+ = AMP + H+",
                                   stoch={"AMP+": -1, "AMP": 1, "H+": 1},
                                   unit={"AMP+": "m", "AMP": "m", "H+": "m"},
                                   equilibrium_constant=self.__amp_dissociation_constant__)

                # self.add_rxn_insta(id="CO2 + AMP = AMPCOO- + H+",
                #                   stoch={"CO2": -1, "AMP": -1, "AMPCOO-": 1, "H+": 1},
                #                   unit={"CO2": "m", "AMP": "m", "AMPCOO-": "m", "H+": "m"},
                #                   equilibrium_constant=self.__amp_stability_carbamate__)

            self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
            self.add_specie(id="CO3-2", molar_mass_kg_kmol=60, charge=-2)
            self.add_specie(id="HCO3-", molar_mass_kg_kmol=61, charge=-1)
            self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
            self.add_specie(id="H+", molar_mass_kg_kmol=1, charge=1)
            self.add_specie(id="OH-", molar_mass_kg_kmol=17, charge=-1)

            self.add_rxn_insta(id="H2O = H+ + OH-",
                               stoch={"H2O": -1, "H+": 1, "OH-": 1},
                               unit={"H2O": "x", "H+": "m", "OH-": "m"},
                               equilibrium_constant=self.__water_autoprotolysis_eq_constant__)

            self.add_rxn_insta(id="CO2 + H2O = HCO3- + H+",
                               stoch={"H2O": -1, "CO2": -1, "H+": 1, "HCO3-": 1},
                               unit={"H2O": "x", "CO2": "m", "H+": "m", "HCO3-": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_1_eq_const__)

            self.add_rxn_insta(id="HCO3- = CO3-2 + H+",
                               stoch={"HCO3-": -1, "CO3-2": 1, "H+": 1},
                               unit={"HCO3-": "m", "CO3-2": "m", "H+": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_2_eq_const__)

            self.add_vapor_pressure_bara_henry(id="CO2(g) = CO2(aq)",
                                               gas_id="CO2",
                                               liq_id="CO2",
                                               liq_unit="m",
                                               henrys_coefficient=self.__CO2_henrys_constant__)

            self.add_vapor_pressure_bara_raoult(id="H2O(g) = H2O(l)",
                                                gas_id="H2O",
                                                liq_id="H2O",
                                                pure_vapor_pressure_bara=self.__H2O_vapor_pressure_bara__)

            # Temperature and Flow
            self.set_solution_temp_K(value=temp_K)
            self.set_solution_flow_kg_h(value=flow_kg_h)

            # Number of moles in 1 kg of Unloaded Solution
            n_MDEA = MDEA_mass_fraction / 119
            n_PZ = PZ_mass_fraction / 86
            n_MEA = MEA_mass_fraction / 61
            n_AMP = AMP_mass_fraction / 89

            # Number of Moles CO2 in 1 kg Unloaded Solution, After Loading
            n_CO2 = CO2Load * (n_MDEA + n_MEA + n_PZ + n_AMP)

            CO2_mass_fraction = 44 * n_CO2

            # Concentrations
            H2O_mass_fraction = 1 - MEA_mass_fraction - MDEA_mass_fraction - PZ_mass_fraction - AMP_mass_fraction

            # Setting all mass fractions to zero initially
            zero = 0 * np.ones(shape=H2O_mass_fraction.shape)
            for id in self.specie.keys():
                self.set_specie_mass_fraction(id=id, value=zero)

            # Mass Fractions
            self.set_specie_mass_fraction(id="CO2", value=CO2_mass_fraction)
            self.set_specie_mass_fraction(id="H2O", value=H2O_mass_fraction)

            if "MDEA" in self.specie.keys():
                self.set_specie_mass_fraction(id="MDEA", value=MDEA_mass_fraction)

            if "PZ" in self.specie.keys():
                self.set_specie_mass_fraction(id="PZ", value=PZ_mass_fraction)

            if "MEA" in self.specie.keys():
                self.set_specie_mass_fraction(id="MEA", value=MEA_mass_fraction)

            if "AMP" in self.specie.keys():
                self.set_specie_mass_fraction(id="AMP", value=AMP_mass_fraction)

            self.normalize_mass_fractions()

        def CO2Load(self, LiquidStream):

            x_CO2 = 0
            x_Amine = 0

            C = {"CO2": 1, "HCO3-": 1, "CO3-2": 1,
                 "MEACOO-": 1, "MEACOO": 1,
                 "PZCOO": 1, "PZCOO-": 1,
                 "PZ(COO)2-2": 2, "PZ(COO)2-": 2,
                 "AMPCOO-": 1}

            A = {"MEA": 1, "MEACOO": 1, "MEACOO-": 1, "MEA+": 1,
                 "MDEA": 1, "MDEA+": 1,
                 "PZ": 1, "PZ+": 1, "PZ+2": 1, "PZCOO": 1, "PZCOO-": 1, "PZ(COO)2-2": 1, "PZ(COO)2-": 1,
                 "AMP": 1, "AMPCOO-": 1, "AMP+": 1}

            for c in C.keys():
                if c in LiquidStream.specie.keys():
                    x_CO2 = x_CO2 + C[c] * LiquidStream.get_specie_molar_fraction(id=c)

            for a in A.keys():
                if a in LiquidStream.specie.keys():
                    x_Amine = x_Amine + A[a] * LiquidStream.get_specie_molar_fraction(id=a)

            alpha = np.maximum(x_CO2 / np.maximum(x_Amine, 10 ** (-9)), 10 ** (-9))
            return alpha

        def __heat_capacity_kJ_kgK__(self, LiquidStream):
            w_H2O = 1 - self.get_info(id="Amine Mass Fraction")
            w_Amine = self.get_info(id="Amine Mass Fraction")
            T = LiquidStream.get_solution_temp_K()
            alpha = LiquidStream.CO2Load(LiquidStream)
            cp = 4.2 * w_H2O + 2.8 * w_Amine + (0.3 / 75) * (T - 298.15) - (0.3 / 0.5) * alpha
            return cp

        def __density_kg_m3__(self, LiquidStream):
            rho = 1050 * np.ones(shape=(LiquidStream.temp_K.shape))
            return rho

        def __activity_coefficient__(self, LiquidStream, id):
            I = LiquidStream.get_solution_ionic_strength_mol_kg()
            T = LiquidStream.temp_K
            z = LiquidStream.get_specie_charge(id)
            log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))
            gamma = 10 ** log10_gamma
            return gamma

        def __CO2_henrys_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            H_CO2 = 1.153 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)
            # I = LiquidStream.get_solution_ionic_strength_mol_kg()
            # H_CO2 = H_CO2 * 10**(- I * 0.09)
            return H_CO2

        def __H2O_vapor_pressure_bara__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            T = np.minimum(T, 273.15 + 150)
            pc = 220.64
            Tc = 647.096
            tau = 1 - T / Tc
            a1 = -7.85951783
            a2 = 1.84408259
            a3 = -11.7866497
            a4 = 22.6807411
            a5 = -15.9618719
            a6 = 1.80122502
            p = pc * np.exp(
                (Tc / T) * (
                            a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 3 + a4 * tau ** 3.5 + a5 * tau ** 4 + a6 * tau ** 7.5))
            return p

        def __water_autoprotolysis_eq_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            Kw = 10 ** (-14) * np.exp(-13445.9 * (1 / T - 1 / 298) - 22.48 * np.log(T / 298))
            return Kw

        def __carbonic_acid_rxn_1_eq_const__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-6.32) * np.exp(5139 * (1 / T - 1 / 298) + 14.5258479 * np.log(T / 298))
            return K

        def __carbonic_acid_rxn_2_eq_const__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-10.33) * np.exp(22062 * (1 / T - 1 / 298) + 67.264072 * np.log(T / 298))
            return K

        def __mea_dissociation_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-9.27) * np.exp(-6375 * (1 / T - 1 / 313))
            return K

        def __mea_carbamate_dissociation_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-7.0) * np.exp(-2766 * (1 / T - 1 / 313))
            return K

        def __mea_stability_carbamate__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 1.3217 * 10 ** (-5) * np.exp(3018 * (1 / T - 1 / 313))
            return K

        def __mdea_dissociation_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-8.75) * np.exp(-4900 * (1 / T - 1 / 298.15))  # 8.35
            return K

        def __pz_dissociation_constant_K1__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-5.07) * np.exp(-3850 * (1 / T - 1 / 313))
            return K

        def __pz_dissociation_constant_K2__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-9.30) * np.exp(-4570 * (1 / T - 1 / 313))
            return K

        def __pz_carbamate_dissociation_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-8.90) * np.exp(-2165 * (1 / T - 1 / 313))
            return K

        def __pz_dicarbamate_dissociation_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-8.44) * np.exp(-7096 * (1 / T - 1 / 313))
            return K

        def __pz_carbamate_stability__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 4.147 * 10 ** (-5) * np.exp(7230 * (1 / T - 1 / 313))
            return K

        def __pz_dicarbamate_stability__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 5.359 * 10 ** (-7) * np.exp(1513 * (1 / T - 1 / 313))
            return K

        def __amp_dissociation_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-9.37) * np.exp(-6682 * (1 / T - 1 / 313))
            return K

        def __amp_stability_carbamate__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 4.28 * 10 ** (-11) * np.exp(4317 * (1 / T - 1 / 313))
            return K

    def __density2_kg_m3__(self, LiquidStream):
        T = LiquidStream.get_solution_temp_K()
        alpha = LiquidStream.get_function_value(id="CO2 Load")
        shape = np.ones(shape=T.shape)

        # Unloaded Mass Fractions
        w = {}
        w["MEA"] = LiquidStream.info["MEA Mass Fraction"] if "MEA Mass Fraction" in LiquidStream.info.keys() else 0 * shape
        w["MDEA"] = LiquidStream.info["MDEA Mass Fraction"] if "MDEA Mass Fraction" in LiquidStream.info.keys() else 0 * shape
        w["PZ"] = LiquidStream.info["PZ Mass Fraction"] if "PZ Mass Fraction" in LiquidStream.info.keys() else 0 * shape

        # Total Mass Fractions of Amines and Aminos
        w_Amino = 0
        for id in w.keys():
            w_Amino = w_Amino + w[id]

        # Mass Ratio of the Amines and Aminos w.r.t. Overall Amine/Amino Mass Fraction
        r = {}
        for id in w.keys():
            r[id] = w[id] / w_Amino

        # Mass Fraction of Water
        w["H2O"] = 1 - w_Amino

        # Molar Masses
        M = {}
        M["MEA"] = 61
        M["PZ"] = 86
        M["MDEA"] = 119

        # Average Molar Mass of Amine/Aminos
        M_Amino = 0
        for id in M.keys():
            M_Amino = M_Amino + r[id] * M[id]

        # Density in pure form
        rho = {}
        rho["H2O"] = density_H2O_kg_m3(LiquidStream)
        rho["MEA"] = 1000 + ((1000 - 920) / (310 - 410)) * (T - 310)
        rho["MDEA"] = 1035 + ((1035 - 920) / (300 - 450)) * (T - 300)
        rho["PZ"] = 1000 + ((1000 - 950) / (313 - 393)) * (T - 313)

        # Density of Ideal, Unloaded Solution
        rho_ideal = 0
        for id in rho.keys():
            rho_ideal = rho_ideal + w[id] / rho[id]
        rho_ideal = rho_ideal ** (-1)

        # Density of Non-Ideal, Unloaded Solution
        rho_unloaded = rho_ideal + 198 * w["H2O"] ** 1.955 * w_Amino ** 1.34

        # Density of Loaded Solution
        rho_loaded = rho_unloaded + 37.76518 * 1000 * alpha * w_Amino / M_Amino
        return rho_loaded

    def Mass_Transfer_CO2_kmol_m3s_OLD(self, Column):
        ap = Column.get_packing_area_m2_m3()
        m = Column.LiquidStream.get_solution_flow_kg_h()
        A = Column.get_cross_sectional_area_m2()
        ae = ap * self.beta
        flux = Column.LiquidStream.__CO2_flux_stirred_cell_reactor_kmol_m2s__(Column.GasStream, Column.LiquidStream)
        r = ae * flux
        return r

    def Mass_Transfer_CO2_Absorber_kmol_m3s_OLD(self, Column):

        # Packing Geometry
        ap = Column.get_packing_area_m2_m3()

        # Various
        #D_CO2 = Column.LiquidStream.get_specie_diffusivity_m2_s(id="CO2")
        D_CO2 = 1.92 * 10**(-9)
        T_liq = Column.LiquidStream.get_solution_temp_K()
        # H_CO2 = Column.LiquidStream.vapor_pressure_bara["CO2(g) = CO2(aq)"]["H"](Column.LiquidStream)
        H_CO2 = 1.153 * np.exp((-T_liq * (1713 * (1 - 0.0015453 * T_liq) ** (1 / 3) + 3680) + 1198506) / T_liq ** 2)

        # Driving Force
        p_CO2_vap = Column.LiquidStream.get_specie_vapor_pressure_bara(gas_id="CO2")
        p_CO2 = Column.GasStream.get_specie_pressure_bara(id="CO2")

        # Mass Transfer Coefficients
        kL, kG, kH, ae = self.Transfer_Coefficients(Column=Column, gas_specie_id="CO2", liq_specie_id="CO2")

        # Reaction Rate Coefficients
        k = {}
        k["MEA"] = 6000 * np.exp(-5000 * (1 / T_liq - 1 / 298.15))
        k["PZ"] = 40000 * np.exp(-5000 * (1 / T_liq - 1 / 298.15))
        k["OH-"] = 8416 * np.exp(-5000 * (1 / T_liq - 1 / 298.15))
        k["Lys-"] = 28000 * np.exp(-5000 * (1 / T_liq - 1 / 298.15))
        # k["MEA"] = 5993 * np.exp(-5400 * (1 / T_liq - 1 / 298.15))
        # k["PZ"] = 58000 * np.exp(-4209 * (1 / T_liq - 1 / 298.15))
        # k["PZCOO-"] = 14500 * np.exp(-4209 * (1 / T_liq - 1 / 298.15))
        # k["OH-"] = 8416 * np.exp(-6667 * (1 / T_liq - 1 / 298.15))
        k["AMP"] = 1500 * np.exp(-5176 * (1 / T_liq - 1 / 298.15))
        k["MDEA"] = 6.06 * np.exp(-5922 * (1 / T_liq - 1 / 298.15))

        # Enhancement Factor times Liquid Mass Transfer Coefficient
        EkL = 0
        for id in k.keys():
            if id in Column.LiquidStream.specie.keys():
                EkL = EkL + np.sqrt(D_CO2 * k[id] * Column.LiquidStream.get_specie_molarity_kmol_m3(id))
        r = EkL * ae * H_CO2 * (p_CO2 - p_CO2_vap)
        return r

    def Mass_Transfer_CO2_Stripper_kmol_m3s_OLD(self, Column):

        # Various
        T_liq = Column.LiquidStream.get_solution_temp_K()
        #H_CO2 = Column.LiquidStream.vapor_pressure_bara["CO2(g) = CO2(aq)"]["H"](Column.LiquidStream)
        H_CO2 = 1.153 * np.exp((-T_liq * (1713 * (1 - 0.0015453 * T_liq) ** (1 / 3) + 3680) + 1198506) / T_liq ** 2)
        g = 9.81
        R = 0.08314
        p_CO2_vap = Column.LiquidStream.get_specie_vapor_pressure_bara(gas_id="CO2")
        p_CO2 = Column.GasStream.get_specie_pressure_bara(id="CO2")

        # Mass Transfer Coefficients
        kL, kG, kH, ae = self.Transfer_Coefficients(Column=Column, gas_specie_id="CO2", liq_specie_id="CO2")

        # Enhancement Factor
        nu = {"MEA": 2, "PZ": 2, "PZCOO-":2, "OH-": 2, "AMP": 2, "MDEA": 1, "Lys-": 2}
        E = 1
        for id in nu.keys():
            if id in Column.LiquidStream.specie.keys():
                E = E + np.sqrt(1 / 2) * (1 / nu[id]) * (Column.LiquidStream.get_specie_molarity_kmol_m3(id) / (H_CO2 * p_CO2))

        KG = ((R * T_liq / kG) + 1 / (H_CO2 * E * kL)) ** (-1)
        r = KG * ae * (p_CO2 - p_CO2_vap)
        return r

    class __StructuredPacking__():

        def __init__(self):
            pass

        def Liquid_Holdup_m3_m3(self, Column):
            theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
            ap = Column.get_packing_area_m2_m3()
            mu = 0.89 * 10 ** (-3)
            rho = Column.LiquidStream.get_solution_density_kg_m3()
            v_LS = Column.get_superficial_liquid_velocity_m_s()
            g = 9.81
            enu = 3 * np.abs(v_LS) * mu * ap ** 2
            den = rho * g * np.sin(theta) ** 2
            h_liq = (enu / den) ** (1 / 3)
            return h_liq

        def Mass_Transfer_H2O_kmol_m3s(self, Column):
            p_H2O = Column.GasStream.get_specie_pressure_bara(id="H2O")
            p_H2O_vap = Column.LiquidStream.get_specie_vapor_pressure_bara(
                gas_id="H2O")  # / Column.LiquidStream.get_specie_molar_fraction(id="H2O")
            T = Column.GasStream.get_gas_temp_K()
            R = 0.08314
            kL, kG, kH, ae = self.Transfer_Coefficients(Column=Column, gas_specie_id="H2O", liq_specie_id="H2O")
            KGa = kG * ae / (R * T)
            r = KGa * (p_H2O - p_H2O_vap)
            return r

        def Mass_Transfer_H2O_kJ_kmol(self, Column):
            T0 = Column.LiquidStream.temp_K
            p0 = Column.LiquidStream.vapor_pressure_bara["H2O(g) = H2O(l)"]["p0"](Column.LiquidStream)
            Column.LiquidStream.temp_K = Column.LiquidStream.temp_K + 0.05
            T1 = Column.LiquidStream.temp_K
            p1 = Column.LiquidStream.vapor_pressure_bara["H2O(g) = H2O(l)"]["p0"](Column.LiquidStream)
            Column.LiquidStream.temp_K = Column.LiquidStream.temp_K - 0.05
            h = 8.314 * (np.log(1 / p1) - np.log(1 / p0)) / ((1 / T1) - (1 / T0))
            return h

        def Mass_Transfer_CO2_Absorber_kmol_m3s(self, Column):
            kL, kG, kH, ae = self.Transfer_Coefficients(Column=Column, gas_specie_id="CO2", liq_specie_id="CO2")
            flux = Column.LiquidStream.CO2_flux_stirred_cell_reactor_kmol_m2s(Column.GasStream, Column.LiquidStream)
            r = ae * flux
            return r

        def Mass_Transfer_CO2_kJ_kmol(self, Column):
            T0 = Column.LiquidStream.temp_K
            H0 = Column.LiquidStream.vapor_pressure_bara["CO2(g) = CO2(aq)"]["H"](Column.LiquidStream)
            Column.LiquidStream.temp_K = Column.LiquidStream.temp_K + 0.05
            T1 = Column.LiquidStream.temp_K
            H1 = Column.LiquidStream.vapor_pressure_bara["CO2(g) = CO2(aq)"]["H"](Column.LiquidStream)
            Column.LiquidStream.temp_K = Column.LiquidStream.temp_K - 0.05
            q = 8.314 * (np.log(H1) - np.log(H0)) / ((1 / T1) - (1 / T0))
            return q

        def Heat_Transfer_kW_m3(self, Column):
            T_gas = Column.GasStream.get_gas_temp_K()
            T_liq = Column.LiquidStream.get_solution_temp_K()
            kL, kG, kH, ae = self.Transfer_Coefficients(Column=Column, gas_specie_id="H2O", liq_specie_id="H2O")
            q = kH * ae * (T_gas - T_liq)
            return q

        def Transfer_Coefficients(self, Column, gas_specie_id, liq_specie_id):
            # Dimensions
            ap = Column.get_packing_area_m2_m3()
            theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
            M = (3 * ap ** 3 * np.sin(theta) * np.cos(theta)) / (16 * (np.sin(theta) ** 2 + 1) ** (3 / 2))
            Mi = M / ap ** 3

            # Superficial Gas Velocity [m/s] and Liquid Load [m/s]
            v_gas = Column.get_superficial_gas_velocity_m_s()
            v_liq = Column.get_superficial_liquid_velocity_m_s()

            # Various
            p = Column.GasStream.get_gas_pressure_bara()
            # mu = Column.LiquidStream.get_solution_viscosity_Pas()
            mu = 0.89 * 10 ** (-3)

            # Mass Transfer Coefficients
            kL = 275 * 10 ** (-6) * (Mi / 0.035) ** 0.42 * (ap / 350) ** 0.26 * (3600 * v_liq / 10) ** 0.5 * (
                        1000 * mu / 0.89) ** (-0.75)  # 32
            kG = 57 * 10 ** (-3) * (v_gas / 2.4) ** 0.8 * (p / 1.0) ** 0.4 * (Mi / 0.035) ** 0.3 * (ap / 350) ** 0.42

            # Heat Transfer Coefficient
            kH = 79 * 10 ** (-3) * (v_gas / 2.4) ** 0.8 * (p / 1.0) ** 0.4 * (Mi / 0.035) ** 0.3 * (ap / 350) ** 0.42

            # Effective Interface Area
            ae = ap * 0.54 * (3600 * v_liq * 350 / (10 * ap)) ** 0.15
            return kL, kG, kH, ae

    class WashWaterSection(lab.Column_StructuredPacking_CounterCurrent):

        def __init__(self, height_m=1.6, num_of_heights=50, cross_sectional_area_m2=0.5, void_fraction_m3_m3=0.98,
                     packing_area_m2_m3=350, corrugation_angle_degree=60, enhancement_factor=1.0):
            self.enhancement_factor = enhancement_factor

            super().__init__(height_m=height_m,
                             num_of_heights=num_of_heights,
                             cross_sectional_area_m2=cross_sectional_area_m2,
                             void_fraction_m3_m3=void_fraction_m3_m3,
                             packing_area_m2_m3=packing_area_m2_m3,
                             corrugation_angle_degree=corrugation_angle_degree)

            self.add_mass_transfer_kmol_m3s(id="H2O(g) -> H2O(aq)",
                                            stoch_gas={"H2O": -1},
                                            stoch_liq={"H2O": 1},
                                            rate_kmol_m3s=self.Mass_Transfer_H2O_kmol_m3s,
                                            exothermic_heat_kJ_kmol=__StructuredPacking__().Mass_Transfer_H2O_kJ_kmol)

            self.add_heat_transfer_kW_m3(heat_transfer_kW_m3=self.Heat_Transfer_kW_m3)
            self.add_liquid_holdup_m3_m3(liquid_holdup_m3_m3=__StructuredPacking__().Liquid_Holdup_m3_m3)
            self.add_pressure_drop_Pa_m(presure_drop_Pa_m=None)

        def Mass_Transfer_H2O_kmol_m3s(self, Column):
            r = __StructuredPacking__().Mass_Transfer_H2O_kmol_m3s(Column)
            r = r * self.enhancement_factor

            T = Column.GasStream.get_gas_temp_K()
            y_H2O = Column.GasStream.get_specie_molar_fraction(id="H2O")

            # r = r * (1 + 10 * y_H2O)

            return r

        def Heat_Transfer_kW_m3(self, Column):
            q = __StructuredPacking__().Heat_Transfer_kW_m3(Column)
            q = q * self.enhancement_factor
            return q

    class WashWaterSection_CoCurrent(lab.Column_StructuredPacking_CoCurrent):

        def __init__(self, height_m=1.6, cross_sectional_area_m2=0.5, void_fraction_m3_m3=0.98, packing_area_m2_m3=350,
                     corrugation_angle_degree=60, enhancement_factor=1.0):
            self.enhancement_factor = enhancement_factor

            super().__init__(height_m=height_m,
                             cross_sectional_area_m2=cross_sectional_area_m2,
                             void_fraction_m3_m3=void_fraction_m3_m3,
                             packing_area_m2_m3=packing_area_m2_m3,
                             corrugation_angle_degree=corrugation_angle_degree)

            self.add_mass_transfer_kmol_m3s(id="H2O(g) -> H2O(aq)",
                                            stoch_gas={"H2O": -1},
                                            stoch_liq={"H2O": 1},
                                            rate_kmol_m3s=self.Mass_Transfer_H2O_kmol_m3s,
                                            exothermic_heat_kJ_kmol=__StructuredPacking__().Mass_Transfer_H2O_kJ_kmol)

            self.add_heat_transfer_kW_m3(heat_transfer_kW_m3=self.Heat_Transfer_kW_m3)
            self.add_liquid_holdup_m3_m3(liquid_holdup_m3_m3=__StructuredPacking__().Liquid_Holdup_m3_m3)
            self.add_pressure_drop_Pa_m(presure_drop_Pa_m=None)

        def Mass_Transfer_H2O_kmol_m3s(self, Column):
            r = __StructuredPacking__().Mass_Transfer_H2O_kmol_m3s(Column)
            r = r * self.enhancement_factor
            return r

        def Heat_Transfer_kW_m3(self, Column):
            q = __StructuredPacking__().Heat_Transfer_kW_m3(Column)
            q = q * self.enhancement_factor
            return q

    class Stripper(lab.Column_StructuredPacking_CounterCurrent):

        def __init__(self, height_m=3.4, num_of_heights=90, cross_sectional_area_m2=0.096, void_fraction_m3_m3=0.98,
                     packing_area_m2_m3=250, corrugation_angle_degree=60, enhancement_factor=0.25):
            self.enhancement_factor = enhancement_factor

            super().__init__(height_m=height_m,
                             num_of_heights=num_of_heights,
                             cross_sectional_area_m2=cross_sectional_area_m2,
                             void_fraction_m3_m3=void_fraction_m3_m3,
                             packing_area_m2_m3=packing_area_m2_m3,
                             corrugation_angle_degree=corrugation_angle_degree)

            self.add_mass_transfer_kmol_m3s(id="CO2(g) -> CO2(aq)",
                                            stoch_gas={"CO2": -1},
                                            stoch_liq={"CO2": 1},
                                            rate_kmol_m3s=self.Mass_Transfer_CO2_kmol_m3s,
                                            exothermic_heat_kJ_kmol=__StructuredPacking__().Mass_Transfer_CO2_kJ_kmol)

            self.add_mass_transfer_kmol_m3s(id="H2O(g) -> H2O(aq)",
                                            stoch_gas={"H2O": -1},
                                            stoch_liq={"H2O": 1},
                                            rate_kmol_m3s=self.Mass_Transfer_H2O_kmol_m3s,
                                            exothermic_heat_kJ_kmol=__StructuredPacking__().Mass_Transfer_H2O_kJ_kmol)

            self.add_heat_transfer_kW_m3(heat_transfer_kW_m3=self.Heat_Transfer_kW_m3)
            self.add_liquid_holdup_m3_m3(liquid_holdup_m3_m3=__StructuredPacking__().Liquid_Holdup_m3_m3)
            self.add_pressure_drop_Pa_m(presure_drop_Pa_m=None)

        def Mass_Transfer_CO2_kmol_m3s(self, Column):
            r = __StructuredPacking__().Mass_Transfer_CO2_Stripper_kmol_m3s(Column)
            r = r * self.enhancement_factor
            return r

        def Mass_Transfer_H2O_kmol_m3s(self, Column):
            r = __StructuredPacking__().Mass_Transfer_H2O_kmol_m3s(Column)
            r = r * self.enhancement_factor
            return r

        def Heat_Transfer_kW_m3(self, Column):
            q = __StructuredPacking__().Heat_Transfer_kW_m3(Column)
            q = q * self.enhancement_factor
            return q

    class Rectifier(lab.Column_StructuredPacking_CounterCurrent):

        def __init__(self, height_m=0.8, num_of_heights=90, cross_sectional_area_m2=0.096, void_fraction_m3_m3=0.98,
                     packing_area_m2_m3=500, corrugation_angle_degree=60, enhancement_factor=0.1):
            self.enhancement_factor = enhancement_factor

            super().__init__(height_m=height_m,
                             num_of_heights=num_of_heights,
                             cross_sectional_area_m2=cross_sectional_area_m2,
                             void_fraction_m3_m3=void_fraction_m3_m3,
                             packing_area_m2_m3=packing_area_m2_m3,
                             corrugation_angle_degree=corrugation_angle_degree)

            self.add_mass_transfer_kmol_m3s(id="H2O(g) -> H2O(aq)",
                                            stoch_gas={"H2O": -1},
                                            stoch_liq={"H2O": 1},
                                            rate_kmol_m3s=self.Mass_Transfer_H2O_kmol_m3s,
                                            exothermic_heat_kJ_kmol=__StructuredPacking__().Mass_Transfer_H2O_kJ_kmol)

            self.add_heat_transfer_kW_m3(heat_transfer_kW_m3=self.Heat_Transfer_kW_m3)
            self.add_liquid_holdup_m3_m3(liquid_holdup_m3_m3=__StructuredPacking__().Liquid_Holdup_m3_m3)
            self.add_pressure_drop_Pa_m(presure_drop_Pa_m=None)

        def Mass_Transfer_H2O_kmol_m3s(self, Column):
            r = __StructuredPacking__().Mass_Transfer_H2O_kmol_m3s(Column)
            r = r * self.enhancement_factor
            return r

        def Heat_Transfer_kW_m3(self, Column):
            q = __StructuredPacking__().Heat_Transfer_kW_m3(Column)
            q = q * self.enhancement_factor
            return q

    class _PackedColumn():

        def __init__(self):
            pass

        def Mass_Transfer_CO2_kmol_m3s(self, Column):

            # Features
            ap = Column.get_packing_area_m2_m3()
            theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
            M = (3 * ap ** 3 * np.sin(theta) * np.cos(theta)) / (16 * (np.sin(theta) ** 2 + 1) ** (3 / 2))
            Mi = M / ap ** 3
            v_gas = Column.get_superficial_gas_velocity_m_s()
            v_liq = Column.get_superficial_liquid_velocity_m_s()
            m_liq = Column.LiquidStream.get_solution_flow_kg_h()
            A = Column.get_cross_sectional_area_m2()
            T = Column.LiquidStreamIn.temp_K
            wA = Column.LiquidStream.get_info(id="Amine Mass Fraction")
            alpha = Column.LiquidStream.CO2Load(Column.LiquidStream)
            rho = Column.LiquidStream.get_solution_density_kg_m3()

            # Driving Force
            p_CO2 = Column.GasStream.get_specie_pressure_bara(id="CO2")
            p_CO2_vap = Column.LiquidStream.get_specie_vapor_pressure_bara(gas_id="CO2")

            # Henry's Law Coefficient
            H_CO2 = 1.153 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)

            # Viscosity
            mu_H2O = 0.89 * np.exp(1800 * (1 / T - 1 / 298))
            mu_Sol = 1.34 * np.exp(
                4.41 * wA ** 1.68 + 6.54 * (wA ** 1.71) * alpha + (1800 + 2500 * wA) * (1 / T - 1 / 298))

            # Diffusivity
            D_CO2_in_H2O = 1.91 * 10 ** (-9) * np.exp(-2119 * (1 / T - 1 / 298))
            D_CO2_in_Sol = D_CO2_in_H2O * (mu_Sol / mu_H2O) ** (-0.8)

            # Effective Interface Area
            ae = ap * 0.45 * (v_gas / 2.4) ** 0.27 * ((m_liq / A) * (0.5 / 5000)) ** 0.19

            # Liquid Mass Transfer Coefficient
            kL = 67.2 * 10 ** (-6)
            # kL = 32 * 10 ** (-6)
            kL = kL * (D_CO2_in_Sol / (1.91 * 10 ** (-9))) ** 0.5
            kL = kL * (Mi / 0.035) ** 0.42
            kL = kL * (ap / 350) ** 0.26
            kL = kL * ((m_liq / A) * (0.5 / 5000)) ** 0.5
            kL = kL * (mu_Sol / 0.89) ** (-0.38)

            # Gas Mass Transfer Coefficient
            kG_RT = 1.84 * 10 ** (-3)
            kG_RT = kG_RT * (Mi / 0.035) ** 0.30
            kG_RT = kG_RT * (ap / 350) ** 0.42
            kG_RT = kG_RT * (v_gas / 2.4) ** 0.8

            # Reaction Rate
            k = {}
            k["MEA"] = 5993 * np.exp(-5400 * (1 / T - 1 / 298.15))
            k["PZ"] = 58000 * np.exp(-4209 * (1 / T - 1 / 298.15))
            k["PZCOO-"] = k["PZ"] / 4
            k["OH-"] = 8416 * np.exp(-6667 * (1 / T - 1 / 298.15))
            k["AMP"] = 1500 * np.exp(-5176 * (1 / T - 1 / 298.15))

            # Stochiometric Coefficients
            nu = {}
            nu["MEA"] = 2
            nu["PZ"] = 2
            nu["PZCOO-"] = 1
            nu["OH-"] = 2
            nu["AMP"] = 1

            # Molarity and Molality
            c = {}
            m = {}
            for id in k.keys():
                if id in Column.LiquidStream.specie.keys():
                    c[id] = rho * Column.LiquidStream.get_specie_mass_fraction(
                        id) / Column.LiquidStream.get_specie_molar_mass_kg_kmol(id)
                    m[id] = Column.LiquidStream.get_specie_molality_mol_kg(id)

            # Hatta Number
            kc = 0
            for id in k.keys():
                if id in Column.LiquidStream.specie.keys():
                    kc = kc + k[id] * c[id]
            Ha = (1 / kL) * np.sqrt(D_CO2_in_Sol * kc)

            # Enhancement Factor, Upper Bound
            m_nu = 0
            for id in k.keys():
                if id in Column.LiquidStream.specie.keys():
                    m_nu = m_nu + m[id] / nu[id]
            E_inf = 1 + np.sqrt(1 / 2) * m_nu / (H_CO2 * p_CO2)
            # E = - Ha**2 / (2*(E_inf - 1)) + np.sqrt( (Ha**4/(4*(E_inf - 1)**2)) + (E_inf*Ha**2/(E_inf-1)) + 1)

            # Enhancement Factor
            E = (1 / Ha + 1 / E_inf) ** (-1)

            # Overall Mass Transfer Coefficient
            KGa = (1 / (kG_RT * ae) + 1 / (E * kL * ae * H_CO2)) ** (-1)

            # Absorption Rate
            r = KGa * (p_CO2 - p_CO2_vap)
            return r

        def Mass_Transfer_CO2_kJ_kmol(self, Column):
            T0 = Column.LiquidStream.temp_K
            H0 = Column.LiquidStream.vapor_pressure_bara["CO2(g) = CO2(aq)"]["H"](Column.LiquidStream)
            Column.LiquidStream.temp_K = Column.LiquidStream.temp_K + 0.05
            T1 = Column.LiquidStream.temp_K
            H1 = Column.LiquidStream.vapor_pressure_bara["CO2(g) = CO2(aq)"]["H"](Column.LiquidStream)
            Column.LiquidStream.temp_K = Column.LiquidStream.temp_K - 0.05
            q = 8.314 * (np.log(H1) - np.log(H0)) / ((1 / T1) - (1 / T0))
            return q

        def Mass_Transfer_H2O_kmol_m3s(self, Column):

            # Features
            ap = Column.get_packing_area_m2_m3()
            theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
            M = (3 * ap ** 3 * np.sin(theta) * np.cos(theta)) / (16 * (np.sin(theta) ** 2 + 1) ** (3 / 2))
            Mi = M / ap ** 3
            v_gas = Column.get_superficial_gas_velocity_m_s()
            v_liq = Column.get_superficial_liquid_velocity_m_s()
            m_liq = Column.LiquidStream.get_solution_flow_kg_h()
            A = Column.get_cross_sectional_area_m2()

            # Driving Force
            p_H2O = Column.GasStream.get_specie_pressure_bara(id="H2O")
            p_H2O_vap = Column.LiquidStream.get_specie_vapor_pressure_bara(gas_id="H2O")

            # Effective Interface Area
            ae = ap * 0.45 * (v_gas / 2.4) ** 0.27 * ((m_liq / A) * (0.5 / 5000)) ** 0.19

            # Gas Mass Transfer Coefficient
            kG_RT = 1.84 * 10 ** (-3)
            kG_RT = kG_RT * (Mi / 0.035) ** 0.30
            kG_RT = kG_RT * (ap / 350) ** 0.42
            kG_RT = kG_RT * (v_gas / 2.4) ** 0.8

            # Overall Mass Transfer Coefficient
            KGa = kG_RT * ae

            # Mass Transfer
            r = KGa * (p_H2O - p_H2O_vap)
            return r

        def Mass_Transfer_H2O_kJ_kmol(self, Column):
            T0 = Column.LiquidStream.temp_K
            p0 = Column.LiquidStream.vapor_pressure_bara["H2O(g) = H2O(l)"]["p0"](Column.LiquidStream)
            Column.LiquidStream.temp_K = Column.LiquidStream.temp_K + 0.05
            T1 = Column.LiquidStream.temp_K
            p1 = Column.LiquidStream.vapor_pressure_bara["H2O(g) = H2O(l)"]["p0"](Column.LiquidStream)
            Column.LiquidStream.temp_K = Column.LiquidStream.temp_K - 0.05
            h = 8.314 * (np.log(1 / p1) - np.log(1 / p0)) / ((1 / T1) - (1 / T0))
            return h

        def Heat_Transfer_kW_m3(self, Column):

            # Features
            ap = Column.get_packing_area_m2_m3()
            theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
            M = (3 * ap ** 3 * np.sin(theta) * np.cos(theta)) / (16 * (np.sin(theta) ** 2 + 1) ** (3 / 2))
            Mi = M / ap ** 3
            m_liq = Column.LiquidStream.get_solution_flow_kg_h()
            A = Column.get_cross_sectional_area_m2()
            T_gas = Column.GasStream.get_gas_temp_K()
            T_liq = Column.LiquidStream.get_solution_temp_K()
            v_gas = Column.get_superficial_gas_velocity_m_s()

            # Effective Interface Area
            ae = ap * 0.45 * (v_gas / 2.4) ** 0.27 * ((m_liq / A) * (0.5 / 5000)) ** 0.19

            # Heat Transfer Coefficient
            kH = 63 * 10 ** (-3)
            kH = kH * (Mi / 0.035) ** 0.3
            kH = kH * (ap / 350) ** 0.42
            kH = kH * (v_gas / 2.4) ** 0.8

            # Heat Transfer [kW/m3]
            q = kH * ae * (T_gas - T_liq)
            return q

        def Liquid_Holdup_m3_m3(self, Column):
            theta = Column.get_corrugation_angle_degree() * (np.pi / 180)
            ap = Column.get_packing_area_m2_m3()
            mu = 0.89 * 10 ** (-3)
            rho = Column.LiquidStream.get_solution_density_kg_m3()
            v_LS = Column.get_superficial_liquid_velocity_m_s()
            g = 9.81
            enu = 3 * np.abs(v_LS) * mu * ap ** 2
            den = rho * g * np.sin(theta) ** 2
            h_liq = (enu / den) ** (1 / 3)
            return h_liq

    class LiquidStream_KLys9(lab.LiquidStream):

        def __init__(self, temp_K, flow_kg_h, KLys_mass_fraction, CO2Load):

            super().__init__(stream_id="", solvent_id="H2O")
            z = np.zeros(shape=temp_K.shape)

            self.load_heat_capacity_kJ_kgK(function=self.__heat_capacity_kJ_kgK__)
            self.load_density_kg_m3(function=self.__density_kg_m3__)
            self.load_activity_coefficient(function=self.__activity_coefficient__)
            self.add_info(key="Amine Mass Fraction", value=0.09 * np.ones(shape=temp_K.shape))

            self.add_function(key="CO2 Load", function=self.CO2Load)

            self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
            self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
            self.add_specie(id="HCO3-", molar_mass_kg_kmol=61, charge=-1)
            self.add_specie(id="CO3-2", molar_mass_kg_kmol=60, charge=-2)
            self.add_specie(id="H+", molar_mass_kg_kmol=1, charge=1)
            self.add_specie(id="OH-", molar_mass_kg_kmol=17, charge=-1)

            # self.add_specie(id="Lys+2", molar_mass_kg_kmol=148, charge=2)
            self.add_specie(id="Lys+", molar_mass_kg_kmol=147, charge=1)
            self.add_specie(id="Lys", molar_mass_kg_kmol=146, charge=0)
            self.add_specie(id="Lys-", molar_mass_kg_kmol=145, charge=-1)
            # self.add_specie(id="LysCOO", molar_mass_kg_kmol=190, charge=0)
            self.add_specie(id="LysCOO-", molar_mass_kg_kmol=189, charge=-1)
            self.add_specie(id="LysCOO-2", molar_mass_kg_kmol=188, charge=-2)
            # self.add_specie(id="Lys(COO)2-2", molar_mass_kg_kmol=232, charge=-2)
            # self.add_specie(id="Lys(COO)2-3", molar_mass_kg_kmol=231, charge=-3)
            self.add_specie(id="K+", molar_mass_kg_kmol=39, charge=1)

            self.add_vapor_pressure_bara_henry(id="CO2(g) = CO2(aq)",
                                               gas_id="CO2",
                                               liq_id="CO2",
                                               liq_unit="m",
                                               henrys_coefficient=self.__CO2_henrys_constant__)

            self.add_vapor_pressure_bara_raoult(id="H2O(g) = H2O(l)",
                                                gas_id="H2O",
                                                liq_id="H2O",
                                                pure_vapor_pressure_bara=self.__H2O_vapor_pressure_bara__)

            self.add_rxn_insta(id="H2O = H+ + OH-",
                               stoch={"H2O": -1, "H+": 1, "OH-": 1},
                               unit={"H2O": "x", "H+": "m", "OH-": "m"},
                               equilibrium_constant=self.__water_autoprotolysis_eq_constant__)

            self.add_rxn_insta(id="CO2 + H2O = HCO3- + H+",
                               stoch={"H2O": -1, "CO2": -1, "H+": 1, "HCO3-": 1},
                               unit={"H2O": "x", "CO2": "m", "H+": "m", "HCO3-": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_1_eq_const__)

            self.add_rxn_insta(id="HCO3- = CO3-2 + H+",
                               stoch={"HCO3-": -1, "CO3-2": 1, "H+": 1},
                               unit={"HCO3-": "m", "CO3-2": "m", "H+": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_2_eq_const__)

            # self.add_rxn_insta(id="Lys+2 = Lys+ + H+",
            #                   stoch={"Lys+2": -1, "Lys+": 1, "H+": 1},
            #                   unit={"Lys+2": "m", "Lys+": "m", "H+": "m"},
            #                   equilibrium_constant=self.__lysine_dissociation_constant_K1__)

            self.add_rxn_insta(id="Lys+ = Lys + H+",
                               stoch={"Lys+": -1, "Lys": 1, "H+": 1},
                               unit={"Lys+": "m", "Lys": "m", "H+": "m"},
                               equilibrium_constant=self.__lysine_dissociation_constant_K2__)

            self.add_rxn_insta(id="Lys = Lys- + H+",
                               stoch={"Lys": -1, "Lys-": 1, "H+": 1},
                               unit={"Lys": "m", "Lys-": "m", "H+": "m"},
                               equilibrium_constant=self.__lysine_dissociation_constant_K3__)

            self.add_rxn_insta(id="CO2 + Lys = H+ + LysCOO-",
                               stoch={"CO2": -1, "Lys": -1, "H+": 1, "LysCOO-": 1},
                               unit={"CO2": "m", "Lys": "m", "H+": "m", "LysCOO-": "m"},
                               equilibrium_constant=self.__lysine_stability_carbamate_1__)

            self.add_rxn_insta(id="CO2 + Lys- = H+ + LysCOO-2",
                               stoch={"CO2": -1, "Lys-": -1, "H+": 1, "LysCOO-2": 1},
                               unit={"CO2": "m", "Lys-": "m", "H+": "m", "LysCOO-2": "m"},
                               equilibrium_constant=self.__lysine_stability_carbamate_2__)

            self.set_solution_temp_K(value=temp_K)
            self.set_solution_flow_kg_h(value=flow_kg_h)

            # Mass Fractions
            w_KLys = KLys_mass_fraction
            w_H2O = 1.0 - KLys_mass_fraction
            w_CO2 = KLys_mass_fraction * (44 / 184) * CO2Load

            # Normalizing
            w = w_KLys + w_H2O + w_CO2
            w_KLys = w_KLys / w
            w_H2O = w_H2O / w
            w_CO2 = w_CO2 / w

            self.set_specie_mass_fraction(id="H2O", value=w_H2O)
            self.set_specie_mass_fraction(id="CO2", value=w_CO2)
            self.set_specie_mass_fraction(id="HCO3-", value=z)
            self.set_specie_mass_fraction(id="CO3-2", value=z)
            self.set_specie_mass_fraction(id="H+", value=z)
            self.set_specie_mass_fraction(id="OH-", value=z)
            self.set_specie_mass_fraction(id="Lys+", value=z)
            self.set_specie_mass_fraction(id="Lys", value=z)
            self.set_specie_mass_fraction(id="Lys-", value=KLys_mass_fraction * 145 / 184)
            self.set_specie_mass_fraction(id="LysCOO-", value=z)
            self.set_specie_mass_fraction(id="LysCOO-2", value=z)
            self.set_specie_mass_fraction(id="K+", value=KLys_mass_fraction * 39 / 184)

        def CO2Load(self, LiquidStream):
            x_CO2 = 0
            x_Amine = 0
            C = {"CO2": 1, "HCO3-": 1, "CO3-2": 1, "LysCOO-2": 1, "LysCOO-": 1, "LysCOO": 1, "Lys(COO)2-3": 2,
                 "Lys(COO)2-2": 2, "Lys(COO)2-": 2}
            A = {"Lys+": 1, "Lys": 1, "Lys-": 1, "LysCOO": 1, "LysCOO-": 1, "LysCOO-2": 1, "Lys(COO)2-": 1,
                 "Lys(COO)2-2": 1, "Lys(COO)2-3": 1}
            for c in C.keys():
                if c in LiquidStream.specie.keys():
                    x_CO2 = x_CO2 + C[c] * LiquidStream.get_specie_molar_fraction(id=c)
            for a in A.keys():
                if a in LiquidStream.specie.keys():
                    x_Amine = x_Amine + A[a] * LiquidStream.get_specie_molar_fraction(id=a)
            alpha = np.maximum(x_CO2 / np.maximum(x_Amine, 10 ** (-9)), 10 ** (-9))
            return alpha

        def __heat_capacity_kJ_kgK__(self, LiquidStream):
            w_AA = 0.09
            T = LiquidStream.temp_K
            cp = ((4.2 + 0.41 * w_AA) / (1 + 1.05 * w_AA)) + (0.3 / 75) * (T - 298)
            return cp

        def __density_kg_m3__(self, LiquidStream):
            return 1050 * np.ones(shape=LiquidStream.temp_K.shape)

        def __activity_coefficient__(self, LiquidStream, id):
            I = LiquidStream.get_solution_ionic_strength_mol_kg()
            z = LiquidStream.get_specie_charge(id)
            log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))
            gamma = 10 ** log10_gamma
            return gamma

        def __CO2_henrys_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            H_CO2 = 1.153 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)
            return H_CO2

        def __H2O_vapor_pressure_bara__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            T = np.minimum(T, 273.15 + 150)
            pc = 220.64
            Tc = 647.096
            tau = 1 - T / Tc
            a1 = -7.85951783
            a2 = 1.84408259
            a3 = -11.7866497
            a4 = 22.6807411
            a5 = -15.9618719
            a6 = 1.80122502
            p = pc * np.exp(
                (Tc / T) * (
                            a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 3 + a4 * tau ** 3.5 + a5 * tau ** 4 + a6 * tau ** 7.5))
            return p

        def __water_autoprotolysis_eq_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            Kw = 10 ** (-14) * np.exp(-13445.9 * (1 / T - 1 / 298) - 22.48 * np.log(T / 298))
            return Kw

        def __carbonic_acid_rxn_1_eq_const__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-6.32) * np.exp(5139 * (1 / T - 1 / 298) + 14.5258479 * np.log(T / 298))
            return K

        def __carbonic_acid_rxn_2_eq_const__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-10.33) * np.exp(22062 * (1 / T - 1 / 298) + 67.264072 * np.log(T / 298))
            return K

        # --------------------------------------------------------------------------------------------

        def __lysine_dissociation_constant_K2__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-9.16) * np.exp(-4900 * (1 / T - 1 / 298.15))
            return K

        def __lysine_dissociation_constant_K3__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-10.9) * np.exp(-5700 * (1 / T - 1 / 298.15))
            return K

        def __lysine_stability_carbamate_1__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 8.671 * 10 ** (-6) * np.exp(4594 * (1 / T - 1 / 298))
            return K

        def __lysine_stability_carbamate_2__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 1.1838 * 10 ** (-4) * np.exp(8302 * (1 / T - 1 / 298))
            return K

    class LiquidStream_KLys25(lab.LiquidStream):

        def __init__(self, temp_K, flow_kg_h, CO2Load):

            super().__init__(stream_id="", solvent_id="H2O")
            z = np.zeros(shape=temp_K.shape)

            self.load_heat_capacity_kJ_kgK(function=self.__heat_capacity_kJ_kgK__)
            self.load_density_kg_m3(function=self.__density_kg_m3__)
            self.load_activity_coefficient(function=self.__activity_coefficient__)
            self.add_info(key="Amine Mass Fraction", value=0.25 * np.ones(shape=temp_K.shape))

            self.add_function(key="CO2 Load", function=self.CO2Load)

            self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
            self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
            self.add_specie(id="HCO3-", molar_mass_kg_kmol=61, charge=-1)
            self.add_specie(id="CO3-2", molar_mass_kg_kmol=60, charge=-2)
            self.add_specie(id="H+", molar_mass_kg_kmol=1, charge=1)
            self.add_specie(id="OH-", molar_mass_kg_kmol=17, charge=-1)

            # self.add_specie(id="Lys+2", molar_mass_kg_kmol=148, charge=2)
            self.add_specie(id="Lys+", molar_mass_kg_kmol=147, charge=1)
            self.add_specie(id="Lys", molar_mass_kg_kmol=146, charge=0)
            self.add_specie(id="Lys-", molar_mass_kg_kmol=145, charge=-1)
            # self.add_specie(id="LysCOO", molar_mass_kg_kmol=190, charge=0)
            self.add_specie(id="LysCOO-", molar_mass_kg_kmol=189, charge=-1)
            self.add_specie(id="LysCOO-2", molar_mass_kg_kmol=188, charge=-2)
            # self.add_specie(id="Lys(COO)2-2", molar_mass_kg_kmol=232, charge=-2)
            # self.add_specie(id="Lys(COO)2-3", molar_mass_kg_kmol=231, charge=-3)
            self.add_specie(id="K+", molar_mass_kg_kmol=39, charge=1)

            self.add_vapor_pressure_bara_henry(id="CO2(g) = CO2(aq)",
                                               gas_id="CO2",
                                               liq_id="CO2",
                                               liq_unit="m",
                                               henrys_coefficient=self.__CO2_henrys_constant__)

            self.add_vapor_pressure_bara_raoult(id="H2O(g) = H2O(l)",
                                                gas_id="H2O",
                                                liq_id="H2O",
                                                pure_vapor_pressure_bara=self.__H2O_vapor_pressure_bara__)

            self.add_rxn_insta(id="H2O = H+ + OH-",
                               stoch={"H2O": -1, "H+": 1, "OH-": 1},
                               unit={"H2O": "x", "H+": "m", "OH-": "m"},
                               equilibrium_constant=self.__water_autoprotolysis_eq_constant__)

            self.add_rxn_insta(id="CO2 + H2O = HCO3- + H+",
                               stoch={"H2O": -1, "CO2": -1, "H+": 1, "HCO3-": 1},
                               unit={"H2O": "x", "CO2": "m", "H+": "m", "HCO3-": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_1_eq_const__)

            self.add_rxn_insta(id="HCO3- = CO3-2 + H+",
                               stoch={"HCO3-": -1, "CO3-2": 1, "H+": 1},
                               unit={"HCO3-": "m", "CO3-2": "m", "H+": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_2_eq_const__)

            # self.add_rxn_insta(id="Lys+2 = Lys+ + H+",
            #                   stoch={"Lys+2": -1, "Lys+": 1, "H+": 1},
            #                   unit={"Lys+2": "m", "Lys+": "m", "H+": "m"},
            #                   equilibrium_constant=self.__lysine_dissociation_constant_K1__)

            self.add_rxn_insta(id="Lys+ = Lys + H+",
                               stoch={"Lys+": -1, "Lys": 1, "H+": 1},
                               unit={"Lys+": "m", "Lys": "m", "H+": "m"},
                               equilibrium_constant=self.__lysine_dissociation_constant_K2__)

            self.add_rxn_insta(id="Lys = Lys- + H+",
                               stoch={"Lys": -1, "Lys-": 1, "H+": 1},
                               unit={"Lys": "m", "Lys-": "m", "H+": "m"},
                               equilibrium_constant=self.__lysine_dissociation_constant_K3__)

            self.add_rxn_insta(id="CO2 + Lys = H+ + LysCOO-",
                               stoch={"CO2": -1, "Lys": -1, "H+": 1, "LysCOO-": 1},
                               unit={"CO2": "m", "Lys": "m", "H+": "m", "LysCOO-": "m"},
                               equilibrium_constant=self.__lysine_stability_carbamate_1__)

            self.set_solution_temp_K(value=temp_K)
            self.set_solution_flow_kg_h(value=flow_kg_h)

            # Mass Fractions
            w_KLys = 0.25 + z
            w_H2O = 1.0 - w_KLys
            w_CO2 = w_KLys * (44 / 184) * CO2Load

            # Normalizing
            w = w_KLys + w_H2O + w_CO2
            w_KLys = w_KLys / w
            w_H2O = w_H2O / w
            w_CO2 = w_CO2 / w

            self.set_specie_mass_fraction(id="H2O", value=w_H2O)
            self.set_specie_mass_fraction(id="CO2", value=w_CO2)
            self.set_specie_mass_fraction(id="HCO3-", value=z)
            self.set_specie_mass_fraction(id="CO3-2", value=z)
            self.set_specie_mass_fraction(id="H+", value=z)
            self.set_specie_mass_fraction(id="OH-", value=z)
            self.set_specie_mass_fraction(id="Lys+", value=z)
            self.set_specie_mass_fraction(id="Lys", value=z)
            self.set_specie_mass_fraction(id="Lys-", value=w_KLys * 145 / 184)
            self.set_specie_mass_fraction(id="LysCOO-", value=z)
            self.set_specie_mass_fraction(id="LysCOO-2", value=z)
            self.set_specie_mass_fraction(id="K+", value=w_KLys * 39 / 184)

        def CO2Load(self, LiquidStream):
            x_CO2 = 0
            x_Amine = 0
            C = {"CO2": 1, "HCO3-": 1, "CO3-2": 1, "LysCOO-2": 1, "LysCOO-": 1, "LysCOO": 1, "Lys(COO)2-3": 2,
                 "Lys(COO)2-2": 2, "Lys(COO)2-": 2}
            A = {"Lys+": 1, "Lys": 1, "Lys-": 1, "LysCOO": 1, "LysCOO-": 1, "LysCOO-2": 1, "Lys(COO)2-": 1,
                 "Lys(COO)2-2": 1, "Lys(COO)2-3": 1}
            for c in C.keys():
                if c in LiquidStream.specie.keys():
                    x_CO2 = x_CO2 + C[c] * LiquidStream.get_specie_molar_fraction(id=c)
            for a in A.keys():
                if a in LiquidStream.specie.keys():
                    x_Amine = x_Amine + A[a] * LiquidStream.get_specie_molar_fraction(id=a)
            alpha = np.maximum(x_CO2 / np.maximum(x_Amine, 10 ** (-9)), 10 ** (-9))
            return alpha

        def __heat_capacity_kJ_kgK__(self, LiquidStream):
            w_AA = 0.25
            T = LiquidStream.temp_K
            cp = ((4.2 + 0.41 * w_AA) / (1 + 1.05 * w_AA)) + (0.3 / 75) * (T - 298)
            return cp

        def __density_kg_m3__(self, LiquidStream):
            return 1050 * np.ones(shape=LiquidStream.temp_K.shape)

        def __activity_coefficient__(self, LiquidStream, id):
            I = LiquidStream.get_solution_ionic_strength_mol_kg()
            z = LiquidStream.get_specie_charge(id)
            log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))
            gamma = 10 ** log10_gamma
            return gamma

        def __CO2_henrys_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            H_CO2 = 1.005 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)
            return H_CO2

        def __H2O_vapor_pressure_bara__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            T = np.minimum(T, 273.15 + 150)
            pc = 220.64
            Tc = 647.096
            tau = 1 - T / Tc
            a1 = -7.85951783
            a2 = 1.84408259
            a3 = -11.7866497
            a4 = 22.6807411
            a5 = -15.9618719
            a6 = 1.80122502
            p = pc * np.exp(
                (Tc / T) * (
                            a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 3 + a4 * tau ** 3.5 + a5 * tau ** 4 + a6 * tau ** 7.5))
            return p

        def __water_autoprotolysis_eq_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            Kw = 10 ** (-14) * np.exp(-13445.9 * (1 / T - 1 / 298) - 22.48 * np.log(T / 298))
            return Kw

        def __carbonic_acid_rxn_1_eq_const__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-6.32) * np.exp(5139 * (1 / T - 1 / 298) + 14.5258479 * np.log(T / 298))
            return K

        def __carbonic_acid_rxn_2_eq_const__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-10.33) * np.exp(22062 * (1 / T - 1 / 298) + 67.264072 * np.log(T / 298))
            return K

        def __lysine_dissociation_constant_K2__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-9.16) * np.exp(-4900 * (1 / T - 1 / 298.15))
            return K

        def __lysine_dissociation_constant_K3__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-10.9) * np.exp(-5700 * (1 / T - 1 / 298.15))
            return K

        def __lysine_stability_carbamate_1__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 2.344 * 10 ** (-6) * np.exp(2282 * (1 / T - 1 / 298))
            return K

    class LiquidStream_KLys40(lab.LiquidStream):

        def __init__(self, temp_K, flow_kg_h, CO2Load):

            super().__init__(stream_id="", solvent_id="H2O")
            z = np.zeros(shape=temp_K.shape)

            self.load_heat_capacity_kJ_kgK(function=self.__heat_capacity_kJ_kgK__)
            self.load_density_kg_m3(function=self.__density_kg_m3__)
            self.load_activity_coefficient(function=self.__activity_coefficient__)
            self.add_info(key="Amine Mass Fraction", value=0.4 * np.ones(shape=temp_K.shape))

            self.add_function(key="CO2 Load", function=self.CO2Load)

            self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
            self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
            self.add_specie(id="HCO3-", molar_mass_kg_kmol=61, charge=-1)
            self.add_specie(id="CO3-2", molar_mass_kg_kmol=60, charge=-2)
            self.add_specie(id="H+", molar_mass_kg_kmol=1, charge=1)
            self.add_specie(id="OH-", molar_mass_kg_kmol=17, charge=-1)

            self.add_specie(id="Lys+2", molar_mass_kg_kmol=148, charge=2)
            self.add_specie(id="Lys+", molar_mass_kg_kmol=147, charge=1)
            self.add_specie(id="Lys", molar_mass_kg_kmol=146, charge=0)
            self.add_specie(id="Lys-", molar_mass_kg_kmol=145, charge=-1)
            # self.add_specie(id="LysCOO", molar_mass_kg_kmol=190, charge=0)
            self.add_specie(id="LysCOO-", molar_mass_kg_kmol=189, charge=-1)
            self.add_specie(id="LysCOO-2", molar_mass_kg_kmol=188, charge=-2)
            # self.add_specie(id="Lys(COO)2-2", molar_mass_kg_kmol=232, charge=-2)
            # self.add_specie(id="Lys(COO)2-3", molar_mass_kg_kmol=231, charge=-3)
            self.add_specie(id="K+", molar_mass_kg_kmol=39, charge=1)

            self.add_vapor_pressure_bara_henry(id="CO2(g) = CO2(aq)",
                                               gas_id="CO2",
                                               liq_id="CO2",
                                               liq_unit="m",
                                               henrys_coefficient=self.__CO2_henrys_constant__)

            self.add_vapor_pressure_bara_raoult(id="H2O(g) = H2O(l)",
                                                gas_id="H2O",
                                                liq_id="H2O",
                                                pure_vapor_pressure_bara=self.__H2O_vapor_pressure_bara__)

            self.add_rxn_insta(id="H2O = H+ + OH-",
                               stoch={"H2O": -1, "H+": 1, "OH-": 1},
                               unit={"H2O": "x", "H+": "m", "OH-": "m"},
                               equilibrium_constant=self.__water_autoprotolysis_eq_constant__)

            self.add_rxn_insta(id="CO2 + H2O = HCO3- + H+",
                               stoch={"H2O": -1, "CO2": -1, "H+": 1, "HCO3-": 1},
                               unit={"H2O": "x", "CO2": "m", "H+": "m", "HCO3-": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_1_eq_const__)

            self.add_rxn_insta(id="HCO3- = CO3-2 + H+",
                               stoch={"HCO3-": -1, "CO3-2": 1, "H+": 1},
                               unit={"HCO3-": "m", "CO3-2": "m", "H+": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_2_eq_const__)

            self.add_rxn_insta(id="Lys+2 = Lys+ + H+",
                               stoch={"Lys+2": -1, "Lys+": 1, "H+": 1},
                               unit={"Lys+2": "m", "Lys+": "m", "H+": "m"},
                               equilibrium_constant=self.__lysine_dissociation_constant_K1__)

            self.add_rxn_insta(id="Lys+ = Lys + H+",
                               stoch={"Lys+": -1, "Lys": 1, "H+": 1},
                               unit={"Lys+": "m", "Lys": "m", "H+": "m"},
                               equilibrium_constant=self.__lysine_dissociation_constant_K2__)

            self.add_rxn_insta(id="Lys = Lys- + H+",
                               stoch={"Lys": -1, "Lys-": 1, "H+": 1},
                               unit={"Lys": "m", "Lys-": "m", "H+": "m"},
                               equilibrium_constant=self.__lysine_dissociation_constant_K3__)

            self.add_rxn_insta(id="CO2 + Lys = H+ + LysCOO-",
                               stoch={"CO2": -1, "Lys": -1, "H+": 1, "LysCOO-": 1},
                               unit={"CO2": "m", "Lys": "m", "H+": "m", "LysCOO-": "m"},
                               equilibrium_constant=self.__lysine_stability_carbamate_1__)

            self.set_solution_temp_K(value=temp_K)
            self.set_solution_flow_kg_h(value=flow_kg_h)

            # Mass Fractions
            w_KLys = 0.4 + z
            w_H2O = 1.0 - w_KLys
            w_CO2 = w_KLys * (44 / 184) * CO2Load

            # Normalizing
            w = w_KLys + w_H2O + w_CO2
            w_KLys = w_KLys / w
            w_H2O = w_H2O / w
            w_CO2 = w_CO2 / w

            self.set_specie_mass_fraction(id="H2O", value=w_H2O)
            self.set_specie_mass_fraction(id="CO2", value=w_CO2)
            self.set_specie_mass_fraction(id="HCO3-", value=z)
            self.set_specie_mass_fraction(id="CO3-2", value=z)
            self.set_specie_mass_fraction(id="H+", value=z)
            self.set_specie_mass_fraction(id="OH-", value=z)
            self.set_specie_mass_fraction(id="Lys+2", value=z)
            self.set_specie_mass_fraction(id="Lys+", value=z)
            self.set_specie_mass_fraction(id="Lys", value=z)
            self.set_specie_mass_fraction(id="Lys-", value=w_KLys * 145 / 184)
            self.set_specie_mass_fraction(id="LysCOO-", value=z)
            self.set_specie_mass_fraction(id="LysCOO-2", value=z)
            self.set_specie_mass_fraction(id="K+", value=w_KLys * 39 / 184)

        def CO2Load(self, LiquidStream):
            x_CO2 = 0
            x_Amine = 0
            C = {"CO2": 1, "HCO3-": 1, "CO3-2": 1, "LysCOO-2": 1, "LysCOO-": 1, "LysCOO": 1, "Lys(COO)2-3": 2,
                 "Lys(COO)2-2": 2, "Lys(COO)2-": 2}
            A = {"Lys+": 1, "Lys": 1, "Lys-": 1, "LysCOO": 1, "LysCOO-": 1, "LysCOO-2": 1, "Lys(COO)2-": 1,
                 "Lys(COO)2-2": 1, "Lys(COO)2-3": 1}
            for c in C.keys():
                if c in LiquidStream.specie.keys():
                    x_CO2 = x_CO2 + C[c] * LiquidStream.get_specie_molar_fraction(id=c)
            for a in A.keys():
                if a in LiquidStream.specie.keys():
                    x_Amine = x_Amine + A[a] * LiquidStream.get_specie_molar_fraction(id=a)
            alpha = np.maximum(x_CO2 / np.maximum(x_Amine, 10 ** (-9)), 10 ** (-9))
            return alpha

        def __CO2_flux_stirred_cell_reactor_kmol_m2s__(self, GasStreamIn, LiquidStreamIn):
            alpha = LiquidStreamIn.CO2Load(LiquidStreamIn)
            T = LiquidStreamIn.temp_K
            H = 1.153 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)
            p_CO2_vap = LiquidStreamIn.get_specie_vapor_pressure_bara(gas_id="CO2")
            p_CO2 = GasStreamIn.get_specie_pressure_bara(id="CO2")
            EkL = 3.5 * 10 ** (-5) * np.sqrt(6000 * LiquidStreamIn.get_specie_molality_mol_kg(id="Lys-"))
            EkL = EkL * np.sqrt(np.exp(- 1.5 * alpha))
            EkL = EkL * np.exp(-7000 * (1 / T - 1 / 298))
            CO2_flux_kmol_m2s = EkL * H * (p_CO2 - p_CO2_vap)
            return CO2_flux_kmol_m2s

        def __heat_capacity_kJ_kgK__(self, LiquidStream):
            w_AA = 0.4
            T = LiquidStream.temp_K
            cp = ((4.2 + 0.41 * w_AA) / (1 + 1.05 * w_AA)) + (0.3 / 75) * (T - 298)
            return cp

        def __density_kg_m3__(self, LiquidStream):
            return 1050 * np.ones(shape=LiquidStream.temp_K.shape)

        def __activity_coefficient__(self, LiquidStream, id):
            I = LiquidStream.get_solution_ionic_strength_mol_kg()
            z = LiquidStream.get_specie_charge(id)
            log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))
            gamma = 10 ** log10_gamma
            return gamma

        def __CO2_henrys_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            H_CO2 = 1.374 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)
            return H_CO2

        def __H2O_vapor_pressure_bara__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            T = np.minimum(T, 273.15 + 150)
            pc = 220.64
            Tc = 647.096
            tau = 1 - T / Tc
            a1 = -7.85951783
            a2 = 1.84408259
            a3 = -11.7866497
            a4 = 22.6807411
            a5 = -15.9618719
            a6 = 1.80122502
            p = pc * np.exp(
                (Tc / T) * (
                            a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 3 + a4 * tau ** 3.5 + a5 * tau ** 4 + a6 * tau ** 7.5))
            return p

        def __water_autoprotolysis_eq_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            Kw = 10 ** (-14) * np.exp(-13445.9 * (1 / T - 1 / 298) - 22.48 * np.log(T / 298))
            return Kw

        def __carbonic_acid_rxn_1_eq_const__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-6.32) * np.exp(5139 * (1 / T - 1 / 298) + 14.5258479 * np.log(T / 298))
            return K

        def __carbonic_acid_rxn_2_eq_const__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-10.33) * np.exp(22062 * (1 / T - 1 / 298) + 67.264072 * np.log(T / 298))
            return K

        def __lysine_dissociation_constant_K1__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            return 10 ** (-2.15) + 0 * T

        def __lysine_dissociation_constant_K2__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-9.16) * np.exp(-4900 * (1 / T - 1 / 298.15))
            return K

        def __lysine_dissociation_constant_K3__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-10.9) * np.exp(-5700 * (1 / T - 1 / 298.15))
            return K

        def __lysine_stability_carbamate_1__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 3.3733 * 10 ** (-6) * np.exp(4592 * (1 / T - 1 / 298))
            return K

    class LiquidStream_KSar40(lab.LiquidStream):

        def __init__(self, temp_K, flow_kg_h, KSar_mass_fraction, CO2Load):

            super().__init__(stream_id="", solvent_id="H2O")
            z = np.zeros(shape=temp_K.shape)

            self.load_heat_capacity_kJ_kgK(function=self.__heat_capacity_kJ_kgK__)
            self.load_density_kg_m3(function=self.__density_kg_m3__)
            self.load_activity_coefficient(function=self.__activity_coefficient__)
            self.add_info(key="Amine Mass Fraction", value=0.4 * np.ones(shape=temp_K.shape))

            self.add_function(key="CO2 Load", function=self.CO2Load)

            self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
            self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
            self.add_specie(id="HCO3-", molar_mass_kg_kmol=61, charge=-1)
            self.add_specie(id="CO3-2", molar_mass_kg_kmol=60, charge=-2)
            self.add_specie(id="H+", molar_mass_kg_kmol=1, charge=1)
            self.add_specie(id="OH-", molar_mass_kg_kmol=17, charge=-1)
            self.add_specie(id="Sar", molar_mass_kg_kmol=89, charge=0)
            self.add_specie(id="Sar-", molar_mass_kg_kmol=88, charge=-1)
            self.add_specie(id="SarCOO-2", molar_mass_kg_kmol=88 + 44 - 1, charge=-2)
            self.add_specie(id="K+", molar_mass_kg_kmol=39, charge=1)

            self.add_vapor_pressure_bara_henry(id="CO2(g) = CO2(aq)",
                                               gas_id="CO2",
                                               liq_id="CO2",
                                               liq_unit="m",
                                               henrys_coefficient=self.__CO2_henrys_constant__)

            self.add_vapor_pressure_bara_raoult(id="H2O(g) = H2O(l)",
                                                gas_id="H2O",
                                                liq_id="H2O",
                                                pure_vapor_pressure_bara=self.__H2O_vapor_pressure_bara__)

            self.add_rxn_insta(id="H2O = H+ + OH-",
                               stoch={"H2O": -1, "H+": 1, "OH-": 1},
                               unit={"H2O": "x", "H+": "m", "OH-": "m"},
                               equilibrium_constant=self.__water_autoprotolysis_eq_constant__)

            self.add_rxn_insta(id="CO2 + H2O = HCO3- + H+",
                               stoch={"H2O": -1, "CO2": -1, "H+": 1, "HCO3-": 1},
                               unit={"H2O": "x", "CO2": "m", "H+": "m", "HCO3-": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_1_eq_const__)

            self.add_rxn_insta(id="HCO3- = CO3-2 + H+",
                               stoch={"HCO3-": -1, "CO3-2": 1, "H+": 1},
                               unit={"HCO3-": "m", "CO3-2": "m", "H+": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_2_eq_const__)

            self.add_rxn_insta(id="Sar = Sar- + H+",
                               stoch={"Sar": -1, "Sar-": 1, "H+": 1},
                               unit={"Sar": "m", "Sar-": "m", "H+": "m"},
                               equilibrium_constant=self.__sarcosine_dissociation_constant_K2__)

            self.add_rxn_insta(id="CO2 + Sar- = H+ + SarCOO-2",
                               stoch={"CO2": -1, "Sar-": -1, "H+": 1, "SarCOO-2": 1},
                               unit={"CO2": "m", "Sar-": "m", "H+": "m", "SarCOO-2": "m"},
                               equilibrium_constant=self.__sarcosine_stability_carbamate__)

            self.set_solution_temp_K(value=temp_K)
            self.set_solution_flow_kg_h(value=flow_kg_h)

            # Mass Fractions
            w_KSar = KSar_mass_fraction
            w_H2O = 1.0 - KSar_mass_fraction
            w_CO2 = KSar_mass_fraction * (44 / 127) * CO2Load

            # Normalizing
            w = w_KSar + w_H2O + w_CO2
            w_KSar = w_KSar / w
            w_H2O = w_H2O / w
            w_CO2 = w_CO2 / w

            self.set_specie_mass_fraction(id="H2O", value=w_H2O)
            self.set_specie_mass_fraction(id="CO2", value=w_CO2)
            self.set_specie_mass_fraction(id="HCO3-", value=z)
            self.set_specie_mass_fraction(id="CO3-2", value=z)
            self.set_specie_mass_fraction(id="H+", value=z)
            self.set_specie_mass_fraction(id="OH-", value=z)
            self.set_specie_mass_fraction(id="Sar", value=z)
            self.set_specie_mass_fraction(id="Sar-", value=KSar_mass_fraction * 88 / 127)
            self.set_specie_mass_fraction(id="SarCOO-2", value=z)
            self.set_specie_mass_fraction(id="K+", value=KSar_mass_fraction * 39 / 127)

        def CO2Load(self, LiquidStream):
            x_CO2 = 0
            x_Amine = 0
            C = {"CO2": 1, "HCO3-": 1, "CO3-2": 1, "SarCOO-2": 1}
            A = {"Sar": 1, "Sar-": 1, "SarCOO-2": 1}
            for c in C.keys():
                if c in LiquidStream.specie.keys():
                    x_CO2 = x_CO2 + C[c] * LiquidStream.get_specie_molar_fraction(id=c)
            for a in A.keys():
                if a in LiquidStream.specie.keys():
                    x_Amine = x_Amine + A[a] * LiquidStream.get_specie_molar_fraction(id=a)
            alpha = np.maximum(x_CO2 / np.maximum(x_Amine, 10 ** (-9)), 10 ** (-9))
            return alpha

        def __heat_capacity_kJ_kgK__(self, LiquidStream):
            return 3.6

        def __density_kg_m3__(self, LiquidStream):
            return 1050

        def __activity_coefficient__(self, LiquidStream, id):
            I = LiquidStream.get_solution_ionic_strength_mol_kg()
            z = LiquidStream.get_specie_charge(id)
            log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))
            gamma = 10 ** log10_gamma
            return gamma

        def __CO2_henrys_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            H_CO2 = 1.153 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)
            return H_CO2

        def __H2O_vapor_pressure_bara__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            T = np.minimum(T, 273.15 + 150)
            pc = 220.64
            Tc = 647.096
            tau = 1 - T / Tc
            a1 = -7.85951783
            a2 = 1.84408259
            a3 = -11.7866497
            a4 = 22.6807411
            a5 = -15.9618719
            a6 = 1.80122502
            p = pc * np.exp((Tc / T) * (
                        a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 3 + a4 * tau ** 3.5 + a5 * tau ** 4 + a6 * tau ** 7.5))
            return p

        def __water_autoprotolysis_eq_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            Kw = 10 ** (-14) * np.exp(-13445.9 * (1 / T - 1 / 298) - 22.48 * np.log(T / 298))
            return Kw

        def __carbonic_acid_rxn_1_eq_const__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-6.32) * np.exp(6275 * 0.81899898 * (1 / T - 1 / 298) + 0.79419617 * 18.29 * np.log(T / 298))
            return K

        def __carbonic_acid_rxn_2_eq_const__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-10.33) * np.exp(14400 * 1.53213073 * (1 / T - 1 / 298) + 1.43115048 * 47.0 * np.log(T / 298))
            return K

        def __sarcosine_dissociation_constant_K2__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-10.8) * np.exp(-4900 * (1 / T - 1 / 298.15))  # 11.64 or 10.15 or 10.01
            return K

        def __sarcosine_stability_carbamate__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 1.3217 * 10 ** (-7) * np.exp(3000 * (1 / T - 1 / 313))
            return K

    class LiquidStream_KPro(lab.LiquidStream):

        def __init__(self, temp_K, flow_kg_h, KPro_mass_fraction, CO2Load):

            super().__init__(stream_id="", solvent_id="H2O")
            z = np.zeros(shape=temp_K.shape)

            self.x = [1.0000000000000029, 0.09855568253686149, 0.9829815072208714, 1.7546290865569796,
                      0.9999395671731712, 1.8073451185021876, -0.027409178466454444, -1.0087389238001296,
                      0.48755731200288127, 0.4234245193556142, -0.09564395676421932, -0.006317767372101072,
                      0.3212720322471745, -0.31651826203417305, 0.8486432255086424, 0.8746450589337607,
                      0.08640887347366291]

            self.load_heat_capacity_kJ_kgK(function=self.__heat_capacity_kJ_kgK__)
            self.load_density_kg_m3(function=self.__density_kg_m3__)
            self.load_activity_coefficient(function=self.__activity_coefficient__)

            self.add_function(key="CO2 Load", function=self.CO2Load)

            self.add_specie(id="H2O", molar_mass_kg_kmol=18, charge=0)
            self.add_specie(id="CO2", molar_mass_kg_kmol=44, charge=0)
            self.add_specie(id="HCO3-", molar_mass_kg_kmol=61, charge=-1)
            self.add_specie(id="CO3-2", molar_mass_kg_kmol=60, charge=-2)
            self.add_specie(id="H+", molar_mass_kg_kmol=1, charge=1)
            self.add_specie(id="OH-", molar_mass_kg_kmol=17, charge=-1)
            self.add_specie(id="K+", molar_mass_kg_kmol=39, charge=1)
            self.add_specie(id="Pro", molar_mass_kg_kmol=115, charge=0)
            self.add_specie(id="Pro-", molar_mass_kg_kmol=114, charge=-1)
            self.add_specie(id="ProCOO-2", molar_mass_kg_kmol=157, charge=-2)

            self.add_vapor_pressure_bara_henry(id="CO2(g) = CO2(aq)",
                                               gas_id="CO2",
                                               liq_id="CO2",
                                               liq_unit="m",
                                               henrys_coefficient=self.__CO2_henrys_constant__)

            self.add_vapor_pressure_bara_raoult(id="H2O(g) = H2O(l)",
                                                gas_id="H2O",
                                                liq_id="H2O",
                                                pure_vapor_pressure_bara=self.__H2O_vapor_pressure_bara__)

            self.add_rxn_insta(id="H2O = H+ + OH-",
                               stoch={"H2O": -1, "H+": 1, "OH-": 1},
                               unit={"H2O": "x", "H+": "m", "OH-": "m"},
                               equilibrium_constant=self.__water_autoprotolysis_eq_constant__)

            self.add_rxn_insta(id="CO2 + H2O = HCO3- + H+",
                               stoch={"H2O": -1, "CO2": -1, "H+": 1, "HCO3-": 1},
                               unit={"H2O": "x", "CO2": "m", "H+": "m", "HCO3-": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_1_eq_const__)

            self.add_rxn_insta(id="HCO3- = CO3-2 + H+",
                               stoch={"HCO3-": -1, "CO3-2": 1, "H+": 1},
                               unit={"HCO3-": "m", "CO3-2": "m", "H+": "m"},
                               equilibrium_constant=self.__carbonic_acid_rxn_2_eq_const__)

            self.add_rxn_insta(id="Pro = Pro- + H+",
                               stoch={"Pro": -1, "Pro-": 1, "H+": 1},
                               unit={"Pro": "m", "Pro-": "m", "H+": "m"},
                               equilibrium_constant=self.__proline_dissociation_constant_K2__)

            self.add_rxn_insta(id="CO2 + Pro- = H+ + ProCOO-2",
                               stoch={"CO2": -1, "Pro-": -1, "H+": 1, "ProCOO-2": 1},
                               unit={"CO2": "m", "Pro-": "m", "H+": "m", "ProCOO-2": "m"},
                               equilibrium_constant=self.__proline_carbamate_stability__)

            self.set_solution_temp_K(value=temp_K)
            self.set_solution_flow_kg_h(value=flow_kg_h)

            # Mass Fractions
            w_KPro = KPro_mass_fraction
            w_H2O = 1.0 - KPro_mass_fraction
            w_CO2 = KPro_mass_fraction * (44 / 153) * CO2Load

            # Normalizing
            w = w_KPro + w_H2O + w_CO2
            w_KPro = w_KPro / w
            w_H2O = w_H2O / w
            w_CO2 = w_CO2 / w

            self.set_specie_mass_fraction(id="H2O", value=w_H2O)
            self.set_specie_mass_fraction(id="CO2", value=w_CO2)
            self.set_specie_mass_fraction(id="HCO3-", value=z)
            self.set_specie_mass_fraction(id="CO3-2", value=z)
            self.set_specie_mass_fraction(id="H+", value=z)
            self.set_specie_mass_fraction(id="OH-", value=z)
            self.set_specie_mass_fraction(id="Pro", value=z)
            self.set_specie_mass_fraction(id="Pro-", value=KPro_mass_fraction * 114 / 153)
            self.set_specie_mass_fraction(id="ProCOO-2", value=z)
            self.set_specie_mass_fraction(id="K+", value=KPro_mass_fraction * 39 / 153)

        def CO2Load(self, LiquidStream):
            x_CO2 = 0
            x_Amine = 0
            C = {"CO2": 1, "HCO3-": 1, "CO3-2": 1, "ProCOO-2": 1}
            A = {"Pro": 1, "Pro-": 1, "ProCOO-2": 1}
            for c in C.keys():
                if c in LiquidStream.specie.keys():
                    x_CO2 = x_CO2 + C[c] * LiquidStream.get_specie_molar_fraction(id=c)
            for a in A.keys():
                if a in LiquidStream.specie.keys():
                    x_Amine = x_Amine + A[a] * LiquidStream.get_specie_molar_fraction(id=a)
            alpha = np.maximum(x_CO2 / np.maximum(x_Amine, 10 ** (-9)), 10 ** (-9))
            return alpha

        def __heat_capacity_kJ_kgK__(self, LiquidStream):
            return 3.6

        def __density_kg_m3__(self, LiquidStream):
            return 1050

        def __activity_coefficient__(self, LiquidStream, id):
            I = LiquidStream.get_solution_ionic_strength_mol_kg()
            z = LiquidStream.get_specie_charge(id)
            log10_gamma = - 0.51 * z ** 2 * np.sqrt(I) / (1 + 1.5 * np.sqrt(I))

            b = {}
            b["H2O"] = 0.0
            b["H+"] = 0.0
            b["CO2"] = LiquidStream.x[8]
            b["CO3-2"] = LiquidStream.x[9]
            b["HCO3-"] = LiquidStream.x[10]
            b["OH-"] = LiquidStream.x[11]
            b["Pro"] = LiquidStream.x[12]
            b["Pro-"] = LiquidStream.x[13]
            b["ProCOO-2"] = LiquidStream.x[14]

            c = {}
            c["H2O"] = 1.0
            c["H+"] = LiquidStream.x[0]
            c["CO2"] = LiquidStream.x[1]
            c["CO3-2"] = LiquidStream.x[2]
            c["HCO3-"] = LiquidStream.x[3]
            c["OH-"] = LiquidStream.x[4]
            c["Pro"] = LiquidStream.x[5]
            c["Pro-"] = LiquidStream.x[6]
            c["ProCOO-2"] = LiquidStream.x[7]

            log10_gamma = log10_gamma + b[id] * I ** c[id]
            gamma = 10 ** log10_gamma
            return gamma

        def __CO2_henrys_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            H_CO2 = 1.153 * np.exp((-T * (1713 * (1 - 0.0015453 * T) ** (1 / 3) + 3680) + 1198506) / T ** 2)
            return H_CO2

        def __H2O_vapor_pressure_bara__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            T = np.minimum(T, 273.15 + 150)
            pc = 220.64
            Tc = 647.096
            tau = 1 - T / Tc
            a1 = -7.85951783
            a2 = 1.84408259
            a3 = -11.7866497
            a4 = 22.6807411
            a5 = -15.9618719
            a6 = 1.80122502
            p = pc * np.exp((Tc / T) * (
                        a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 3 + a4 * tau ** 3.5 + a5 * tau ** 4 + a6 * tau ** 7.5))
            return p

        def __water_autoprotolysis_eq_constant__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            Kw = 10 ** (-14) * np.exp(-12612 * (1 / T - 1 / 298) - 23.6397 * np.log(T / 298))
            return Kw

        def __carbonic_acid_rxn_1_eq_const__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-6.32) * np.exp(5262 * (1 / T - 1 / 298) + 15.01453 * np.log(T / 298))
            return K

        def __carbonic_acid_rxn_2_eq_const__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-10.33) * np.exp(13699 * (1 / T - 1 / 298) + 45.7353 * np.log(T / 298))
            return K

        def __proline_dissociation_constant_K2__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 10 ** (-10.6) * np.exp(-4900 * (1 / T - 1 / 298.15))  # -10.60
            return K

        def __proline_carbamate_stability__(self, LiquidStream):
            T = LiquidStream.get_solution_temp_K()
            K = 7.31 * 10 ** (-6 + LiquidStream.x[15]) * np.exp(3000 * LiquidStream.x[16] * (1 / T - 1 / 298.15))
            return K



