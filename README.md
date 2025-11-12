# testlablib

## 1. About

testlablib is a Python Library for simulating Physical and Chemical Processes.  

For example:
- Calculate Equilibrium Concentrations in Gases and Liquids
- Flash-Calculations
- Plug-Flow Reactors
- Continuous Stirred Tank Reactors
- ... and much more




## 2. Installation

```
pip install git+https://github.com/vegardlarsen85/testlablib.git
```


## 3. Abbriviations

| Description                   | Symbol    | Unit                   |
|:------------------------------|:----------|:-----------------------|
| *Molar Flow*                  | $\dot{n}$ | $kmol \cdot h^{-1}$    |
| *Mass Flow*                   | $\dot{m}$ | $kg \cdot h^{-1}$      |
| *Volume Flow*                 | $\dot{V}$ | $m^3 \cdot h^{-1}$     |
| *Gas Phase Molar Fraction*    | $y_i$     | $kmol \cdot kmol^{-1}$ |
| *Liquid Phase Molar Fraction* | $x_i$     | $kmol \cdot kmol^{-1}$ |
| *Mass Fraction*               | $w_i$     | $kmol \cdot kmol^{-1}$ |
| *Molarity*                    | $c_i$     | $kmol \cdot m^{-3}$    |
| *Molality of solute i*        | $b_i$     | $mol \cdot kg^{-1}$    |
| *Gas Pressure*                | $p$       | $bar(a)$               |
| *Temperature*                 | $T$       | $K$                    |


## 4. Formula Cheat Sheets



### 4.1 Solvents

Conversions between Molarity, Molality, Molar Fraction and Mass Fraction.


| Dilute Aqueous Solution                      | Arbitrary Solution                                                                                                 | Note       |
|:---------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:-----------|
| $c=\rho/18$                                  | $c=\rho \sum_i \frac{w_i}{M_i}$                                                                                    |            |
| $c=\rho/18$                                  | $c=\rho \frac{1}{\sum_k M_k x_k}$                                                                                  |            |
| $c=c_{solvent}$                              | $c=\sum_i c_i$                                                                                                     |            |
| $c_i=\rho \frac{w_i}{M_i}$                   | $c_i=\rho \frac{w_i}{M_i}$                                                                                         |            |
| $c_i=\rho \frac{x_i}{18}$                    | $c_i=\rho \frac{x_i}{\sum_k M_k x_k}$                                                                              |            |
| $c_i=\frac{1}{1000} \cdot m_i \cdot \rho$    | $c_i = \frac{x_{solvent}}{1000} \cdot m_i \cdot M_{solvent} \cdot c$                                               | *i=solute* |
| $m_i=1000 \left( \frac{w_i}{M_i} \right)$    | $m_i= \left( \frac{1000}{w_{solvent}} \right) \left( \frac{w_i}{M_i} \right) $                                     | *i=solute* |
| $m_i=1000 \left( \frac{x_i}{18} \right)$     | $m_i = \left( \frac{1000}{x_{solvent}} \right) \left( \frac{x_i}{M_{solvent}} \right) $                            | *i=solute* |
| $m_i = \left( \frac{1000}{\rho} \right) c_i$ | $m_i = \left( \frac{1000}{x_{solvent}} \right) \left( \frac{1}{M_{solvent}} \right) \left( \frac{c_i}{c} \right) $ | *i=solute* |
| $x_i = \left( \frac{18}{M_i} \right) w_i$    | $x_i = \left( \frac{1/M_i}{ \sum_k w_k/M_k} \right) w_i$                                                           |            |
| $x_i = \frac{c_i}{c}$                        | $x_i = \frac{c_i}{c}$                                                                                              |            |
| $x_{solvent} = 1$                            | $x_{solvent}=\frac{1000}{1000+M_{solvent} \sum_i m_i}$                                                             | *i=solute* |
| $x_{solvent} = 1$                            | $x_{solvent}=\frac{1000}{1000+M_{solvent} \sum_i m_i}$                                                             | *i=solute* |

### 4.2 Gases


$f_i=\phi_i \cdot y_i \cdot p$  
$c_i = \frac{1}{RT} \cdot f_i$



## 5. Gas Streams, Liquid Streams and Custom Reactors



## 6. Quick Start Guide


To get started with the library we will set up a simulation using Potassium Carbonate (K2CO3) dissolved in water as an example.



```
def  codeblock():
    return None
```

$\sum_{i=1}^{n} X_i$

*Cursive*  
**Bold**  
***Bold & Cursive***



## 7. API Reference


