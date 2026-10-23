# Domain Glossary & Engineering Standards

A comprehensive reference for the petroleum engineering, drilling mechanics, subsurface well integrity standards, and petrophysical data formats implemented in the Wellbore Knowledge Graph.

---

## 1. Well Control & Integrity Standards

### Two-Barrier Rule (API RP 53 / NORSOK D-010)
- **Principle**: At all times during well construction and intervention operations, at least two independent, tested, and verified well barriers must be present between an influx source and the external environment.
- **Primary Barrier**: Typically the active drilling fluid column (hydrostatic pressure $\ge$ formation pore pressure) or downhole casing/liner shoe cement sheath.
- **Secondary Barrier**: Typically the Blowout Preventer (BOP) stack (pipe rams, blind shear rams, annular preventer) or high-pressure surface casing string.

### MASP (Maximum Allowable Surface Pressure)
- The highest anticipated surface pressure during a well shut-in or gas kick circulation:
  $$\text{MASP} = P_{\text{frac}} - (0.052 \times \rho_{\text{gas}} \times \text{TVD})$$
  Used to rate surface wellheads, BOP equipment, and casing burst safety factors.

---

## 2. Drilling Hydraulics & ECD

### Bingham-Plastic Rheology
- **Plastic Viscosity ($PV$, cP)**: Resistance of mud to flow caused by mechanical friction between solid particles:
  $$PV = \theta_{600} - \theta_{300}$$
- **Yield Point ($YP$, $\text{lbf}/100\,\text{ft}^2$)**: Electro-chemical attractive forces between particles under flowing conditions:
  $$YP = \theta_{300} - PV$$
- **Annular Velocity ($V_a$, $\text{ft}/\text{min}$)**:
  $$V_a = \frac{24.5 \times Q}{d_h^2 - d_p^2}$$
  where $Q$ is flow rate (gpm), $d_h$ is hole diameter (in), and $d_p$ is pipe outer diameter (in).
- **Equivalent Circulating Density ($ECD$, ppg)**:
  $$ECD = \rho_{\text{mud}} + \frac{\Delta P_{\text{annular}}}{0.052 \times TVD}$$

---

## 3. Directional Drilling & Anti-Collision

### Minimum Curvature Method
- Industry standard mathematical method (promoted by the Industry Steering Committee on Wellbore Survey Accuracy - **ISCWSA**) for computing 3D wellbore coordinates $(N, E, TVD)$ from measured depth ($MD$), inclination ($Inc$), and azimuth ($Azi$) survey stations. Uses a circular arc between consecutive stations to model dogleg severity ($DLS$).

### Separation Factor ($SF$)
- Ratio of the center-to-center distance ($CC$) between two wellbores to the combined uncertainty radii ($S_1 + S_2$) of their position error ellipsoids:
  $$SF = \frac{CC}{S_1 + S_2}$$
  - $SF > 1.5$: Safe clearance.
  - $1.0 \le SF \le 1.5$: Warning / Alert zone (frequent gyro surveys required).
  - $SF < 1.0$: Critical collision risk (drilling must cease immediately).

---

## 4. Drillstring Vibrations & Dynamics

### Surface Stick-Slip Index ($SSSI\%$)
- Measure of severe torsional drillstring oscillation where the bit stalls downhole while the surface rotary table continues rotating:
  $$SSSI\% = \frac{\Omega_{\max} - \Omega_{\min}}{2 \times \Omega_{\text{avg}}} \times 100\%$$
  - $SSSI < 50\%$: Benign / Smooth drilling.
  - $50\% \le SSSI \le 100\%$: Moderate stick-slip warning.
  - $SSSI > 100\%$: Severe full-stall stick-slip (risk of twist-off, MWD electronic failure, cutter shearing).

### Bit Bounce & Backward Whirl
- **Bit Bounce**: Axial drillstring resonance causing rapid, large cyclic variations in Weight on Bit ($WOB$).
- **Backward Whirl**: High-frequency eccentric precession of the bottomhole assembly (BHA) against the borehole wall, generating high lateral $g$-shocks and accelerated collar fatigue.

---

## 5. Industry Data Exchange Formats

### CWLS LAS 2.0 (Log ASCII Standard)
- Standard format defined by the Canadian Well Logging Society for petrophysical curve logs. Contains:
  - `~WELL`: Rig metadata, operator, field, well name, surface coordinates.
  - `~CURVE`: Curve mnemonics (e.g., `DEPT`, `GR`, `RES`, `CALI`, `NPHI`, `RHOB`, `ROP`).
  - `~PARAMETER`: Mud properties, casing depth, bottomhole temperature.
  - `~ASCII`: Tabular curve values ordered by measured depth.

### Energistics WITSML 1.4.1.1
- Wellsite Information Transfer Standard Markup Language. The XML-based global oil & gas standard for exchanging real-time drilling telemetry, wellbore architectures, and directional survey trajectories across distributed enterprise systems.
