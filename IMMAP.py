import streamlit as st
import numpy as np
import pandas as pd
import casadi as ca

st.set_page_config(layout="wide")
st.title("CNG ECU INJECTOR MULTIPLIER MAP GENERATOR")
st.markdown("##### Model Predictive Control (MPC) Optimization for A.E.B. & MIJO Calibration Systems")

# ================= 1. VEHICLE & CNG ECU CONFIGURATION =================
st.sidebar.header("1. Engine & Vehicle Setup")
n_cyl = st.sidebar.selectbox("No. of Cylinders", [2, 3, 4, 5, 6, 8], index=2)
disp_l = st.sidebar.number_input("Engine Displacement (L)", 0.5, 15.0, 2.0)
inj_type = st.sidebar.selectbox("Injector Type", ["I-PLUS", "APA", "IG1", "VALTEK"], index=0)
inj_mode = st.sidebar.selectbox("Injection Mode", ["Sequential", "Semi-Sequential", "Full Group"], index=0)

st.sidebar.header("2. CNG Calibration Parameters (AEB / MIJO)")
fuel_type = st.sidebar.selectbox("Fuel Type", ["CNG", "LPG"], index=0)
reducer_press = st.sidebar.number_input("Reducer Pressure (bar)", 0.50, 3.50, 1.80 if fuel_type == "CNG" else 0.95, step=0.05)
temp_reducer = st.sidebar.number_input("Reducer Temp. for Change-over (°C)", 10, 90, 25)
changeover_rpm = st.sidebar.number_input("Revs. Threshold for Change-over (RPM)", 800, 3000, 1600, step=100)

st.sidebar.header("3. Fuel Targets")
AFR_ref = st.sidebar.number_input("Target Stoichiometric AFR (CNG λ=1.0)", 14.0, 18.0, 16.5)

# Fixed MPC Horizon and Operating Baseline Defaults
p_horizon = 15
c_horizon = 5
lhv_cng = 50.0  # Lower Heating Value of CNG (MJ/kg)

# Standard AEB / MIJO Calibration Grid Axes
rpm_grid = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000])
inj_grid_ms = np.array([2.00, 2.50, 3.00, 3.50, 4.50, 6.00, 8.00, 10.00, 12.00, 14.00, 16.00, 18.00])

# ================= 2. STATE-SPACE MPC CALIBRATION ENGINE =================

def build_continuous_state_space():
    """Constructs 5-state system matrices A, B, E."""
    J = 0.20        
    tau_t = 0.04    
    sigma = 25.0    
    
    A = np.array([
        [-12.5,   0.0,     0.0,   -0.0002,  0.0],
        [  0.0, -25.0,     0.0,    0.0,     0.0],
        [ 85.0, -1400.0,  -8.0,    0.0,     0.0],
        [  0.0,   0.0,     0.0,   -0.55,   (1.0/J)*(30.0/np.pi)],
        [  0.0, (2.2e6/tau_t), (-8.5/tau_t), 0.0, -sigma]
    ])
    
    B = np.array([
        [0.0,        0.0],
        [0.028,      0.0],
        [0.0,       -0.12],
        [0.0,        0.0],
        [0.0,   (3.5/tau_t)]
    ])
    
    return A, B

def discretize_system(A, B, dt=0.02):
    I = np.eye(A.shape[0])
    return I + A * dt, B * dt

def generate_mpc_multiplier_map(p_press, target_afr):
    """
    Computes optimal gas injection percentage multipliers for the AEB/MIJO map grid
    using receding horizon optimal fuel corrections scaled to gas reducer pressure.
    """
    A, B = build_continuous_state_space()
    A_d, B_d = discretize_system(A, B, dt=0.02)
    
    # CasADi MPC Setup
    u_sym = ca.MX.sym('U', 2, p_horizon)
    x_0 = ca.MX.sym('x0', 5)
    
    cost = 0
    x_k = x_0
    A_ca, B_ca = ca.MX(A_d), ca.MX(B_d)
    
    for k in range(p_horizon):
        cost += 1500.0 * (x_k[2] - target_afr)**2 + 5.0 * (x_k[4] - 150.0)**2
        cost += 10.0 * (u_sym[0, k] - 0.003)**2 + 2.0 * (u_sym[1, k] - 12.0)**2
        x_k = ca.mtimes(A_ca, x_k) + ca.mtimes(B_ca, u_sym[:, k])
        
    nlp = {'x': ca.reshape(u_sym, -1, 1), 'f': cost, 'p': x_0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})
    
    # Pressure compensation factor relative to nominal CNG pressure (1.80 bar)
    press_factor = np.sqrt(1.80 / p_press) if p_press > 0 else 1.0
    
    multiplier_matrix = np.zeros((len(inj_grid_ms), len(rpm_grid)))
    
    for i, t_inj in enumerate(inj_grid_ms):
        for j, rpm in enumerate(rpm_grid):
            # Evaluate optimal state pulse width requirement
            x_operating = np.array([0.0012 * (rpm / 2000.0), 0.0000727, target_afr, rpm, 150.0])
            sol = solver(
                x0=np.tile([t_inj * 0.001, 12.0], p_horizon),
                p=x_operating,
                lbx=np.tile([0.001, 0.0], p_horizon),
                ubx=np.tile([0.020, 35.0], p_horizon)
            )
            pw_opt = float(sol['x'][0]) * 1000.0  # Optimal pulse width (ms)
            
            # AEB/MIJO map value percentage conversion (100 = 100%)
            base_mult = (pw_opt / t_inj) * 100.0 * press_factor
            
            # Non-linear volumetric efficiency and pressure loss shaping
            rpm_correction = 1.0 + 0.08 * np.sin(rpm / 1500.0)
            load_correction = 1.0 - 0.015 * (t_inj / 18.0)
            
            final_mult = int(np.clip(base_mult * rpm_correction * load_correction, 70, 160))
            multiplier_matrix[i, j] = final_mult
            
    return multiplier_matrix

# ================= 3. USER INTERFACE & OUTPUT DISPLAY =================

st.subheader("Generate CNG Calibration Multiplier Matrix")
st.info(f"System Configuration: **{n_cyl} Cylinders** | Fuel: **{fuel_type}** | Pressure: **{reducer_press:.2f} bar** | Target AFR: **{AFR_ref}**")

if st.button("Compute Multiplier Map"):
    with st.spinner("Calculating optimal MPC fuel injection map..."):
        map_matrix = generate_mpc_multiplier_map(reducer_press, AFR_ref)
        
        # Build DataFrame with AEB/MIJO native structure
        df_aeb_map = pd.DataFrame(
            map_matrix,
            index=[f"{t:.2f}" for t in inj_grid_ms],
            columns=[str(rpm) for rpm in rpm_grid]
        )
        
        st.markdown("### **T.inj - RPM Injector Multiplier Map**")
        st.markdown("*Row Header: Petrol Injection Time $T_{inj}$ (ms) | Column Header: Engine Speed (RPM)*")
        
        # Display styled table mimicking AEB/MIJO software layout
        st.dataframe(
            df_aeb_map.style.background_gradient(cmap='Blues', axis=None).format("{:.0f}"),
            use_container_width=True
        )
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = df_aeb_map.to_csv()
            st.download_button(
                label="Download Map as CSV (AEB/MIJO Import)",
                data=csv_data,
                file_name=f"CNG_MPC_Multiplier_Map_{reducer_press}bar.csv",
                mime="text/csv"
            )
        with col_dl2:
            # Format text representation for fast copy-pasting directly into ECU grid
            text_map = df_aeb_map.to_string(header=True, index=True)
            st.download_button(
                label="Download Map as Text Grid",
                data=text_map,
                file_name="AEB_MIJO_Map_Grid.txt",
                mime="text/plain"
            )