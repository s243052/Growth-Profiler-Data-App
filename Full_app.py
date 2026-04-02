import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import re
import pickle
import io
from scipy.signal import savgol_filter

# ============================================================
# 1. PAGE CONFIG & SESSION STATE
# ============================================================
st.set_page_config(page_title="Growth Profiler Analysis", layout="wide")
st.title("🧪 High-Throughput Microbial Analyzer ")

STRAIN_NAMES = {1: "MUCL 28849", 2: "MUCL 29853", 3: "MUCL 29989", 4: "Y-01481", 
                5: "Y-00587", 6: "Y-7784", 7: "Y-00879", 8: "W29", 9: "Y-1095", 10: "Y-01087"}
LABEL_PATTERN = re.compile(r"M(\d+)_Y(\d+)_R(\d+)")

if 'datasets' not in st.session_state:
    st.session_state.datasets = {} 
if 'step' not in st.session_state:
    st.session_state.step = 0

# ============================================================
# PERSISTENCE HELPERS (Save/Load to Disk)
# ============================================================
def save_session():
    # Bundles the dictionary into a binary format
    return pickle.dumps({
        "datasets": st.session_state.datasets,
        "step": st.session_state.step
    })

def load_session(uploaded_file):
    try:
        data = pickle.load(uploaded_file)
        st.session_state.datasets = data["datasets"]
        st.session_state.step = data["step"]
        st.success("Session loaded successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to load session: {e}")

# Sidebar Persistence Controls
with st.sidebar:
    st.header("💾 Save/Load Session")
    # Download current state
    if st.session_state.datasets:
        st.download_button(
            label="📥 Download Session File",
            data=save_session(),
            file_name="od_analyzer_session.pkl",
            mime="application/octet-stream",
            help="Save all datasets and maps to your computer."
        )
    
    # Upload previous state
    uploaded_session = st.file_uploader("📤 Load Session File", type=["pkl"])
    if uploaded_session is not None:
        if st.button("🔄 Restore Session"):
            load_session(uploaded_session)

# ============================================================
# HELPER: ROBUST CSV LOADER
# ============================================================
def load_od_csv(uploaded_file):
    uploaded_file.seek(0)
    try:
        raw = uploaded_file.read()
        uploaded_file.seek(0)
        text = raw.decode("utf-8", errors="ignore")
        lines = text.splitlines()

        start_line = None
        for i, line in enumerate(lines):
            line_clean = line.strip().replace('"', '')
            if any(h in line_clean for h in ["Time(min)", "Time (min)", "Time [min]"]) or line_clean.startswith("Time"):
                start_line = i
                break

        if start_line is None: return None

        csv_text = "\n".join(lines[start_line:])
        df = pd.read_csv(io.StringIO(csv_text))
        if df.shape[1] == 1:
            df = pd.read_csv(io.StringIO(csv_text), sep=';')

        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        return df if df.shape[1] >= 2 else None
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return None

# ============================================================
# STEP 0: DATASET MANAGER
# ============================================================
if st.session_state.step == 0:
    st.header("Step 0: Manage Experiment Groups")
    
    if st.session_state.datasets:
        st.subheader("Current Datasets:")
        for name, info in st.session_state.datasets.items():
            num_f = len(info.get("data", {}))
            st.write(f"✅ **{name}** ({num_f} plates loaded)")
        st.divider()

    st.subheader("Add New Dataset")
    new_name = st.text_input("Enter a name for the new experiment")
    
    col1, col2 = st.columns(2)
    if col1.button("➕ Add Dataset"):
        if new_name and new_name not in st.session_state.datasets:
            st.session_state.datasets[new_name] = {"data": {}, "maps": {}, "media": {}}
            st.rerun()
            
    if st.session_state.datasets:
        if col2.button("Next: Upload Files ➡️"):
            st.session_state.step = 1
            st.rerun()
            
    if st.button("🗑️ Reset Everything", type="secondary"):
        st.session_state.clear()
        st.rerun()

# ============================================================
# STEP 1: UPLOAD
# ============================================================
elif st.session_state.step == 1:
    st.header("Step 1: Upload CSVs")
    for name in st.session_state.datasets.keys():
        with st.expander(f"📁 Files for {name}", expanded=True):
            ups = st.file_uploader(f"Choose CSVs for {name}", accept_multiple_files=True, key=f"up_{name}")
            if ups:
                for f in ups:
                    if f.name not in st.session_state.datasets[name]["data"]:
                        df = load_od_csv(f)
                        if df is not None:
                            st.session_state.datasets[name]["data"][f.name] = df
            if st.session_state.datasets[name]["data"]:
                st.write("**Currently Loaded:**")
                for fn in st.session_state.datasets[name]["data"].keys():
                    st.write(f"📄 {fn}")

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Back"):
        st.session_state.step = 0
        st.rerun()
    if col2.button("Next: Plate Maps ➡️"):
        if any(len(d["data"]) > 0 for d in st.session_state.datasets.values()):
            st.session_state.step = 2
            st.rerun()
        else:
            st.error("Upload at least one file.")

# ============================================================
# STEP 2: PLATE MAPS
# ============================================================
elif st.session_state.step == 2:
    st.header("Step 2: Assign Plate Maps")
    st.info("""
    **📋 How to paste your Plate Maps:**
    Please paste the well labels in order (starting from A1, A2...). The software expects the format: 
    **Mx_Yx_Rx** where **M** is Media ID, **Y** is Strain ID, and **R** is Replicate.
    
    **Example Format:**
    ```text
    M1_Y1_R1  M1_Y1_R2  M1_Y1_R3  M1_Y2_R1
    M1_Y2_R2  M1_Y2_R3  M2_Y1_R1  M2_Y1_R2
    ```
    """)

    for ds_name, ds_info in st.session_state.datasets.items():
        with st.expander(f"🗺️ Maps for {ds_name}", expanded=True):
            saved_filenames = sorted(list(ds_info["data"].keys()))
            if not saved_filenames:
                st.warning(f"No files for '{ds_name}'.")
            else:
                for idx, fname in enumerate(saved_filenames):
                    m_val = st.text_area(f"Plate {idx+1} Layout (File: {fname})", 
                                         value=ds_info["maps"].get(fname, ""),
                                         key=f"txt_{ds_name}_{fname}", height=120)
                    st.session_state.datasets[ds_name]["maps"][fname] = m_val

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Back to Uploads"):
        st.session_state.step = 1
        st.rerun()
    if col2.button("Next: View Results ➡️"):
        st.session_state.step = 3
        st.rerun()

# ============================================================
# STEP 3: RESULTS
# ============================================================
elif st.session_state.step == 3:
    st.header("Step 3: Results & Filtering")
    grouped_curves = {} 
    time_points = {}

    for ds_name, ds_info in st.session_state.datasets.items():
        for fname, df in ds_info["data"].items():
            t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values / 60
            time_points[ds_name] = t
            well_vals = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values
            p_map = ds_info["maps"].get(fname, "")
            labels = [l.strip().upper() for l in p_map.split() if l.strip()]
            for idx, label in enumerate(labels):
                match = LABEL_PATTERN.search(label)
                if match and idx < well_vals.shape[1]:
                    mid, sid, rep = map(int, match.groups())
                    curve = np.where(well_vals[:, idx] < 0, np.nan, well_vals[:, idx])
                    grouped_curves.setdefault((ds_name, sid, mid), []).append(curve)

    with st.sidebar:
        st.subheader("🧪 Media Dictionary")
        for ds_name in st.session_state.datasets.keys():
            with st.expander(f"Media for {ds_name}"):
                m_ids = sorted(list(set(k[2] for k in grouped_curves.keys() if k[0] == ds_name)))
                for mid in m_ids:
                    st.session_state.datasets[ds_name]["media"][mid] = st.text_input(
                        f"M{mid}", value=st.session_state.datasets[ds_name]["media"].get(mid, f"Media {mid}"),
                        key=f"med_{ds_name}_{mid}")
        
        st.divider()
        st.subheader("🔍 Filters")
        all_strains = sorted(list(set(STRAIN_NAMES.get(k[1], f"Y{k[1]}") for k in grouped_curves.keys())))
        sel_strains = st.multiselect("Select Strains", all_strains, default=all_strains)
        sel_ds = st.multiselect("Select Data Sets", list(st.session_state.datasets.keys()), default=list(st.session_state.datasets.keys()))
        smooth = st.checkbox("Smooth Curves", True); win = st.slider("Window", 5, 31, 11, 2); show_sd = st.checkbox("Show SD", True)

    fig_curves = go.Figure(); final_stats = []; colors = px.colors.qualitative.Plotly + px.colors.qualitative.Safe; color_idx = 0

    for (ds_name, sid, mid), reps_list in grouped_curves.items():
        s_name = STRAIN_NAMES.get(sid, f"Y{sid}")
        m_name = st.session_state.datasets[ds_name]["media"].get(mid, f"Media {mid}")
        if s_name in sel_strains and ds_name in sel_ds:
            min_l = min(len(r) for r in reps_list); arr = np.array([r[:min_l] for r in reps_list], dtype=np.float64)
            y_mean = np.nanmean(arr, axis=0); y_sd = np.nanstd(arr, axis=0); x = time_points[ds_name][:min_l]
            plot_y = savgol_filter(y_mean, win, 2) if (smooth and len(y_mean) > win) else y_mean
            color = colors[color_idx % len(colors)]; leg = f"{ds_name} | {s_name} | {m_name}"
            if show_sd:
                rgba = f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}" if color.startswith('#') else color.replace("rgb", "rgba").replace(")", ", 0.15)")
                fig_curves.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([plot_y+y_sd, (plot_y-y_sd)[::-1]]), fill='toself', fillcolor=rgba, line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            fig_curves.add_trace(go.Scatter(x=x, y=plot_y, name=leg, line=dict(width=3, color=color)))
            mu = 0; mask = (y_mean > 0.01) & (~np.isnan(y_mean))
            if np.sum(mask) > 5:
                try:
                    slp = [np.polyfit(x[mask][i:i+5], np.log(y_mean[mask][i:i+5]), 1)[0] for i in range(len(y_mean[mask])-5)]
                    mu = round(max(slp), 3) if slp else 0
                except: mu = 0
            final_stats.append({"Legend": leg, "Mu": mu, "Color": color}); color_idx += 1

    if final_stats:
        st.plotly_chart(fig_curves, use_container_width=True)
        st.plotly_chart(px.bar(pd.DataFrame(final_stats), x="Legend", y="Mu", color="Legend", color_discrete_map={s['Legend']: s['Color'] for s in final_stats}, text="Mu"), use_container_width=True)
    
    if st.button("⬅️ Add/Edit Datasets"):
        st.session_state.step = 0; st.rerun()
