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
APP_TITLE = "High-Throughput Growth Profiler"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"🔬 {APP_TITLE}")

STRAIN_NAMES = {1: "MUCL 28849", 2: "MUCL 29853", 3: "MUCL 29989", 4: "Y-01481", 
                5: "Y-00587", 6: "Y-7784", 7: "Y-00879", 8: "W29", 9: "Y-1095", 10: "Y-01087"}
LABEL_PATTERN = re.compile(r"M(\d+)_Y(\d+)_R(\d+)")

if 'datasets' not in st.session_state:
    st.session_state.datasets = {} 
if 'step' not in st.session_state:
    st.session_state.step = 0

# ============================================================
# PERSISTENCE HELPERS
# ============================================================
def save_session():
    return pickle.dumps({"datasets": st.session_state.datasets, "step": st.session_state.step})

def load_session(uploaded_file):
    try:
        data = pickle.load(uploaded_file)
        st.session_state.datasets = data["datasets"]
        st.session_state.step = data["step"]
        st.success("Session loaded!")
        st.rerun()
    except Exception as e:
        st.error(f"Load failed: {e}")

with st.sidebar:
    st.header("💾 Session Manager")
    if st.session_state.datasets:
        st.download_button("📥 Download Session", save_session(), "growth_session.pkl", "application/octet-stream")
    up_session = st.file_uploader("📤 Load Session", type=["pkl"])
    if up_session and st.button("🔄 Restore"): load_session(up_session)

# ============================================================
# HELPER: ROBUST CSV LOADER
# ============================================================
def load_od_csv(uploaded_file):
    uploaded_file.seek(0)
    try:
        raw = uploaded_file.read().decode("utf-8", errors="ignore")
        lines = raw.splitlines()
        start_idx = None
        for i, line in enumerate(lines):
            if any(h in line for h in ["Time(min)", "Time (min)", "Time [min]"]):
                start_idx = i
                break
        if start_idx is None: return None
        data_body = "\n".join(lines[start_idx:])
        df = pd.read_csv(io.StringIO(data_body), sep=None, engine='python', on_bad_lines='skip')
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        return df if df.shape[1] >= 2 else None
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return None

# ============================================================
# STEP 0: DATASET MANAGER (RESTORED DELETE)
# ============================================================
if st.session_state.step == 0:
    st.header("Step 0: Manage Experiment Groups")
    
    if st.session_state.datasets:
        st.subheader("Current Datasets:")
        for name in list(st.session_state.datasets.keys()):
            col_a, col_b = st.columns([4, 1])
            col_a.write(f"✅ **{name}** ({len(st.session_state.datasets[name].get('data', {}))} plates)")
            if col_b.button("🗑️ Remove", key=f"del_{name}"):
                del st.session_state.datasets[name]
                st.rerun()
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
            st.session_state.step = 1; st.rerun()

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
                        if df is not None: st.session_state.datasets[name]["data"][f.name] = df
            if st.session_state.datasets[name]["data"]:
                for fn in st.session_state.datasets[name]["data"].keys(): st.write(f"📄 {fn}")

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Back"): st.session_state.step = 0; st.rerun()
    if col2.button("Next: Plate Maps ➡️"):
        if any(len(d["data"]) > 0 for d in st.session_state.datasets.values()):
            st.session_state.step = 2; st.rerun()

# ============================================================
# STEP 2: PLATE MAPS
# ============================================================
elif st.session_state.step == 2:
    st.header("Step 2: Assign Plate Maps")
    st.info("""
    **📋 How to paste your Plate Maps:**
    Please paste the well labels in order (starting from A1, A2...). The software expects the format: 
    **Mx_Yx_Rx** where **M** is Media ID, **Y** is Strain ID, and **R** is Replicate.
    
    * **Bad Wells:** If a well is contaminated or messed up, type **MISSING** in its place to omit it.
    
    **Example Format:**
    ```text
    M1_Y1_R1  M1_Y1_R2  M1_Y1_R3  MISSING
    M1_Y2_R2  M1_Y2_R3  M2_Y1_R1  M2_Y1_R2
    ```
    """)

    for ds_name, ds_info in st.session_state.datasets.items():
        with st.expander(f"🗺️ Maps for {ds_name}", expanded=True):
            saved_filenames = sorted(list(ds_info["data"].keys()))
            for idx, fname in enumerate(saved_filenames):
                m_val = st.text_area(f"Plate {idx+1} Layout ({fname})", 
                                     value=ds_info["maps"].get(fname, ""), key=f"txt_{ds_name}_{fname}", height=120)
                st.session_state.datasets[ds_name]["maps"][fname] = m_val

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Back"): st.session_state.step = 1; st.rerun()
    if col2.button("Next: View Results ➡️"): st.session_state.step = 3; st.rerun()

# ============================================================
# STEP 3: RESULTS (AXIS LABELS + ALL FILTERS RESTORED)
# ============================================================
elif st.session_state.step == 3:
    st.header("Step 3: Kinetic Analysis & Filtering")
    grouped_curves = {}; time_points = {}

    for ds_name, ds_info in st.session_state.datasets.items():
        for fname, df in ds_info["data"].items():
            t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values / 60
            time_points[ds_name] = t
            well_vals = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values
            labels = [l.strip().upper() for l in ds_info["maps"].get(fname, "").split() if l.strip()]
            
            for idx, label in enumerate(labels):
                if "MISSING" in label: continue
                match = LABEL_PATTERN.search(label)
                if match and idx < well_vals.shape[1]:
                    mid, sid, rep = map(int, match.groups())
                    curve = well_vals[:, idx]
                    if not np.all(np.isnan(curve)):
                        grouped_curves.setdefault((ds_name, sid, mid), []).append(np.where(curve < 0, np.nan, curve))

    with st.sidebar:
        st.subheader("🧪 Media Dictionary")
        for ds_name in st.session_state.datasets.keys():
            with st.expander(f"Media Names for {ds_name}"):
                m_ids = sorted(list(set(k[2] for k in grouped_curves.keys() if k[0] == ds_name)))
                for mid in m_ids:
                    st.session_state.datasets[ds_name]["media"][mid] = st.text_input(
                        f"M{mid} Name", value=st.session_state.datasets[ds_name]["media"].get(mid, f"Media {mid}"),
                        key=f"m_in_{ds_name}_{mid}")
        
        st.divider()
        st.subheader("🔍 Filter View")
        all_strains = sorted(list(set(STRAIN_NAMES.get(k[1], f"Y{k[1]}") for k in grouped_curves.keys())))
        sel_strains = st.multiselect("Strains", all_strains, default=all_strains)
        
        all_media_labels = sorted(list(set(st.session_state.datasets[k[0]]["media"].get(k[2], f"Media {k[2]}") for k in grouped_curves.keys())))
        sel_media = st.multiselect("Media Types", all_media_labels, default=all_media_labels)
        
        sel_ds = st.multiselect("Datasets", list(st.session_state.datasets.keys()), default=list(st.session_state.datasets.keys()))
        
        smooth = st.checkbox("Apply Smoothing", True); win = st.slider("Smoothing Window", 5, 31, 11, 2)
        show_sd = st.checkbox("Show Standard Deviation", True)

    fig_curves = go.Figure(); final_stats = []; colors = px.colors.qualitative.Plotly + px.colors.qualitative.Safe; color_idx = 0

    for (ds_name, sid, mid), reps_list in grouped_curves.items():
        s_name = STRAIN_NAMES.get(sid, f"Y{sid}")
        m_name = st.session_state.datasets[ds_name]["media"].get(mid, f"Media {mid}")
        
        if s_name in sel_strains and m_name in sel_media and ds_name in sel_ds:
            min_l = min(len(r) for r in reps_list); x = time_points[ds_name][:min_l]
            arr = np.array([r[:min_l] for r in reps_list])
            y_mean = np.nanmean(arr, axis=0); y_sd = np.nanstd(arr, axis=0)
            
            y_clean = pd.Series(y_mean).interpolate().bfill().ffill().values if np.isnan(y_mean).any() else y_mean
            plot_y = savgol_filter(y_clean, win, 2) if (smooth and len(y_clean) > win) else y_clean
            
            color = colors[color_idx % len(colors)]; leg = f"{ds_name} | {s_name} | {m_name}"

            if show_sd:
                rgba = f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}" if color.startswith('#') else color.replace("rgb", "rgba").replace(")", ", 0.15)")
                fig_curves.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([plot_y+y_sd, (plot_y-y_sd)[::-1]]), fill='toself', fillcolor=rgba, line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            
            fig_curves.add_trace(go.Scatter(x=x, y=plot_y, name=leg, line=dict(width=3, color=color)))
            
            # Kinetic μmax calculation
            mu = 0; mask = (y_clean > 0.01) & (~np.isnan(y_clean))
            if np.sum(mask) > 5:
                try:
                    slp = [np.polyfit(x[mask][i:i+5], np.log(y_clean[mask][i:i+5]), 1)[0] for i in range(len(y_clean[mask])-5)]
                    mu = round(max(slp), 3) if slp else 0
                except: mu = 0
            final_stats.append({"Legend": leg, "Mu": mu, "Color": color}); color_idx += 1

    # --- SET AXIS TITLES HERE ---
    fig_curves.update_layout(xaxis_title="Time (h)", yaxis_title="OD600", legend_title="Experiment Groups")

    if final_stats:
        st.subheader("📈 Growth Kinetics Visualization")
        st.plotly_chart(fig_curves, use_container_width=True)
        st.subheader("🚀 Calculated Maximum Growth Rates (μmax)")
        st.plotly_chart(px.bar(pd.DataFrame(final_stats), x="Legend", y="Mu", color="Legend", color_discrete_map={s['Legend']: s['Color'] for s in final_stats}, text="Mu", labels={"Mu": "μmax (h⁻¹)", "Legend": "Group"}), use_container_width=True)
    
    if st.button("⬅️ Back to Data Entry"): st.session_state.step = 0; st.rerun()
