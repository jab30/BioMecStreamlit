import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Motion Capture Analyzer",
    page_icon="🟪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #2d3748;
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4299e1;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-box h4 {
        color: #63b3ed;
        margin-bottom: 0.5rem;
    }
    .metric-box li {
        color: #cbd5e0;
        margin-bottom: 0.3rem;
    }
    .improvement-box {
        background-color: #742a2a;
        color: #fed7d7;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #f56565;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .improvement-box h4 {
        color: #feb2b2;
        margin-bottom: 0.5rem;
    }
    .improvement-box li {
        color: #fbb6ce;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


def load_mocap_data_from_file(file_path):
    """Load and process OptiTrack CSV data from file path"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None, None, None, 0

        # Read the file content
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Parse header information
        header_info = {}
        for i, line in enumerate(lines[:6]):
            if 'Capture Frame Rate' in line:
                header_info['frame_rate'] = float(line.split(',')[7]) if len(line.split(',')) > 7 else 100.0
            if 'Total Frames' in line:
                header_info['total_frames'] = int(line.split(',')[13]) if len(line.split(',')) > 13 else 0

        # Read data starting from row 6
        df = pd.read_csv(file_path, header=6)

        # Get column information from header lines
        name_line = lines[3].strip().split(',')[1:] if len(lines) > 3 else []
        coord_line = lines[5].strip().split(',')[1:] if len(lines) > 5 else []

        # Find Kenya Brewer columns
        kenya_columns = []
        for i, name in enumerate(name_line):
            clean_name = name.replace('"', '').strip()
            if 'Kenya Brewer' in clean_name and 'Unlabeled' not in clean_name:
                kenya_columns.append({
                    'index': i,
                    'name': clean_name,
                    'coordinate': coord_line[i] if i < len(coord_line) else 'Unknown'
                })

        # Create body part mapping
        body_parts = {}
        for col in kenya_columns:
            parts = col['name'].split(':')
            if len(parts) == 2:
                body_part = parts[1]
                coord = col['coordinate'].lower().strip()

                if body_part not in body_parts:
                    body_parts[body_part] = {}

                if 'position' in coord:
                    if 'x' not in body_parts[body_part]:
                        body_parts[body_part]['x'] = col['index'] + 2
                    elif 'y' not in body_parts[body_part]:
                        body_parts[body_part]['y'] = col['index'] + 2
                    elif 'z' not in body_parts[body_part]:
                        body_parts[body_part]['z'] = col['index'] + 2

        return df, body_parts, header_info, len(kenya_columns)

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, None, 0


def load_mocap_data(uploaded_file):
    """Load and process OptiTrack CSV data from uploaded file"""
    try:
        # Read the file content
        lines = uploaded_file.getvalue().decode("utf-8").split('\n')

        # Parse header information
        header_info = {}
        for i, line in enumerate(lines[:6]):
            if 'Capture Frame Rate' in line:
                header_info['frame_rate'] = float(line.split(',')[7]) if len(line.split(',')) > 7 else 100.0
            if 'Total Frames' in line:
                header_info['total_frames'] = int(line.split(',')[13]) if len(line.split(',')) > 13 else 0

        # Read data starting from row 6
        df = pd.read_csv(uploaded_file, header=6)

        # Get column information from header lines
        name_line = lines[3].strip().split(',')[1:] if len(lines) > 3 else []
        coord_line = lines[5].strip().split(',')[1:] if len(lines) > 5 else []

        # Find Kenya Brewer columns
        kenya_columns = []
        for i, name in enumerate(name_line):
            clean_name = name.replace('"', '').strip()
            if 'Kenya Brewer' in clean_name and 'Unlabeled' not in clean_name:
                kenya_columns.append({
                    'index': i,
                    'name': clean_name,
                    'coordinate': coord_line[i] if i < len(coord_line) else 'Unknown'
                })

        # Create body part mapping
        body_parts = {}
        for col in kenya_columns:
            parts = col['name'].split(':')
            if len(parts) == 2:
                body_part = parts[1]
                coord = col['coordinate'].lower().strip()

                if body_part not in body_parts:
                    body_parts[body_part] = {}

                if 'position' in coord:
                    if 'x' not in body_parts[body_part]:
                        body_parts[body_part]['x'] = col['index'] + 2
                    elif 'y' not in body_parts[body_part]:
                        body_parts[body_part]['y'] = col['index'] + 2
                    elif 'z' not in body_parts[body_part]:
                        body_parts[body_part]['z'] = col['index'] + 2

        return df, body_parts, header_info, len(kenya_columns)

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, None, 0


def create_movement_timeline(df, body_parts):
    """Create an interactive timeline showing key movement phases"""
    if not body_parts or 'Hip' not in body_parts:
        return None

    hip_data = body_parts['Hip']
    if not all(coord in hip_data for coord in ['x', 'y', 'z']):
        return None

    time_col = df.columns[1]
    time_data = df[time_col].values

    # Calculate movement velocity
    x_vals = df.iloc[:, hip_data['x']].values
    y_vals = df.iloc[:, hip_data['y']].values
    z_vals = df.iloc[:, hip_data['z']].values

    # Calculate 3D velocity
    velocity = []
    for i in range(1, len(x_vals)):
        dx = x_vals[i] - x_vals[i - 1]
        dy = y_vals[i] - y_vals[i - 1]
        dz = z_vals[i] - z_vals[i - 1]
        dt = time_data[i] - time_data[i - 1]

        if dt > 0:
            vel = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) / dt
            velocity.append(vel)
        else:
            velocity.append(0)

    velocity = [0] + velocity  # Add initial zero

    # Smooth the velocity
    window = max(5, len(velocity) // 20)
    velocity_smooth = pd.Series(velocity).rolling(window=window, center=True).mean().fillna(pd.Series(velocity))

    # Create the plot
    fig = go.Figure()

    # Add velocity line
    fig.add_trace(go.Scatter(
        x=time_data,
        y=velocity_smooth,
        mode='lines',
        line=dict(color='#1f77b4', width=3),
        name='Hip Velocity',
        fill='tonexty'
    ))

    # Identify movement phases
    vel_threshold = np.percentile(velocity_smooth, 75)
    high_activity = velocity_smooth > vel_threshold

    # Add phase annotations
    phase_starts = []
    in_phase = False
    for i, high_vel in enumerate(high_activity):
        if high_vel and not in_phase:
            phase_starts.append(time_data[i])
            in_phase = True
        elif not high_vel and in_phase:
            in_phase = False

    for i, start_time in enumerate(phase_starts):
        fig.add_vline(
            x=start_time,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Phase {i + 1}",
            annotation_position="top"
        )

    fig.update_layout(
        title="Movement Timeline - Hip Velocity Analysis",
        xaxis_title="Time (seconds)",
        yaxis_title="Velocity (m/s)",
        template="plotly_white",
        height=400,
        showlegend=False
    )

    return fig


def create_body_heat_map(df, body_parts):
    """Create a heatmap showing movement intensity by body part"""
    movement_data = []

    key_parts = ['Hip', 'Head', 'LHand', 'RHand', 'LFoot', 'RFoot', 'Chest', 'LShoulder', 'RShoulder']

    for part in key_parts:
        if part in body_parts:
            coords = body_parts[part]
            if all(coord in coords for coord in ['x', 'y', 'z']):
                try:
                    x_data = df.iloc[:, coords['x']].dropna()
                    y_data = df.iloc[:, coords['y']].dropna()
                    z_data = df.iloc[:, coords['z']].dropna()

                    if len(x_data) > 0 and len(y_data) > 0 and len(z_data) > 0:
                        # Calculate range of movement in feet
                        x_range = (x_data.max() - x_data.min()) * 3.28084
                        y_range = (y_data.max() - y_data.min()) * 3.28084
                        z_range = (z_data.max() - z_data.min()) * 3.28084
                        total_range = np.sqrt(x_range ** 2 + y_range ** 2 + z_range ** 2)

                        movement_data.append({
                            'body_part': part,
                            'x_movement': x_range,
                            'y_movement': y_range,
                            'z_movement': z_range,
                            'total_movement': total_range
                        })
                except:
                    continue

    if not movement_data:
        return None

    df_movement = pd.DataFrame(movement_data)
    df_movement = df_movement.sort_values('total_movement', ascending=True)

    # Create horizontal bar chart
    fig = go.Figure()

    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0', '#ffb3e6', '#c4e17f', '#ffb3b3']

    fig.add_trace(go.Bar(
        y=df_movement['body_part'],
        x=df_movement['total_movement'],
        orientation='h',
        marker_color=colors[:len(df_movement)],
        text=[f'{val:.1f} ft' for val in df_movement['total_movement']],
        textposition='outside'
    ))

    fig.update_layout(
        title="Movement Range by Body Part",
        xaxis_title="Total Movement Range (feet)",
        yaxis_title="Body Parts",
        template="plotly_white",
        height=500,
        showlegend=False
    )

    return fig


def create_3d_trajectory(df, body_parts):
    """Create a 3D trajectory plot"""
    if 'Hip' not in body_parts:
        return None

    hip_coords = body_parts['Hip']
    if not all(coord in hip_coords for coord in ['x', 'y', 'z']):
        return None

    x_data = df.iloc[:, hip_coords['x']].values
    y_data = df.iloc[:, hip_coords['y']].values
    z_data = df.iloc[:, hip_coords['z']].values
    time_data = df.iloc[:, 1].values

    # Remove NaN values
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data) | np.isnan(z_data))
    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]
    z_clean = z_data[valid_mask]
    time_clean = time_data[valid_mask]

    if len(x_clean) == 0:
        return None

    fig = go.Figure(data=[go.Scatter3d(
        x=x_clean,
        y=y_clean,
        z=z_clean,
        mode='markers+lines',
        marker=dict(
            size=4,
            color=time_clean,
            colorscale='Viridis',
            colorbar=dict(title="Time (s)"),
            showscale=True
        ),
        line=dict(
            color='darkblue',
            width=6
        ),
        name='Hip Movement Path'
    )])

    # Add start and end markers
    fig.add_trace(go.Scatter3d(
        x=[x_clean[0]], y=[y_clean[0]], z=[z_clean[0]],
        mode='markers',
        marker=dict(size=12, color='green', symbol='diamond'),
        name='Start Position'
    ))

    fig.add_trace(go.Scatter3d(
        x=[x_clean[-1]], y=[y_clean[-1]], z=[z_clean[-1]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='square'),
        name='End Position'
    ))

    fig.update_layout(
        title="3D Hip Movement Trajectory",
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Z Position (m)'
        ),
        template="plotly_white",
        height=600
    )

    return fig


def create_coordination_plot(df, body_parts):
    """Create a plot showing coordination between left and right sides"""
    left_parts = ['LHand', 'LFoot']
    right_parts = ['RHand', 'RFoot']

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Hand Coordination', 'Foot Coordination'),
        vertical_spacing=0.1
    )

    time_data = df.iloc[:, 1].values
    colors = ['#ff7f0e', '#2ca02c']

    for i, (left_part, right_part) in enumerate(zip(left_parts, right_parts)):
        if left_part in body_parts and right_part in body_parts:
            left_coords = body_parts[left_part]
            right_coords = body_parts[right_part]

            if all(coord in left_coords for coord in ['x', 'y', 'z']) and \
                    all(coord in right_coords for coord in ['x', 'y', 'z']):
                # Calculate movement magnitude for each side
                left_x = df.iloc[:, left_coords['x']].values
                left_y = df.iloc[:, left_coords['y']].values
                left_z = df.iloc[:, left_coords['z']].values

                right_x = df.iloc[:, right_coords['x']].values
                right_y = df.iloc[:, right_coords['y']].values
                right_z = df.iloc[:, right_coords['z']].values

                # Calculate distances from starting position
                left_dist = np.sqrt((left_x - left_x[0]) ** 2 + (left_y - left_y[0]) ** 2 + (left_z - left_z[0]) ** 2)
                right_dist = np.sqrt(
                    (right_x - right_x[0]) ** 2 + (right_y - right_y[0]) ** 2 + (right_z - right_z[0]) ** 2)

                # Add traces
                fig.add_trace(
                    go.Scatter(x=time_data, y=left_dist, name=f'Left {left_part[1:]}',
                               line=dict(color=colors[0]), mode='lines'),
                    row=i + 1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=time_data, y=right_dist, name=f'Right {right_part[1:]}',
                               line=dict(color=colors[1]), mode='lines'),
                    row=i + 1, col=1
                )

    fig.update_layout(
        title="Left vs Right Side Coordination",
        template="plotly_white",
        height=600
    )

    fig.update_xaxes(title_text="Time (seconds)")
    fig.update_yaxes(title_text="Distance from Start (meters)")

    return fig


# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">Motion Capture Analyzer</h1>', unsafe_allow_html=True)

    st.markdown("""
    Upload your OptiTrack CSV file to analyze movement patterns, identify performance insights, 
    and get actionable recommendations for improvement.
    """)

    # Sidebar
    st.sidebar.header("Data Source")

    # Option to choose between demo file or upload
    data_source = st.sidebar.radio(
        "Choose data source:",
        ("Demo File (Opti.csv)", "Upload Your Own File")
    )

    df, body_parts, header_info, num_markers = None, None, None, 0

    if data_source == "Demo File (Opti.csv)":
        # Try to load the demo file
        with st.spinner('Loading demo data from Opti.csv...'):
            df, body_parts, header_info, num_markers = load_mocap_data_from_file('Opti.csv')

        if df is None:
            st.warning("Demo file 'Opti.csv' not found in the current directory. Please upload your own file.")
            data_source = "Upload Your Own File"

    if data_source == "Upload Your Own File":
        uploaded_file = st.sidebar.file_uploader("Choose your CSV file", type=['csv'])
        if uploaded_file is not None:
            with st.spinner('Processing your motion capture data...'):
                df, body_parts, header_info, num_markers = load_mocap_data(uploaded_file)

    if df is not None and body_parts is not None:
        # Display basic metrics
        col1, col2, col3, col4 = st.columns(4)

        duration = float(df.iloc[-1, 1]) if len(df) > 0 else 0
        frame_rate = header_info.get('frame_rate', 100)

        with col1:
            st.metric("Session Duration", f"{duration:.1f}s")
        with col2:
            st.metric("Frame Rate", f"{frame_rate:.0f} fps")
        with col3:
            st.metric("Total Frames", len(df))
        with col4:
            st.metric("Tracking Points", num_markers)

        # Chart selection
        st.sidebar.header("View Options")
        chart_options = [
            "Movement Timeline",
            "Body Part Activity",
            "3D Trajectory",
            "Left/Right Coordination"
        ]

        selected_charts = st.sidebar.multiselect(
            "Select charts to display:",
            chart_options,
            default=chart_options[:2]
        )

        # Display selected charts
        if "Movement Timeline" in selected_charts:
            st.subheader("Movement Timeline Analysis")
            timeline_fig = create_movement_timeline(df, body_parts)
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
                st.markdown("""
                <div class="improvement-box">
                <h4>Key Insights:</h4>
                <ul>
                    <li>Red dashed lines indicate high-activity movement phases</li>
                    <li>Focus on smooth transitions between phases</li>
                    <li>Consistent velocity patterns indicate better movement control</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

        if "Body Part Activity" in selected_charts:
            st.subheader("Body Part Movement Analysis")
            heatmap_fig = create_body_heat_map(df, body_parts)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)

        if "3D Trajectory" in selected_charts:
            st.subheader("3D Movement Path")
            trajectory_fig = create_3d_trajectory(df, body_parts)
            if trajectory_fig:
                st.plotly_chart(trajectory_fig, use_container_width=True)
                st.markdown("""
                <div class="improvement-box">
                <h4>Spatial Analysis:</h4>
                <ul>
                    <li>Green diamond = starting position, Red square = ending position</li>
                    <li>Look for smooth, efficient paths between positions</li>
                    <li>Minimize unnecessary detours or corrections in the path</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

        if "Left/Right Coordination" in selected_charts:
            st.subheader("Bilateral Coordination")
            coordination_fig = create_coordination_plot(df, body_parts)
            if coordination_fig:
                st.plotly_chart(coordination_fig, use_container_width=True)
                st.markdown("""
                <div class="improvement-box">
                <h4>Symmetry Check:</h4>
                <ul>
                    <li>Similar patterns between left/right indicate good coordination</li>
                    <li>Large differences may suggest imbalances to address</li>
                    <li>Practice exercises to improve weaker side coordination</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

    else:
        if data_source == "Demo File (Opti.csv)":
            st.info(
                "Demo file 'Opti.csv' not found. Please place the Opti.csv file in the same directory as this script, or upload your own file.")
        else:
            st.info("Please upload a CSV file from your OptiTrack motion capture system to begin analysis.")

        # Show sample visualization
        st.subheader("Sample Analysis Preview")
        st.image("https://via.placeholder.com/800x400/1f77b4/ffffff?text=Upload+Your+Data+to+See+Analysis",
                 caption="Your motion capture visualizations will appear here")


if __name__ == "__main__":
    main()