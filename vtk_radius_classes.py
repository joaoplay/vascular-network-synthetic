"""
VTK real-time construction of the vascular network with 7 radius classes.
Grows the graph incrementally via BFS from the highest-degree node, capturing
each step as a frame. Edges are colored and sized by radius class.
"""
import vtk
import numpy as np
import imageio
import os
from collections import deque

from settings import OUTPUT_PATH
from vascular_network.dataset_generation import generate_training_graph
from sgg.radius_classes import R_EDGES

# --- Configuration ---
GIF_OUTPUT_PATH = "images/vascular_network_7classes.gif"
EDGES_PER_FRAME = 5           # how many edges to add per frame
FRAME_DURATION_MS = 50        # ms per frame in the GIF
SLOW_ROTATION_DEG = 0.3       # slow camera rotation per frame
WINDOW_SIZE = (1024, 768)
BACKGROUND_COLOR = (1.0, 1.0, 1.0)

CLASS_LABELS = ['tiny', 'small', 'medium', 'normal', 'large', 'big', 'huge']
CLASS_COLORS = [
    (0.122, 0.467, 0.706),  # tiny    - #1f77b4
    (0.090, 0.745, 0.812),  # small   - #17becf
    (0.173, 0.627, 0.173),  # medium  - #2ca02c
    (0.737, 0.741, 0.133),  # normal  - #bcbd22
    (1.000, 0.498, 0.055),  # large   - #ff7f0e
    (0.839, 0.153, 0.157),  # big     - #d62728
    (0.580, 0.404, 0.741),  # huge    - #9467bd
]
CLASS_TUBE_RADII = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

# --- Load graph ---
print("Loading vascular network graph...")
G, _ = generate_training_graph(OUTPUT_PATH)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# --- Classify each edge by radius ---
r_edges = np.array(R_EDGES, dtype=float)
pos_dict = {node: np.array(G.nodes[node]['node_label'], dtype=float) for node in G.nodes}

edge_classes = {}
for u, v, data in G.edges(data=True):
    radius = float(data.get('avgRadiusAvg', 3.0) or 3.0)
    class_idx = int(np.searchsorted(r_edges, radius, side='left'))
    class_idx = max(0, min(6, class_idx))
    edge_classes[(u, v)] = class_idx

# --- BFS edge ordering from highest-degree node ---
start_node = max(G.nodes, key=lambda n: G.degree(n))
visited = {start_node}
queue = deque([start_node])
bfs_edges = []

while queue:
    node = queue.popleft()
    for neighbor in G.neighbors(node):
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)
            # find the edge key (may be (node, neighbor) or (neighbor, node))
            key = (node, neighbor) if (node, neighbor) in edge_classes else (neighbor, node)
            bfs_edges.append(key)

print(f"BFS traversal: {len(bfs_edges)} edges from node {start_node}")
total_frames = (len(bfs_edges) + EDGES_PER_FRAME - 1) // EDGES_PER_FRAME
print(f"Will produce ~{total_frames} construction frames")

# --- Per-class VTK structures (grow incrementally) ---
class_points = {}
class_lines = {}
class_polydata = {}
class_tube_filters = {}
class_node_maps = {}

renderer = vtk.vtkRenderer()
renderer.SetBackground(*BACKGROUND_COLOR)

for ci in range(7):
    pts = vtk.vtkPoints()
    lns = vtk.vtkCellArray()
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lns)

    tf = vtk.vtkTubeFilter()
    tf.SetInputData(pd)
    tf.SetRadius(CLASS_TUBE_RADII[ci])
    tf.SetNumberOfSides(12)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tf.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*CLASS_COLORS[ci])
    actor.GetProperty().SetOpacity(0.85)

    renderer.AddActor(actor)

    class_points[ci] = pts
    class_lines[ci] = lns
    class_polydata[ci] = pd
    class_tube_filters[ci] = tf
    class_node_maps[ci] = {}

# --- Color legend ---
RADIUS_RANGES = ['< 2', '2 - 3', '3 - 4', '4 - 5', '5 - 7', '7 - 10', '> 10']
legend = vtk.vtkLegendBoxActor()
legend.SetNumberOfEntries(7)
for ci in range(7):
    # Create a small colored sphere as the legend icon
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(0.5)
    sphere.Update()
    legend.SetEntry(ci, sphere.GetOutput(), f"{CLASS_LABELS[ci]}  ({RADIUS_RANGES[ci]})",
                    [*CLASS_COLORS[ci]])
legend.UseBackgroundOn()
legend.SetBackgroundColor(0.95, 0.95, 0.95)
legend.SetBackgroundOpacity(0.8)
legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
legend.GetPositionCoordinate().SetValue(0.01, 0.02)
legend.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
legend.GetPosition2Coordinate().SetValue(0.22, 0.38)
legend.GetEntryTextProperty().SetFontSize(14)
legend.GetEntryTextProperty().SetColor(0.0, 0.0, 0.0)
renderer.AddActor(legend)

# --- Progress annotation ---
progress_annotation = vtk.vtkCornerAnnotation()
progress_annotation.SetLinearFontScaleFactor(2)
progress_annotation.SetNonlinearFontScaleFactor(1)
progress_annotation.SetMaximumFontSize(16)
progress_annotation.GetTextProperty().SetColor(0.0, 0.0, 0.0)
renderer.AddViewProp(progress_annotation)

# --- Off-screen rendering ---
window = vtk.vtkRenderWindow()
window.SetOffScreenRendering(1)
window.AddRenderer(renderer)
window.SetSize(*WINDOW_SIZE)

# Set camera to encompass the full graph, centered properly
all_positions = np.array(list(pos_dict.values()))
center = all_positions.mean(axis=0)
extent = all_positions.max(axis=0) - all_positions.min(axis=0)
max_extent = float(np.linalg.norm(extent))  # diagonal extent

camera = renderer.GetActiveCamera()
camera.SetFocalPoint(*center)
# Position camera looking slightly from above, at a distance that fits the whole graph
cam_distance = max_extent * 1.2
camera.SetPosition(
    center[0],
    center[1] - cam_distance * 0.9,
    center[2] + cam_distance * 0.45,
)
camera.SetViewUp(0, 0, 1)
camera.SetClippingRange(cam_distance * 0.01, cam_distance * 10)
renderer.ResetCameraClippingRange()

# --- Helper to add an edge to its class VTK structures ---
def add_edge_vtk(u, v, ci):
    pts = class_points[ci]
    lns = class_lines[ci]
    nmap = class_node_maps[ci]

    for node in (u, v):
        if node not in nmap:
            nmap[node] = pts.GetNumberOfPoints()
            pts.InsertNextPoint(pos_dict[node])

    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, nmap[u])
    line.GetPointIds().SetId(1, nmap[v])
    lns.InsertNextCell(line)

    class_polydata[ci].Modified()
    class_tube_filters[ci].Modified()


def capture_frame(w2i_filter):
    w2i_filter.Modified()
    w2i_filter.Update()
    vtk_image = w2i_filter.GetOutput()
    w, h, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    np_array = np.frombuffer(
        memoryview(vtk_array), dtype=np.uint8
    ).reshape(h, w, 3)
    return np.flipud(np_array).copy()


# --- Grow the network frame by frame ---
w2i = vtk.vtkWindowToImageFilter()
w2i.SetInput(window)
w2i.SetInputBufferTypeToRGB()

frames = []
edge_idx = 0

while edge_idx < len(bfs_edges):
    # Add a batch of edges
    batch_end = min(edge_idx + EDGES_PER_FRAME, len(bfs_edges))
    for i in range(edge_idx, batch_end):
        u, v = bfs_edges[i]
        ci = edge_classes[(u, v)]
        add_edge_vtk(u, v, ci)
    edge_idx = batch_end

    # Update progress text
    pct = int(100 * edge_idx / len(bfs_edges))
    progress_annotation.SetText(2, f"Edges: {edge_idx}/{len(bfs_edges)}  ({pct}%)")

    # Slow rotation
    camera.Azimuth(SLOW_ROTATION_DEG)
    renderer.ResetCameraClippingRange()

    window.Render()
    frames.append(capture_frame(w2i))

    if edge_idx % 100 == 0 or edge_idx == len(bfs_edges):
        print(f"  {edge_idx}/{len(bfs_edges)} edges")

# Hold on the final frame for a moment
for _ in range(30):
    camera.Azimuth(SLOW_ROTATION_DEG)
    renderer.ResetCameraClippingRange()
    window.Render()
    frames.append(capture_frame(w2i))

# --- Save GIF ---
os.makedirs(os.path.dirname(GIF_OUTPUT_PATH), exist_ok=True)
print(f"Saving GIF ({len(frames)} frames) to {GIF_OUTPUT_PATH}...")
imageio.mimsave(
    GIF_OUTPUT_PATH,
    frames,
    duration=FRAME_DURATION_MS / 1000.0,
    loop=0,
)
print(f"Done! GIF saved to {GIF_OUTPUT_PATH}")
