from typing import TYPE_CHECKING
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_error
from qtpy.QtWidgets import QPushButton
import numpy as np

if TYPE_CHECKING :
    import napari


@magic_factory(
        call_button="Run CoTracker",
        online_mode={"widget_type": "CheckBox", "value":True}
)
def cotracker_widget(
    viewer: "napari.viewer.Viewer",
    image: "napari.types.ImageData", 
    points: "napari.layers.Points",
    start: int,
    end: int,
    online_mode: bool,
) -> "napari.types.PointsData" :
    
    @thread_worker
    def script_worker():
        from .core import run_cotracker_blocking
        queries = points.data[:, [0, 2, 1]]
        tracks = run_cotracker_blocking(image, queries, online_mode, start, end)
        tracks = tracks[:, [0, 2, 1]]
        return tracks

    worker = script_worker()
    
    # Visual cue that the process is running
    widget = cotracker_widget
    run_button = None
    for child in widget.native.findChildren(QPushButton) :
        if "Run" in child.text() :
            run_button = child
            break
    
    def set_running_state():
        if run_button:
            run_button.setText("Processing...")
            run_button.setStyleSheet("background-color: orange; color:white;")
            run_button.setEnabled(False)

    def set_idle_state():
        if run_button:
            run_button.setText("Run")
            run_button.setStyleSheet("")
            run_button.setEnabled(True)

    @worker.started.connect
    def on_started():
        set_running_state()
        show_info("CoTracker task started...")

    @worker.returned.connect
    def on_returned(tracks):
        set_idle_state()
        viewer.add_points(tracks, name="Tracks", face_color="red")
        show_info("CoTracker task complete")

    @worker.errored.connect
    def on_errored(e):
        set_idle_state()
        show_error(f"CoTracker task failed: {e}")

    worker.start()


@magic_factory
def polyline_test(viewer: "napari.viewer.Viewer", shape: "napari.layers.Shapes") -> None :
    print(shape.data)
    print(shape.shape_type)
    print(shape.edge_color)
    # viewer.add_shapes([np.array([[100, 100], [400, 400], [250, 596]]), np.array([[50, 50], [60, 60], [80, 110]])], shape_type=["path", "path"], edge_color=["red", "blue"])




