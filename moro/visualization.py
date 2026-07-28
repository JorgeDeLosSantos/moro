"""
Numython R&D, (c) 2026
Moro is a Python library for kinematic and dynamic modeling of serial robots.
This library has been designed, mainly, for academic and research purposes,
using SymPy as base library.

visualization module provides tools for rendering the kinematic diagram
of a serial robot using different backends (matplotlib, Three.js).
"""
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, eye

from moro.core import Robot
from moro.util import sympy_matrix_to_numpy_float
from dataclasses import dataclass

__all__ = [
    "RobotVisualizer",
    "MatplotlibBackend",
    "ThreeJSBackend",
    "evaluate_robot",
]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SceneData:
    """
    Evaluated data of a robot configuration.
    """
    joints: list[np.ndarray]
    frames: list["FrameData"]
    dimension: float


@dataclass(slots=True)
class FrameData:
    """
    Numerical homogeneous transformation of one robot frame.
    """

    T: np.ndarray

    @property
    def position(self):
        return self.T[:3, 3]

    @property
    def rotation(self):
        return self.T[:3, :3]

    @property
    def x(self):
        return self.rotation[:, 0]

    @property
    def y(self):
        return self.rotation[:, 1]

    @property
    def z(self):
        return self.rotation[:, 2]



# class SceneData:
#     """
#     Holds the evaluated (numerical) data of a robot at a given configuration.

#     Attributes
#     ----------
#     joints : list of list of float
#         Position [x, y, z] of each joint (including the base frame at index 0).
#         Length = dof + 1.
#     frames : list of dict
#         Frame data for each link frame. Each dict has keys:
#             'position' : [x, y, z]
#             'x'        : [x, y, z]   direction of the x-axis
#             'y'        : [x, y, z]   direction of the y-axis
#             'z'        : [x, y, z]   direction of the z-axis
#         Length = dof + 1 (frame 0 is the base frame).
#     dimension : float
#         A characteristic dimension of the robot (used for scaling the view).
#     """
#     def __init__(self, joints, frames, dimension):
#         self.joints = joints
#         self.frames = frames
#         self.dimension = dimension


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_robot(robot, num_vals):
    """
    Evaluate the robot at a given numerical configuration and return a
    :class:`SceneData` instance containing joint positions and frame
    orientations.

    Parameters
    ----------
    robot : Robot
        A serial-link robot instance.
    num_vals : dict
        Dictionary mapping symbolic joint variables (or any other symbols)
        to numerical values, e.g. ``{q1: 0.5, q2: 1.2}``.

    Returns
    -------
    SceneData
        Evaluated scene data ready for rendering.
    """
    joints = []
    frames = []

    # Base frame (index 0)
    joints.append([0.0, 0.0, 0.0])
    frames.append(FrameData(np.eye(4)))

    # Link frames
    for i in range(1, robot.dof + 1):
        Ti = robot.T_i0(i).subs(num_vals)
        Ti_num = sympy_matrix_to_numpy_float(Ti)

        pos = [float(Ti_num[j, 3]) for j in range(3)]
        joints.append(pos)

        frames.append(FrameData(Ti_num))

    # Characteristic dimension for view scaling
    all_coords = [c for joint in joints for c in joint]
    max_coord = max(abs(c) for c in all_coords) if all_coords else 1.0
    dimension = max(max_coord * 1.5, 1.0)

    return SceneData(joints=joints, frames=frames, dimension=dimension)


# ---------------------------------------------------------------------------
# Matplotlib backend
# ---------------------------------------------------------------------------

class MatplotlibBackend:
    """
    Backend that renders the robot kinematic diagram using Matplotlib 3D.
    """

    @staticmethod
    def render(
        scene_data,
        show_frames=True,
        show_links=True,
        show_joints=True,
        show_base=True,
        show_grid=True,
        ax=None,
        figsize=(10, 8),
        view_init=(30, 30),
        frame_scale=None,
        link_color="#778877",
        link_linewidth=3,
        joint_color="#ff1493",
        base_color="#ff00ff",
        joint_size=None,
        base_size=None,
    ):
        """
        Render the robot scene on a Matplotlib 3D axis.

        Parameters
        ----------
        scene_data : SceneData
            Evaluated robot data.
        show_frames : bool
            Draw coordinate frames at each joint.
        show_links : bool
            Draw links as lines connecting joints.
        show_joints : bool
            Draw spheres at joint positions.
        show_base : bool
            Highlight the base joint.
        show_grid : bool
            Show the 3D grid.
        ax : Axes3D or None
            Existing axis to draw on. If None, a new figure and axis are created.
        figsize : tuple
            Figure size in inches.
        view_init : tuple
            Initial view angles ``(elev, azim)``.
        frame_scale : float or None
            Length of the frame axes. If None, derived from scene dimension.
        link_color : str
            Color for the links.
        link_linewidth : float
            Line width for links.
        joint_color : str
            Color for joint spheres.
        base_color : str
            Color for the base joint sphere.
        joint_size : float or None
            Size of joint spheres. If None, derived from scene dimension.
        base_size : float or None
            Size of base sphere. If None, derived from scene dimension.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : mpl_toolkits.mplot3d.Axes3D
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        joints = scene_data.joints
        frames = scene_data.frames
        dim = scene_data.dimension

        # Derived sizes
        if frame_scale is None:
            frame_scale = dim / 5
        if joint_size is None:
            joint_size = max(dim * 0.025, 0.1)
        if base_size is None:
            base_size = max(dim * 0.035, 0.15)

        # --- Links ---
        if show_links:
            xs = [j[0] for j in joints]
            ys = [j[1] for j in joints]
            zs = [j[2] for j in joints]
            ax.plot(
                xs, ys, zs,
                "o-",
                color=link_color,
                linewidth=link_linewidth,
                markersize=0,  # we draw custom joint spheres
            )

        # --- Joints ---
        if show_joints:
            for idx, pos in enumerate(joints):
                if idx == 0 and not show_base:
                    continue
                color = base_color if idx == 0 else joint_color
                size = base_size if idx == 0 else joint_size
                ax.scatter(
                    pos[0], pos[1], pos[2],
                    color=color,
                    s=size * 100,  # scatter uses points^2
                    zorder=10,
                )

        # --- Frames ---
        if show_frames:
            for frame in frames:
                origin = frame.position
                for axis_name, color in zip(("x", "y", "z"), ("r", "g", "b")):
                    direction = getattr(frame, axis_name)
                    ax.quiver(
                        origin[0], origin[1], origin[2],
                        direction[0], direction[1], direction[2],
                        color=color,
                        length=frame_scale,
                        arrow_length_ratio=0.2,
                    )

        # --- Axes setup ---
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)
        ax.set_zlim(-dim, dim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=view_init[0], azim=view_init[1])

        if not show_grid:
            ax.grid(False)

        return fig, ax

    @staticmethod
    def animate(
        scene_data_list,
        interval=100,
        show_frames=True,
        show_links=True,
        show_joints=True,
        show_base=True,
        figsize=(10, 8),
        **kwargs,
    ):
        """
        Create a matplotlib animation from a list of scene data.

        Parameters
        ----------
        scene_data_list : list of SceneData
            Sequence of evaluated robot configurations.
        interval : int
            Delay between frames in milliseconds.
        figsize : tuple
            Figure size in inches.
        **kwargs
            Additional arguments forwarded to :meth:`render`.

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Determine scene dimension from the first frame
        dim = scene_data_list[0].dimension

        def init():
            ax.set_xlim(-dim, dim)
            ax.set_ylim(-dim, dim)
            ax.set_zlim(-dim, dim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            return (ax,)

        def update(frame_idx):
            ax.clear()
            MatplotlibBackend.render(
                scene_data_list[frame_idx],
                show_frames=show_frames,
                show_links=show_links,
                show_joints=show_joints,
                show_base=show_base,
                ax=ax,
                **kwargs,
            )
            return (ax,)

        anim = FuncAnimation(
            fig, update, frames=len(scene_data_list),
            init_func=init, interval=interval, blit=False,
        )
        plt.show()
        return anim


# ---------------------------------------------------------------------------
# Three.js backend
# ---------------------------------------------------------------------------

class ThreeJSBackend:
    """
    Backend that renders the robot kinematic diagram using Three.js.

    The output is an HTML string that can be displayed in a Jupyter notebook
    (via ``IPython.display.HTML``) or saved to a standalone file.

    Controls (CAD-like):
        - Scroll wheel: orbit (rotate scene)
        - Left button: pan
        - Right button: pan
        - Scroll wheel: zoom
    """

    @staticmethod
    def render(scene_data, width=800, height=600):
        """
        Generate an HTML page with an interactive Three.js 3D view of the robot.

        Parameters
        ----------
        scene_data : SceneData
            Evaluated robot data.
        width : int
            Canvas width in pixels.
        height : int
            Canvas height in pixels.

        Returns
        -------
        html : str
            A self-contained HTML document with embedded Three.js.
        """
        import json
        import uuid

        unique_id = str(uuid.uuid4())[:8]

        # Prepare data payload
        payload = {
            "joints": scene_data.joints,
            "frames": [
                {
                    "position": f.position.tolist(),
                    "x": f.x.tolist(),
                    "y": f.y.tolist(),
                    "z": f.z.tolist(),
                }
                for f in scene_data.frames
            ],
            "dimension": float(scene_data.dimension),
        }
        robot_json = json.dumps(payload)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #f5f5f5; }}
        #controls-{unique_id} {{
            position: absolute; top: 10px; left: 10px;
            background: rgba(255, 255, 255, 0.95);
            padding: 10px; border-radius: 5px;
            font-family: Arial, sans-serif; font-size: 12px;
            z-index: 100; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        #controls-{unique_id} button {{
            margin: 2px; padding: 5px 10px; cursor: pointer;
            border: none; color: white; border-radius: 3px;
        }}
        #status-{unique_id} {{ margin-top: 5px; padding: 5px; font-size: 10px; color: #666; }}
    </style>
</head>
<body>
    <div id="controls-{unique_id}">
        <button onclick="window.robot_{unique_id}.toggleRotation()"
                style="background:#4CAF50;">&#9654; Rotate</button>
        <button onclick="window.robot_{unique_id}.resetView()"
                style="background:#4CAF50;">&#x21bb; Reset</button>
        <div id="status-{unique_id}">Loading...</div>
    </div>
    <div id="container-{unique_id}"></div>

    <script>
    (function() {{
        if (typeof THREE !== 'undefined' && typeof THREE.OrbitControls !== 'undefined') {{
            initRobot_{unique_id}();
        }} else {{
            var scripts = [
                'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js',
                'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'
            ];
            var loaded = 0;
            scripts.forEach(function(src) {{
                var s = document.createElement('script');
                s.src = src;
                s.onload = function() {{
                    loaded++;
                    if (loaded === scripts.length) initRobot_{unique_id}();
                }};
                s.onerror = function() {{
                    document.getElementById('status-{unique_id}').innerHTML =
                        'Error loading Three.js';
                    document.getElementById('status-{unique_id}').style.color = 'red';
                }};
                document.head.appendChild(s);
            }});
        }}

        function initRobot_{unique_id}() {{
            var data = {robot_json};
            var container = document.getElementById('container-{unique_id}');

            var scene, camera, renderer, controls, robotGroup;
            var isAutoRotate = false;

            try {{
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf5f5f5);

                camera = new THREE.PerspectiveCamera(
                    50, {width} / {height}, 0.1, data.dimension * 10
                );
                var camDist = data.dimension * 2.5;
                camera.position.set(camDist, camDist * 0.8, camDist);
                camera.lookAt(0, 0, data.dimension / 3);

                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize({width}, {height});
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);

                // OrbitControls with CAD-like mapping
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.target.set(0, 0, data.dimension / 3);
                controls.enableDamping = true;
                controls.dampingFactor = 0.1;
                controls.mouseButtons = {{
                    LEFT: THREE.MOUSE.PAN,
                    MIDDLE: THREE.MOUSE.ROTATE,
                    RIGHT: THREE.MOUSE.PAN
                }};
                controls.touches = {{
                    ONE: THREE.TOUCH.PAN,
                    TWO: THREE.TOUCH.DOLLY_PAN
                }};
                controls.update();

                // Lights
                scene.add(new THREE.AmbientLight(0xffffff, 0.6));
                var dl1 = new THREE.DirectionalLight(0xffffff, 0.5);
                dl1.position.set(data.dimension, data.dimension, data.dimension);
                scene.add(dl1);
                var dl2 = new THREE.DirectionalLight(0xffffff, 0.3);
                dl2.position.set(-data.dimension, -data.dimension, -data.dimension);
                scene.add(dl2);

                // Grid
                var gridSize = data.dimension * 2.5;
                scene.add(new THREE.GridHelper(gridSize, 20, 0x888888, 0xcccccc));
                scene.add(new THREE.AxesHelper(data.dimension / 4));

                robotGroup = new THREE.Group();
                scene.add(robotGroup);

                drawRobot(data);

                animate();

                document.getElementById('status-{unique_id}').innerHTML =
                    'Ready &mdash; Scroll to orbit | Left/Right drag to pan | Wheel to zoom';
                document.getElementById('status-{unique_id}').style.color = 'green';
            }} catch (e) {{
                document.getElementById('status-{unique_id}').innerHTML =
                    'Error: ' + e.message;
                document.getElementById('status-{unique_id}').style.color = 'red';
            }}

            function drawRobot(d) {{
                var joints = d.joints, frames = d.frames;

                // --- Links ---
                var linkMat = new THREE.MeshPhongMaterial({{
                    color: 0x778877, shininess: 30, side: THREE.DoubleSide
                }});
                for (var i = 0; i < joints.length - 1; i++) {{
                    var start = new THREE.Vector3().fromArray(joints[i]);
                    var end   = new THREE.Vector3().fromArray(joints[i+1]);
                    var dir   = new THREE.Vector3().subVectors(end, start);
                    var len   = dir.length();
                    if (len < 1e-6) continue;
                    var radius = Math.max(d.dimension * 0.015, 0.3);
                    var cyl = new THREE.Mesh(
                        new THREE.CylinderGeometry(radius, radius, len, 8),
                        linkMat
                    );
                    var mid = start.clone().add(dir.clone().multiplyScalar(0.5));
                    cyl.position.copy(mid);
                    cyl.quaternion.setFromUnitVectors(
                        new THREE.Vector3(0, 1, 0),
                        dir.clone().normalize()
                    );
                    robotGroup.add(cyl);
                }}

                // --- Joints ---
                var jointMat = new THREE.MeshPhongMaterial({{
                    color: 0xff1493, shininess: 50
                }});
                var baseMat = new THREE.MeshPhongMaterial({{
                    color: 0xff00ff, shininess: 50
                }});
                joints.forEach(function(pos, idx) {{
                    var radius = idx === 0
                        ? Math.max(d.dimension * 0.03, 0.6)
                        : Math.max(d.dimension * 0.02, 0.4);
                    var mat = idx === 0 ? baseMat : jointMat;
                    var sphere = new THREE.Mesh(
                        new THREE.SphereGeometry(radius, 16, 16), mat
                    );
                    sphere.position.fromArray(pos);
                    robotGroup.add(sphere);
                }});

                // --- Frames (using AxesHelper) ---
                var axesLen = Math.max(d.dimension / 6, 1.0);
                frames.forEach(function(f) {{
                    var origin = new THREE.Vector3().fromArray(f.position);
                    var axes = new THREE.AxesHelper(axesLen);
                    axes.position.copy(origin);
                    robotGroup.add(axes);
                }});
            }}

            function animate() {{
                requestAnimationFrame(animate);
                if (isAutoRotate) {{
                    var target = controls.target;
                    var offset = camera.position.clone().sub(target);
                    var theta = 0.005;
                    var cos = Math.cos(theta);
                    var sin = Math.sin(theta);
                    var x = offset.x * cos - offset.z * sin;
                    var z = offset.x * sin + offset.z * cos;
                    camera.position.set(target.x + x, camera.position.y, target.z + z);
                    camera.lookAt(target);
                }}
                controls.update();
                renderer.render(scene, camera);
            }}

            window.robot_{unique_id} = {{
                toggleRotation: function() {{
                    isAutoRotate = !isAutoRotate;
                }},
                resetView: function() {{
                    var d = data.dimension;
                    camera.position.set(d * 2.5, d * 2.0, d * 2.5);
                    controls.target.set(0, 0, d / 3);
                    controls.update();
                }}
            }};
        }}
    }})();
    </script>
</body>
</html>"""
        return html

    @staticmethod
    def render_notebook(scene_data, width=800, height=600):
        """
        Render the robot scene as an interactive HTML view inside a Jupyter
        notebook.

        Parameters
        ----------
        scene_data : SceneData
            Evaluated robot data.
        width : int
            Canvas width in pixels.
        height : int
            Canvas height in pixels.

        Returns
        -------
        display : IPython.display.HTML
        """
        from IPython.display import HTML

        html = ThreeJSBackend.render(scene_data, width=width, height=height)
        return HTML(html)

    @staticmethod
    def animate(
        scene_data_list,
        width=800,
        height=600,
    ):
        """
        Create an interactive animation from a list of scene data.

        The animation uses a timer inside Three.js to cycle through the frames.
        A slider is provided for frame-by-frame navigation.

        Parameters
        ----------
        scene_data_list : list of SceneData
            Sequence of evaluated robot configurations.
        width : int
            Canvas width in pixels.
        height : int
            Canvas height in pixels.

        Returns
        -------
        display : IPython.display.HTML
        """
        import json
        import uuid

        unique_id = str(uuid.uuid4())[:8]

        payloads = []
        for sd in scene_data_list:
            payloads.append({
                "joints": sd.joints,
                "frames": [
                    {
                        "position": f.position.tolist(),
                        "x": f.x.tolist(),
                        "y": f.y.tolist(),
                        "z": f.z.tolist(),
                    }
                    for f in sd.frames
                ],
                "dimension": float(sd.dimension),
            })

        data_json = json.dumps(payloads)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #f5f5f5; }}
        #controls-{unique_id} {{
            position: absolute; top: 10px; left: 10px;
            background: rgba(255, 255, 255, 0.95);
            padding: 10px; border-radius: 5px;
            font-family: Arial, sans-serif; font-size: 12px;
            z-index: 100; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            min-width: 200px;
        }}
        #controls-{unique_id} button {{
            margin: 2px; padding: 5px 10px; cursor: pointer;
            border: none; color: white; border-radius: 3px;
        }}
        #controls-{unique_id} input[type=range] {{ width: 100%; }}
        #status-{unique_id} {{ margin-top: 5px; padding: 5px; font-size: 10px; color: #666; }}
    </style>
</head>
<body>
    <div id="controls-{unique_id}">
        <button onclick="window.robot_{unique_id}.toggleRotation()"
                style="background:#4CAF50;">&#9654; Rotate</button>
        <button onclick="window.robot_{unique_id}.togglePlay()"
                style="background:#2196F3;">&#9654; Play</button>
        <button onclick="window.robot_{unique_id}.resetView()"
                style="background:#4CAF50;">&#x21bb; Reset</button>
        <div>
            <label>Frame:
                <span id="frameLabel-{unique_id}">0</span> / {len(payloads) - 1}
            </label>
            <input type="range" id="slider-{unique_id}" min="0"
                   max="{len(payloads) - 1}" value="0" step="1"
                   oninput="window.robot_{unique_id}.goToFrame(parseInt(this.value))">
        </div>
        <div id="status-{unique_id}">Loading...</div>
    </div>
    <div id="container-{unique_id}"></div>

    <script>
    (function() {{
        if (typeof THREE !== 'undefined' && typeof THREE.OrbitControls !== 'undefined') {{
            initRobot_{unique_id}();
        }} else {{
            var scripts = [
                'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js',
                'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'
            ];
            var loaded = 0;
            scripts.forEach(function(src) {{
                var s = document.createElement('script');
                s.src = src;
                s.onload = function() {{
                    loaded++;
                    if (loaded === scripts.length) initRobot_{unique_id}();
                }};
                s.onerror = function() {{
                    document.getElementById('status-{unique_id}').innerHTML =
                        'Error loading Three.js';
                    document.getElementById('status-{unique_id}').style.color = 'red';
                }};
                document.head.appendChild(s);
            }});
        }}

        function initRobot_{unique_id}() {{
            var framesData = {data_json};
            var container = document.getElementById('container-{unique_id}');
            var slider = document.getElementById('slider-{unique_id}');
            var frameLabel = document.getElementById('frameLabel-{unique_id}');

            var scene, camera, renderer, controls, robotGroup;
            var currentFrame = 0;
            var isAutoRotate = false, isPlaying = false;
            var playInterval = null;

            var dim = framesData[0].dimension;

            try {{
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf5f5f5);

                camera = new THREE.PerspectiveCamera(
                    50, {width} / {height}, 0.1, dim * 10
                );
                var camDist = dim * 2.5;
                camera.position.set(camDist, camDist * 0.8, camDist);
                camera.lookAt(0, 0, dim / 3);

                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize({width}, {height});
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);

                // OrbitControls with CAD-like mapping
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.target.set(0, 0, dim / 3);
                controls.enableDamping = true;
                controls.dampingFactor = 0.1;
                controls.mouseButtons = {{
                    LEFT: THREE.MOUSE.PAN,
                    MIDDLE: THREE.MOUSE.ROTATE,
                    RIGHT: THREE.MOUSE.PAN
                }};
                controls.touches = {{
                    ONE: THREE.TOUCH.PAN,
                    TWO: THREE.TOUCH.DOLLY_PAN
                }};
                controls.update();

                // Lights
                scene.add(new THREE.AmbientLight(0xffffff, 0.6));
                var dl1 = new THREE.DirectionalLight(0xffffff, 0.5);
                dl1.position.set(dim, dim, dim);
                scene.add(dl1);
                var dl2 = new THREE.DirectionalLight(0xffffff, 0.3);
                dl2.position.set(-dim, -dim, -dim);
                scene.add(dl2);

                var gridSize = dim * 2.5;
                scene.add(new THREE.GridHelper(gridSize, 20, 0x888888, 0xcccccc));
                scene.add(new THREE.AxesHelper(dim / 4));

                robotGroup = new THREE.Group();
                scene.add(robotGroup);

                // Draw initial frame
                drawRobot(framesData[0]);

                animate();

                document.getElementById('status-{unique_id}').innerHTML =
                    'Ready &mdash; ' + framesData.length + ' frames';
                document.getElementById('status-{unique_id}').style.color = 'green';
            }} catch (e) {{
                document.getElementById('status-{unique_id}').innerHTML =
                    'Error: ' + e.message;
                document.getElementById('status-{unique_id}').style.color = 'red';
            }}

            function drawRobot(d) {{
                while (robotGroup.children.length > 0) {{
                    robotGroup.remove(robotGroup.children[0]);
                }}

                var joints = d.joints, frames = d.frames;

                // Links
                var linkMat = new THREE.MeshPhongMaterial({{
                    color: 0x778877, shininess: 30, side: THREE.DoubleSide
                }});
                for (var i = 0; i < joints.length - 1; i++) {{
                    var start = new THREE.Vector3().fromArray(joints[i]);
                    var end   = new THREE.Vector3().fromArray(joints[i+1]);
                    var dir   = new THREE.Vector3().subVectors(end, start);
                    var len   = dir.length();
                    if (len < 1e-6) continue;
                    var radius = Math.max(dim * 0.015, 0.3);
                    var cyl = new THREE.Mesh(
                        new THREE.CylinderGeometry(radius, radius, len, 8),
                        linkMat
                    );
                    var mid = start.clone().add(dir.clone().multiplyScalar(0.5));
                    cyl.position.copy(mid);
                    cyl.quaternion.setFromUnitVectors(
                        new THREE.Vector3(0, 1, 0),
                        dir.clone().normalize()
                    );
                    robotGroup.add(cyl);
                }}

                // Joints
                var jointMat = new THREE.MeshPhongMaterial({{
                    color: 0xff1493, shininess: 50
                }});
                var baseMat = new THREE.MeshPhongMaterial({{
                    color: 0xff00ff, shininess: 50
                }});
                joints.forEach(function(pos, idx) {{
                    var radius = idx === 0
                        ? Math.max(dim * 0.03, 0.6)
                        : Math.max(dim * 0.02, 0.4);
                    var mat = idx === 0 ? baseMat : jointMat;
                    var sphere = new THREE.Mesh(
                        new THREE.SphereGeometry(radius, 16, 16), mat
                    );
                    sphere.position.fromArray(pos);
                    robotGroup.add(sphere);
                }});

                // Frames (using AxesHelper)
                var axesLen = Math.max(dim / 6, 1.0);
                frames.forEach(function(f) {{
                    var origin = new THREE.Vector3().fromArray(f.position);
                    var axes = new THREE.AxesHelper(axesLen);
                    axes.position.copy(origin);
                    robotGroup.add(axes);
                }});
            }}

            function goToFrame(idx) {{
                currentFrame = idx;
                drawRobot(framesData[idx]);
                slider.value = idx;
                frameLabel.textContent = idx;
            }}

            function animate() {{
                requestAnimationFrame(animate);
                if (isAutoRotate) {{
                    var target = controls.target;
                    var offset = camera.position.clone().sub(target);
                    var theta = 0.005;
                    var cos = Math.cos(theta);
                    var sin = Math.sin(theta);
                    var x = offset.x * cos - offset.z * sin;
                    var z = offset.x * sin + offset.z * cos;
                    camera.position.set(target.x + x, camera.position.y, target.z + z);
                    camera.lookAt(target);
                }}
                controls.update();
                renderer.render(scene, camera);
            }}

            window.robot_{unique_id} = {{
                toggleRotation: function() {{
                    isAutoRotate = !isAutoRotate;
                }},
                togglePlay: function() {{
                    isPlaying = !isPlaying;
                    if (isPlaying) {{
                        playInterval = setInterval(function() {{
                            var next = (currentFrame + 1) % framesData.length;
                            goToFrame(next);
                        }}, 100);
                    }} else {{
                        clearInterval(playInterval);
                    }}
                }},
                goToFrame: goToFrame,
                resetView: function() {{
                    camera.position.set(dim * 2.5, dim * 2.0, dim * 2.5);
                    controls.target.set(0, 0, dim / 3);
                    controls.update();
                }}
            }};
        }}
    }})();
    </script>
</body>
</html>"""
        from IPython.display import HTML

        return HTML(html)


# ---------------------------------------------------------------------------
# Main visualizer (orchestrator)
# ---------------------------------------------------------------------------

class RobotVisualizer:
    """
    Orchestrator for rendering a :class:`Robot` in different visualization
    backends.

    Parameters
    ----------
    robot : Robot
        A serial-link robot instance.
    """

    def __init__(self, robot):
        if not isinstance(robot, Robot):
            raise TypeError(
                f"Expected a Robot instance, got {type(robot).__name__}."
            )
        self.robot = robot

    # ---- Single-configuration plot ----------------------------------------

    def plot(self, num_vals, backend="matplotlib", **kwargs):
        """
        Render the robot kinematic diagram at a given configuration.

        Parameters
        ----------
        num_vals : dict
            Dictionary mapping symbolic joint variables to numerical values,
            e.g. ``{q1: 0.5, q2: 1.2}``.
        backend : str
            Visualization backend. ``"matplotlib"`` (default) or ``"threejs"``.
        **kwargs
            Additional arguments forwarded to the backend's ``render`` method.

        Returns
        -------
        Depends on the backend:
            - ``"matplotlib"`` : ``(fig, ax)`` tuple.
            - ``"threejs"``    : HTML string.
        """
        scene_data = evaluate_robot(self.robot, num_vals)

        if backend == "matplotlib":
            return MatplotlibBackend.render(scene_data, **kwargs)
        elif backend == "threejs":
            return ThreeJSBackend.render(scene_data, **kwargs)
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Available backends: 'matplotlib', 'threejs'."
            )

    def plot_notebook(self, num_vals, width=800, height=600):
        """
        Convenience method to render the robot inside a Jupyter notebook
        using Three.js.

        Parameters
        ----------
        num_vals : dict
            Dictionary mapping symbolic joint variables to numerical values.
        width : int
            Canvas width in pixels.
        height : int
            Canvas height in pixels.

        Returns
        -------
        IPython.display.HTML
        """
        scene_data = evaluate_robot(self.robot, num_vals)
        return ThreeJSBackend.render_notebook(scene_data, width=width, height=height)

    # ---- Animation --------------------------------------------------------

    def animate(self, num_vals_list, backend="matplotlib", **kwargs):
        """
        Create an animation from a sequence of robot configurations.

        Parameters
        ----------
        num_vals_list : list of dict
            Each element is a ``dict`` mapping symbolic joint variables to
            numerical values for one frame of the animation.
        backend : str
            ``"matplotlib"`` (returns a FuncAnimation) or ``"threejs"``
            (returns an IPython HTML display).
        **kwargs
            Additional arguments forwarded to the backend's ``animate`` method.

        Returns
        -------
        Depends on the backend:
            - ``"matplotlib"`` : ``matplotlib.animation.FuncAnimation``.
            - ``"threejs"``    : ``IPython.display.HTML``.
        """
        scene_data_list = [
            evaluate_robot(self.robot, nv) for nv in num_vals_list
        ]

        if backend == "matplotlib":
            return MatplotlibBackend.animate(scene_data_list, **kwargs)
        elif backend == "threejs":
            return ThreeJSBackend.animate(scene_data_list, **kwargs)
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Available backends: 'matplotlib', 'threejs'."
            )

    def animate_notebook(self, num_vals_list, width=800, height=600):
        """
        Convenience method to create an interactive animation inside a Jupyter
        notebook using Three.js.

        Parameters
        ----------
        num_vals_list : list of dict
            Each element is a ``dict`` mapping symbolic joint variables to
            numerical values for one frame of the animation.
        width : int
            Canvas width in pixels.
        height : int
            Canvas height in pixels.

        Returns
        -------
        IPython.display.HTML
        """
        scene_data_list = [
            evaluate_robot(self.robot, nv) for nv in num_vals_list
        ]
        return ThreeJSBackend.animate(scene_data_list, width=width, height=height)