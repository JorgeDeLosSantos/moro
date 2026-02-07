"""

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import *
from sympy.matrices import Matrix,eye
from moro.transformations import *
from moro.util import *

__all__ = ["plot_euler", "draw_uv", "draw_uvw"]

def plot_euler(phi,theta,psi,seq="zxz"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if seq in ("zxz","ZXZ","313",313):
        R1 = rotz(phi)
        R2 = R1*rotx(theta)
        R3 = R2*rotz(psi)
    elif seq in ("zyz","ZYZ","323",323):
        R1 = rotz(phi)
        R2 = R1*roty(theta)
        R3 = R2*rotz(psi)
    else:
        R1 = R2 = R3 = eye(4)
    draw_uvw(eye(4), ax, sz=6, alpha=0.4)
    draw_uvw(R1, ax, sz=6, alpha=0.6)
    draw_uvw(R2, ax, sz=6, alpha=0.8)
    draw_uvw(R3, ax, sz=6, alpha=1.0)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    # ax.set_aspect("equal")
    ax.axis('off')

    
def draw_uvw(H,ax,color=("r","g","b"),sz=1,alpha=1.0):
    u = H[:3,0]
    v = H[:3,1]
    w = H[:3,2]
    if ishtm(H):
        o = H[:3,3]
    else:
        o = Matrix([0,0,0])
    L = sz/5
    if isinstance(color,str):
        colorl = (color,color,color)
    else:
        colorl = color
    ax.quiver(o[0],o[1],o[2],u[0],u[1],u[2], color=colorl[0], 
              length=L, arrow_length_ratio=0.2, alpha=alpha)
    ax.quiver(o[0],o[1],o[2],v[0],v[1],v[2], color=colorl[1], 
              length=L, arrow_length_ratio=0.2, alpha=alpha)
    ax.quiver(o[0],o[1],o[2],w[0],w[1],w[2], color=colorl[2], 
              length=L, arrow_length_ratio=0.2, alpha=alpha)


def draw_xyz(*args, **kwargs):
    return draw_uvw(*args, **kwargs)

def draw_frame(*args, **kwargs):
    return draw_uvw(*args, **kwargs)

def draw_uv(H, ax, name="S0", color=("r","g"), sz=1):
    tpos = H*Matrix([1,1,0,1])
    H = sympy2float(H)
    u = H[:3,0]             
    v = H[:3,1]
    w = H[:3,2]
    if ishtm(H):
        o = H[:3,3]
    else:
        o = Matrix([0,0,0])
    L = sz/5
    if isinstance(color,str):
        colorl = (color,color)
    else:
        colorl = color
    # ~ print(o, u)
    ax.arrow(o[0],o[1],u[0],u[1], color=colorl[0])
    ax.arrow(o[0],o[1],v[0],v[1], color=colorl[1])
    ax.text(tpos[0], tpos[1], "{"+name+"}", fontsize=8)
    ax.set_aspect("auto")


def plot_diagram_threejs(self, num_vals, width=800, height=600):
    """
    Dibuja el diagrama cinemático del robot usando Three.js en Jupyter.
    
    Parameters
    ----------
    num_vals : dict
        Diccionario con valores numéricos para las variables simbólicas
    width : int
        Ancho del canvas en pixels
    height : int
        Alto del canvas en pixels
    """
    from IPython.display import HTML
    import json
    import uuid
    
    # Generar ID único para evitar conflictos
    unique_id = str(uuid.uuid4())[:8]
    
    # Extraer posiciones de joints
    joints = []
    frames = []
    
    # Frame base
    joints.append([0.0, 0.0, 0.0])
    frames.append({
        'position': [0.0, 0.0, 0.0],
        'x': [1.0, 0.0, 0.0],
        'y': [0.0, 1.0, 0.0],
        'z': [0.0, 0.0, 1.0]
    })
    
    # Para cada joint del robot
    for i in range(self.dof):
        Ti = self.T_i0(i + 1).subs(num_vals)
        
        # Posición del joint
        pos = [float(Ti[j, 3]) for j in range(3)]
        joints.append(pos)
        
        # Orientación del frame (ejes x, y, z)
        frames.append({
            'position': pos,
            'x': [float(Ti[j, 0]) for j in range(3)],
            'y': [float(Ti[j, 1]) for j in range(3)],
            'z': [float(Ti[j, 2]) for j in range(3)]
        })
    
    # Calcular dimensión para escalar la vista
    all_coords = [coord for joint in joints for coord in joint]
    max_coord = max(abs(c) for c in all_coords) if all_coords else 100
    dim = max(max_coord * 1.5, 50)
    
    robot_data = {
        'joints': joints,
        'frames': frames,
        'dimension': float(dim)
    }
    
    # Convertir a JSON
    robot_json = json.dumps(robot_data)
    
    # Template HTML con Three.js usando IDs únicos
    html_template = f"""
    <div id="container-{unique_id}" style="width: {width}px; height: {height}px; border: 1px solid #ccc; position: relative;">
        <div id="controls-{unique_id}" style="position: absolute; top: 10px; left: 10px; background: rgba(255, 255, 255, 0.95); padding: 10px; border-radius: 5px; font-family: Arial, sans-serif; font-size: 12px; z-index: 100; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
            <button onclick="window.robot_{unique_id}.toggleRotation()" style="margin: 2px; padding: 5px 10px; cursor: pointer; border: none; background: #4CAF50; color: white; border-radius: 3px;">⏯ Rotar</button>
            <button onclick="window.robot_{unique_id}.resetView()" style="margin: 2px; padding: 5px 10px; cursor: pointer; border: none; background: #4CAF50; color: white; border-radius: 3px;">🔄 Reset</button>
            <div id="status-{unique_id}" style="margin-top: 5px; padding: 5px; font-size: 10px; color: #666;">Cargando...</div>
        </div>
    </div>
    
    <script>
    (function() {{
        // Verificar si THREE ya está cargado
        if (typeof THREE !== 'undefined') {{
            initRobot_{unique_id}();
        }} else {{
            // Cargar Three.js
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
            script.onload = function() {{
                initRobot_{unique_id}();
            }};
            script.onerror = function() {{
                document.getElementById('status-{unique_id}').innerHTML = 'Error cargando Three.js ✗';
                document.getElementById('status-{unique_id}').style.color = 'red';
            }};
            document.head.appendChild(script);
        }}
        
        function initRobot_{unique_id}() {{
            const robotData = {robot_json};
            const container = document.getElementById('container-{unique_id}');
            
            // Variables locales para este robot
            let scene, camera, renderer, robotGroup;
            let isRotating = false;
            let isDragging = false;
            let previousMousePosition = {{ x: 0, y: 0 }};
            
            try {{
                // Escena
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf5f5f5);
                
                // Cámara
                camera = new THREE.PerspectiveCamera(
                    50,
                    {width} / {height},
                    0.1,
                    robotData.dimension * 10
                );
                const camDist = robotData.dimension * 2;
                camera.position.set(camDist, camDist, camDist);
                camera.lookAt(0, 0, robotData.dimension / 2);
                
                // Renderer
                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize({width}, {height});
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);
                
                // Luces
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);
                
                const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
                dirLight1.position.set(robotData.dimension, robotData.dimension, robotData.dimension);
                scene.add(dirLight1);
                
                const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
                dirLight2.position.set(-robotData.dimension, -robotData.dimension, -robotData.dimension);
                scene.add(dirLight2);
                
                // Grid
                const gridSize = robotData.dimension * 2;
                const gridHelper = new THREE.GridHelper(gridSize, 20, 0x888888, 0xcccccc);
                scene.add(gridHelper);
                
                // Ejes principales
                const axesHelper = new THREE.AxesHelper(robotData.dimension / 4);
                scene.add(axesHelper);
                
                // Grupo del robot
                robotGroup = new THREE.Group();
                scene.add(robotGroup);
                
                // Dibujar robot
                drawRobot();
                
                // Event listeners
                renderer.domElement.addEventListener('mousedown', onMouseDown);
                renderer.domElement.addEventListener('mousemove', onMouseMove);
                renderer.domElement.addEventListener('mouseup', onMouseUp);
                renderer.domElement.addEventListener('mouseleave', onMouseUp);
                renderer.domElement.addEventListener('wheel', onWheel, {{ passive: false }});
                
                // Animación
                function animate() {{
                    requestAnimationFrame(animate);
                    
                    if (isRotating && robotGroup) {{
                        robotGroup.rotation.y += 0.005;
                    }}
                    
                    renderer.render(scene, camera);
                }}
                animate();
                
                document.getElementById('status-{unique_id}').innerHTML = 
                    'Listo - Arrastra para rotar | Scroll para zoom';
                document.getElementById('status-{unique_id}').style.color = 'green';
                
            }} catch(error) {{
                console.error('Error:', error);
                document.getElementById('status-{unique_id}').innerHTML = 
                    'Error: ' + error.message;
                document.getElementById('status-{unique_id}').style.color = 'red';
            }}
            
            function drawRobot() {{
                const {{ joints, frames }} = robotData;
                
                // Material para links
                const linkMaterial = new THREE.MeshPhongMaterial({{
                    color: 0x778877,
                    shininess: 30,
                    side: THREE.DoubleSide
                }});
                
                // Dibujar links
                for (let i = 0; i < joints.length - 1; i++) {{
                    const start = new THREE.Vector3(...joints[i]);
                    const end = new THREE.Vector3(...joints[i + 1]);
                    
                    const direction = new THREE.Vector3().subVectors(end, start);
                    const length = direction.length();
                    
                    if (length > 0.001) {{
                        const radius = Math.max(robotData.dimension * 0.015, 1);
                        const geometry = new THREE.CylinderGeometry(radius, radius, length, 8);
                        const link = new THREE.Mesh(geometry, linkMaterial);
                        
                        const midpoint = start.clone().add(direction.clone().multiplyScalar(0.5));
                        link.position.copy(midpoint);
                        
                        const axis = new THREE.Vector3(0, 1, 0);
                        link.quaternion.setFromUnitVectors(axis, direction.clone().normalize());
                        
                        robotGroup.add(link);
                    }}
                }}
                
                // Dibujar joints
                joints.forEach((joint, index) => {{
                    const radius = index === 0 ? 
                        Math.max(robotData.dimension * 0.03, 2) : 
                        Math.max(robotData.dimension * 0.025, 1.5);
                    const geometry = new THREE.SphereGeometry(radius, 16, 16);
                    const material = new THREE.MeshPhongMaterial({{
                        color: index === 0 ? 0xff00ff : 0xff1493,
                        shininess: 50
                    }});
                    const sphere = new THREE.Mesh(geometry, material);
                    sphere.position.set(...joint);
                    robotGroup.add(sphere);
                }});
                
                // Dibujar sistemas de coordenadas
                const arrowLength = Math.max(robotData.dimension / 5, 10);
                const arrowHeadLength = arrowLength * 0.2;
                const arrowHeadWidth = arrowLength * 0.15;
                
                frames.forEach((frame) => {{
                    const origin = new THREE.Vector3(...frame.position);
                    
                    // Eje X (rojo)
                    const xDir = new THREE.Vector3(...frame.x).normalize();
                    const xArrow = new THREE.ArrowHelper(
                        xDir, origin, arrowLength, 0xff0000, 
                        arrowHeadLength, arrowHeadWidth
                    );
                    robotGroup.add(xArrow);
                    
                    // Eje Y (verde)
                    const yDir = new THREE.Vector3(...frame.y).normalize();
                    const yArrow = new THREE.ArrowHelper(
                        yDir, origin, arrowLength, 0x00ff00,
                        arrowHeadLength, arrowHeadWidth
                    );
                    robotGroup.add(yArrow);
                    
                    // Eje Z (azul)
                    const zDir = new THREE.Vector3(...frame.z).normalize();
                    const zArrow = new THREE.ArrowHelper(
                        zDir, origin, arrowLength, 0x0000ff,
                        arrowHeadLength, arrowHeadWidth
                    );
                    robotGroup.add(zArrow);
                }});
            }}
            
            function onMouseDown(event) {{
                isDragging = true;
                previousMousePosition = {{ x: event.clientX, y: event.clientY }};
            }}
            
            function onMouseMove(event) {{
                if (!isDragging || !robotGroup) return;
                
                const deltaX = event.clientX - previousMousePosition.x;
                const deltaY = event.clientY - previousMousePosition.y;
                
                robotGroup.rotation.y += deltaX * 0.01;
                robotGroup.rotation.x += deltaY * 0.01;
                
                previousMousePosition = {{ x: event.clientX, y: event.clientY }};
            }}
            
            function onMouseUp() {{
                isDragging = false;
            }}
            
            function onWheel(event) {{
                event.preventDefault();
                if (camera) {{
                    const delta = event.deltaY * 0.001;
                    const scale = 1 + delta;
                    camera.position.multiplyScalar(scale);
                }}
            }}
            
            // Exponer funciones públicas
            window.robot_{unique_id} = {{
                toggleRotation: function() {{
                    isRotating = !isRotating;
                }},
                resetView: function() {{
                    const camDist = robotData.dimension * 2;
                    camera.position.set(camDist, camDist, camDist);
                    camera.lookAt(0, 0, robotData.dimension / 2);
                    robotGroup.rotation.set(0, 0, 0);
                }}
            }};
        }}
    }})();
    </script>
    """
    
    return HTML(html_template)



if __name__=="__main__":
    plot_euler(pi/3, pi/3, 0.5)
    plt.show()
    # ~ fig = plt.figure()
    # ~ ax = fig.add_subplot(111)
    # ~ H1 = eye(4)*htmrot(pi/3)
    # ~ H2 = H1*htmtra([10,5,0])
    # ~ H3 = H2*htmtra([-4,5,0])*htmrot(pi/4)
    # ~ draw_uv(H1, ax, "A", "b")
    # ~ draw_uv(H2, ax, "B")
    # ~ draw_uv(H3, ax, "C")
    # ~ plt.grid(ls="--")
    # ~ plt.axis([-20,20,-20,20])
    # ~ plt.show()
