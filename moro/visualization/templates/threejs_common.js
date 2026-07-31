window.MoroThreeJS = (function() {
    'use strict';

    function toFiniteNumber(value, fallback) {
        var num = Number(value);
        if (!Number.isFinite(num)) {
            return fallback;
        }
        return num;
    }

    function resolveStyle(style, sceneScale) {
        return {
            frameScale: style.frame_scale !== null
                ? Number(style.frame_scale)
                : Math.max(sceneScale / 6, 1.0),
            jointRadius: style.joint_size !== null
                ? Number(style.joint_size)
                : Math.max(sceneScale * 0.02, 0.4),
            baseRadius: style.base_size !== null
                ? Number(style.base_size)
                : Math.max(sceneScale * 0.03, 0.6),
            linkRadius: Math.max(sceneScale * 0.015, 0.3)
                * (Number(style.link_linewidth) / 3)
        };
    }

    function createCameras(options) {
        var sceneScale = Math.max(toFiniteNumber(options.sceneScale, 0), 1);
        var aspect = toFiniteNumber(options.aspect, 1);

        if (aspect <= 0) {
            aspect = 1;
        }

        var nearPlane = options.nearPlane;
        if (nearPlane === undefined) {
            nearPlane = Math.max(sceneScale * 0.001, 0.01);
        }

        var farPlane = options.farPlane;
        if (farPlane === undefined) {
            farPlane = Math.max(sceneScale * 50, 1000);
        }

        var perspectiveCamera = new THREE.PerspectiveCamera(
            50,
            aspect,
            nearPlane,
            farPlane
        );

        var orthoHeight = sceneScale * 3.0;
        var orthoWidth = orthoHeight * aspect;

        var orthographicNear = options.orthographicNear;
        if (orthographicNear === undefined) {
            orthographicNear = nearPlane;
        }

        var orthographicFar = options.orthographicFar;
        if (orthographicFar === undefined) {
            orthographicFar = farPlane;
        }

        var orthographicCamera = new THREE.OrthographicCamera(
            -orthoWidth / 2,
            orthoWidth / 2,
            orthoHeight / 2,
            -orthoHeight / 2,
            orthographicNear,
            orthographicFar
        );

        perspectiveCamera.up.set(0, 0, 1);
        orthographicCamera.up.set(0, 0, 1);

        return {
            perspectiveCamera: perspectiveCamera,
            orthographicCamera: orthographicCamera,
            nearPlane: nearPlane,
            farPlane: farPlane,
            aspect: aspect,
            sceneScale: sceneScale
        };
    }

    function setupOrbitControls(options) {
        var controls = new THREE.OrbitControls(
            options.camera,
            options.domElement
        );

        if (options.target) {
            controls.target.copy(options.target);
        }

        controls.enableDamping = true;
        controls.dampingFactor = 0.1;
        controls.mouseButtons = {
            LEFT: THREE.MOUSE.PAN,
            MIDDLE: THREE.MOUSE.ROTATE,
            RIGHT: THREE.MOUSE.PAN
        };
        controls.touches = {
            ONE: THREE.TOUCH.PAN,
            TWO: THREE.TOUCH.DOLLY_PAN
        };

        if (typeof options.onStart === 'function') {
            controls.addEventListener('start', options.onStart);
        }

        return controls;
    }

    function setPresetView(options) {
        var camera = options.camera;
        var target = options.target;
        var distance = options.distance;
        var viewName = options.viewName;

        if (!camera || !target || !Number.isFinite(distance)) {
            return false;
        }

        var topUp = options.topUp || [0, 1, 0];
        var position = new THREE.Vector3();

        switch (viewName) {
            case 'front':
                camera.up.set(0, 0, 1);
                position.set(
                    target.x,
                    target.y - distance,
                    target.z
                );
                break;
            case 'top':
                camera.up.set(topUp[0], topUp[1], topUp[2]);
                position.set(
                    target.x,
                    target.y,
                    target.z + distance
                );
                break;
            case 'isometric':
                camera.up.set(0, 0, 1);
                position.set(
                    target.x + distance,
                    target.y + distance * 0.8,
                    target.z + distance
                );
                break;
            default:
                return false;
        }

        camera.position.copy(position);
        camera.lookAt(target);
        camera.updateProjectionMatrix();
        return true;
    }

    function syncOrthographicFromPerspective(options) {
        var perspectiveCamera = options.perspectiveCamera;
        var orthographicCamera = options.orthographicCamera;
        var target = options.target;
        var aspect = options.aspect;

        var distance = perspectiveCamera.position.distanceTo(target);
        var fov = THREE.MathUtils.degToRad(perspectiveCamera.fov);
        var visibleHeight = 2 * distance * Math.tan(fov / 2);
        var visibleWidth = visibleHeight * aspect;

        orthographicCamera.left = -visibleWidth / 2;
        orthographicCamera.right = visibleWidth / 2;
        orthographicCamera.top = visibleHeight / 2;
        orthographicCamera.bottom = -visibleHeight / 2;
        orthographicCamera.zoom = 1;
        orthographicCamera.updateProjectionMatrix();
    }

    function syncPerspectiveFromOrthographic(options) {
        var perspectiveCamera = options.perspectiveCamera;
        var orthographicCamera = options.orthographicCamera;
        var target = options.target;

        var visibleHeight = (
            orthographicCamera.top - orthographicCamera.bottom
        ) / orthographicCamera.zoom;

        var fov = THREE.MathUtils.degToRad(perspectiveCamera.fov);
        var distance = visibleHeight / (2 * Math.tan(fov / 2));
        var direction = orthographicCamera.position
            .clone()
            .sub(target)
            .normalize();

        perspectiveCamera.position.copy(
            target.clone().add(direction.multiplyScalar(distance))
        );
    }

    function switchCameraType(options) {
        var type = options.type;
        var activeCameraType = options.activeCameraType;
        var camera = options.camera;
        var perspectiveCamera = options.perspectiveCamera;
        var orthographicCamera = options.orthographicCamera;
        var controls = options.controls;

        if (type !== 'perspective' && type !== 'orthographic') {
            return {
                changed: false,
                camera: camera,
                activeCameraType: activeCameraType
            };
        }

        if (type === activeCameraType) {
            return {
                changed: false,
                camera: camera,
                activeCameraType: activeCameraType
            };
        }

        var previousCamera = camera;
        var nextCamera;
        var synchronizeProjection = options.synchronizeProjection !== false;

        if (type === 'orthographic') {
            if (synchronizeProjection) {
                syncOrthographicFromPerspective({
                    perspectiveCamera: perspectiveCamera,
                    orthographicCamera: orthographicCamera,
                    target: controls.target,
                    aspect: options.aspect
                });
            }

            nextCamera = orthographicCamera;
            nextCamera.position.copy(previousCamera.position);
        } else {
            nextCamera = perspectiveCamera;

            if (synchronizeProjection) {
                syncPerspectiveFromOrthographic({
                    perspectiveCamera: perspectiveCamera,
                    orthographicCamera: orthographicCamera,
                    target: controls.target
                });
            } else {
                nextCamera.position.copy(previousCamera.position);
            }
        }

        nextCamera.quaternion.copy(previousCamera.quaternion);
        nextCamera.up.copy(previousCamera.up);
        nextCamera.updateProjectionMatrix();

        controls.object = nextCamera;
        controls.update();

        return {
            changed: true,
            camera: nextCamera,
            activeCameraType: type
        };
    }

    function createLights(scene, sceneScale) {
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));

        var dl1 = new THREE.DirectionalLight(0xffffff, 0.5);
        dl1.position.set(sceneScale, sceneScale, sceneScale);
        scene.add(dl1);

        var dl2 = new THREE.DirectionalLight(0xffffff, 0.3);
        dl2.position.set(-sceneScale, -sceneScale, -sceneScale);
        scene.add(dl2);

        return {
            ambient: scene.children[scene.children.length - 3],
            directionalPrimary: dl1,
            directionalSecondary: dl2
        };
    }

    function createGrid(scene, options) {
        var grid = new THREE.GridHelper(
            options.gridSize,
            options.gridDivisions || 20,
            options.primaryColor || 0x888888,
            options.secondaryColor || 0xcccccc
        );

        if (options.rotateToXY !== false) {
            grid.rotation.x = Math.PI / 2;
        }

        if (options.position) {
            grid.position.set(
                options.position[0],
                options.position[1],
                options.position[2]
            );
        }

        scene.add(grid);
        return grid;
    }

    function createCylinderBetweenPoints(
        start,
        end,
        radius,
        radialSegments,
        material
    ) {
        var direction = new THREE.Vector3().subVectors(end, start);
        var length = direction.length();

        if (length < 1e-6) {
            return null;
        }

        var cylinder = new THREE.Mesh(
            new THREE.CylinderGeometry(
                radius,
                radius,
                length,
                radialSegments || 8
            ),
            material
        );

        var midpoint = start.clone().add(direction.clone().multiplyScalar(0.5));
        cylinder.position.copy(midpoint);
        cylinder.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            direction.clone().normalize()
        );

        return cylinder;
    }

    function createRobotMeshes(options) {
        var robotGroup = options.robotGroup;
        var data = options.data;
        var style = options.style;
        var resolvedStyle = options.resolvedStyle;
        var normalizeFrameAxes = options.normalizeFrameAxes === true;

        var joints = data.joints;
        var frames = data.frames;

        if (style.show_links) {
            var linkMat = new THREE.MeshPhongMaterial({
                color: style.link_color,
                shininess: 30,
                side: THREE.DoubleSide
            });

            for (var i = 0; i < joints.length - 1; i++) {
                var start = new THREE.Vector3().fromArray(joints[i]);
                var end = new THREE.Vector3().fromArray(joints[i + 1]);
                var link = createCylinderBetweenPoints(
                    start,
                    end,
                    resolvedStyle.linkRadius,
                    8,
                    linkMat
                );

                if (link !== null) {
                    robotGroup.add(link);
                }
            }
        }

        if (style.show_joints) {
            var jointMat = new THREE.MeshPhongMaterial({
                color: style.joint_color,
                shininess: 50
            });
            var baseMat = new THREE.MeshPhongMaterial({
                color: style.base_color,
                shininess: 50
            });

            joints.forEach(function(position, index) {
                if (index === 0 && !style.show_base) {
                    return;
                }

                var radius = index === 0
                    ? resolvedStyle.baseRadius
                    : resolvedStyle.jointRadius;

                var material = index === 0 ? baseMat : jointMat;
                var sphere = new THREE.Mesh(
                    new THREE.SphereGeometry(radius, 16, 16),
                    material
                );

                sphere.position.fromArray(position);
                robotGroup.add(sphere);
            });
        }

        if (style.show_frames) {
            frames.forEach(function(frame) {
                var xAxis = new THREE.Vector3().fromArray(frame.x);
                var yAxis = new THREE.Vector3().fromArray(frame.y);
                var zAxis = new THREE.Vector3().fromArray(frame.z);

                if (normalizeFrameAxes) {
                    xAxis.normalize();
                    yAxis.normalize();
                    zAxis.normalize();
                }

                var rotationMatrix = new THREE.Matrix4();
                rotationMatrix.set(
                    xAxis.x, yAxis.x, zAxis.x, 0,
                    xAxis.y, yAxis.y, zAxis.y, 0,
                    xAxis.z, yAxis.z, zAxis.z, 0,
                    0,       0,       0,       1
                );

                var axes = new THREE.AxesHelper(resolvedStyle.frameScale);
                axes.position.fromArray(frame.position);
                axes.setRotationFromMatrix(rotationMatrix);
                robotGroup.add(axes);
            });
        }
    }

    function disposeObject(object3d) {
        if (object3d.geometry) {
            object3d.geometry.dispose();
        }

        if (!object3d.material) {
            return;
        }

        if (Array.isArray(object3d.material)) {
            object3d.material.forEach(function(material) {
                material.dispose();
            });
            return;
        }

        object3d.material.dispose();
    }

    function clearGroup(group) {
        while (group.children.length > 0) {
            var child = group.children[0];
            group.remove(child);
            disposeObject(child);
        }
    }

    return {
        clearGroup: clearGroup,
        createCameras: createCameras,
        createCylinderBetweenPoints: createCylinderBetweenPoints,
        createGrid: createGrid,
        createLights: createLights,
        createRobotMeshes: createRobotMeshes,
        resolveStyle: resolveStyle,
        setPresetView: setPresetView,
        setupOrbitControls: setupOrbitControls,
        switchCameraType: switchCameraType,
        syncOrthographicFromPerspective: syncOrthographicFromPerspective,
        syncPerspectiveFromOrthographic: syncPerspectiveFromOrthographic
    };
})();
