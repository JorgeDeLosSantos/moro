window.MoroThreeJS = (function() {
    'use strict';

    function toFiniteNumber(value, fallback) {
        var num = Number(value);
        if (!Number.isFinite(num)) {
            return fallback;
        }
        return num;
    }

    function loadScriptWithId(scriptId, src, onLoad, onError) {
        var existing = document.getElementById(scriptId);

        if (existing) {
            if (existing.getAttribute('data-loaded') === 'true') {
                onLoad();
                return;
            }

            existing.addEventListener('load', onLoad, { once: true });
            existing.addEventListener('error', function() {
                onError(new Error('Failed to load script: ' + src));
            }, { once: true });
            return;
        }

        var script = document.createElement('script');
        script.id = scriptId;
        script.src = src;
        script.async = true;
        script.onload = function() {
            script.setAttribute('data-loaded', 'true');
            onLoad();
        };
        script.onerror = function() {
            onError(new Error('Failed to load script: ' + src));
        };
        document.head.appendChild(script);
    }

    function ensureThreeReady(onReady, onError) {
        if (
            typeof THREE !== 'undefined' &&
            typeof THREE.OrbitControls !== 'undefined'
        ) {
            onReady();
            return;
        }

        if (!window.__moroThreeReadyPromise) {
            window.__moroThreeReadyPromise = new Promise(function(resolve, reject) {
                loadScriptWithId(
                    '__moro-three-r128__',
                    'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js',
                    function() {
                        loadScriptWithId(
                            '__moro-orbitcontrols-r128__',
                            'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js',
                            function() {
                                if (
                                    typeof THREE === 'undefined' ||
                                    typeof THREE.OrbitControls === 'undefined'
                                ) {
                                    reject(new Error('THREE or OrbitControls did not initialize correctly.'));
                                    return;
                                }

                                resolve();
                            },
                            reject
                        );
                    },
                    reject
                );
            });
        }

        window.__moroThreeReadyPromise.then(onReady).catch(onError);
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

    function resolveTrajectoryStyle(style) {
        return {
            showTrajectory: style.show_trajectory === true,
            trajectoryColor: style.trajectory_color || '#1565c0',
            trajectoryLinewidth: toFiniteNumber(
                style.trajectory_linewidth,
                2
            ),
            trajectoryMode: style.trajectory_mode || 'full'
        };
    }

    function createTrajectoryLine(trajectoryData, resolvedStyle) {
        if (!Array.isArray(trajectoryData) || trajectoryData.length === 0) {
            return null;
        }

        var points = trajectoryData
            .map(function(position) {
                if (!Array.isArray(position) || position.length < 3) {
                    return null;
                }

                var x = Number(position[0]);
                var y = Number(position[1]);
                var z = Number(position[2]);

                if (
                    !Number.isFinite(x) ||
                    !Number.isFinite(y) ||
                    !Number.isFinite(z)
                ) {
                    return null;
                }

                return new THREE.Vector3(x, y, z);
            })
            .filter(function(point) {
                return point !== null;
            });

        if (points.length === 0) {
            return null;
        }

        var trajectoryGeometry = new THREE.BufferGeometry().setFromPoints(points);
        var trajectoryMaterial = new THREE.LineBasicMaterial({
            color: resolvedStyle.trajectoryColor,
            linewidth: resolvedStyle.trajectoryLinewidth
        });

        var trajectoryLine = new THREE.Line(
            trajectoryGeometry,
            trajectoryMaterial
        );

        trajectoryLine.userData.trajectoryPointCount = points.length;
        return trajectoryLine;
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

    function assertRobotTopology(referenceData, data) {
        var expectedJointCount = Array.isArray(referenceData.joints)
            ? referenceData.joints.length
            : 0;
        var expectedFrameCount = Array.isArray(referenceData.frames)
            ? referenceData.frames.length
            : 0;
        var actualJointCount = Array.isArray(data.joints)
            ? data.joints.length
            : 0;
        var actualFrameCount = Array.isArray(data.frames)
            ? data.frames.length
            : 0;

        if (
            expectedJointCount !== actualJointCount ||
            expectedFrameCount !== actualFrameCount
        ) {
            throw new Error(
                'Inconsistent robot topology between animation frames.'
            );
        }
    }

    function createPersistentRobotObjects(options) {
        var robotGroup = options.robotGroup;
        var data = options.data;
        var style = options.style;
        var resolvedStyle = options.resolvedStyle;
        var normalizeFrameAxes = options.normalizeFrameAxes === true;

        var joints = data.joints || [];
        var frames = data.frames || [];

        var linkMeshes = [];
        var jointMeshes = [];
        var frameHelpers = [];

        if (style.show_links) {
            var linkMaterial = new THREE.MeshPhongMaterial({
                color: style.link_color,
                shininess: 30,
                side: THREE.DoubleSide
            });
            var linkGeometry = new THREE.CylinderGeometry(
                resolvedStyle.linkRadius,
                resolvedStyle.linkRadius,
                1,
                8
            );

            for (var i = 0; i < joints.length - 1; i++) {
                var linkMesh = new THREE.Mesh(linkGeometry, linkMaterial);
                robotGroup.add(linkMesh);
                linkMeshes.push(linkMesh);
            }
        }

        if (style.show_joints) {
            var jointMaterial = new THREE.MeshPhongMaterial({
                color: style.joint_color,
                shininess: 50
            });
            var baseMaterial = new THREE.MeshPhongMaterial({
                color: style.base_color,
                shininess: 50
            });
            var jointGeometry = new THREE.SphereGeometry(
                resolvedStyle.jointRadius,
                16,
                16
            );
            var baseGeometry = new THREE.SphereGeometry(
                resolvedStyle.baseRadius,
                16,
                16
            );

            for (var j = 0; j < joints.length; j++) {
                var isBase = j === 0;
                var sphere = new THREE.Mesh(
                    isBase ? baseGeometry : jointGeometry,
                    isBase ? baseMaterial : jointMaterial
                );

                robotGroup.add(sphere);
                jointMeshes.push(sphere);
            }
        }

        if (style.show_frames) {
            for (var k = 0; k < frames.length; k++) {
                var axes = new THREE.AxesHelper(resolvedStyle.frameScale);
                robotGroup.add(axes);
                frameHelpers.push(axes);
            }
        }

        return {
            style: style,
            normalizeFrameAxes: normalizeFrameAxes,
            referenceData: data,
            linkMeshes: linkMeshes,
            jointMeshes: jointMeshes,
            frameHelpers: frameHelpers,
            _tmp: {
                start: new THREE.Vector3(),
                end: new THREE.Vector3(),
                direction: new THREE.Vector3(),
                xAxis: new THREE.Vector3(),
                yAxis: new THREE.Vector3(),
                zAxis: new THREE.Vector3(),
                yUnit: new THREE.Vector3(0, 1, 0),
                rotationMatrix: new THREE.Matrix4(),
                quaternion: new THREE.Quaternion()
            }
        };
    }

    function updateLinkMesh(mesh, start, end, tmp) {
        tmp.start.fromArray(start);
        tmp.end.fromArray(end);
        tmp.direction.subVectors(tmp.end, tmp.start);

        var length = tmp.direction.length();

        if (!Number.isFinite(length) || length < 1e-6) {
            mesh.visible = false;
            return;
        }

        mesh.visible = true;
        mesh.position.copy(tmp.start).addScaledVector(tmp.direction, 0.5);

        tmp.direction.multiplyScalar(1 / length);
        mesh.quaternion.setFromUnitVectors(tmp.yUnit, tmp.direction);
        mesh.scale.set(1, length, 1);
    }

    function updatePersistentRobotObjects(robotObjects, data) {
        assertRobotTopology(robotObjects.referenceData, data);

        var style = robotObjects.style;
        var joints = data.joints || [];
        var frames = data.frames || [];
        var tmp = robotObjects._tmp;
        var i;

        if (style.show_links) {
            for (i = 0; i < robotObjects.linkMeshes.length; i++) {
                updateLinkMesh(
                    robotObjects.linkMeshes[i],
                    joints[i],
                    joints[i + 1],
                    tmp
                );
            }
        }

        if (style.show_joints) {
            for (i = 0; i < robotObjects.jointMeshes.length; i++) {
                var jointMesh = robotObjects.jointMeshes[i];
                jointMesh.position.fromArray(joints[i]);
                jointMesh.visible = i !== 0 || style.show_base;
            }
        }

        if (style.show_frames) {
            for (i = 0; i < robotObjects.frameHelpers.length; i++) {
                var helper = robotObjects.frameHelpers[i];
                var frame = frames[i];

                helper.position.fromArray(frame.position);

                tmp.xAxis.fromArray(frame.x);
                tmp.yAxis.fromArray(frame.y);
                tmp.zAxis.fromArray(frame.z);

                if (robotObjects.normalizeFrameAxes) {
                    tmp.xAxis.normalize();
                    tmp.yAxis.normalize();
                    tmp.zAxis.normalize();
                }

                tmp.rotationMatrix.set(
                    tmp.xAxis.x, tmp.yAxis.x, tmp.zAxis.x, 0,
                    tmp.xAxis.y, tmp.yAxis.y, tmp.zAxis.y, 0,
                    tmp.xAxis.z, tmp.yAxis.z, tmp.zAxis.z, 0,
                    0,           0,           0,           1
                );

                tmp.quaternion.setFromRotationMatrix(tmp.rotationMatrix);
                helper.quaternion.copy(tmp.quaternion);
                helper.visible = true;
            }
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
        assertRobotTopology: assertRobotTopology,
        clearGroup: clearGroup,
        createCameras: createCameras,
        createCylinderBetweenPoints: createCylinderBetweenPoints,
        createGrid: createGrid,
        createLights: createLights,
        createPersistentRobotObjects: createPersistentRobotObjects,
        createRobotMeshes: createRobotMeshes,
        ensureThreeReady: ensureThreeReady,
        resolveStyle: resolveStyle,
        resolveTrajectoryStyle: resolveTrajectoryStyle,
        setPresetView: setPresetView,
        setupOrbitControls: setupOrbitControls,
        switchCameraType: switchCameraType,
        syncOrthographicFromPerspective: syncOrthographicFromPerspective,
        syncPerspectiveFromOrthographic: syncPerspectiveFromOrthographic,
        createTrajectoryLine: createTrajectoryLine,
        updatePersistentRobotObjects: updatePersistentRobotObjects
    };
})();
