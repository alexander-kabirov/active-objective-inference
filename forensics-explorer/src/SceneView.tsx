import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { TrialRecord } from "./types";

type Props = {
  trial?: TrialRecord;
  time: number;
  viewMode: "2d" | "3d";
};

const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value));

function movePath(actions: string[] = []): Array<[number, number]> {
  const points: Array<[number, number]> = [[-4, 0]];
  const pattern = /move_to\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)/i;
  for (const action of actions) {
    const match = action.match(pattern);
    if (match) points.push([Number(match[1]), Number(match[2])]);
  }
  return points;
}

function pointOnPath(points: Array<[number, number]>, progress: number): [number, number] {
  if (points.length < 2) return points[0] ?? [-4, 0];
  const lengths = points.slice(1).map((point, index) =>
    Math.hypot(point[0] - points[index][0], point[1] - points[index][1]),
  );
  const total = lengths.reduce((sum, length) => sum + length, 0);
  let remaining = clamp(progress) * total;
  for (let index = 0; index < lengths.length; index += 1) {
    if (remaining <= lengths[index]) {
      const local = lengths[index] ? remaining / lengths[index] : 0;
      return [
        THREE.MathUtils.lerp(points[index][0], points[index + 1][0], local),
        THREE.MathUtils.lerp(points[index][1], points[index + 1][1], local),
      ];
    }
    remaining -= lengths[index];
  }
  return points[points.length - 1];
}

export function SceneView({ trial, time, viewMode }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    block: THREE.Mesh;
    ball: THREE.Mesh;
    robot: THREE.Group;
    renderer: THREE.WebGLRenderer;
    perspective: THREE.PerspectiveCamera;
    orthographic: THREE.OrthographicCamera;
    controls: OrbitControls;
    path: THREE.Line;
    frame: number;
  } | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d10);
    scene.fog = new THREE.Fog(0x0b0d10, 14, 24);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    host.appendChild(renderer.domElement);

    const perspective = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    perspective.position.set(7.5, 7.2, 8.6);
    const orthographic = new THREE.OrthographicCamera(-6, 6, 4.2, -4.2, 0.1, 100);
    orthographic.position.set(0, 14, 0.001);
    orthographic.up.set(0, 0, -1);
    orthographic.lookAt(0, 0, 0);

    const controls = new OrbitControls(perspective, renderer.domElement);
    controls.target.set(0, 0, 1.7);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI / 2.05;

    scene.add(new THREE.HemisphereLight(0xa9c8ff, 0x101217, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 3.2);
    key.position.set(-4, 9, -3);
    key.castShadow = true;
    scene.add(key);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 13),
      new THREE.MeshStandardMaterial({ color: 0x15191e, roughness: 0.92 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.z = 1.25;
    floor.receiveShadow = true;
    scene.add(floor);

    const grid = new THREE.GridHelper(12, 24, 0x303740, 0x20262d);
    grid.position.set(0, 0.006, 1.25);
    scene.add(grid);

    const corridor = new THREE.Mesh(
      new THREE.PlaneGeometry(9.2, 0.72),
      new THREE.MeshBasicMaterial({ color: 0x202b31, transparent: true, opacity: 0.88 }),
    );
    corridor.rotation.x = -Math.PI / 2;
    corridor.position.set(0, 0.012, 0);
    scene.add(corridor);

    const restricted = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 2.3),
      new THREE.MeshBasicMaterial({ color: 0xff425c, transparent: true, opacity: 0.18 }),
    );
    restricted.rotation.x = -Math.PI / 2;
    restricted.position.set(0, 0.018, 6.2);
    scene.add(restricted);

    const boundary = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-6, 0.03, 5.05),
        new THREE.Vector3(6, 0.03, 5.05),
      ]),
      new THREE.LineBasicMaterial({ color: 0xff5368 }),
    );
    scene.add(boundary);

    const block = new THREE.Mesh(
      new THREE.BoxGeometry(0.5, 0.5, 0.5),
      new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.32, metalness: 0.12 }),
    );
    block.position.set(0, 0.25, 0);
    block.castShadow = true;
    scene.add(block);

    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(0.2, 32, 20),
      new THREE.MeshStandardMaterial({ color: 0xb8c0ca, roughness: 0.2, metalness: 0.82 }),
    );
    ball.position.set(0, 0.21, 1.45);
    ball.castShadow = true;
    scene.add(ball);

    const guide = new THREE.Mesh(
      new THREE.BoxGeometry(0.62, 0.08, 4.2),
      new THREE.MeshStandardMaterial({ color: 0x313942, roughness: 0.75 }),
    );
    guide.position.set(0, 0.04, 3.2);
    guide.receiveShadow = true;
    scene.add(guide);

    const robot = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.58, 0.3, 0.46),
      new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.4 }),
    );
    body.position.y = 0.24;
    body.castShadow = true;
    robot.add(body);
    for (const x of [-0.22, 0.22]) {
      for (const z of [-0.18, 0.18]) {
        const wheel = new THREE.Mesh(
          new THREE.CylinderGeometry(0.09, 0.09, 0.08, 16),
          new THREE.MeshStandardMaterial({ color: 0x101215, roughness: 0.8 }),
        );
        wheel.rotation.z = Math.PI / 2;
        wheel.position.set(x, 0.1, z);
        robot.add(wheel);
      }
    }
    scene.add(robot);

    const target = new THREE.Mesh(
      new THREE.RingGeometry(0.34, 0.46, 36),
      new THREE.MeshBasicMaterial({ color: 0x41d59b, side: THREE.DoubleSide }),
    );
    target.rotation.x = -Math.PI / 2;
    target.position.set(4, 0.035, 0);
    scene.add(target);

    const path = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-4, 0.045, 0),
        new THREE.Vector3(4, 0.045, 0),
      ]),
      new THREE.LineDashedMaterial({ color: 0x5eead4, dashSize: 0.18, gapSize: 0.12 }),
    );
    path.computeLineDistances();
    scene.add(path);

    const resize = () => {
      const width = host.clientWidth;
      const height = host.clientHeight;
      renderer.setSize(width, height, false);
      perspective.aspect = width / Math.max(height, 1);
      perspective.updateProjectionMatrix();
      const aspect = width / Math.max(height, 1);
      const span = 8;
      orthographic.left = (-span * aspect) / 2;
      orthographic.right = (span * aspect) / 2;
      orthographic.top = span / 2;
      orthographic.bottom = -span / 2;
      orthographic.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    const animate = () => {
      const state = sceneRef.current;
      if (!state) return;
      state.controls.enabled = viewMode === "3d";
      if (viewMode === "3d") state.controls.update();
      renderer.render(scene, viewMode === "3d" ? perspective : orthographic);
      state.frame = requestAnimationFrame(animate);
    };

    sceneRef.current = { block, ball, robot, renderer, perspective, orthographic, controls, path, frame: 0 };
    animate();

    return () => {
      observer.disconnect();
      if (sceneRef.current) cancelAnimationFrame(sceneRef.current.frame);
      controls.dispose();
      renderer.dispose();
      host.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, [viewMode]);

  useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;
    const execution = trial?.execution;
    const completion = execution?.completion_time_seconds;
    if (!execution || typeof completion !== "number") {
      state.block.position.z = 0;
      state.ball.position.z = 1.45;
      state.robot.position.set(-4, 0, 0);
      return;
    }

    const triggered = execution.ball_latch_released === true;
    const causalEvents = execution.causal_chain ?? [];
    const releaseTime = causalEvents.find((event) =>
      event.event.toLowerCase().includes("begins moving"),
    )?.time_seconds;
    const violationTime = execution.violation_time_seconds;
    const pushEnd = typeof releaseTime === "number" ? releaseTime : null;
    const pushProgress = triggered && pushEnd !== null ? clamp(time / pushEnd) : 0;
    state.block.position.z = pushProgress;
    state.ball.position.z =
      triggered && typeof releaseTime === "number" && typeof violationTime === "number"
        ? 1.45 + 3.6 * clamp((time - releaseTime) / (violationTime - releaseTime))
        : 1.45;

    const points = movePath(trial?.decision?.actions);
    state.path.geometry.dispose();
    state.path.geometry = new THREE.BufferGeometry().setFromPoints(
      points.map(([x, y]) => new THREE.Vector3(x, 0.045, y)),
    );
    state.path.computeLineDistances();
    const travel = triggered && pushEnd !== null
      ? clamp((time - pushEnd) / Math.max(0.1, completion - pushEnd))
      : clamp(time / Math.max(0.1, completion));
    const [robotX, robotY] = pointOnPath(points, travel);
    state.robot.position.set(robotX, 0, robotY);
  }, [trial, time, viewMode]);

  return <div className="scene-host" ref={hostRef} aria-label="Interactive physical scenario playback" />;
}
