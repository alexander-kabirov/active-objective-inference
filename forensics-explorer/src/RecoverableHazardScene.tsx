import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { TrialRecord } from "./types";

type Props = { trial?: TrialRecord; time: number; viewMode: "2d" | "3d" };

const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value));

export function RecoverableHazardScene({ trial, time, viewMode }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    load: THREE.Mesh;
    robot: THREE.Group;
    catcher: THREE.Group;
    violationGlow: THREE.Mesh;
    renderer: THREE.WebGLRenderer;
    perspective: THREE.PerspectiveCamera;
    orthographic: THREE.OrthographicCamera;
    controls: OrbitControls;
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
    host.appendChild(renderer.domElement);

    const perspective = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    perspective.position.set(8.5, 7.4, 8.8);
    const orthographic = new THREE.OrthographicCamera(-6, 6, 4.5, -4.5, 0.1, 100);
    orthographic.position.set(0, 14, 0.001);
    orthographic.up.set(0, 0, -1);
    orthographic.lookAt(0, 0, 0);
    const controls = new OrbitControls(perspective, renderer.domElement);
    controls.target.set(0, 1.4, 1.1);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI / 2.05;

    scene.add(new THREE.HemisphereLight(0xa9c8ff, 0x101217, 2.3));
    const key = new THREE.DirectionalLight(0xffffff, 3.4);
    key.position.set(-4, 10, -3);
    key.castShadow = true;
    scene.add(key);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 9),
      new THREE.MeshStandardMaterial({ color: 0x15191e, roughness: 0.92 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    const grid = new THREE.GridHelper(12, 24, 0x303740, 0x20262d);
    grid.position.y = 0.006;
    scene.add(grid);

    const corridor = new THREE.Mesh(
      new THREE.PlaneGeometry(9.2, 0.8),
      new THREE.MeshBasicMaterial({ color: 0x203139, transparent: true, opacity: 0.9 }),
    );
    corridor.rotation.x = -Math.PI / 2;
    corridor.position.y = 0.012;
    scene.add(corridor);

    const protectedZone = new THREE.Mesh(
      new THREE.CylinderGeometry(0.68, 0.68, 0.035, 48),
      new THREE.MeshBasicMaterial({ color: 0xff425c, transparent: true, opacity: 0.28 }),
    );
    protectedZone.position.set(0, 0.025, 2);
    scene.add(protectedZone);

    const worker = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.2, 0.26, 0.9, 24),
      new THREE.MeshStandardMaterial({ color: 0x60a5fa, roughness: 0.65 }),
    );
    body.position.y = 0.56;
    worker.add(body);
    const head = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, 24, 16),
      new THREE.MeshStandardMaterial({ color: 0xf2c6a0, roughness: 0.72 }),
    );
    head.position.y = 1.18;
    worker.add(head);
    worker.position.set(0, 0, 2);
    scene.add(worker);

    const gantryMaterial = new THREE.MeshStandardMaterial({ color: 0x3d4650, roughness: 0.7, metalness: 0.4 });
    for (const x of [-1.1, 1.1]) {
      const post = new THREE.Mesh(new THREE.BoxGeometry(0.12, 5.2, 0.12), gantryMaterial);
      post.position.set(x, 2.6, 2);
      scene.add(post);
    }
    const beam = new THREE.Mesh(new THREE.BoxGeometry(2.35, 0.14, 0.14), gantryMaterial);
    beam.position.set(0, 5.15, 2);
    scene.add(beam);

    const cable = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 5.1, 2),
        new THREE.Vector3(0, 4.85, 2),
      ]),
      new THREE.LineBasicMaterial({ color: 0x8d99a6 }),
    );
    scene.add(cable);

    const load = new THREE.Mesh(
      new THREE.BoxGeometry(0.78, 0.78, 0.78),
      new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.42, metalness: 0.08 }),
    );
    load.position.set(0, 4.5, 2);
    load.castShadow = true;
    scene.add(load);

    const catcher = new THREE.Group();
    for (const x of [-0.65, 0.65]) {
      const arm = new THREE.Mesh(
        new THREE.BoxGeometry(0.5, 0.12, 0.18),
        new THREE.MeshStandardMaterial({ color: 0x34d399, roughness: 0.35, metalness: 0.32 }),
      );
      arm.position.x = x;
      catcher.add(arm);
    }
    catcher.position.set(0, 1.95, 2);
    catcher.visible = false;
    scene.add(catcher);

    const robot = new THREE.Group();
    const robotBody = new THREE.Mesh(
      new THREE.BoxGeometry(0.58, 0.42, 0.52),
      new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.4 }),
    );
    robotBody.position.y = 0.28;
    robotBody.castShadow = true;
    robot.add(robotBody);
    robot.position.set(-4, 0, 0);
    scene.add(robot);

    const target = new THREE.Mesh(
      new THREE.RingGeometry(0.34, 0.47, 36),
      new THREE.MeshBasicMaterial({ color: 0x41d59b, side: THREE.DoubleSide }),
    );
    target.rotation.x = -Math.PI / 2;
    target.position.set(4, 0.035, 0);
    scene.add(target);

    const violationGlow = new THREE.Mesh(
      new THREE.CylinderGeometry(0.92, 0.92, 0.05, 48),
      new THREE.MeshBasicMaterial({ color: 0xff203f, transparent: true, opacity: 0.62 }),
    );
    violationGlow.position.set(0, 0.04, 2);
    violationGlow.visible = false;
    scene.add(violationGlow);

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
    stateRef.current = { load, robot, catcher, violationGlow, renderer, perspective, orthographic, controls, frame: 0 };
    const animate = () => {
      const state = stateRef.current;
      if (!state) return;
      state.controls.enabled = viewMode === "3d";
      if (viewMode === "3d") state.controls.update();
      renderer.render(scene, viewMode === "3d" ? perspective : orthographic);
      state.frame = requestAnimationFrame(animate);
    };
    animate();
    return () => {
      observer.disconnect();
      if (stateRef.current) cancelAnimationFrame(stateRef.current.frame);
      controls.dispose();
      renderer.dispose();
      host.removeChild(renderer.domElement);
      stateRef.current = null;
    };
  }, [viewMode]);

  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    const execution = trial?.execution;
    const initiated = execution?.hazard_initiated === true;
    const caught = execution?.routeops_caught_load === true;
    const releaseTime = 2;
    const catchTime = 7;
    const contactTime = 8;
    let height = 4.5;
    if (initiated && time > releaseTime) {
      height = 4.5 - 0.5 * Math.min(time - releaseTime, contactTime - releaseTime);
      if (caught && time >= catchTime) height = 2;
    }
    state.load.position.y = height;
    state.catcher.visible = caught && time >= catchTime;
    state.violationGlow.visible = initiated && !caught && time >= contactTime;
    const completion = execution?.completion_time_seconds ?? 6;
    state.robot.position.x = THREE.MathUtils.lerp(-4, 4, clamp(time / completion));
  }, [trial, time]);

  return <div className="scene-host" ref={hostRef} aria-label="Recoverable falling-load scenario playback" />;
}
