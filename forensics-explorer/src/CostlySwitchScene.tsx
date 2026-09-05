import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { TrialRecord } from "./types";

type Props = { trial?: TrialRecord; time: number; viewMode: "2d" | "3d" };

const clamp = (value: number) => Math.min(1, Math.max(0, value));

function vehicle(color: number) {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.7, 0.36, 0.5),
    new THREE.MeshStandardMaterial({ color, roughness: 0.42 }),
  );
  body.position.y = 0.27;
  body.castShadow = true;
  group.add(body);
  return group;
}

export function CostlySwitchScene({ trial, time, viewMode }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    vehicleA: THREE.Group;
    vehicleB: THREE.Group;
    divider: THREE.Mesh;
    restricted: THREE.Mesh;
    violation: THREE.Mesh;
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
    perspective.position.set(8, 8, 9);
    const orthographic = new THREE.OrthographicCamera(-6, 6, 4.5, -4.5, 0.1, 100);
    orthographic.position.set(0, 14, 0.001);
    orthographic.up.set(0, 0, -1);
    orthographic.lookAt(0, 0, 0);
    const controls = new OrbitControls(perspective, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI / 2.05;

    scene.add(new THREE.HemisphereLight(0xa9c8ff, 0x101217, 2.3));
    const key = new THREE.DirectionalLight(0xffffff, 3.2);
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

    for (const z of [-3, -2, 0, 2, 3]) {
      const lane = new THREE.Mesh(
        new THREE.PlaneGeometry(10, 0.55),
        new THREE.MeshBasicMaterial({ color: 0x22313a, transparent: true, opacity: 0.82 }),
      );
      lane.rotation.x = -Math.PI / 2;
      lane.position.set(0, 0.012, z);
      scene.add(lane);
    }

    const restricted = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 3),
      new THREE.MeshBasicMaterial({ color: 0xff425c, transparent: true, opacity: 0.2 }),
    );
    restricted.rotation.x = -Math.PI / 2;
    restricted.position.set(0, 0.018, 3);
    scene.add(restricted);

    const divider = new THREE.Mesh(
      new THREE.BoxGeometry(1.1, 0.42, 0.35),
      new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.35 }),
    );
    divider.position.set(0, 0.22, 3);
    divider.castShadow = true;
    scene.add(divider);

    const vehicleA = vehicle(0xf59e0b);
    const vehicleB = vehicle(0xb8c0ca);
    scene.add(vehicleA, vehicleB);

    const violation = new THREE.Mesh(
      new THREE.RingGeometry(0.45, 0.62, 36),
      new THREE.MeshBasicMaterial({ color: 0xff203f, side: THREE.DoubleSide }),
    );
    violation.rotation.x = -Math.PI / 2;
    violation.position.y = 0.035;
    violation.visible = false;
    scene.add(violation);

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
    stateRef.current = { vehicleA, vehicleB, divider, restricted, violation, renderer, perspective, orthographic, controls, frame: 0 };
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
    const mirrored = trial?.layout?.mirrored === true;
    const direction = mirrored ? -1 : 1;
    const isCratePlacement = trial?.condition === "causal-crate-placement";
    const selectedPadY = trial?.decision?.action === "PLACE_CRATE_PAD_ALPHA"
      ? trial?.layout?.pad_alpha_y
      : trial?.layout?.pad_beta_y;
    const parkedY = selectedPadY ?? trial?.layout?.parked_y ?? direction * 3;
    const bypassY = trial?.layout?.bypass_y ?? direction * 2;
    const junctionX = trial?.layout?.junction_x ?? 0;
    const shifted = trial?.decision?.action === "SHIFT_DIVIDER" || trial?.execution?.crate_blocked_other_lane === true;

    state.divider.position.x = junctionX;
    state.restricted.position.z = direction * 3;
    state.divider.position.z = isCratePlacement
      ? parkedY
      : shifted
        ? THREE.MathUtils.lerp(parkedY, 0, clamp(time))
        : parkedY;
    state.divider.scale.z = isCratePlacement ? ((trial?.layout?.crate_half_extent_y ?? 0.5) * 2) / 0.35 : 1;
    const aLane = -direction * 3;
    const aCompletion = trial?.execution?.vehicle_a_completion_time_seconds ?? 12;
    state.vehicleA.position.set(THREE.MathUtils.lerp(-4.5, 4.5, clamp(time / aCompletion)), 0, aLane);
    const bLane = shifted ? THREE.MathUtils.lerp(0, bypassY, clamp((time - 4.5) / 1.5)) : 0;
    state.vehicleB.position.set(junctionX + Math.min(11, Math.max(0, time)) - 6, 0, bLane);
    state.violation.position.set(state.vehicleB.position.x, 0.035, bypassY);
    state.violation.visible = shifted && time >= 7;
  }, [trial, time]);

  return <div className="scene-host" ref={hostRef} aria-label="Costly Switch scenario playback" />;
}
