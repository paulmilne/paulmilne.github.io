// quad3d.js — interactive 3D quadcopter: commands -> attitude controller -> mixer -> four rotors -> rigid body.
// Usage: mountQuad(container, THREE). Frame: x forward, y left, z up. X configuration.
export function mountQuad(host, THREE) {
  host.innerHTML = `
  <div class="qd-view"></div>
  <div class="qd-hud">
    <div class="qd-motors">${["FL", "FR", "RR", "RL"].map((m, i) => `<div class="qd-m"><div class="qd-bar"><div class="qd-fill" data-m="${i}"></div></div><span>${m}<br><small>${i % 2 ? "CCW" : "CW"}</small></span></div>`).join("")}</div>
    <div class="qd-readout"></div>
  </div>
  <div class="qd-controls">
    <label>Throttle <input type="range" data-c="thr" min="-1" max="1" step="0.01" value="0"><output></output></label>
    <label>Roll (right +) <input type="range" data-c="roll" min="-25" max="25" step="1" value="0"><output></output></label>
    <label>Pitch (forward +) <input type="range" data-c="pitch" min="-25" max="25" step="1" value="0"><output></output></label>
    <label>Yaw rate (left +) <input type="range" data-c="yaw" min="-90" max="90" step="5" value="0"><output></output></label>
    <label>Motor mismatch <input type="range" data-c="mis" min="0" max="10" step="0.5" value="0"><output></output></label>
  </div>
  <div class="qd-buttons"><label><input type="checkbox" data-c="stab" checked> Stabilisation (control loop on)</label>
    <button data-c="centre">Centre sticks</button><button data-c="reset">Reset</button></div>`;
  const view = host.querySelector(".qd-view"), W = () => view.clientWidth, H = () => view.clientHeight;
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); renderer.setSize(W(), H()); view.appendChild(renderer.domElement);
  const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(45, W() / H(), 0.05, 200);
  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const sun = new THREE.DirectionalLight(0xffffff, 0.8); sun.position.set(4, -6, 10); scene.add(sun);
  // world: z up. three.js is y-up, so build everything in a group rotated -90° about x.
  const world = new THREE.Group(); world.rotation.x = -Math.PI / 2; scene.add(world);
  const grid = new THREE.GridHelper(10, 20, 0x999999, 0xcccccc); grid.rotation.x = Math.PI / 2; world.add(grid);
  const quad = new THREE.Group(); world.add(quad);
  const arm = 0.11, a = arm / Math.SQRT2, scale = 4;  // drawn 4x size for visibility
  const pos = [[a, a], [a, -a], [-a, -a], [-a, a]];     // FL, FR, RR, RL
  const spin = [1, -1, 1, -1];                            // CW (viewed from above) = -z rotation; store sign for drawing
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.07 * scale, 0.05 * scale, 0.025 * scale), new THREE.MeshStandardMaterial({color: 0x333a44}));
  quad.add(body);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.012 * scale, 0.04 * scale, 12), new THREE.MeshStandardMaterial({color: 0xe76f51}));
  nose.rotation.z = -Math.PI / 2; nose.position.x = 0.055 * scale; quad.add(nose);
  const discs = [], arrows = [];
  pos.forEach(([x, y], i) => {
    const armM = new THREE.Mesh(new THREE.CylinderGeometry(0.004 * scale, 0.004 * scale, arm * scale, 8), new THREE.MeshStandardMaterial({color: 0x555555}));
    armM.position.set(x * scale / 2, y * scale / 2, 0); armM.rotation.z = Math.atan2(y, x) - Math.PI / 2; quad.add(armM);
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.0635 * scale, 0.0635 * scale, 0.002 * scale, 32),
      new THREE.MeshStandardMaterial({color: i % 2 ? 0x2a9d8f : 0x6fa8dc, transparent: true, opacity: 0.45}));
    disc.rotation.x = Math.PI / 2; disc.position.set(x * scale, y * scale, 0.02 * scale); quad.add(disc);
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.12 * scale, 0.01 * scale, 0.003 * scale), new THREE.MeshStandardMaterial({color: 0x222222}));
    blade.position.copy(disc.position); quad.add(blade); discs.push(blade);
    const ar = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(x * scale, y * scale, 0.03 * scale), 0.3, 0xe9c46a, 0.06, 0.04);
    quad.add(ar); arrows.push(ar);
  });
  const trailN = 600, trailPos = new Float32Array(trailN * 3), trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
  world.add(new THREE.Line(trailGeo, new THREE.LineBasicMaterial({color: 0xe76f51, transparent: true, opacity: 0.6})));
  let trailCount = 0;
  // physics
  const m = 0.6, g = 9.81, J = [0.004, 0.004, 0.007], kQ = 0.016, Tmax = 4.5, D2R = Math.PI / 180;
  let s;
  const reset = () => { s = {p: [0, 0, 1.5], v: [0, 0, 0], R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], w: [0, 0, 0], T: [0, 0, 0, 0], blade: [0, 0, 0, 0]}; trailCount = 0; };
  reset();
  const c = {thr: 0, roll: 0, pitch: 0, yaw: 0, mis: 0, stab: true};
  host.querySelectorAll("input[type=range]").forEach(inp => {
    const o = inp.nextElementSibling, k = inp.dataset.c;
    const upd = () => { c[k] = +inp.value; o.textContent = k === "thr" ? (c.thr >= 0 ? "+" : "") + Math.round(c.thr * 100) + "%" : k === "mis" ? c.mis + "%" : k === "yaw" ? c.yaw + "°/s" : c[k] + "°"; };
    inp.addEventListener("input", upd); upd();
  });
  host.querySelector("[data-c=stab]").addEventListener("change", e => c.stab = e.target.checked);
  host.querySelector("[data-c=reset]").onclick = reset;
  host.querySelector("[data-c=centre]").onclick = () => host.querySelectorAll("input[type=range]").forEach(inp => { if (inp.dataset.c !== "mis") { inp.value = 0; inp.dispatchEvent(new Event("input")); } });
  const mul = (A, B) => A.map((r, i) => [0, 1, 2].map(j => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
  const euler = R => ({roll: Math.atan2(R[2][1], R[2][2]), pitch: -Math.asin(Math.max(-1, Math.min(1, R[2][0]))), yaw: Math.atan2(R[1][0], R[0][0])});
  function step(dt) {
    const e = euler(s.R);
    // body rates in body frame are s.w. In x-forward, y-left, z-up axes a positive rotation about x lowers the right side (roll right)
    // and a positive rotation about y lowers the nose (pitch forward).
    let thrust = m * g * (1 + 0.6 * c.thr), tau;
    if (c.stab) {
      const rollDes = c.roll * D2R, pitchDes = c.pitch * D2R, rDes = [8 * (rollDes - e.roll), 8 * (pitchDes - e.pitch), c.yaw * D2R];
      tau = [0, 1, 2].map(i => J[i] * (i === 2 ? 6 : 40) * (rDes[i] - s.w[i]));
      thrust /= Math.max(0.6, s.R[2][2]);
      thrust += m * 2 * (0 - s.v[2]) * (Math.abs(c.thr) < 0.02 ? 1 : 0);   // hold height when throttle centred
    } else {
      tau = [c.roll * 0.0006, c.pitch * 0.0006, c.yaw * 0.00004];        // open loop: sticks set torque directly
    }
    // mixer: [T, tx, ty, tz] = M [T_FL, T_FR, T_RR, T_RL]
    const k = kQ, inv = [
      [0.25, 0.25 / a, -0.25 / a, 0.25 / k], [0.25, -0.25 / a, -0.25 / a, -0.25 / k],
      [0.25, -0.25 / a, 0.25 / a, 0.25 / k], [0.25, 0.25 / a, 0.25 / a, -0.25 / k]];
    const u = [thrust, tau[0], tau[1], tau[2]];
    s.T = inv.map((r, i) => Math.max(0, Math.min(Tmax, r[0] * u[0] + r[1] * u[1] + r[2] * u[2] + r[3] * u[3])) * (i === 0 ? 1 - c.mis / 100 : 1));
    const [f1, f2, f3, f4] = s.T, Tt = f1 + f2 + f3 + f4;
    const tb = [a * (f1 + f4 - f2 - f3), a * (f3 + f4 - f1 - f2), k * (f1 + f3 - f2 - f4)];
    // rotational dynamics (Euler's equations) with light aerodynamic damping
    const w = s.w, Jw = [J[0] * w[0], J[1] * w[1], J[2] * w[2]];
    const cr = [w[1] * Jw[2] - w[2] * Jw[1], w[2] * Jw[0] - w[0] * Jw[2], w[0] * Jw[1] - w[1] * Jw[0]];
    s.w = w.map((wi, i) => wi + dt * (tb[i] - cr[i] - 0.002 * wi) / J[i]);
    const [p, q, r] = s.w.map(v => v * dt), Om = [[1, -r, q], [r, 1, -p], [-q, p, 1]];
    s.R = mul(s.R, Om);
    // re-orthonormalise
    const c0 = [s.R[0][0], s.R[1][0], s.R[2][0]], n0 = Math.hypot(...c0); const x0 = c0.map(v => v / n0);
    let c1 = [s.R[0][1], s.R[1][1], s.R[2][1]]; const d = c1.reduce((acc, v, i) => acc + v * x0[i], 0); c1 = c1.map((v, i) => v - d * x0[i]);
    const n1 = Math.hypot(...c1), y0 = c1.map(v => v / n1), z0 = [x0[1] * y0[2] - x0[2] * y0[1], x0[2] * y0[0] - x0[0] * y0[2], x0[0] * y0[1] - x0[1] * y0[0]];
    s.R = [[x0[0], y0[0], z0[0]], [x0[1], y0[1], z0[1]], [x0[2], y0[2], z0[2]]];
    // translation: thrust along body z, gravity, drag
    const acc = [0, 1, 2].map(i => s.R[i][2] * Tt / m - (i === 2 ? g : 0) - 0.3 * s.v[i] / m);
    s.v = s.v.map((v, i) => v + acc[i] * dt); s.p = s.p.map((v, i) => v + s.v[i] * dt);
    for (let i = 0; i < 2; i++) { if (s.p[i] > 5) s.p[i] -= 10; if (s.p[i] < -5) s.p[i] += 10; }
    if (s.p[2] < 0.05) { if (Math.abs(euler(s.R).roll) > 1 || Math.abs(euler(s.R).pitch) > 1 || s.v[2] < -4) { reset(); return; } s.p[2] = 0.05; s.v = [s.v[0] * 0.5, s.v[1] * 0.5, Math.max(0, s.v[2])]; }
    if (s.p[2] > 8) { s.p[2] = 8; s.v[2] = Math.min(0, s.v[2]); }
    s.blade = s.blade.map((ang, i) => ang + spin[i] * Math.sqrt(s.T[i]) * 1.4);
  }
  const fills = [...host.querySelectorAll(".qd-fill")], readout = host.querySelector(".qd-readout");
  let th = 0.8, ph = 1.1, rad = 4.2, drag = false, lx = 0, ly = 0;
  const el = renderer.domElement;
  el.addEventListener("pointerdown", e => { drag = true; lx = e.clientX; ly = e.clientY; el.setPointerCapture(e.pointerId); });
  el.addEventListener("pointermove", e => { if (!drag) return; th -= (e.clientX - lx) * 0.01; ph = Math.min(1.5, Math.max(0.2, ph - (e.clientY - ly) * 0.01)); lx = e.clientX; ly = e.clientY; });
  el.addEventListener("pointerup", () => drag = false);
  const ro = new ResizeObserver(() => { renderer.setSize(W(), H()); cam.aspect = W() / H(); cam.updateProjectionMatrix(); }); ro.observe(view);
  let last = performance.now(), frame = 0, alive = true;
  function loop(now) {
    if (!alive) return;
    if (!host.isConnected) { alive = false; renderer.dispose(); ro.disconnect(); return; }
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    for (let i = 0; i < 20; i++) step(dt / 20);
    const M = new THREE.Matrix4().set(s.R[0][0], s.R[0][1], s.R[0][2], s.p[0], s.R[1][0], s.R[1][1], s.R[1][2], s.p[1], s.R[2][0], s.R[2][1], s.R[2][2], s.p[2], 0, 0, 0, 1);
    quad.matrixAutoUpdate = false; quad.matrix.copy(M);
    discs.forEach((b, i) => b.rotation.z = s.blade[i]);
    arrows.forEach((ar, i) => ar.setLength(Math.max(0.02, s.T[i] / Tmax * 1.2), 0.06, 0.04));
    fills.forEach((f, i) => f.style.height = (100 * s.T[i] / Tmax).toFixed(0) + "%");
    if (++frame % 3 === 0) {
      if (trailCount < trailN) trailCount++; else trailPos.copyWithin(0, 3);
      trailPos.set(s.p, (trailCount - 1) * 3); trailGeo.setDrawRange(0, trailCount); trailGeo.attributes.position.needsUpdate = true;
      const e = euler(s.R);
      readout.innerHTML = `roll ${(e.roll / D2R).toFixed(0)}° · pitch ${(e.pitch / D2R).toFixed(0)}° · yaw ${(e.yaw / D2R).toFixed(0)}°<br>height ${s.p[2].toFixed(1)} m · speed ${Math.hypot(...s.v).toFixed(1)} m/s`;
    }
    const tgt = new THREE.Vector3(s.p[0], s.p[2], -s.p[1]);
    cam.position.set(tgt.x + rad * Math.sin(ph) * Math.sin(th), tgt.y + rad * Math.cos(ph), tgt.z + rad * Math.sin(ph) * Math.cos(th));
    cam.lookAt(tgt); renderer.render(scene, cam); requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}
