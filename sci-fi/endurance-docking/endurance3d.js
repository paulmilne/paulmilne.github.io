// endurance3d.js: 3D view of the Ranger docking with the spinning Endurance.
// Usage: mountDocking(host, THREE, sim) with sim = await import("dockingsim.js").
export function mountDocking(host, THREE, S) {
  host.innerHTML = `
  <div class="ed-wrap"><div class="ed-view"></div><div class="ed-reticle"></div><div class="ed-hud"></div><div class="ed-msg"></div></div>
  <div class="ed-controls">
    <div class="ed-row"><span>Spin</span>
      <button data-rpm="5.5">5.5 rpm (design)</button><button data-rpm="20">20 rpm (on screen)</button><button data-rpm="68">68 rpm (dialogue)</button></div>
    <div class="ed-row"><label><input type="checkbox" data-c="damaged" checked> Damaged (engine and cargo module lost)</label></div>
    <label>Wobble: spin tilted off the hub axis <input type="range" data-c="wobble" min="0" max="10" step="0.5" value="0"><output></output></label>
    <label>Ranger thrust (g) <input type="range" data-c="acc" min="-1" max="1.6" step="0.05" value="0.4"><output></output></label>
    <label>Ranger angular acceleration (rad/s²) <input type="range" data-c="alpha" min="-0.7" max="1.3" step="0.05" value="-0.2"><output></output></label>
    <div class="ed-row"><span>View</span>
      <button data-cam="chase" class="on">Outside</button><button data-cam="cockpit">Cooper's seat</button><button data-cam="axis">Along the spin axis</button>
      <span class="ed-gap"></span><span>Speed</span><button data-ts="1" class="on">1×</button><button data-ts="0.25">¼×</button>
      <button data-c="pause">Pause</button><button data-c="restart">Restart</button></div>
  </div>`;
  const $ = q => host.querySelector(q), $$ = q => [...host.querySelectorAll(q)];
  const view = $(".ed-view"), hud = $(".ed-hud"), msg = $(".ed-msg");
  const W = () => view.clientWidth, H = () => view.clientHeight;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); renderer.setSize(W(), H());
  renderer.setClearColor(0x02040a); view.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(50, W() / H(), 0.1, 20000);
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const sun = new THREE.DirectionalLight(0xfff4e0, 1.1); sun.position.set(300, 200, 500); scene.add(sun);
  const fill = new THREE.DirectionalLight(0x8fb6ff, 0.35); fill.position.set(0, -500, 0); scene.add(fill);   // planet-shine

  // stars and Mann's planet
  { const n = 2500, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { const u = Math.random() * 2 - 1, t = Math.random() * 2 * Math.PI, r = 6000, s = Math.sqrt(1 - u * u);
      pos[3 * i] = r * s * Math.cos(t); pos[3 * i + 1] = r * u; pos[3 * i + 2] = r * s * Math.sin(t); }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xffffff, size: 1.6, sizeAttenuation: false }))); }
  const planet = new THREE.Mesh(new THREE.SphereGeometry(2600, 64, 48), new THREE.MeshStandardMaterial({ color: 0xc9d6e2, roughness: 0.95 }));
  planet.position.set(400, -3100, -900); scene.add(planet);
  const haze = new THREE.Mesh(new THREE.SphereGeometry(2680, 64, 48), new THREE.MeshBasicMaterial({ color: 0x9cc3ff, transparent: true, opacity: 0.12 }));
  haze.position.copy(planet.position); scene.add(haze);

  // axes: angular momentum (fixed, +Y) and the hub's axis (moves with the ship)
  const lineMat = c => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.7 });
  const Laxis = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -70, 0), new THREE.Vector3(0, 40, 0)]), lineMat(0xf2c14e));
  scene.add(Laxis);

  const mat = c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.6, metalness: 0.3 });
  const COL = { engine: 0x9a8f86, cargo: 0xc8c2b4, habitat: 0xe6e6e1, cryo: 0xd9dde6, command: 0xe9e4d5 };
  // ---------- the Endurance ----------
  const E = new THREE.Group(), P = new THREE.Group(), RING = new THREE.Group(); E.add(P); P.add(RING); scene.add(E);
  let hubAxis;
  function buildEndurance(damaged) {
    while (RING.children.length) RING.remove(RING.children[0]);
    const torus = new THREE.Mesh(new THREE.TorusGeometry(S.RING_R, 0.7, 10, 96), mat(0xb8b8b0)); RING.add(torus);
    for (let k = 0; k < 4; k++) {
      const a = Math.PI / 4 + k * Math.PI / 2, L = S.RING_R - 3;
      const sp = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.45, L, 8), mat(0xa0a0a0));
      sp.position.set(Math.cos(a) * (L / 2 + 3), Math.sin(a) * (L / 2 + 3), 0); sp.rotation.z = a - Math.PI / 2; RING.add(sp);
    }
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(3, 3, 6, 24), mat(0xd0d0cc)); hub.rotation.x = Math.PI / 2; RING.add(hub);
    const port = new THREE.Mesh(new THREE.TorusGeometry(1.5, 0.3, 10, 32), mat(0x444444)); port.position.z = -S.HUB_PORT - 0.05; RING.add(port);
    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.35, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff3b30 })); lamp.position.set(0, 2.2, -S.HUB_PORT); RING.add(lamp);
    S.LAYOUT.forEach((t, k) => {
      const a = k * Math.PI / 6, lost = damaged && S.LOST.includes(k);
      if (lost) { // broken stub
        const stub = new THREE.Mesh(new THREE.BoxGeometry(2, 3, 3), mat(0x3a3a3a)); stub.position.set(Math.cos(a) * (S.RING_R - 3), Math.sin(a) * (S.RING_R - 3), 0); stub.rotation.z = a + 0.5; RING.add(stub); return; }
      const g = new THREE.Group(); g.position.set(Math.cos(a) * S.RING_R, Math.sin(a) * S.RING_R, 0); g.rotation.z = a;
      const box = new THREE.Mesh(new THREE.BoxGeometry(S.BOX[1], S.BOX[0], S.BOX[2]), mat(COL[t])); g.add(box);
      if (t === "engine") for (let j = -1; j <= 1; j++) {
        const nz = new THREE.Mesh(new THREE.ConeGeometry(1.1, 2.2, 14, 1, true), mat(0x333333)); nz.position.set(0, j * 3, S.BOX[2] / 2 + 1); nz.rotation.x = -Math.PI / 2; g.add(nz); }
      if (t === "command") { const w = new THREE.Mesh(new THREE.BoxGeometry(0.3, 6, 1.5), new THREE.MeshBasicMaterial({ color: 0x223344 })); w.position.x = S.BOX[1] / 2 + 0.1; g.add(w); }
      const stripe = new THREE.Mesh(new THREE.BoxGeometry(S.BOX[1] + 0.05, 0.5, S.BOX[2] + 0.05), new THREE.MeshBasicMaterial({ color: t === "engine" ? 0xd9622b : 0x1f5fa8 })); g.add(stripe);
      RING.add(g);
    });
    hubAxis = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, -70), new THREE.Vector3(0, 0, 30)]), lineMat(0xff6b6b));
    RING.add(hubAxis);
  }
  // ---------- the Ranger (body: x forward, y up, z right) ----------
  const RG = new THREE.Group(); scene.add(RG);
  { const hull = mat(0xdcdcd6), dark = mat(0x2b2b2b);
    const body = new THREE.Mesh(new THREE.BoxGeometry(13, 2.2, 4.2), hull); body.position.x = -1.5; RG.add(body);
    const nose = new THREE.Mesh(new THREE.ConeGeometry(2.1, 4.5, 4), hull); nose.rotation.z = -Math.PI / 2; nose.rotation.x = Math.PI / 4; nose.scale.set(1, 1, 0.55); nose.position.x = 7.2; RG.add(nose);
    const wing = new THREE.Mesh(new THREE.BoxGeometry(6.5, 0.35, 13), hull); wing.position.set(-3.5, -0.6, 0); RG.add(wing);
    for (const z of [-6.3, 6.3]) { const fin = new THREE.Mesh(new THREE.BoxGeometry(3, 2.2, 0.3), hull); fin.position.set(-5.5, 0.6, z); RG.add(fin); }
    const win = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.4, 3.0), new THREE.MeshBasicMaterial({ color: 0x0b1a2a })); win.position.set(3.2, 1.05, 0); RG.add(win);
    const collar = new THREE.Mesh(new THREE.CylinderGeometry(1.4, 1.6, 0.6, 24), dark); collar.position.set(S.RANGER.port[0], S.RANGER.port[1] - 0.3, S.RANGER.port[2]); RG.add(collar);
    for (const z of [-0.9, 0.9]) { const eng = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.9, 1.2, 14), dark); eng.rotation.z = Math.PI / 2; eng.position.set(-8.5, 0, z); RG.add(eng); }
    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.25, 10, 8), new THREE.MeshBasicMaterial({ color: 0x4dd8ff })); lamp.position.set(-0.2, 1.7, 1.8); RG.add(lamp); }

  // ---------- state and controls ----------
  const st = { rpm: 20, damaged: true, wobble: 0, acc: 0.4, alpha: -0.2, camMode: "chase", ts: 1, paused: false };
  let sim;
  const accG = () => Math.pow(10, st.acc), alphaVal = () => Math.pow(10, st.alpha);
  function restart() {
    sim = S.createSim({ rpm: st.rpm, damaged: st.damaged, wobble: st.wobble, aMax: accG() * 9.81, alphaMax: alphaVal() });
    buildEndurance(st.damaged);
    P.quaternion.set(-sim.mp.qP[1], -sim.mp.qP[2], -sim.mp.qP[3], sim.mp.qP[0]);   // conj(qP): ring -> principal
    RING.position.set(-sim.mp.com[0], -sim.mp.com[1], -sim.mp.com[2]);
  }
  const outs = {};
  $$(".ed-controls input[type=range]").forEach(inp => {
    const k = inp.dataset.c, o = inp.parentElement.querySelector("output"); outs[k] = o;
    const show = () => { st[k] = +inp.value; o.textContent = k === "acc" ? accG().toFixed(accG() < 1 ? 2 : 1) : k === "alpha" ? alphaVal().toFixed(alphaVal() < 1 ? 2 : 1) : `${st[k]}°`; };
    show(); inp.addEventListener("input", show); inp.addEventListener("change", restart);
  });
  $("[data-c=damaged]").addEventListener("change", e => { st.damaged = e.target.checked; restart(); });
  const pick = (sel, el) => { $$(sel).forEach(b => b.classList.remove("on")); el.classList.add("on"); };
  $$("[data-rpm]").forEach(b => b.addEventListener("click", () => { st.rpm = +b.dataset.rpm; pick("[data-rpm]", b); restart(); }));
  $$("[data-cam]").forEach(b => b.addEventListener("click", () => { st.camMode = b.dataset.cam; pick("[data-cam]", b); }));
  $$("[data-ts]").forEach(b => b.addEventListener("click", () => { st.ts = +b.dataset.ts; pick("[data-ts]", b); }));
  $("[data-c=pause]").addEventListener("click", e => { st.paused = !st.paused; e.target.textContent = st.paused ? "Play" : "Pause"; });
  $("[data-c=restart]").addEventListener("click", restart);
  pick("[data-rpm]", $("[data-rpm='20']"));

  // orbit camera for the outside view
  let az = 0.9, el = 0.25, dist = 120, drag = false, lx = 0, ly = 0;
  view.addEventListener("pointerdown", e => { drag = true; lx = e.clientX; ly = e.clientY; view.setPointerCapture(e.pointerId); });
  view.addEventListener("pointerup", () => { drag = false; });
  view.addEventListener("pointermove", e => { if (!drag) return; az -= (e.clientX - lx) * 0.006; el = Math.max(-1.45, Math.min(1.45, el + (e.clientY - ly) * 0.006)); lx = e.clientX; ly = e.clientY; });
  view.addEventListener("wheel", e => { e.preventDefault(); dist = Math.max(25, Math.min(600, dist * Math.exp(e.deltaY * 0.001))); }, { passive: false });

  const tq = q => new THREE.Quaternion(q[1], q[2], q[3], q[0]);
  const f1 = (x, d = 1) => (Math.abs(x) < 1e-9 ? 0 : x).toFixed(d);
  let last = performance.now(), alive = true;
  function frame(now) {
    if (!alive) return;
    if (!host.isConnected) { alive = false; renderer.dispose(); return; }
    const dtReal = Math.min(0.05, (now - last) / 1000); last = now;
    if (!st.paused) { const dt = dtReal * st.ts, n = Math.max(1, Math.ceil(dt * 800)); for (let i = 0; i < n; i++) sim.step(dt / n); }
    E.quaternion.copy(tq(sim.qE));
    RG.position.set(...sim.p); RG.quaternion.copy(tq(sim.qR));
    // camera
    RG.visible = st.camMode !== "cockpit"; $(".ed-reticle").style.display = st.camMode === "cockpit" ? "block" : "none";
    if (st.camMode === "cockpit") {
      const q = tq(sim.qR), seat = new THREE.Vector3(...S.RANGER.seat).applyQuaternion(q).add(RG.position);
      const look = new THREE.Vector3(S.RANGER.port[0], S.RANGER.port[1] + 12, S.RANGER.port[2]).applyQuaternion(q).add(RG.position);
      cam.fov = 75; cam.position.copy(seat); cam.up.set(1, 0, 0).applyQuaternion(q); cam.lookAt(look);
    } else if (st.camMode === "axis") {
      cam.fov = 50; cam.position.set(0, -dist * 1.2, 0.01); cam.up.set(0, 0, -1); cam.lookAt(0, 0, 0);
    } else {
      cam.fov = 50; cam.up.set(0, 1, 0);
      cam.position.set(dist * Math.cos(el) * Math.sin(az), dist * Math.sin(el) - 15, dist * Math.cos(el) * Math.cos(az)); cam.lookAt(0, -15, 0);
    }
    cam.updateProjectionMatrix();
    if (renderer.domElement.width !== Math.floor(W() * renderer.getPixelRatio())) { renderer.setSize(W(), H()); cam.aspect = W() / H(); }
    renderer.render(scene, cam);
    // HUD
    if (now % 3 < 1.5 || true) {
      const g = sim.aNeed / 9.81, cap = accG();
      hud.innerHTML = `<b>Endurance</b> ${f1(sim.endRpm())} rpm · hub axis ${f1(sim.axisTilt())}° off the spin axis · docking port moving ${f1(sim.portSpeed())} m/s<br>
        <b>Ranger</b> ${f1(sim.rangerRpm())} rpm · attitude error ${f1(sim.angErr)}° · position error ${f1(sim.posErr, 2)} m · Cooper's seat ${f1(sim.seatG)} g<br>
        <b>Needs</b> <span class="${g > cap ? "bad" : ""}">${f1(g, 2)} g</span> of ${f1(cap, cap < 1 ? 2 : 1)} g thrust · <span class="${sim.alphaNeed > alphaVal() ? "bad" : ""}">${f1(sim.alphaNeed, 2)} rad/s²</span> of ${f1(alphaVal(), 2)} rad/s²`;
      let m = "";
      if (sim.phase === "station") m = `Holding ${f1(sim.standoff, 0)} m below the hub and matching the spin` +
        (g > cap ? ` — cannot: keeping station on the port needs ${f1(g, 1)} g` : sim.alphaNeed > alphaVal() ? ` — cannot: the wobble needs ${f1(sim.alphaNeed, 1)} rad/s² of angular acceleration` : "");
      else if (sim.phase === "approach") m = `Closing at 1 m/s: ${f1(sim.standoff, 0)} m to go`;
      else if (sim.phase === "latched") m = `Docked. Cooper's seat: ${f1(sim.seatG)} g`;
      else if (sim.phase === "missed") m = "Missed the port: restart";
      msg.textContent = m; msg.className = "ed-msg " + (sim.phase === "latched" ? "ok" : (g > cap || sim.alphaNeed > alphaVal()) && sim.phase === "station" ? "bad" : "");
    }
    requestAnimationFrame(frame);
  }
  restart(); requestAnimationFrame(frame);
}
