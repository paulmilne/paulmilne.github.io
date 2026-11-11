// tars3d.js: plays back MuJoCo runs of the four-slab robot (walk, star roll, cross roll).
// Frames: [t, x, z, pitch, h0, s0, h1, s1, h2, s2, h3, s3] in MuJoCo's slope frame (x downhill, z up).
export async function mountTars(host, THREE, url) {
  const data = await (await fetch(url)).json();
  host.innerHTML = `<div class="tr-view"></div><div class="tr-hud"></div>
    <div class="tr-controls"><span>Run</span><button data-m="walk" class="on">Crutch walk</button><button data-m="star">Roll: star, 8 spokes</button><button data-m="cross">Roll: cross, 4 spokes</button>
    <span class="tr-gap"></span><button data-ts="1" class="on">1×</button><button data-ts="0.25">¼×</button><button data-c="restart">Restart</button></div>`;
  const view = host.querySelector(".tr-view"), hud = host.querySelector(".tr-hud");
  const W = () => view.clientWidth, H = () => view.clientHeight;
  const renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(W(), H());
  renderer.setClearColor(0xdfe6ec); view.appendChild(renderer.domElement);
  const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(40, W() / H(), 0.05, 200);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f96, 1.0));
  const sun = new THREE.DirectionalLight(0xffffff, 0.7); sun.position.set(-3, 8, 6); scene.add(sun);
  const world = new THREE.Group(); scene.add(world);                       // tilted by the slope
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshLambertMaterial({ color: 0x7d8577 }));
  ground.rotation.x = -Math.PI / 2; ground.position.x = 30; world.add(ground);
  const stripeMat = new THREE.MeshBasicMaterial({ color: 0x5e665a });
  for (let x = -10; x <= 70; x++) { const st = new THREE.Mesh(new THREE.PlaneGeometry(x % 5 === 0 ? 0.06 : 0.025, 6), stripeMat); st.rotation.x = -Math.PI / 2; st.position.set(x, 0.005, 0); world.add(st); }
  // robot: mj (x,y,z) -> three (x, z, -y)
  const LS = 1.6, Wd = 0.30, D = 0.22, YS = [-0.48, -0.16, 0.16, 0.48];
  const robot = new THREE.Group(); world.add(robot);
  const axle = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1.2, 12), new THREE.MeshStandardMaterial({ color: 0x333333 }));
  axle.rotation.x = Math.PI / 2; robot.add(axle);
  const arms = YS.map((y, i) => {
    const arm = new THREE.Group(); arm.position.z = -y; robot.add(arm);
    const slab = new THREE.Group(); arm.add(slab);
    const m = new THREE.Mesh(new THREE.BoxGeometry(D, LS, Wd - 0.01), new THREE.MeshStandardMaterial({ color: i % 3 === 0 ? 0xd9d9d3 : 0xc2c2bc, metalness: 0.5, roughness: 0.45 }));
    slab.add(m);
    if (i === 1) { const scr = new THREE.Mesh(new THREE.PlaneGeometry(0.16, 0.22), new THREE.MeshBasicMaterial({ color: 0x0c1418 })); scr.position.set(D / 2 + 0.002, 0.25, 0); scr.rotation.y = Math.PI / 2; slab.add(scr);
      const txt = new THREE.Mesh(new THREE.PlaneGeometry(0.1, 0.012), new THREE.MeshBasicMaterial({ color: 0x9fd8ff })); txt.position.set(D / 2 + 0.004, 0.28, 0); txt.rotation.y = Math.PI / 2; slab.add(txt); }
    return { arm, slab };
  });
  let mode = "walk", ts = 1, t = 0, k = 0, last = performance.now(), alive = true;
  const pick = (sel, el) => { host.querySelectorAll(sel).forEach(b => b.classList.remove("on")); el.classList.add("on"); };
  host.querySelectorAll("[data-m]").forEach(b => b.addEventListener("click", () => { mode = b.dataset.m; pick("[data-m]", b); t = 0; k = 0; }));
  host.querySelectorAll("[data-ts]").forEach(b => b.addEventListener("click", () => { ts = +b.dataset.ts; pick("[data-ts]", b); }));
  host.querySelector("[data-c=restart]").addEventListener("click", () => { t = 0; k = 0; });
  let az = 0.35, el = 0.18, dist = 5.5, drag = false, lx = 0, ly = 0;
  view.addEventListener("pointerdown", e => { drag = true; lx = e.clientX; ly = e.clientY; view.setPointerCapture(e.pointerId); });
  view.addEventListener("pointerup", () => drag = false);
  view.addEventListener("pointermove", e => { if (!drag) return; az -= (e.clientX - lx) * 0.006; el = Math.max(0.02, Math.min(1.3, el + (e.clientY - ly) * 0.006)); lx = e.clientX; ly = e.clientY; });
  view.addEventListener("wheel", e => { e.preventDefault(); dist = Math.max(2, Math.min(30, dist * Math.exp(e.deltaY * 0.001))); }, { passive: false });
  function frame(now) {
    if (!alive) return;
    if (!host.isConnected) { alive = false; renderer.dispose(); return; }
    const run = data[mode], F = run.frames, dt = Math.min(0.05, (now - last) / 1000); last = now;
    t += dt * ts; if (t > F[F.length - 1][0] + 1.5) { t = 0; k = 0; }
    while (k < F.length - 1 && F[k + 1][0] <= t) k++; if (F[k][0] > t) k = 0;
    const f = F[Math.min(k, F.length - 1)];
    world.rotation.z = -run.slope * Math.PI / 180;
    robot.position.set(f[1], 1.0 + f[2], 0); robot.rotation.z = -f[3];
    arms.forEach((a, i) => { a.arm.rotation.z = -f[4 + 2 * i]; a.slab.position.y = f[5 + 2 * i]; });
    const p = new THREE.Vector3(); robot.getWorldPosition(p);
    cam.position.set(p.x + dist * Math.cos(el) * Math.sin(az), p.y + dist * Math.sin(el), p.z + dist * Math.cos(el) * Math.cos(az)); cam.lookAt(p.x, p.y - 0.2, p.z);
    if (renderer.domElement.width !== Math.floor(W() * renderer.getPixelRatio())) { renderer.setSize(W(), H()); cam.aspect = W() / H(); cam.updateProjectionMatrix(); }
    renderer.render(scene, cam);
    const j = Math.max(0, k - 30), v = k > j ? (F[k][1] - F[j][1]) / (F[k][0] - F[j][0]) : 0;
    hud.textContent = `${mode === "walk" ? "Crutch walk on level ground" : `Rolling down a ${run.slope}° slope`} · t = ${t.toFixed(1)} s · speed ${v.toFixed(2)} m/s`;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
