// rose_ring.js - interactive 3D view of the H-frame NdFeB magnet with Rose-rim pole caps,
// using the finite-difference field profiles from magnet_fd3d.m / magnet_design.m.
// Requires three.js (r128) loaded first. Renders into <div id="rose-widget">.
(function () {
  // Field deviation B/B0 - 1 (ppm) through the gap centre, from magnet_design_results.mat
  // (0.5 mm grid, cell centres at 0.25, 0.75, ... mm).
  const PROF = {
    x: { // along B0, between the pole caps (cap face at 10 mm)
      s: [0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75],
      flat: [0,33,99,196,322,475,652,849,1063,1289,1523,1761,1998,2231,2456,2669,2867,3048,3210,3350],
      rim: [0,1,3,6,11,18,29,42,60,83,112,149,193,246,309,383,467,564,673,795] },
    z: { // along the sample tube (vertical)
      s: [0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75],
      flat: [0,-17,-50,-101,-170,-259,-368,-499,-655,-837,-1049,-1295,-1577,-1902,-2273,-2698,-3184,-3739,-4373,-5097],
      rim: [0,-1,-1,-3,-4,-6,-7,-8,-8,-6,-3,3,12,25,43,66,95,132,177,231] }
  };
  function init() {
    const host = document.getElementById('rose-widget');
    if (!host || typeof THREE === 'undefined') return;
    host.className = 'nmrw';
    host.innerHTML = `
      <div class="nmrw-view"></div>
      <div class="nmrw-row">
        <label><input type="checkbox" data-k="rim" checked> Rose rims</label>
        <label><input type="checkbox" data-k="yoke" checked> Yoke</label>
        <label><input type="checkbox" data-k="cut" checked> Cutaway</label>
        <button data-k="spin">Pause rotation</button>
      </div>
      <div class="nmrw-sl"><span>Zoom</span><input type="range" data-k="zoom" min="70" max="300" step="1" value="190"><span></span></div>
      <div class="nmrw-lg">
        <div><span class="nmrw-d" style="background:#5F5E5A"></span>NdFeB discs Ø50 × 15</div>
        <div><span class="nmrw-d" style="background:#B4B2A9"></span>Soft-iron caps and yoke</div>
        <div><span class="nmrw-d" style="background:#D85A30"></span>Rose rim 1.5 × 2 mm</div>
        <div><span class="nmrw-d" style="background:#378ADD"></span>Field lines, sample</div>
        <div><span class="nmrw-d" style="background:#1D9E75"></span>Region within ±50 ppm</div>
        <div><span class="nmrw-d" style="background:#EF9F27"></span>RF coil</div>
        <div><span class="nmrw-d" style="background:#444441"></span>B<sub>0</sub> direction</div>
      </div>
      <div class="nmrw-row">
        <button data-k="px">Profile along B<sub>0</sub></button>
        <button data-k="pz">Profile along the tube</button>
      </div>
      <canvas class="nmrw-chart"></canvas>
      <div class="nmrw-note" data-k="sum"></div>`;
    const wrap = host.querySelector('.nmrw-view'), H = 360, W = () => wrap.clientWidth;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); renderer.setSize(W(), H);
    renderer.localClippingEnabled = true; wrap.appendChild(renderer.domElement);
    const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(40, W() / H, 1, 3000);
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const l1 = new THREE.DirectionalLight(0xffffff, 0.85); l1.position.set(80, 140, 160); scene.add(l1);
    const l2 = new THREE.DirectionalLight(0xffffff, 0.35); l2.position.set(-100, -40, -120); scene.add(l2);
    const clip = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    const mat = (c, o) => new THREE.MeshStandardMaterial(Object.assign({ color: c, metalness: 0.5, roughness: 0.5,
      side: THREE.DoubleSide, clippingPlanes: [clip] }, o || {}));
    const ironM = mat(0xB4B2A9), magM = mat(0x5F5E5A), rimM = mat(0xD85A30),
          yokeM = mat(0xB4B2A9, { transparent: true, opacity: 0.35, depthWrite: false });
    const clipped = [ironM, magM, rimM, yokeM];
    const G = 10, R = 25; // half gap, cap radius (mm)
    const cylX = (r, len, xc, m) => { const c = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 72), m); c.rotation.z = Math.PI / 2; c.position.x = xc; scene.add(c); return c; };
    const rims = [];
    [-1, 1].forEach(s => {
      cylX(R, 5, s * (G + 2.5), ironM);            // pole cap 5 mm
      cylX(R, 15, s * (G + 5 + 7.5), magM);        // NdFeB disc
      const prof = [new THREE.Vector2(23, 0), new THREE.Vector2(R, 0), new THREE.Vector2(R, 1.5), new THREE.Vector2(23, 1.5), new THREE.Vector2(23, 0)];
      const ring = new THREE.Mesh(new THREE.LatheGeometry(prof, 120), rimM);
      ring.rotation.z = s > 0 ? Math.PI / 2 : -Math.PI / 2; ring.position.x = s * G; scene.add(ring); rims.push(ring);
    });
    const yoke = new THREE.Group();
    const box = (sx, sy, sz, x, y) => { const b = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), yokeM); b.position.set(x, y, 0); yoke.add(b); };
    box(20, 70, 60, -40, 0); box(20, 70, 60, 40, 0); box(100, 20, 60, 0, 45); box(100, 20, 60, 0, -45);
    scene.add(yoke);
    const tubeM = new THREE.MeshStandardMaterial({ color: 0xdfe8f0, transparent: true, opacity: 0.3, depthWrite: false });
    const tube = new THREE.Mesh(new THREE.CylinderGeometry(5.5, 5.5, 110, 32, 1, true), tubeM); tube.position.y = 30; scene.add(tube);
    const water = new THREE.Mesh(new THREE.CylinderGeometry(5, 5, 10, 32), new THREE.MeshStandardMaterial({ color: 0x378ADD, transparent: true, opacity: 0.55 }));
    scene.add(water);
    class Helix extends THREE.Curve { getPoint(t, o = new THREE.Vector3()) { const a = t * 8 * 2 * Math.PI; return o.set(6.6 * Math.cos(a), -6 + 12 * t, 6.6 * Math.sin(a)); } }
    scene.add(new THREE.Mesh(new THREE.TubeGeometry(new Helix(), 400, 0.35, 6, false),
      new THREE.MeshStandardMaterial({ color: 0xEF9F27, metalness: 0.6, roughness: 0.35 })));
    const uniM = new THREE.MeshStandardMaterial({ color: 0x1D9E75, transparent: true, opacity: 0.35, depthWrite: false });
    const uni = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 20), uniM); scene.add(uni);
    const lineM = new THREE.LineBasicMaterial({ color: 0x378ADD, transparent: true, opacity: 0.8 });
    let lines = new THREE.Group(); scene.add(lines);
    scene.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(-8, -18, 0), 16, 0x444441, 4, 2.5)); // B0 direction
    let on = true, prof = 'z';
    function build() {
      scene.remove(lines); lines = new THREE.Group();
      [0, 7, 13, 18, 22].forEach(r => { for (let a = 0; a < 6; a++) {
        const th = a * Math.PI / 3 + 0.3, pts = [];
        const bow = on ? 0.15 * Math.pow(r / 22, 6) : 3.2 * Math.pow(r / 22, 3);
        for (let i = 0; i <= 30; i++) { const x = -G + 2 * G * i / 30, u = x / G, rr = r + bow * (1 - u * u);
          pts.push(new THREE.Vector3(x, rr * Math.cos(th), rr * Math.sin(th))); }
        lines.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), lineM)); } });
      scene.add(lines);
      // +/-50 ppm region from the FD profiles: flat 0.9 x 1.3 mm, rim 4.0 x 7.3 mm (semi-axes)
      if (on) uni.scale.set(4.0, 7.3, 7.3); else uni.scale.set(0.9, 1.3, 1.3);
      rims.forEach(m => m.visible = on);
      chart();
    }
    const cv = host.querySelector('canvas.nmrw-chart'), cx = cv.getContext('2d');
    function chart() {
      const P = PROF[prof], w = cv.clientWidth, h = 190, dpr = Math.min(window.devicePixelRatio, 2);
      if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = h * dpr; }
      cx.setTransform(dpr, 0, 0, dpr, 0, 0); cx.clearRect(0, 0, w, h);
      const all = P.flat.concat(P.rim), ymin = Math.min(0, ...all), ymax = Math.max(0, ...all);
      const x0 = 56, x1 = w - 10, y0 = 22, y1 = h - 28, smax = 10;
      const X = s => x0 + (x1 - x0) * s / smax, Y = v => y0 + (y1 - y0) * (ymax - v) / (ymax - ymin || 1);
      cx.fillStyle = 'rgba(29,158,117,0.10)'; cx.fillRect(X(0), y0, X(5) - X(0), y1 - y0);
      cx.strokeStyle = '#ddd'; cx.lineWidth = 0.5; cx.fillStyle = '#666'; cx.font = '11px sans-serif';
      const step = Math.pow(10, Math.floor(Math.log10((ymax - ymin) / 4))) * (((ymax - ymin) / 4) / Math.pow(10, Math.floor(Math.log10((ymax - ymin) / 4))) > 5 ? 5 : ((ymax - ymin) / 4) / Math.pow(10, Math.floor(Math.log10((ymax - ymin) / 4))) > 2 ? 2 : 1);
      for (let v = Math.ceil(ymin / step) * step; v <= ymax; v += step) {
        cx.beginPath(); cx.moveTo(x0, Y(v)); cx.lineTo(x1, Y(v)); cx.stroke(); cx.textAlign = 'right'; cx.fillText(v, x0 - 6, Y(v) + 4); }
      cx.textAlign = 'center'; for (let s = 0; s <= 10; s += 2) cx.fillText(s, X(s), y1 + 14);
      cx.fillText(prof === 'z' ? 'distance from centre along the tube (mm)' : 'distance from centre along B0 (mm); cap face at 10 mm', (x0 + x1) / 2, h - 2);
      cx.textAlign = 'left'; cx.fillText('ΔB/B0 (ppm), FD calculation; shaded = Ø10 × 10 mm sample', x0, 12);
      const line = (d, col, wid, dash) => { cx.strokeStyle = col; cx.lineWidth = wid; cx.setLineDash(dash); cx.beginPath();
        P.s.forEach((s, i) => { i ? cx.lineTo(X(s), Y(d[i])) : cx.moveTo(X(s), Y(d[i])); }); cx.stroke(); cx.setLineDash([]); };
      line(P.flat, '#888780', on ? 1.2 : 2.4, on ? [4, 3] : []);
      line(P.rim, '#D85A30', on ? 2.4 : 1.2, on ? [] : [4, 3]);
      cx.fillStyle = '#888780'; cx.fillText('flat caps', X(6.2), Y(P.flat[12]) + (prof === 'z' ? 14 : -6));
      cx.fillStyle = '#D85A30'; cx.fillText('Rose rim', X(6.2), Y(P.rim[12]) + (prof === 'z' ? -8 : -8));
      host.querySelector('[data-k="sum"]').innerHTML = on
        ? 'With the Rose rim: B<sub>0</sub> = 0.423 T (18.0 MHz); over the Ø10 × 10 mm sample 164 ppm peak-to-peak, σ ≈ 9 µT.'
        : 'Flat caps: B<sub>0</sub> = 0.445 T (18.9 MHz); over the Ø10 × 10 mm sample 3160 ppm peak-to-peak, σ ≈ 276 µT.';
    }
    let th = 0.75, ph = 1.15, rad = 190, auto = true, drag = false, lx = 0, ly = 0;
    const el = renderer.domElement;
    el.addEventListener('pointerdown', e => { drag = true; lx = e.clientX; ly = e.clientY; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', e => { if (!drag) return; th -= (e.clientX - lx) * 0.01; ph = Math.min(2.9, Math.max(0.2, ph - (e.clientY - ly) * 0.01)); lx = e.clientX; ly = e.clientY; });
    el.addEventListener('pointerup', () => drag = false);
    const q = k => host.querySelector(`[data-k="${k}"]`);
    q('rim').onchange = e => { on = e.target.checked; build(); };
    q('yoke').onchange = e => { yoke.visible = e.target.checked; };
    q('cut').onchange = e => { const c = e.target.checked ? [clip] : []; clipped.forEach(m => { m.clippingPlanes = c; m.needsUpdate = true; }); };
    q('zoom').oninput = e => rad = +e.target.value;
    q('spin').onclick = e => { auto = !auto; e.target.textContent = auto ? 'Pause rotation' : 'Auto-rotate'; };
    q('px').onclick = () => { prof = 'x'; chart(); };
    q('pz').onclick = () => { prof = 'z'; chart(); };
    window.addEventListener('resize', () => { renderer.setSize(W(), H); cam.aspect = W() / H; cam.updateProjectionMatrix(); chart(); });
    build();
    (function loop() {
      if (auto && !drag) th += 0.004;
      cam.position.set(rad * Math.sin(ph) * Math.sin(th), rad * Math.cos(ph), rad * Math.sin(ph) * Math.cos(th)); cam.lookAt(0, 0, 0);
      renderer.render(scene, cam); requestAnimationFrame(loop);
    })();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
