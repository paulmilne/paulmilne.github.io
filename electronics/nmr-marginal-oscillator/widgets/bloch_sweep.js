// bloch_sweep.js - interactive 3D Bloch-sphere view of isochromats in a swept field.
// Requires three.js (r128) loaded first. Renders into <div id="bloch-widget">.
(function () {
  function init() {
    const host = document.getElementById('bloch-widget');
    if (!host || typeof THREE === 'undefined') return;
    host.className = 'nmrw';
    host.innerHTML = `
      <div class="nmrw-view"></div>
      <div class="nmrw-row">
        <button data-k="frame">Switch to lab frame</button>
        <button data-k="pause">Pause</button>
        <button data-k="reset">Reset</button>
      </div>
      <div class="nmrw-sl"><span>Sweep amplitude</span><input type="range" data-s="A" min="0" max="3" step="0.1" value="1.5"><span data-v="A">1.5</span></div>
      <div class="nmrw-sl"><span>B<sub>1</sub> strength</span><input type="range" data-s="B" min="0" max="1.5" step="0.05" value="0.5"><span data-v="B">0.50</span></div>
      <div class="nmrw-sl"><span>Inhomogeneity</span><input type="range" data-s="I" min="0" max="1.5" step="0.05" value="0.4"><span data-v="I">0.40</span></div>
      <div class="nmrw-sl"><span>Field offset</span><input type="range" data-s="O" min="-2" max="2" step="0.1" value="0"><span data-v="O">0.0</span></div>
      <div class="nmrw-lg">
        <div><span class="nmrw-d" style="background:#888780"></span>B<sub>0</sub> (static field)</div>
        <div><span class="nmrw-d" style="background:#EF9F27"></span>B<sub>1</sub> (RF field)</div>
        <div><span class="nmrw-d" style="background:#1D9E75"></span>B<sub>eff</sub> (rotating frame)</div>
        <div><span class="nmrw-d" style="background:#AFA9EC"></span>Isochromats</div>
        <div><span class="nmrw-d" style="background:#D85A30"></span>Total M and its trail</div>
        <div><span class="nmrw-d" style="background:#378ADD"></span>Field sweep (chart)</div>
      </div>
      <canvas class="nmrw-chart"></canvas>
      <div class="nmrw-note">Drag to rotate. Time and field are scaled for visibility: T<sub>1</sub> = 6 s, T<sub>2</sub> = 3 s, sweep period 12.5 s; Larmor precession in the lab frame is slowed by ~10<sup>7</sup>.</div>`;
    const wrap = host.querySelector('.nmrw-view'), H = 360, W = () => wrap.clientWidth;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); renderer.setSize(W(), H);
    wrap.appendChild(renderer.domElement);
    const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(40, W() / H, 1, 1000);
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const RS = 40;
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(RS, 32, 20),
      new THREE.MeshBasicMaterial({ color: 0x888780, wireframe: true, transparent: true, opacity: 0.15 })));
    const eq = new THREE.Mesh(new THREE.TorusGeometry(RS, 0.25, 6, 96),
      new THREE.MeshBasicMaterial({ color: 0x888780, transparent: true, opacity: 0.6 }));
    eq.rotation.x = Math.PI / 2; scene.add(eq);
    const axM = new THREE.LineBasicMaterial({ color: 0x888780, transparent: true, opacity: 0.4 });
    [[1, 0], [0, 1]].forEach(a => scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(-RS * a[0], 0, -RS * a[1]), new THREE.Vector3(RS * a[0], 0, RS * a[1])]), axM)));
    const arr = (c, hl, hw) => { const a = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1, c, hl, hw); scene.add(a); return a; };
    const aB0 = arr(0x888780, 6, 3.5); aB0.setLength(RS * 1.45, 6, 3.5);
    const aB1 = arr(0xEF9F27, 5, 3), aBe = arr(0x1D9E75, 5, 3);
    const N = 9, iso = []; for (let k = 0; k < N; k++) iso.push(arr(0xAFA9EC, 3, 1.6));
    const aM = arr(0xD85A30, 7, 4);
    const TR = 500, trPos = new Float32Array(TR * 3), trGeo = new THREE.BufferGeometry();
    trGeo.setAttribute('position', new THREE.BufferAttribute(trPos, 3));
    scene.add(new THREE.Line(trGeo, new THREE.LineBasicMaterial({ color: 0xD85A30, transparent: true, opacity: 0.55 })));
    let trN = 0;
    const TWO = 2 * Math.PI, T1 = 6, T2 = 3, fm = 0.08, wL = TWO * 0.6;
    let sweepA = 1.5, b1 = 0.5, inh = 0.4, off = 0, lab = false, run = true, t = 0, M = [];
    const hist = [];
    function reset() { M = []; for (let k = 0; k < N; k++) M.push([0, 0, 1]); t = 0; trN = 0; hist.length = 0; }
    reset();
    const dc = tt => TWO * (off + sweepA * Math.sin(TWO * fm * tt));
    const f = (m, D, w1) => [D * m[1] - m[0] / T2, -D * m[0] + w1 * m[2] - m[1] / T2, -w1 * m[1] - (m[2] - 1) / T1];
    function step(h) {
      const w1 = TWO * b1;
      for (let k = 0; k < N; k++) {
        const dk = TWO * inh * (2 * k / (N - 1) - 1), D0 = dc(t) + dk, Dh = dc(t + h / 2) + dk, D1 = dc(t + h) + dk, m = M[k];
        const k1 = f(m, D0, w1), k2 = f(m.map((v, i) => v + h / 2 * k1[i]), Dh, w1),
              k3 = f(m.map((v, i) => v + h / 2 * k2[i]), Dh, w1), k4 = f(m.map((v, i) => v + h * k3[i]), D1, w1);
        M[k] = m.map((v, i) => v + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
      }
      t += h;
    }
    function toV(x, y, z) {
      let X = x, Y = z, Z = -y;
      if (lab) { const p = wL * t, c = Math.cos(p), s = Math.sin(p); const nx = X * c + Z * s, nz = -X * s + Z * c; X = nx; Z = nz; }
      return new THREE.Vector3(X, Y, Z);
    }
    function setA(a, v, scale, hl, hw) {
      const L = v.length() * scale; if (L < 0.5) { a.visible = false; return; }
      a.visible = true; a.setDirection(v.clone().normalize()); a.setLength(L, Math.min(hl, L * 0.4), hw);
    }
    function update() {
      let sx = 0, sy = 0, sz = 0;
      for (let k = 0; k < N; k++) { const m = M[k]; sx += m[0]; sy += m[1]; sz += m[2]; setA(iso[k], toV(m[0], m[1], m[2]), RS, 3, 1.6); }
      sx /= N; sy /= N; sz /= N;
      const vM = toV(sx, sy, sz); setA(aM, vM, RS, 7, 4);
      const p = vM.clone().multiplyScalar(RS);
      if (trN < TR) trN++; else trPos.copyWithin(0, 3);
      trPos[(trN - 1) * 3] = p.x; trPos[(trN - 1) * 3 + 1] = p.y; trPos[(trN - 1) * 3 + 2] = p.z;
      trGeo.setDrawRange(0, trN); trGeo.attributes.position.needsUpdate = true;
      if (b1 === 0) aB1.visible = false; else setA(aB1, toV(1, 0, 0), RS * 1.25 * Math.min(1, b1 / 0.8), 5, 3);
      const Dn = dc(t) / TWO;
      if (lab) aBe.visible = false; else setA(aBe, new THREE.Vector3(b1, Dn, 0).normalize(), RS * 1.2, 5, 3);
      hist.push([t, Dn, sy]); while (hist.length && hist[0][0] < t - 30) hist.shift();
    }
    const cv = host.querySelector('canvas.nmrw-chart'), cx = cv.getContext('2d');
    function strip() {
      const w = cv.clientWidth, h = 190, dpr = Math.min(window.devicePixelRatio, 2);
      if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = h * dpr; }
      cx.setTransform(dpr, 0, 0, dpr, 0, 0); cx.clearRect(0, 0, w, h);
      cx.strokeStyle = '#ccc'; cx.lineWidth = 0.5; cx.beginPath(); cx.moveTo(30, h / 2); cx.lineTo(w, h / 2); cx.stroke();
      cx.fillStyle = '#666'; cx.font = '12px sans-serif';
      cx.fillText('Field sweep (blue) and absorption m_y (coral), last 30 s', 30, 14); cx.fillText('0', 16, h / 2 + 4);
      if (hist.length < 2) return;
      const t0 = t - 30, X = tt => 30 + (w - 34) * (tt - t0) / 30;
      const tr = (idx, sc, col) => { cx.strokeStyle = col; cx.lineWidth = 1.5; cx.beginPath();
        hist.forEach((q, i) => { const y = h / 2 - q[idx] * sc; i ? cx.lineTo(X(q[0]), y) : cx.moveTo(X(q[0]), y); }); cx.stroke(); };
      tr(1, (h / 2 - 22) / 5, '#378ADD'); tr(2, (h / 2 - 22) / 1.0, '#D85A30');
    }
    let th = 0.9, ph = 1.15, rad = 150, drag = false, lx = 0, ly = 0;
    const el = renderer.domElement;
    el.addEventListener('pointerdown', e => { drag = true; lx = e.clientX; ly = e.clientY; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', e => { if (!drag) return; th -= (e.clientX - lx) * 0.01; ph = Math.min(2.9, Math.max(0.2, ph - (e.clientY - ly) * 0.01)); lx = e.clientX; ly = e.clientY; });
    el.addEventListener('pointerup', () => drag = false);
    const setters = { A: x => sweepA = x, B: x => b1 = x, I: x => inh = x, O: x => off = x };
    host.querySelectorAll('input[data-s]').forEach(s => s.addEventListener('input', () => {
      const k = s.dataset.s, x = +s.value; host.querySelector(`[data-v="${k}"]`).textContent = x.toFixed(k === 'B' || k === 'I' ? 2 : 1); setters[k](x); }));
    const fb = host.querySelector('[data-k="frame"]');
    fb.onclick = () => { lab = !lab; fb.textContent = lab ? 'Switch to rotating frame' : 'Switch to lab frame'; trN = 0; };
    const pb = host.querySelector('[data-k="pause"]');
    pb.onclick = () => { run = !run; pb.textContent = run ? 'Pause' : 'Resume'; };
    host.querySelector('[data-k="reset"]').onclick = reset;
    window.addEventListener('resize', () => { renderer.setSize(W(), H); cam.aspect = W() / H; cam.updateProjectionMatrix(); });
    let last = performance.now();
    (function loop(now) {
      const dt = Math.min(0.05, Math.max(0, (now - last) / 1000)); last = now;
      if (run) { const n = Math.max(1, Math.ceil(dt / 0.004)); for (let i = 0; i < n; i++) step(dt / n); update(); }
      strip();
      cam.position.set(rad * Math.sin(ph) * Math.sin(th), rad * Math.cos(ph), rad * Math.sin(ph) * Math.cos(th)); cam.lookAt(0, 0, 0);
      renderer.render(scene, cam); requestAnimationFrame(loop);
    })(last);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
