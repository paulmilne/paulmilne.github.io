// Small three.js views of tensors for "The Transformer, Part 1".
// mode "heads": a T x d activation splitting into h heads of width d_h.
// mode "kv": the key-value cache of an L-layer decoder growing one token at a time.

export function createTensorScene(THREE, opts = {}) {
  const W = opts.width || 720, H = opts.height || 420;
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(W, H);
  const canvas = renderer.domElement;
  canvas.style.width = "100%"; canvas.style.height = "auto"; canvas.style.touchAction = "none"; canvas.style.display = "block";
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, W / H, 0.1, 500);
  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const sun = new THREE.DirectionalLight(0xffffff, 0.7); sun.position.set(10, 20, 15); scene.add(sun);

  const group = new THREE.Group(); scene.add(group);
  const geo = new THREE.BoxGeometry(0.9, 0.9, 0.9);
  const headCols = [0xe76f51, 0x2a9d8f, 0x6fa8dc, 0xe9c46a, 0x9b5de5, 0x8ab17d];
  const mats = new Map();
  const mat = (c, o = 1) => { const k = c + ":" + o; if (!mats.has(k)) mats.set(k, new THREE.MeshLambertMaterial({color: c, transparent: o < 1, opacity: o})); return mats.get(k); };
  let cells = [];
  const clear = () => { for (const m of cells) group.remove(m); cells = []; };

  let az = -0.6, el = 0.45, zoom = 1, drag = null, centre = new THREE.Vector3(), dist = 30;
  canvas.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", e => {
    if (!drag) return;
    az -= (e.clientX - drag[0]) * 0.008; el = Math.max(-1.2, Math.min(1.4, el + (e.clientY - drag[1]) * 0.006));
    drag = [e.clientX, e.clientY]; render();
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("wheel", e => { e.preventDefault(); zoom = Math.max(0.4, Math.min(2.5, zoom * Math.exp(e.deltaY * 0.001))); render(); }, {passive: false});

  function render() {
    const d = dist * zoom;
    camera.position.set(centre.x + d * Math.cos(el) * Math.sin(az), centre.y + d * Math.sin(el), centre.z + d * Math.cos(el) * Math.cos(az));
    camera.lookAt(centre);
    renderer.render(scene, camera);
  }

  // ---- heads: X (T x d) -> (h, T, d_h) ---------------------------------------------------
  function heads({T = 6, d = 12, h = 3, s = 0}) {
    clear();
    const dh = Math.floor(d / h), gap = 2.2;
    const e = s < 0.5 ? 2 * s * s : 1 - Math.pow(-2 * s + 2, 2) / 2;   // ease in-out
    for (let t = 0; t < T; t++) for (let j = 0; j < d; j++) {
      const hd = Math.floor(j / dh), jj = j % dh;
      const x0 = j, z0 = 0;                                              // one wide matrix
      const x1 = jj + (dh + 1) * 0, z1 = -hd * gap;                     // stacked slabs
      const m = new THREE.Mesh(geo, mat(headCols[hd % headCols.length]));
      m.position.set(x0 + (x1 - x0) * e, T - 1 - t, z0 + (z1 - z0) * e);
      group.add(m); cells.push(m);
    }
    centre.set((d - 1) / 2 * (1 - e) + (dh - 1) / 2 * e, (T - 1) / 2, -(h - 1) * gap / 2 * e);
    dist = 1.45 * Math.max(d, T) + 5;
    render();
  }

  // ---- KV cache: L layers, K and V each (t x d) ----------------------------------------
  function kv({L = 4, d = 8, t = 3, Tmax = 8}) {
    clear();
    const layerGap = 3.2, kvGap = d + 2;
    for (let l = 0; l < L; l++) for (const [side, col] of [[0, 0x1d3557], [1, 0x2a9d8f]]) {
      for (let r = 0; r < Tmax; r++) for (let j = 0; j < d; j++) {
        const filled = r < t, newest = r === t - 1;
        if (!filled && j % 2) continue;  // outline future rows sparsely
        const m = new THREE.Mesh(geo, filled ? mat(newest ? 0xe76f51 : col) : mat(0xbbbbbb, 0.12));
        m.position.set(side * kvGap + j, Tmax - 1 - r, -l * layerGap);
        group.add(m); cells.push(m);
      }
    }
    centre.set((2 * d + 2) / 2 - 0.5, (Tmax - 1) / 2, -(L - 1) * layerGap / 2);
    dist = 1.25 * (2 * d + 2) + 6;
    render();
  }

  function dispose() { clear(); renderer.dispose(); geo.dispose(); for (const m of mats.values()) m.dispose(); }
  return {canvas, heads, kv, render, dispose};
}
