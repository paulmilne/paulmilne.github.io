// Gargantua in the browser: Kerr null geodesics integrated per pixel on the GPU,
// in 32-bit floats, with the same Hamiltonian, step-size rule and camera as
// src/compute/kerr.py. Includes a benchmark of the reader's own device.

const VS = `#version 300 es
in vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }`;

const FS = `#version 300 es
precision highp float; precision highp int;
uniform vec2 uRes; uniform float uA, uR0, uTh0, uFov, uPhi0, uH0, uDin, uDout, uTime;
uniform int uMax, uMode, uDebug;
out vec4 col;
const float PI = 3.14159265358979;

void deriv(float r, float th, float pr, float pth, float L, float a,
           out float dr, out float dth, out float dph, out float dpr, out float dpth) {
  float c = cos(th), s = sin(th);
  if (abs(s) < 1e-3) s = s < 0.0 ? -1e-3 : 1e-3;
  float r2 = r*r, a2 = a*a;
  float S = r2 + a2*c*c, D = r2 - 2.0*r + a2, P = r2 + a2 - a*L;
  float q = L/s - a*s;
  float A = D*pr*pr + pth*pth - P*P/D + q*q;
  float inv2S = 1.0/(2.0*S);
  dr = D*pr/S; dth = pth/S;
  dph = (a*P/D + q/s)/S;
  float Dp = 2.0*r - 2.0;
  float dAdr = Dp*pr*pr - 4.0*r*P/D + P*P*Dp/(D*D);
  float dAdth = 2.0*q*(-L/(s*s) - a)*c;
  dpr = -(dAdr*inv2S - A*inv2S*(2.0*r/S));
  dpth = -(dAdth*inv2S - A*inv2S*(-2.0*a2*c*s)/S);
}

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7)))*43758.5453); }

vec3 sky(float th, float ph) {
  ph = mod(ph + uPhi0, 2.0*PI); th = clamp(th, 0.0, PI);
  vec3 d = vec3(sin(th)*cos(ph), sin(th)*sin(ph), cos(th));
  // a tilted band of faint light, like a galaxy seen edge on
  float lat = dot(d, normalize(vec3(0.0, 0.45, 1.0)));
  vec3 c = vec3(0.006, 0.008, 0.014) + vec3(0.045, 0.040, 0.036)*exp(-lat*lat/0.012);
  // stars: one candidate per cell of a grid in (phi, cos theta)
  for (int k = 0; k < 2; k++) {
    float n = k == 0 ? 220.0 : 90.0;
    vec2 g = vec2(ph/(2.0*PI)*n*2.0, (d.z*0.5 + 0.5)*n);
    vec2 cell = floor(g), f = fract(g);
    float h = hash(cell + float(k)*17.0);
    if (h > (k == 0 ? 0.93 : 0.97)) {
      vec2 at = vec2(hash(cell + 3.1), hash(cell + 7.7));
      float b = exp(-dot(f - at, f - at)*(k == 0 ? 900.0 : 300.0));
      vec3 tint = mix(vec3(1.0, 0.85, 0.7), vec3(0.75, 0.85, 1.0), hash(cell + 11.3));
      c += tint*b*(0.8 + 3.0*(h - 0.9));
    }
  }
  return c;
}

vec3 heat(float t) {          // a black-body-like ramp: deep red, orange, then near white
  t = max(t, 0.0);
  vec3 c = t < 0.6 ? mix(vec3(0.75, 0.12, 0.02), vec3(1.0, 0.52, 0.16), t/0.6)
                   : mix(vec3(1.0, 0.52, 0.16), vec3(1.0, 0.93, 0.80), clamp((t - 0.6)/0.9, 0.0, 1.0));
  return c*pow(t, 1.5)*2.4;
}

void main() {
  float a = uA;
  // camera: pinhole carried by a zero-angular-momentum observer at (uR0, uTh0)
  vec2 pix = gl_FragCoord.xy;
  float f = 0.5*uRes.x/tan(uFov*0.5);
  float x = (pix.x - 0.5*uRes.x)/f, y = (pix.y - 0.5*uRes.y)/f;
  vec3 n = normalize(vec3(-1.0, y, x));
  float nr = n.x, nth = -n.y, nph = n.z;
  float r0 = uR0, th0 = uTh0, sn0 = sin(th0), cs0 = cos(th0);
  float S0 = r0*r0 + a*a*cs0*cs0, D0 = r0*r0 - 2.0*r0 + a*a;
  float Aq = (r0*r0 + a*a)*(r0*r0 + a*a) - a*a*D0*sn0*sn0;
  float alpha = sqrt(S0*D0/Aq), om = 2.0*a*r0/Aq, varpi = sqrt(Aq/S0)*sn0;
  float Ecam = alpha + om*varpi*nph;
  float pr = sqrt(S0/D0)*nr/Ecam, pth = sqrt(S0)*nth/Ecam, L = varpi*nph/Ecam;
  float r = r0, th = th0, ph = 0.0;
  float rh = 1.0 + sqrt(1.0 - a*a);
  int fate = 3; float rhit = 0.0; int steps = 0;
  for (int k = 0; k < 4000; k++) {
    if (k >= uMax) break;
    steps = k + 1;
    float h = uH0*max(r - rh, 0.05)*min(1.0, 20.0/r) + 0.002;
    h = min(h, 1.5)*clamp(5.0*abs(sin(th)), 0.02, 1.0);
    float k1r,k1t,k1p,k1pr,k1pt, k2r,k2t,k2p,k2pr,k2pt, k3r,k3t,k3p,k3pr,k3pt, k4r,k4t,k4p,k4pr,k4pt;
    deriv(r, th, pr, pth, L, a, k1r,k1t,k1p,k1pr,k1pt);
    deriv(r+0.5*h*k1r, th+0.5*h*k1t, pr+0.5*h*k1pr, pth+0.5*h*k1pt, L, a, k2r,k2t,k2p,k2pr,k2pt);
    deriv(r+0.5*h*k2r, th+0.5*h*k2t, pr+0.5*h*k2pr, pth+0.5*h*k2pt, L, a, k3r,k3t,k3p,k3pr,k3pt);
    deriv(r+h*k3r, th+h*k3t, pr+h*k3pr, pth+h*k3pt, L, a, k4r,k4t,k4p,k4pr,k4pt);
    float rn = r + h/6.0*(k1r + 2.0*k2r + 2.0*k3r + k4r);
    float tn = th + h/6.0*(k1t + 2.0*k2t + 2.0*k3t + k4t);
    float pn = ph + h/6.0*(k1p + 2.0*k2p + 2.0*k3p + k4p);
    pr += h/6.0*(k1pr + 2.0*k2pr + 2.0*k3pr + k4pr);
    pth += h/6.0*(k1pt + 2.0*k2pt + 2.0*k3pt + k4pt);
    if ((th - 0.5*PI)*(tn - 0.5*PI) < 0.0) {
      float fr = (0.5*PI - th)/(tn - th), rc = r + fr*(rn - r);
      if (rc >= uDin && rc <= uDout) { fate = 2; rhit = rc; ph = ph + fr*(pn - ph); r = rn; th = tn; break; }
    }
    r = rn; th = tn; ph = pn;
    if (th < 0.0) { th = -th; pth = -pth; ph += PI; }                 // a ray through the pole comes out on the far side
    else if (th > PI) { th = 2.0*PI - th; pth = -pth; ph += PI; }
    if (r < rh*1.01 || isnan(r)) { fate = 1; break; }
    if (r > 60.0) { fate = 0; break; }
  }
  if (uDebug == 1) { col = vec4(float(fate)/3.0, th/PI, fract(ph/(2.0*PI)), float(min(steps, 1020))/1020.0); return; }
  vec3 c = vec3(0.0);
  if (fate == 0) c = sky(th, ph);
  else if (fate == 2) {
    // thin disk: temperature profile of a standard accretion disk, T ~ r^(-3/4) (1 - sqrt(r_in/r))^(1/4)
    float T = pow(rhit/uDin, -0.75)*pow(max(1.0 - sqrt(uDin/rhit), 0.0), 0.25)*1.6;
    float Om = 1.0/(pow(rhit, 1.5) + a);
    float ang = ph - Om*uTime;                               // pattern carried round by the orbiting gas
    float tex = 0.75 + 0.25*sin(rhit*7.0 + 3.0*sin(ang*3.0 + rhit))*sin(ang*5.0 - rhit*2.0);
    if (uMode == 1) {
      float ut = (pow(rhit, 1.5) + a)/(pow(rhit, 0.75)*sqrt(pow(rhit, 1.5) - 3.0*sqrt(rhit) + 2.0*a));
      float g = 1.0/(Ecam*ut*(1.0 - Om*L));                 // observed / emitted frequency
      c = heat(T*g)*pow(g, 3.0)*tex;
    } else {
      c = heat(T)*tex;
    }
  }
  c = 1.0 - exp(-1.3*c);                                   // tone map
  col = vec4(pow(c, vec3(1.0/1.5)), 1.0);
}`;

export async function mountGargantua(host) {
  host.innerHTML = "";
  const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const cv = el("canvas", "gg-canvas"); const side = el("div", "gg-side"); const wrap = el("div", "gg-wrap");
  const fpsEl = el("div", "gg-fps"); const left = el("div"); left.append(cv, fpsEl); wrap.append(left, side); host.append(wrap);
  const gl = cv.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true });
  if (!gl) { host.append(el("p", null, "This interactive needs WebGL 2, which this browser does not provide.")); return host; }
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const U = {}; for (const n of ["uRes", "uA", "uR0", "uTh0", "uFov", "uPhi0", "uH0", "uDin", "uDout", "uTime", "uMax", "uMode", "uDebug"]) U[n] = gl.getUniformLocation(prog, n);

  const risco = (a) => { const z1 = 1 + Math.cbrt(1 - a * a) * (Math.cbrt(1 + a) + Math.cbrt(1 - a)); const z2 = Math.sqrt(3 * a * a + z1 * z1); return 3 + z2 - Math.sqrt((3 - z1) * (3 + z1 + 2 * z2)); };
  const P = { a: 0.6, r0: 25, incl: 85, fov: 60, scale: 0.6, mode: 0, spin: true, phi0: 0, t: 0 };
  const info = gl.getExtension("WEBGL_debug_renderer_info");
  const gpuName = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);

  function setUniforms(w, h, debug = 0) {
    gl.uniform2f(U.uRes, w, h); gl.uniform1f(U.uA, P.a); gl.uniform1f(U.uR0, P.r0); gl.uniform1f(U.uTh0, (P.incl * Math.PI) / 180);
    gl.uniform1f(U.uFov, (P.fov * Math.PI) / 180); gl.uniform1f(U.uPhi0, P.phi0); gl.uniform1f(U.uH0, 0.05); gl.uniform1f(U.uDin, risco(P.a));
    gl.uniform1f(U.uDout, 12); gl.uniform1f(U.uTime, P.t); gl.uniform1i(U.uMax, 2000); gl.uniform1i(U.uMode, P.mode); gl.uniform1i(U.uDebug, debug);
  }
  let W = 0, H = 0;
  function size() {
    const cw = Math.min(left.clientWidth || 640, 900); const dpr = window.devicePixelRatio || 1;
    cv.style.width = cw + "px"; cv.style.height = cw / 2 + "px";
    W = Math.max(64, Math.round(cw * dpr * P.scale)); H = Math.round(W / 2); cv.width = W; cv.height = H;
  }
  let last = performance.now(), frames = 0, acc = 0, dirty = true;
  function frame(now) {
    const dt = now - last; last = now;
    if (P.spin) { P.t += dt * 0.004; dirty = true; }
    if (dirty && visible) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H); setUniforms(W, H); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      frames++; acc += dt; dirty = false;
      if (acc > 600) { fpsEl.textContent = `${(1000 * frames / acc).toFixed(1)} frames a second at ${W}×${H} (${((W * H * frames) / acc / 1e3).toFixed(1)} million rays a second, including display)`; frames = 0; acc = 0; }
    }
    requestAnimationFrame(frame);
  }
  let visible = true; new IntersectionObserver((e) => (visible = e[0].isIntersecting)).observe(cv);

  // controls
  const slider = (label, key, min, max, step, fmt) => {
    const l = el("label"), s = el("span"), i = el("input"); Object.assign(i, { type: "range", min, max, step, value: P[key] });
    const show = () => (s.textContent = `${label}: ${fmt(P[key])}`); i.oninput = () => { P[key] = +i.value; show(); if (key === "scale") size(); dirty = true; }; show(); l.append(s, i); return l;
  };
  const modeSel = el("select"); for (const [v, t] of [[0, "as in the film (no Doppler shift)"], [1, "as a real observer would see it"]]) { const o = el("option", null, t); o.value = v; modeSel.append(o); }
  modeSel.onchange = () => { P.mode = +modeSel.value; dirty = true; };
  const spin = el("label", "gg-check"); const sc = el("input"); sc.type = "checkbox"; sc.checked = true; sc.onchange = () => { P.spin = sc.checked; dirty = true; }; spin.append(sc, " Rotate the disk");
  const bench = el("button", "gg-go", "Benchmark this device"); const out = el("div", "gg-out");
  side.append(slider("Black hole spin a/M", "a", 0, 0.99, 0.01, (v) => v.toFixed(2)), slider("Camera distance", "r0", 8, 50, 0.5, (v) => `${v} M`),
    slider("Camera above the disk", "incl", 60, 120, 0.5, (v) => `${(90 - v).toFixed(1)}°`), slider("Resolution", "scale", 0.2, 1, 0.05, (v) => `${Math.round(v * 100)}% of the screen`),
    el("div", "gg-lab", "Disk colours"), modeSel, spin, bench, out, el("div", "gg-dim", `Drag the image to look around. Graphics processor reported by the browser: ${gpuName}`));

  // drag to turn the sky and tilt the camera
  let drag = null;
  cv.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, phi0: P.phi0, incl: P.incl }; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", (e) => { if (!drag) return; P.phi0 = drag.phi0 - (e.clientX - drag.x) * 0.006; P.incl = Math.min(120, Math.max(60, drag.incl + (e.clientY - drag.y) * 0.15)); dirty = true; });
  cv.addEventListener("pointerup", () => (drag = null));

  // benchmark: render IMAX-shaped frames offscreen and time them, waiting for the GPU each time
  bench.onclick = async () => {
    bench.disabled = true; out.textContent = "Measuring…"; await new Promise((r) => setTimeout(r, 50));
    const bw = 1920, bh = 960, tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, bw, bh, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.viewport(0, 0, bw, bh); const px = new Uint8Array(4);
    // one debug pass to count integration steps
    setUniforms(bw, bh, 1); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); const all = new Uint8Array(bw * bh * 4); gl.readPixels(0, 0, bw, bh, gl.RGBA, gl.UNSIGNED_BYTE, all);
    let st = 0; for (let i = 3; i < all.length; i += 4) st += all[i]; const meanSteps = (st / (bw * bh)) * (1020 / 255);
    setUniforms(bw, bh, 0); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);   // warm up
    const times = [];
    for (let k = 0; k < 6; k++) { const t0 = performance.now(); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); times.push(performance.now() - t0); await new Promise((r) => setTimeout(r, 0)); }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.deleteFramebuffer(fb); gl.deleteTexture(tex);
    times.sort((x, y) => x - y); const ms = times[Math.floor(times.length / 2)];
    const raysPerS = (bw * bh) / (ms / 1000), stepsPerS = raysPerS * meanSteps, imax = 23e6 / raysPerS;
    out.innerHTML = `<b>${ms.toFixed(0)} ms</b> for a ${bw}×${bh} frame: <b>${(raysPerS / 1e6).toFixed(1)} million rays a second</b>, averaging ${meanSteps.toFixed(0)} integration steps each, so ${(stepsPerS / 1e9).toFixed(2)} billion photon-steps a second.<br>
      A 23-million-pixel IMAX frame of single rays would take <b>${imax < 1 ? (imax * 1000).toFixed(0) + " ms" : imax.toFixed(1) + " s"}</b> here.
      For comparison: GRay, a 2013 GPU code, managed about 1 billion photon-steps a second; DNGR took from 30 minutes to several hours per frame on 10 CPU cores, tracing ray bundles rather than single rays.`;
    bench.disabled = false; dirty = true;
  };

  new ResizeObserver(() => { size(); dirty = true; }).observe(left); size(); requestAnimationFrame(frame);
  // expose a hook for automated checks
  host.gargantua = { gl, render: (w, h, debug) => { gl.bindFramebuffer(gl.FRAMEBUFFER, null); cv.width = w; cv.height = h; gl.viewport(0, 0, w, h); setUniforms(w, h, debug); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); const a = new Uint8Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, a); return a; }, P };
  return host;
}
