// Listening Brain AR: the poster figure plays.
// Image tracking (MindAR, on-device) anchors a canvas overlay exactly on the printed figure and a 3D brain above it.
// Data: session 2 of one listener; per network, 3 hearings of each 30-s excerpt (whole-brain signal and the response
// shared by all excerpts removed), shifted back 5 s onto the music clock; lock-on moments = cluster-tested wavelet
// coherence between hearings (p < .05). Built from claudeNM/results/report/web/data.json by scripts/prep_data.py.
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

const $ = (s) => document.querySelector(s);
const REG = {   // key -> [plain label, short label, colour]
  "auditory": ["Hearing", "Hearing", "#2C7FA8"],
  "Visual": ["Vision", "Vision", "#A9A29C"],
  "Somatomotor": ["Body movement", "Body", "#5FB3D4"],
  "Dorsal attention": ["Outward attention", "Outward", "#8B78C2"],
  "Salience (ventral attention)": ["What stands out", "Salience", "#E09A36"],
  "Frontoparietal control": ["Goal-directed control", "Control", "#5E9E6D"],
  "Default mode": ["Inward thought", "Inward", "#C2476C"],
  "Limbic": ["Limbic", "Limbic", "#A3B547"],
};
// print colours of the poster figure (the overlay draws in these so it matches the paper)
const PRINT = { "auditory": "#2E6F8E", "Visual": "#8C8686", "Somatomotor": "#5FA3BF", "Dorsal attention": "#6B5B95",
  "Salience (ventral attention)": "#D08C2E", "Frontoparietal control": "#4E7D5B", "Default mode": "#A23B5A", "Limbic": "#8A9A3B" };
const YEO = { 1: "Visual", 2: "Somatomotor", 3: "Dorsal attention", 4: "Salience (ventral attention)", 5: "Limbic", 6: "Frontoparietal control", 7: "Default mode" };
const ROWS = Object.keys(REG);

let DATA, BRAIN, LAY, tune, traces, sdScale;
const audio = new Audio(); audio.loop = true; audio.preload = "auto"; audio.crossOrigin = "anonymous";
const DEBUG_T = new URLSearchParams(location.search).get("t"), DEBUG_ROT = new URLSearchParams(location.search).get("rot");
let userPaused = false, mode = null, mindar = null, r3d = null, foundOnce = false, lastLockKey = "";

// ---------------------------------------------------------------- data helpers
function gauss(x, sigma) {            // scipy.ndimage.gaussian_filter1d, mode "reflect", truncate 4
  const r = Math.ceil(4 * sigma), k = []; let s = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * (i / sigma) ** 2); k.push(v); s += v; }
  const n = x.length, y = new Float32Array(n);
  for (let i = 0; i < n; i++) { let a = 0; for (let j = -r; j <= r; j++) { let q = i + j; if (q < 0) q = -q - 1; if (q >= n) q = 2 * n - q - 1; a += k[j + r] * x[q]; } y[i] = a / s; }
  return y;
}
function prepTune(t) {
  const sig = LAY.smooth_sigma_s / LAY.tr; const out = {}; const all = [];
  for (const key of ROWS) { const hs = t.hearings[key].map(h => gauss(h, sig)); out[key] = { hs }; hs.forEach(h => all.push(...h)); }
  const m = all.reduce((a, b) => a + b, 0) / all.length; const sd = Math.sqrt(all.reduce((a, b) => a + (b - m) ** 2, 0) / all.length);
  sdScale = LAY.sd_factor * sd;
  for (const key of ROWS) {
    out[key].hs = out[key].hs.map(h => h.map(v => v / sdScale));
    out[key].mean = out[key].hs[0].map((_, i) => (out[key].hs[0][i] + out[key].hs[1][i] + out[key].hs[2][i]) / 3);
  }
  return out;
}
function valueAt(arr, t) {           // linear interpolation on the music clock
  const T = DATA.music_t; if (t <= T[0]) return arr[0]; if (t >= T[T.length - 1]) return arr[arr.length - 1];
  const dt = T[1] - T[0]; const f = (t - T[0]) / dt; const i = Math.floor(f); return arr[i] + (arr[i + 1] - arr[i]) * (f - i);
}
const activeLocks = (t) => tune.locks.filter(l => REG[l.region] && t >= l.start && t <= l.start + l.dur);

// ---------------------------------------------------------------- brain
function buildBrain() {
  const group = new THREE.Group(); const meshes = [];
  const toThree = (x, y, z) => [-y, z, -x];          // RAS -> three: left lateral surface faces the viewer
  for (const h of BRAIN.hemis) {
    const n = h.v.length / 3, pos = new Float32Array(n * 3), col = new Float32Array(n * 3), reg = new Int8Array(n), shade = new Float32Array(n);
    for (let i = 0; i < n; i++) { const p = toThree(h.v[3 * i], h.v[3 * i + 1], h.v[3 * i + 2]); pos.set(p, 3 * i); }
    const net = h.labels.network, aud = h.labels.auditory;
    for (let i = 0; i < n; i++) {
      let key = YEO[net[i]] || null; if (aud && aud[i] > 0) key = "auditory";
      reg[i] = key ? ROWS.indexOf(key) : -1; shade[i] = 0.80 + 0.20 * (1 - h.sulc[i]);
    }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3)); g.setIndex(h.f); g.computeVertexNormals(); g.computeBoundingBox();
    const m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true }));
    meshes.push({ mesh: m, reg, shade, col }); group.add(m);
  }
  const box = new THREE.Box3().setFromObject(group); const c = box.getCenter(new THREE.Vector3()); const size = box.getSize(new THREE.Vector3());
  group.children.forEach(m => m.position.sub(c));
  const holder = new THREE.Group(); holder.add(group); holder.userData = { meshes, inner: group, span: Math.max(size.x, size.y) };
  return holder;
}
const GREY = new THREE.Color("#B3ADA8"), WHITE = new THREE.Color("#FFFFFF");
const regCols = ROWS.map(k => new THREE.Color(REG[k][2]));
function paintBrain(brain, t, time) {
  const act = ROWS.map(k => valueAt(traces[k].mean, t)); const locks = activeLocks(t).map(l => ROWS.indexOf(l.region));
  const pulse = 0.5 + 0.5 * Math.sin(time * 7.0);
  const cols = ROWS.map((k, r) => {
    const L = Math.min(1, Math.max(0, 0.5 + act[r] / 2.6)); const w = 0.45 + 0.55 * L;
    const c = GREY.clone().lerp(regCols[r], w).multiplyScalar(0.62 + 0.62 * L);
    if (L > 0.6) c.lerp(WHITE, (L - 0.6) * 0.45);
    if (locks.includes(r)) c.lerp(WHITE, 0.25 + 0.35 * pulse);
    return c;
  });
  for (const { mesh, reg, shade, col } of brain.userData.meshes) {
    for (let i = 0; i < reg.length; i++) {
      const r = reg[i]; const c = r >= 0 ? cols[r] : GREY; const s = shade[i] * (r >= 0 ? 1 : 0.92);
      col[3 * i] = c.r * s; col[3 * i + 1] = c.g * s; col[3 * i + 2] = c.b * s;
    }
    mesh.geometry.attributes.color.needsUpdate = true;
  }
}

// ---------------------------------------------------------------- overlay (canvas registered to the printed figure)
const OW = 2048; let OH, ocv, octx, otex, staticLayer;
function px(fx) { return fx * OW; } function py(fy) { return fy * OH; }
function rowY(r, v) { const [yt, h] = LAY.row_boxes[r]; const [lo, hi] = LAY.ylim; return py(yt + (hi - v) / (hi - lo) * h); }
function tx(t) { return px(LAY.x0 + (t - LAY.t0) / (LAY.t1 - LAY.t0) * (LAY.x1 - LAY.x0)); }
const SPEC_STOPS = ["#FFFFFF", "#D9E4E8", "#7FA3B3", "#2B5468", "#14222B"].map(h => new THREE.Color(h));
function specColour(u) { u = Math.min(1, Math.max(0, u)) * (SPEC_STOPS.length - 1); const i = Math.min(SPEC_STOPS.length - 2, Math.floor(u)); const c = SPEC_STOPS[i].clone().lerp(SPEC_STOPS[i + 1], u - i); return c; }
function drawStatic() {
  // the printed figure shows LAY.tune; for any other piece, repaint the data area with that piece
  staticLayer = document.createElement("canvas"); staticLayer.width = OW; staticLayer.height = OH; const g = staticLayer.getContext("2d");
  if (tune.id === LAY.tune) return;
  const xL = px(LAY.x0 - 0.006), xR = px(LAY.x1 + 0.012), yT = py(LAY.spec[0] - 0.012), yB = py(LAY.row_boxes[ROWS.length - 1][0] + LAY.row_boxes[0][1] + 0.004);
  g.fillStyle = "#FFFFFF"; g.fillRect(xL, yT, xR - xL, yB - yT);
  const [sx, sy, sw, sh] = LAY.subtitle; g.fillRect(px(sx), py(sy), px(sw), py(sh));
  g.fillStyle = "#7A7474"; g.font = `italic ${Math.round(OH * 0.0205)}px system-ui, sans-serif`; g.textBaseline = "middle";
  g.fillText(`“${tune.name}” (${tune.genre}), a 30-s excerpt heard three times, 10–25 min apart. Session 2, one listener.`, px(0.012), py(sy + sh / 2));
  // spectrogram
  const mel = tune.mel, nb = mel.length, nt = mel[0].length, flat = mel.flat().sort((a, b) => a - b);
  const lo = flat[Math.floor(0.25 * flat.length)], hi = flat[Math.floor(0.995 * (flat.length - 1))];
  const im = new ImageData(nt, nb);
  for (let b = 0; b < nb; b++) for (let i = 0; i < nt; i++) { const c = specColour((mel[b][i] - lo) / (hi - lo)); const o = 4 * ((nb - 1 - b) * nt + i); im.data[o] = c.r * 255; im.data[o + 1] = c.g * 255; im.data[o + 2] = c.b * 255; im.data[o + 3] = 255; }
  const tmp = document.createElement("canvas"); tmp.width = nt; tmp.height = nb; tmp.getContext("2d").putImageData(im, 0, 0);
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high"; g.drawImage(tmp, px(LAY.x0), py(LAY.spec[0]), px(LAY.x1 - LAY.x0), py(LAY.spec[1] - LAY.spec[0]));
  // traces and lock-on boxes
  const T = DATA.music_t; const lw = OW / 1000;
  ROWS.forEach((key, r) => {
    const col = PRINT[key];
    g.strokeStyle = "#D8D4D4"; g.lineWidth = 0.6 * lw; g.beginPath(); g.moveTo(tx(0), rowY(r, 0)); g.lineTo(tx(30), rowY(r, 0)); g.stroke();
    for (const l of tune.locks.filter(l => l.region === key)) {
      const x0 = tx(l.start), x1 = tx(l.start + l.dur), y0 = rowY(r, 2.3), y1 = rowY(r, -2.4);
      g.fillStyle = col + "22"; g.strokeStyle = col; g.lineWidth = 1.6 * lw; g.beginPath(); g.roundRect(x0, y0, x1 - x0, y1 - y0, 8); g.fill(); g.stroke();
      g.fillStyle = col; g.font = `600 ${Math.round(OH * 0.018)}px system-ui, sans-serif`; g.textAlign = "right"; g.textBaseline = "top";
      g.fillText(`lock-on  p = ${l.p.toFixed(3).replace("0.", ".")}`, x1 - 6, y0 + 4); g.textAlign = "left";
    }
    const path = (arr, w, a) => { g.globalAlpha = a; g.strokeStyle = col; g.lineWidth = w; g.beginPath(); let st = false;
      for (let i = 0; i < T.length; i++) { if (T[i] < -0.5 || T[i] > 30.5) continue; const X = tx(T[i]), Y = rowY(r, Math.max(-2.5, Math.min(2.4, arr[i]))); st ? g.lineTo(X, Y) : g.moveTo(X, Y); st = true; }
      g.stroke(); g.globalAlpha = 1; };
    g.save(); g.beginPath(); g.rect(tx(0), py(LAY.row_boxes[r][0]), tx(30) - tx(0), py(LAY.row_boxes[r][1])); g.clip();
    traces[key].hs.forEach(h => path(h, 0.9 * lw, 0.55)); path(traces[key].mean, 2.4 * lw, 1); g.restore();
  });
}
function drawOverlay(t, time) {
  octx.clearRect(0, 0, OW, OH); octx.drawImage(staticLayer, 0, 0);
  const X = tx(Math.min(30, Math.max(0, t))); const yT = py(LAY.spec[0]), yB = py(LAY.row_boxes[ROWS.length - 1][0] + LAY.row_boxes[0][1]);
  const pulse = 0.5 + 0.5 * Math.sin(time * 7.0);
  for (const l of activeLocks(t)) {      // glowing lock-on box
    const r = ROWS.indexOf(l.region), col = PRINT[l.region]; const x0 = tx(l.start), x1 = tx(l.start + l.dur), y0 = rowY(r, 2.3), y1 = rowY(r, -2.4);
    octx.save(); octx.shadowColor = col; octx.shadowBlur = 30 + 25 * pulse; octx.strokeStyle = col; octx.lineWidth = 7;
    octx.fillStyle = col + Math.round(40 + 50 * pulse).toString(16).padStart(2, "0"); octx.beginPath(); octx.roundRect(x0, y0, x1 - x0, y1 - y0, 10); octx.fill(); octx.stroke(); octx.restore();
  }
  // played part of the spectrogram is lightly tinted, then the playhead
  octx.fillStyle = "rgba(232,116,47,0.10)"; octx.fillRect(tx(0), yT, X - tx(0), py(LAY.spec[1]) - yT);
  octx.save(); octx.shadowColor = "rgba(232,116,47,0.9)"; octx.shadowBlur = 18; octx.strokeStyle = "#E8742F"; octx.lineWidth = 6;
  octx.beginPath(); octx.moveTo(X, yT - 6); octx.lineTo(X, yB); octx.stroke(); octx.restore();
  ROWS.forEach((key, r) => {          // a bead rides each network's mean course
    const v = Math.max(-2.5, Math.min(2.4, valueAt(traces[key].mean, t))); const Y = rowY(r, v);
    octx.beginPath(); octx.arc(X, Y, 15, 0, 2 * Math.PI); octx.fillStyle = "#FFFFFF"; octx.fill();
    octx.beginPath(); octx.arc(X, Y, 10, 0, 2 * Math.PI); octx.fillStyle = PRINT[key]; octx.fill();
  });
  otex.needsUpdate = true;
}

// ---------------------------------------------------------------- scene content shared by AR and 3D modes
let content, brain, figurePlane;
function buildContent() {
  content = new THREE.Group(); const h = 1 / LAY.aspect; OH = Math.round(OW * h);
  ocv = document.createElement("canvas"); ocv.width = OW; ocv.height = OH; octx = ocv.getContext("2d");
  otex = new THREE.CanvasTexture(ocv); otex.colorSpace = THREE.SRGBColorSpace; otex.anisotropy = 4;
  const ov = new THREE.Mesh(new THREE.PlaneGeometry(1, h), new THREE.MeshBasicMaterial({ map: otex, transparent: true, depthWrite: false }));
  ov.position.z = 0.002; ov.renderOrder = 2; content.add(ov);
  brain = buildBrain(); const bb = LAY.brain_box;   // over the printed brain renders, floating towards the viewer
  const bx = bb[0] + bb[2] / 2 - 0.5, by = 0.5 * h - (bb[1] + bb[3] / 2) * h;
  brain.position.set(bx - 0.01, by + 0.01, 0.10); brain.scale.setScalar(0.27 / brain.userData.span); content.add(brain);
  const sh = new THREE.Mesh(new THREE.CircleGeometry(0.15, 48), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.16, depthWrite: false }));
  sh.position.set(bx - 0.01, by - 0.02, 0.004); sh.scale.set(1, 0.55, 1); content.add(sh);
  content.add(new THREE.HemisphereLight(0xffffff, 0x8a8078, 2.2)); const d = new THREE.DirectionalLight(0xffffff, 1.4); d.position.set(0.4, 0.8, 1.2); content.add(d);
}
function figureBacking() {             // 3D mode only: the printed figure itself
  const h = 1 / LAY.aspect; const tex = new THREE.TextureLoader().load("data/hero_target.jpg"); tex.colorSpace = THREE.SRGBColorSpace;
  const g = new THREE.Group();
  const card = new THREE.Mesh(new THREE.PlaneGeometry(1.04, h + 0.04), new THREE.MeshBasicMaterial({ color: 0xE3E3E3 })); card.position.z = -0.003;
  const pic = new THREE.Mesh(new THREE.PlaneGeometry(1, h), new THREE.MeshBasicMaterial({ map: tex })); g.add(card, pic); return g;
}
let rise = 0, t0clock = performance.now();
function frame() {
  const time = (performance.now() - t0clock) / 1000; const t = DEBUG_T !== null ? +DEBUG_T : (audio.currentTime || 0);
  rise = Math.min(1, rise + 0.035); const e = 1 - Math.pow(1 - rise, 3);
  brain.userData.inner.rotation.y = DEBUG_ROT !== null ? +DEBUG_ROT : Math.sin(time * 0.35) * 0.75; brain.position.z = 0.03 + 0.08 * e; brain.scale.setScalar((0.27 / brain.userData.span) * (0.2 + 0.8 * e));
  paintBrain(brain, t, time); drawOverlay(t, time); updateHud(t, time);
}

// ---------------------------------------------------------------- HUD
function fmt(s) { s = Math.max(0, s); return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }
function buildHud() {
  $("#meters").innerHTML = ROWS.map(k => `<div class="m" data-k="${k}" style="color:${REG[k][2]}"><div class="bar"><div class="mid"></div><div class="fill" style="background:${REG[k][2]}"></div></div><div class="lab">${REG[k][1]}</div></div>`).join("");
  $("#tunes").innerHTML = DATA.tunes.map((t, i) => `<button class="chip" data-i="${i}" aria-pressed="${t.id === tune.id}">${t.name}</button>`).join("");
  $("#tunes").onclick = (ev) => { const b = ev.target.closest(".chip"); if (b) selectTune(+b.dataset.i, true); };
  $("#fold").onclick = () => { const sh = $(".sheet"); sh.classList.toggle("min"); $("#fold").textContent = sh.classList.contains("min") ? "unfold ▴" : "fold ▾"; };
  $("#play").onclick = () => { if (audio.paused) { userPaused = false; audio.play(); } else { userPaused = true; audio.pause(); } };
  $("#bar").onclick = (ev) => { const r = ev.currentTarget.getBoundingClientRect(); audio.currentTime = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)) * tune.duration; };
  audio.onplay = audio.onpause = () => $("#playIcon").setAttribute("d", audio.paused ? "M5 3l12 7-12 7z" : "M5 3h4v14H5zM11 3h4v14h-4z");
}
function refreshBar() {
  $("#bar").querySelectorAll(".lk").forEach(e => e.remove());
  for (const l of tune.locks.filter(l => REG[l.region])) {
    const d = document.createElement("div"); d.className = "lk"; d.style.left = `${100 * l.start / tune.duration}%`; d.style.width = `${100 * l.dur / tune.duration}%`;
    d.style.background = REG[l.region][2]; d.title = REG[l.region][0]; $("#bar").prepend(d);
  }
  $("#credit").innerHTML = `Music: “${tune.title}” · ${tune.artist} · <a href="${tune.url}" target="_blank" rel="noopener">Internet Archive</a> · ${tune.licence}. 30-s excerpt as heard in the scanner.`;
  document.querySelectorAll(".chip").forEach(c => c.setAttribute("aria-pressed", DATA.tunes[+c.dataset.i].id === tune.id));
}
function summary() {
  const ls = tune.locks.filter(l => REG[l.region]);
  if (!ls.length) return "No lock-on moment in this excerpt.";
  const names = [...new Set(ls.map(l => REG[l.region][0].toLowerCase()))];
  return `This excerpt: ${ls.length} lock-on moment${ls.length > 1 ? "s" : ""} (${names.join(", ")}). The brain is shown 5 s behind the music, the time blood flow takes to respond.`;
}
function updateHud(t, time) {
  const act = ROWS.map(k => valueAt(traces[k].mean, t)); const locks = activeLocks(t);
  document.querySelectorAll(".m").forEach((m, r) => {
    const v = Math.max(-2.4, Math.min(2.4, act[r])) / 2.4; const f = m.querySelector(".fill");
    f.style.height = `${Math.abs(v) * 50}%`; f.style.bottom = v >= 0 ? "50%" : `${50 - Math.abs(v) * 50}%`;
    m.classList.toggle("lock", locks.some(l => l.region === ROWS[r]));
  });
  const key = locks.map(l => l.region).join("|");
  if (key !== lastLockKey) {
    if (locks.length) {
      $("#caption").innerHTML = locks.map(l => `<span class="k" style="color:${REG[l.region][2]}">${REG[l.region][0]}</span>`).join(" and ") +
        ` lock${locks.length > 1 ? "" : "s"} on: the same course on all three hearings (p = ${locks.map(l => l.p.toFixed(3).replace("0.", ".")).join(", ")}).`;
      if (navigator.vibrate && !audio.paused) navigator.vibrate(35);
    } else $("#caption").textContent = summary();
    lastLockKey = key;
  }
  $(".prog").style.width = `${100 * t / tune.duration}%`; $("#time").textContent = fmt(t);
}
function selectTune(i, play) {
  tune = DATA.tunes[i]; traces = prepTune(tune); drawStatic(); refreshBar(); lastLockKey = "-";
  const was = !audio.paused; audio.src = tune.audio; audio.currentTime = 0;
  if (play || was) { userPaused = false; audio.play().catch(() => {}); }
}

// ---------------------------------------------------------------- modes
function showUi() { $("#splash").classList.add("hidden"); $("#top").classList.remove("hidden"); $("#hud").classList.remove("hidden"); }
function setTracking(on, text) { $("#trackDot").classList.toggle("on", on); $("#trackTxt").textContent = text; $("#hint").classList.toggle("hide", on); }
async function startAR() {
  if (r3d) stop3D();
  mode = "ar"; document.body.classList.add("ar"); $("#modeBtn").textContent = "3D view"; $("#hint").classList.remove("hidden"); setTracking(false, "Starting camera…");
  const { MindARThree } = await import("mindar-image-three");
  mindar = new MindARThree({ container: $("#stage"), imageTargetSrc: "data/targets.mind", maxTrack: 1, uiLoading: "no", uiScanning: "no", uiError: "no",
    filterMinCF: 0.0005, filterBeta: 0.004 });
  const { renderer, scene, camera } = mindar; renderer.outputColorSpace = THREE.SRGBColorSpace;
  const anchor = mindar.addAnchor(0); anchor.group.add(content);
  anchor.onTargetFound = () => { setTracking(true, "Figure found"); rise = 0;
    if (!foundOnce || (!userPaused && audio.paused)) { foundOnce = true; if (!userPaused) audio.play().catch(() => {}); } };
  anchor.onTargetLost = () => setTracking(false, "Looking for the figure");
  try { await mindar.start(); } catch (e) { return fail("The camera could not start (" + (e && e.message ? e.message : e) + "). You can still use the 3D view."); }
  setTracking(false, "Looking for the figure");
  // re-fit once the camera reports its final frame size (some cameras change it after the stream starts)
  const vid = $("#stage video"); if (vid) vid.addEventListener("resize", () => mindar && mindar.resize());
  setTimeout(() => mindar && mindar.resize(), 800);
  renderer.setAnimationLoop(() => { frame(); renderer.render(scene, camera); });
}
function stopAR() { if (!mindar) return; mindar.renderer.setAnimationLoop(null); mindar.stop(); content.removeFromParent(); $("#stage").innerHTML = ""; mindar = null; }
function start3D() {
  if (mindar) stopAR();
  mode = "3d"; document.body.classList.remove("ar"); $("#modeBtn").textContent = "Camera (AR)"; $("#hint").classList.add("hidden"); setTracking(true, "3D view");
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false }); renderer.setPixelRatio(Math.min(2, devicePixelRatio)); renderer.setClearColor(0x14191B);
  renderer.outputColorSpace = THREE.SRGBColorSpace; $("#stage").appendChild(renderer.domElement);
  const scene = new THREE.Scene(); const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 20);
  const fig = figureBacking(); scene.add(fig); scene.add(content);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.minDistance = 0.4; controls.maxDistance = 8;
  controls.minAzimuthAngle = -1.2; controls.maxAzimuthAngle = 1.2; controls.minPolarAngle = 0.5; controls.maxPolarAngle = 2.6;
  const fit = () => { const w = innerWidth, h = innerHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
    const hud = $("#hud .sheet").getBoundingClientRect().height || 220; const vis = Math.max(0.35, (h - hud - 60) / h);
    const vf = THREE.MathUtils.degToRad(camera.fov) / 2; const hf = Math.atan(Math.tan(vf) * camera.aspect);
    const d = Math.max(0.56 / Math.tan(hf), (0.5 / LAY.aspect + 0.08) / (Math.tan(vf) * vis));
    const off = (1 - vis) / 2 * 2 * d * Math.tan(vf);          // centre the figure in the space above the player
    camera.position.set(0.0, -off, d); controls.target.set(0, -off, 0); controls.update(); };
  fit(); addEventListener("resize", fit);
  r3d = { renderer, scene, fit, controls }; rise = 0;
  renderer.setAnimationLoop(() => { frame(); controls.update(); renderer.render(scene, camera); });
}
function stop3D() { if (!r3d) return; r3d.renderer.setAnimationLoop(null); removeEventListener("resize", r3d.fit); content.removeFromParent(); r3d.renderer.dispose(); $("#stage").innerHTML = ""; r3d = null; }
function fail(msg) { const d = document.createElement("div"); d.id = "err"; d.innerHTML = `<p style="margin:0 0 10px">${msg}</p><button class="btn primary" id="go3d">Open the 3D view</button>`;
  document.body.appendChild(d); $("#go3d").onclick = () => { d.remove(); start3D(); }; }

// ---------------------------------------------------------------- boot
async function boot() {
  [DATA, BRAIN, LAY] = await Promise.all(["data/tunes.json", "data/brain.json", "data/hero_layout.json"].map(u => fetch(u).then(r => r.json())));
  buildContent(); const hi = DATA.tunes.findIndex(t => t.id === LAY.tune); tune = DATA.tunes[hi]; traces = prepTune(tune);
  buildHud(); drawStatic(); refreshBar(); audio.src = tune.audio; $("#caption").textContent = summary();
  const unlock = () => { audio.play().then(() => { audio.pause(); audio.currentTime = 0; }).catch(() => {}); };
  $("#startAR").onclick = () => { unlock(); showUi(); startAR(); };
  $("#start3D").onclick = () => { unlock(); showUi(); start3D(); userPaused = false; setTimeout(() => audio.play().catch(() => {}), 300); };
  $("#modeBtn").onclick = () => (mode === "ar" ? start3D() : startAR());
  const q = new URLSearchParams(location.search); if (q.get("tune")) selectTune(+q.get("tune"), false); if (q.get("mode") === "3d") { showUi(); start3D(); } else if (q.get("mode") === "ar") { showUi(); startAR(); }
}
boot().catch(e => fail("Could not load the data (" + e.message + ")."));
