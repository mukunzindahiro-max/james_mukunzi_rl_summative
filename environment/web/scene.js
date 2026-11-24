import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/CSS2DRenderer.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const MAP_W = 40;
const MAP_H = 30;
const INNER_W = 64;
const INNER_H = 52;
const CELL = 0.125;
const NX = INNER_W / CELL;
const NZ = INNER_H / CELL;
const VEHICLE_SCALE = 0.1;
const LANE_OFFSET = 0.13;
const GRASS_CHUNK = 3;
const ROAD_WINDING = 1.35;
export const PRIORITY_COLORS = { 1: 0xff3b3b, 2: 0xff9f1c, 3: 0xffd23f };
const ROAD_STYLE = {
  paved: { width: 0.75, color: 0x3b3e43, verge: 'rgba(132,118,94,0.6)', rank: 3, bump: 0.15 },
  murram: { width: 0.6, color: 0xa5552d, verge: 'rgba(170,96,52,0.5)', rank: 2, bump: 0.6 },
  rough: { width: 0.48, color: 0x87613f, verge: 'rgba(128,96,64,0.5)', rank: 1, bump: 1 },
};
const LAYOUT = {
  hospital: { offset: 2.75, park: 1.2, lane: 1.55, xa: -2.1, xb: 1.9, slot0: -1.75, gap: 0.66, approach: 0.7, building: [-2.0, 2.0, -1.0, 0.9], yard: [-2.5, 2.3, 0.9, 2.6], padX: -0.1, padZ: 0.9, inner: 3.3, outer: 4.9 },
  health_centre: { offset: 1.75, park: 0.82, lane: 1.15, xa: -1.6, xb: 1.6, slot0: -1.3, gap: 0.64, approach: 0.6, building: [-1.3, 1.3, -1.35, 0.6], yard: [-1.9, 1.9, 0.6, 1.75], padX: 0, padZ: 0.2, inner: 2.6, outer: 3.9 },
};
const VIEWS = {
  overview: { pos: new THREE.Vector3(0, 24, 28), target: new THREE.Vector3(0, 1, 1) },
};
export const CAMERA_MODES = ['chase', 'cinematic', 'top', 'overview'];
const SHOP_SIGNS = [
  { text: 'ALIMENTATION', bg: '#f2c500', fg: '#1f2937', awning: '#f2c500' },
  { text: 'BOUTIQUE', bg: '#c62828', fg: '#ffffff', awning: '#c62828' },
  { text: 'PHARMACIE', bg: '#15803d', fg: '#ffffff', awning: '#16a34a' },
  { text: 'SALON', bg: '#1d4ed8', fg: '#ffffff', awning: '#2563eb' },
  { text: 'RESTAURANT', bg: '#7c2d12', fg: '#fde68a', awning: '#b45309' },
];
const WALL_COLOURS = ['#e9dcc0', '#d8b47e', '#f1ede4', '#d9a48e', '#c9d6dd', '#e6cf9a', '#cf8f62'];
const ROOF_COLOURS = ['#9a4126', '#b4552f', '#7d2f22', '#9aa2a8', '#7f8a92', '#a64a2c', '#3f6fae', '#5f3a2a', '#8c3b24'];
const SHOP_COLOURS = ['#f4f1e8', '#f4f1e8', '#e9dcc0', '#f2c500', '#d9e4ec', '#c9473d'];
const FIELD_COLOURS = ['#6e9b3d', '#88a843', '#a3ad4f', '#5d8a34', '#7aa64a', '#b7a95c', '#9c7b4a', '#6f8f2f', '#4f7d32', '#c2b46a', '#8b6a45', '#92b25a'];
const SKY = {
  dayTop: new THREE.Color('#3f7fd0'),
  dayHorizon: new THREE.Color('#c3dcea'),
  nightTop: new THREE.Color('#030814'),
  nightHorizon: new THREE.Color('#14213a'),
  dusk: new THREE.Color('#f0a466'),
  rainTop: new THREE.Color('#59626b'),
  rainHorizon: new THREE.Color('#97a1a9'),
  sun: new THREE.Color('#fff1d6'),
  sunset: new THREE.Color('#ffb070'),
  moon: new THREE.Color('#8fa8ff'),
};

const clamp = THREE.MathUtils.clamp;
const smoothstep = (u) => {
  const v = Math.min(1, Math.max(0, u));
  return v * v * (3 - 2 * v);
};

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const PADS = [];

function baseHeight(x, z) {
  const hills =
    0.9 * Math.sin(x * 0.21 + 0.5) * Math.cos(z * 0.24) +
    0.6 * Math.sin(x * 0.11 + 1.3) * Math.sin(z * 0.17 + 0.4) +
    0.25 * Math.cos(x * 0.53 + z * 0.41) +
    0.08 * Math.sin(x * 1.3 - z * 1.1);
  const ex = Math.max(0, Math.abs(x) - MAP_W / 2 - 2);
  const ez = Math.max(0, Math.abs(z) - MAP_H / 2 - 2);
  const edge = Math.hypot(ex, ez);
  const ridge = 0.6 + 0.4 * Math.sin(x * 0.09 + Math.cos(z * 0.07) * 2.5) * Math.cos(z * 0.08 - x * 0.04);
  return 1.8 + hills * 1.2 + edge * edge * 0.02 * ridge + edge * 0.3;
}

export function terrainHeight(x, z) {
  let h = baseHeight(x, z);
  for (const p of PADS) {
    const d = Math.hypot(x - p.x, z - p.z);
    if (d < p.outer) h += (p.h - h) * (1 - smoothstep((d - p.inner) / (p.outer - p.inner)));
  }
  return h;
}

export function toWorld(x, y) {
  return new THREE.Vector3(x - MAP_W / 2, 0, MAP_H / 2 - y);
}

function lerpAngle(a, b, k) {
  return a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * k;
}

function canvasTexture(width, height, draw) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  draw(canvas.getContext('2d'), width, height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

function makeLabel(html, className) {
  const div = document.createElement('div');
  div.className = className;
  div.innerHTML = html;
  return new CSS2DObject(div);
}

function textTexture(text, { w = 1024, h = 128, color = '#ffffff', bg = null, font = 'bold 80px Arial', mirror = false, glow = null, sub = null } = {}) {
  return canvasTexture(w, h, (ctx) => {
    if (bg) {
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);
    }
    if (mirror) {
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    if (glow) {
      ctx.shadowColor = glow;
      ctx.shadowBlur = 16;
    }
    ctx.fillStyle = color;
    ctx.font = font;
    if (sub) {
      ctx.fillText(text, w / 2, h * 0.36, w * 0.94);
      ctx.font = `bold ${Math.round(h * 0.22)}px Arial`;
      ctx.fillText(sub, w / 2, h * 0.76, w * 0.94);
    } else {
      ctx.fillText(text, w / 2, h / 2 + 4, w * 0.94);
    }
  });
}

function starOfLife(ctx, cx, cy, r, color) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.fillStyle = color;
  for (let k = 0; k < 3; k++) {
    ctx.save();
    ctx.rotate((k * Math.PI) / 3);
    ctx.fillRect(-r * 0.22, -r, r * 0.44, r * 2);
    ctx.restore();
  }
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = r * 0.08;
  ctx.beginPath();
  ctx.moveTo(0, -r * 0.78);
  ctx.lineTo(0, r * 0.78);
  ctx.stroke();
  ctx.beginPath();
  for (let i = 0; i <= 24; i++) {
    const y = -r * 0.6 + (i / 24) * r * 1.2;
    const x = Math.sin(i * 0.8) * r * 0.15;
    if (i) ctx.lineTo(x, y);
    else ctx.moveTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

function glowTexture() {
  return canvasTexture(128, 128, (ctx, w, h) => {
    const g = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.25, 'rgba(255,255,255,0.55)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  });
}

function liveryTexture() {
  const W = 1024, H = 512;
  const texture = canvasTexture(W, H, (ctx) => {
    ctx.setTransform(W / 6.1, 0, 0, -H / 2.4, (3.05 / 6.1) * W, H * (1 + 0.3 / 2.4));
    ctx.fillStyle = '#f6f6f1';
    ctx.fillRect(-3.1, 0.2, 6.2, 2.6);
    ctx.fillStyle = '#2f3237';
    ctx.fillRect(-3.1, 0.2, 6.2, 0.42);
    const sq = 0.38;
    for (let x = -3.05, k = 0; x < 3.05; x += sq, k++) {
      for (let row = 0; row < 2; row++) {
        ctx.fillStyle = (k + row) % 2 ? '#f5d000' : '#cc1626';
        ctx.fillRect(x, 0.64 + row * 0.2, sq + 0.002, 0.2);
      }
    }
    ctx.fillStyle = '#cc1626';
    ctx.fillRect(-3.1, 1.88, 6.2, 0.1);
    ctx.fillStyle = '#16202b';
    ctx.beginPath();
    ctx.moveTo(0.98, 1.62);
    ctx.lineTo(0.98, 2.36);
    ctx.lineTo(1.38, 2.36);
    ctx.lineTo(2.02, 1.62);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.14)';
    ctx.beginPath();
    ctx.moveTo(1.05, 2.3);
    ctx.lineTo(1.3, 2.3);
    ctx.lineTo(1.05, 1.95);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = '#cfd8de';
    ctx.fillRect(-2.75, 2.05, 0.75, 0.33);
    ctx.strokeStyle = '#a4aab0';
    ctx.lineWidth = 0.02;
    for (const x of [0.9, -0.85, 0.6]) {
      ctx.beginPath();
      ctx.moveTo(x, 0.64);
      ctx.lineTo(x, 2.44);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(-0.85, 2.44);
    ctx.lineTo(0.6, 2.44);
    ctx.stroke();
    ctx.fillStyle = '#3a3d42';
    ctx.fillRect(0.72, 1.45, 0.12, 0.04);
    ctx.fillRect(-0.78, 1.45, 0.12, 0.04);
    starOfLife(ctx, -1.95, 1.46, 0.36, '#1f5fbf');
  });
  texture.repeat.set(1 / 6.1, 1 / 2.4);
  texture.offset.set(0.5, -0.3 / 2.4);
  return texture;
}

function rearTexture() {
  return canvasTexture(512, 540, (ctx, w, h) => {
    ctx.fillStyle = '#f6f6f1';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#16202b';
    ctx.fillRect(40, 30, 196, 150);
    ctx.fillRect(276, 30, 196, 150);
    ctx.fillStyle = '#cc1626';
    ctx.font = 'bold 66px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('AMBULANCE', w / 2, 245, w * 0.9);
    for (const [x0, x1, dir] of [[0, w / 2, 1], [w / 2, w, -1]]) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, 300, x1 - x0, 210);
      ctx.clip();
      for (let k = -12; k < 16; k++) {
        ctx.fillStyle = k % 2 ? '#f5d000' : '#cc1626';
        const x = x0 + k * 48;
        ctx.beginPath();
        ctx.moveTo(x, 510);
        ctx.lineTo(x + 48, 510);
        ctx.lineTo(x + 48 + dir * 210, 300);
        ctx.lineTo(x + dir * 210, 300);
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    }
    ctx.fillStyle = '#9aa0a6';
    ctx.fillRect(w / 2 - 2, 0, 4, h);
  });
}

function hubTexture() {
  return canvasTexture(128, 128, (ctx, w) => {
    const c = w / 2;
    ctx.fillStyle = '#151515';
    ctx.fillRect(0, 0, w, w);
    ctx.fillStyle = '#c9ced2';
    ctx.beginPath();
    ctx.arc(c, c, 40, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#5b6066';
    for (let k = 0; k < 6; k++) {
      const a = (k / 6) * Math.PI * 2;
      ctx.beginPath();
      ctx.ellipse(c + Math.cos(a) * 26, c + Math.sin(a) * 26, 7, 10, a, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = '#9aa0a5';
    ctx.beginPath();
    ctx.arc(c, c, 12, 0, Math.PI * 2);
    ctx.fill();
  });
}

function grilleTexture() {
  return canvasTexture(256, 80, (ctx, w, h) => {
    ctx.fillStyle = '#1b1e22';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#5d6268';
    for (let y = 10; y < h; y += 14) ctx.fillRect(8, y, w - 16, 4);
    ctx.fillStyle = '#c9ced2';
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 16, 0, Math.PI * 2);
    ctx.fill();
  });
}

function speckle(ctx, w, h, rand, count, colours, min, max) {
  for (let i = 0; i < count; i++) {
    ctx.fillStyle = colours[Math.floor(rand() * colours.length)];
    const s = min + rand() * (max - min);
    ctx.fillRect(rand() * w, rand() * h, s, s);
  }
}

function drawWindow(ctx, x, y, w, h, frame) {
  ctx.fillStyle = 'rgba(0,0,0,0.25)';
  ctx.fillRect(x - 3, y - 3, w + 8, h + 9);
  ctx.fillStyle = frame;
  ctx.fillRect(x - 4, y - 4, w + 8, h + 8);
  const g = ctx.createLinearGradient(x, y, x + w, y + h);
  g.addColorStop(0, '#5b7184');
  g.addColorStop(0.45, '#22303b');
  g.addColorStop(1, '#3c4f5e');
  ctx.fillStyle = g;
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = 'rgba(20,22,24,0.85)';
  for (let gx = x + 4; gx < x + w; gx += 6) ctx.fillRect(gx, y, 1.5, h);
  ctx.fillRect(x, y + h / 2 - 1, w, 2);
  ctx.fillStyle = '#d9d3c6';
  ctx.fillRect(x - 6, y + h + 3, w + 12, 4);
}

function wallTexture(front, brick) {
  return canvasTexture(256, 128, (ctx, w, h) => {
    const rand = mulberry32((front ? 11 : 12) + (brick ? 5 : 0));
    if (brick) {
      ctx.fillStyle = '#c9a487';
      ctx.fillRect(0, 0, w, h);
      const tones = ['#b0603f', '#9c4e33', '#a85a3c', '#8f4630', '#b86a48', '#a35238'];
      for (let row = 0, y = 0; y < h; row++, y += 6) {
        for (let x = row % 2 ? -7 : 0; x < w; x += 15) {
          ctx.fillStyle = tones[Math.floor(rand() * tones.length)];
          ctx.fillRect(x + 0.6, y + 0.6, 13.8, 4.8);
        }
      }
      speckle(ctx, w, h, rand, 500, ['rgba(0,0,0,0.12)', 'rgba(255,220,190,0.1)'], 1, 2.5);
    } else {
      ctx.fillStyle = '#f1ede4';
      ctx.fillRect(0, 0, w, h);
      speckle(ctx, w, h, rand, 1400, ['rgba(0,0,0,0.05)', 'rgba(255,255,255,0.08)', 'rgba(120,90,60,0.05)'], 1, 4);
      for (let i = 0; i < 6; i++) {
        const x = rand() * w;
        const g = ctx.createLinearGradient(x, 0, x, h * 0.8);
        g.addColorStop(0, 'rgba(90,70,50,0.12)');
        g.addColorStop(1, 'rgba(90,70,50,0)');
        ctx.fillStyle = g;
        ctx.fillRect(x, 0, 3 + rand() * 6, h * 0.8);
      }
    }
    const splash = ctx.createLinearGradient(0, h * 0.68, 0, h);
    splash.addColorStop(0, 'rgba(150,72,36,0)');
    splash.addColorStop(1, brick ? 'rgba(120,60,30,0.45)' : 'rgba(160,78,38,0.6)');
    ctx.fillStyle = splash;
    ctx.fillRect(0, h * 0.68, w, h * 0.32);
    ctx.fillStyle = 'rgba(60,45,38,0.45)';
    ctx.fillRect(0, h - 8, w, 8);
    const eave = ctx.createLinearGradient(0, 0, 0, h * 0.28);
    eave.addColorStop(0, 'rgba(0,0,0,0.5)');
    eave.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = eave;
    ctx.fillRect(0, 0, w, h * 0.28);
    const frame = brick ? '#e2dccf' : '#cfc8b8';
    if (front) {
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.fillRect(w / 2 - 17, h * 0.3, 36, h * 0.7 - 6);
      ctx.fillStyle = '#2e5b85';
      ctx.fillRect(w / 2 - 15, h * 0.32, 30, h * 0.68 - 8);
      ctx.strokeStyle = 'rgba(0,0,0,0.35)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(w / 2 - 11, h * 0.37, 22, h * 0.22);
      ctx.strokeRect(w / 2 - 11, h * 0.63, 22, h * 0.22);
      ctx.fillStyle = '#d8c27a';
      ctx.fillRect(w / 2 + 9, h * 0.62, 3, 3);
      drawWindow(ctx, 34, h * 0.34, 40, 34, frame);
      drawWindow(ctx, w - 74, h * 0.34, 40, 34, frame);
    } else {
      drawWindow(ctx, w * 0.22, h * 0.34, 40, 34, frame);
      drawWindow(ctx, w * 0.62, h * 0.34, 40, 34, frame);
    }
  });
}

function roofTexture() {
  const texture = canvasTexture(128, 128, (ctx, w, h) => {
    const rand = mulberry32(31);
    for (let x = 0; x < w; x++) {
      const v = 0.5 + 0.5 * Math.sin((x / w) * Math.PI * 2 * 8);
      const c = Math.round(150 + v * 105);
      ctx.fillStyle = `rgb(${c},${c},${c})`;
      ctx.fillRect(x, 0, 1, h);
    }
    for (let y = 0; y < h; y += 64) {
      ctx.fillStyle = 'rgba(0,0,0,0.25)';
      ctx.fillRect(0, y, w, 2);
    }
    for (let i = 0; i < 14; i++) {
      const g = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
      g.addColorStop(0, 'rgba(110,52,22,0.35)');
      g.addColorStop(1, 'rgba(110,52,22,0)');
      ctx.save();
      ctx.translate(rand() * w, rand() * h);
      ctx.scale(6 + rand() * 14, 4 + rand() * 10);
      ctx.fillStyle = g;
      ctx.fillRect(-1, -1, 2, 2);
      ctx.restore();
    }
  });
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function shopFrontTexture(sign) {
  return canvasTexture(512, 256, (ctx, w, h) => {
    const rand = mulberry32(sign.text.length * 7);
    ctx.fillStyle = '#efeadf';
    ctx.fillRect(0, 0, w, h);
    speckle(ctx, w, h, rand, 1200, ['rgba(0,0,0,0.05)', 'rgba(255,255,255,0.08)'], 1, 4);
    ctx.fillStyle = sign.bg;
    ctx.fillRect(0, 22, w, 62);
    ctx.fillStyle = 'rgba(0,0,0,0.18)';
    ctx.fillRect(0, 80, w, 4);
    ctx.fillStyle = sign.fg;
    ctx.font = 'bold 46px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(sign.text, w / 2, 54, w * 0.9);
    for (const x of [24, 268]) {
      ctx.fillStyle = '#3a3d41';
      ctx.fillRect(x, 108, 220, 148);
      for (let y = 112; y < 256; y += 7) {
        ctx.fillStyle = '#8e949a';
        ctx.fillRect(x, y, 220, 3);
      }
    }
    ctx.fillStyle = 'rgba(25,25,25,0.9)';
    ctx.fillRect(140, 150, 108, 106);
    ctx.fillStyle = '#f5f5f5';
    for (let k = 0; k < 4; k++) ctx.fillRect(150 + k * 24, 168, 16, 22 + (k % 2) * 10);
    const splash = ctx.createLinearGradient(0, h * 0.8, 0, h);
    splash.addColorStop(0, 'rgba(160,78,38,0)');
    splash.addColorStop(1, 'rgba(160,78,38,0.45)');
    ctx.fillStyle = splash;
    ctx.fillRect(0, h * 0.8, w, h * 0.2);
  });
}

function shopSideTexture() {
  return canvasTexture(256, 128, (ctx, w, h) => {
    const rand = mulberry32(77);
    ctx.fillStyle = '#ece6da';
    ctx.fillRect(0, 0, w, h);
    speckle(ctx, w, h, rand, 900, ['rgba(0,0,0,0.05)', 'rgba(255,255,255,0.08)'], 1, 4);
    drawWindow(ctx, w * 0.4, h * 0.36, 44, 30, '#cfc8b8');
    const splash = ctx.createLinearGradient(0, h * 0.75, 0, h);
    splash.addColorStop(0, 'rgba(160,78,38,0)');
    splash.addColorStop(1, 'rgba(160,78,38,0.5)');
    ctx.fillStyle = splash;
    ctx.fillRect(0, h * 0.75, w, h * 0.25);
  });
}

function asphaltTexture() {
  const texture = canvasTexture(512, 512, (ctx, w, h) => {
    const rand = mulberry32(101);
    ctx.fillStyle = '#4a4d52';
    ctx.fillRect(0, 0, w, h);
    speckle(ctx, w, h, rand, 26000, ['#3b3e43', '#55585d', '#44474c', '#5f6166', '#36393d', '#6a6a68'], 1, 2.5);
    for (const lane of [0.27, 0.73]) {
      for (const off of [-0.07, 0.07]) {
        const x = (lane + off) * w;
        const g = ctx.createLinearGradient(x - 22, 0, x + 22, 0);
        g.addColorStop(0, 'rgba(20,20,22,0)');
        g.addColorStop(0.5, 'rgba(20,20,22,0.22)');
        g.addColorStop(1, 'rgba(20,20,22,0)');
        ctx.fillStyle = g;
        ctx.fillRect(x - 22, 0, 44, h);
      }
    }
    for (let i = 0; i < 6; i++) {
      ctx.fillStyle = rand() < 0.5 ? 'rgba(30,31,34,0.35)' : 'rgba(110,110,108,0.18)';
      const pw = 40 + rand() * 120, ph = 30 + rand() * 140;
      ctx.fillRect(rand() * (w - pw), rand() * h, pw, ph);
    }
    ctx.strokeStyle = 'rgba(15,15,16,0.55)';
    ctx.lineWidth = 1.2;
    for (let i = 0; i < 18; i++) {
      let x = rand() * w, y = rand() * h;
      ctx.beginPath();
      ctx.moveTo(x, y);
      for (let k = 0; k < 8; k++) {
        x += (rand() - 0.5) * 22;
        y += 4 + rand() * 12;
        ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    const edge = (x0, x1) => {
      const g = ctx.createLinearGradient(x0, 0, x1, 0);
      g.addColorStop(0, 'rgba(95,82,64,0.55)');
      g.addColorStop(1, 'rgba(95,82,64,0)');
      ctx.fillStyle = g;
      ctx.fillRect(Math.min(x0, x1), 0, Math.abs(x1 - x0), h);
    };
    edge(0, 26);
    edge(w, w - 26);
  });
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function murramTexture() {
  const texture = canvasTexture(512, 512, (ctx, w, h) => {
    const rand = mulberry32(202);
    ctx.fillStyle = '#b4643a';
    ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 260; i++) {
      const g = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
      const tone = rand() < 0.5 ? '150,74,40' : '196,124,78';
      g.addColorStop(0, `rgba(${tone},0.35)`);
      g.addColorStop(1, `rgba(${tone},0)`);
      ctx.save();
      ctx.translate(rand() * w, rand() * h);
      ctx.scale(10 + rand() * 40, 10 + rand() * 40);
      ctx.fillStyle = g;
      ctx.fillRect(-1, -1, 2, 2);
      ctx.restore();
    }
    speckle(ctx, w, h, rand, 16000, ['#8c4a2a', '#c98a5a', '#a0562f', '#d9a074', '#7a3f22', '#bfae9a'], 1, 3);
    for (const lane of [0.3, 0.7]) {
      for (const off of [-0.06, 0.06]) {
        const x = (lane + off) * w;
        const g = ctx.createLinearGradient(x - 16, 0, x + 16, 0);
        g.addColorStop(0, 'rgba(90,40,20,0)');
        g.addColorStop(0.5, 'rgba(90,40,20,0.4)');
        g.addColorStop(1, 'rgba(90,40,20,0)');
        ctx.fillStyle = g;
        ctx.fillRect(x - 16, 0, 32, h);
      }
    }
    for (let i = 0; i < 7; i++) {
      const x = 60 + rand() * (w - 120), y = rand() * h, rx = 8 + rand() * 18, ry = 6 + rand() * 12;
      const g = ctx.createRadialGradient(x, y, 0, x, y, rx);
      g.addColorStop(0, 'rgba(70,32,16,0.7)');
      g.addColorStop(1, 'rgba(70,32,16,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.ellipse(x, y, rx, ry, rand() * 3, 0, Math.PI * 2);
      ctx.fill();
    }
  });
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function shoulderTexture() {
  const texture = canvasTexture(64, 256, (ctx, w, h) => {
    const rand = mulberry32(303);
    const g = ctx.createLinearGradient(0, 0, w, 0);
    g.addColorStop(0, 'rgba(255,255,255,0.95)');
    g.addColorStop(0.55, 'rgba(255,255,255,0.55)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    ctx.globalCompositeOperation = 'destination-out';
    for (let i = 0; i < 400; i++) {
      ctx.fillStyle = `rgba(0,0,0,${0.2 + rand() * 0.6})`;
      const x = w * (0.35 + rand() * 0.65);
      ctx.fillRect(x, rand() * h, 2 + rand() * 6, 2 + rand() * 6);
    }
  });
  texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function leafTexture() {
  const texture = canvasTexture(256, 256, (ctx, w, h) => {
    const rand = mulberry32(404);
    ctx.fillStyle = '#9fb98a';
    ctx.fillRect(0, 0, w, h);
    const tones = ['#e4efd2', '#c8dbb0', '#b2c99a', '#8faa78', '#6f8c5c', '#d6e6bf', '#58744a'];
    for (let i = 0; i < 2600; i++) {
      ctx.save();
      ctx.translate(rand() * w, rand() * h);
      ctx.rotate(rand() * Math.PI);
      ctx.fillStyle = tones[Math.floor(rand() * tones.length)];
      ctx.beginPath();
      ctx.ellipse(0, 0, 3 + rand() * 4, 1.4 + rand() * 1.8, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  });
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function detailTexture() {
  const texture = canvasTexture(512, 512, (ctx, w, h) => {
    const rand = mulberry32(505);
    ctx.fillStyle = 'rgb(160,160,160)';
    ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 400; i++) {
      const g = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
      const v = rand() < 0.5 ? 110 : 205;
      g.addColorStop(0, `rgba(${v},${v},${v},0.35)`);
      g.addColorStop(1, `rgba(${v},${v},${v},0)`);
      ctx.save();
      ctx.translate(rand() * w, rand() * h);
      ctx.scale(8 + rand() * 30, 8 + rand() * 30);
      ctx.fillStyle = g;
      ctx.fillRect(-1, -1, 2, 2);
      ctx.restore();
    }
    for (let i = 0; i < 30000; i++) {
      const v = 90 + Math.floor(rand() * 150);
      ctx.fillStyle = `rgba(${v},${v},${v},0.5)`;
      ctx.fillRect(rand() * w, rand() * h, 1 + rand() * 1.5, 1 + rand() * 3);
    }
  });
  texture.colorSpace = THREE.NoColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function helipadTexture() {
  return canvasTexture(256, 256, (ctx, w) => {
    const c = w / 2;
    ctx.fillStyle = '#3a3e44';
    ctx.fillRect(0, 0, w, w);
    ctx.strokeStyle = '#f5d000';
    ctx.lineWidth = 10;
    ctx.beginPath();
    ctx.arc(c, c, c - 16, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 150px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('H', c, c + 8);
  });
}

function flagTexture() {
  return canvasTexture(192, 128, (ctx, w, h) => {
    ctx.fillStyle = '#00a1de';
    ctx.fillRect(0, 0, w, h * 0.5);
    ctx.fillStyle = '#fad201';
    ctx.fillRect(0, h * 0.5, w, h * 0.25);
    ctx.fillStyle = '#20603d';
    ctx.fillRect(0, h * 0.75, w, h * 0.25);
    ctx.fillStyle = '#e5be01';
    const cx = w * 0.8, cy = h * 0.22;
    for (let k = 0; k < 24; k++) {
      const a = (k / 24) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * 10, cy + Math.sin(a) * 10);
      ctx.lineTo(cx + Math.cos(a + 0.13) * 22, cy + Math.sin(a + 0.13) * 22);
      ctx.lineTo(cx + Math.cos(a + 0.26) * 10, cy + Math.sin(a + 0.26) * 10);
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(cx, cy, 10, 0, Math.PI * 2);
    ctx.fill();
  });
}

function starTexture() {
  return canvasTexture(256, 256, (ctx) => starOfLife(ctx, 128, 128, 112, '#d81e28'));
}

function forecourtTexture(layout, hospital) {
  const [x0, x1, z0, z1] = layout.yard;
  const W = 1024, H = Math.round((1024 * (z1 - z0)) / (x1 - x0));
  return canvasTexture(W, H, (ctx) => {
    ctx.setTransform(W / (x1 - x0), 0, 0, H / (z1 - z0), (-x0 * W) / (x1 - x0), (-z0 * H) / (z1 - z0));
    ctx.fillStyle = hospital ? '#8f9295' : '#a88d6c';
    ctx.fillRect(x0, z0, x1 - x0, z1 - z0);
    const rand = mulberry32(hospital ? 3 : 4);
    for (let i = 0; i < 400; i++) {
      ctx.fillStyle = `rgba(${rand() < 0.5 ? '0,0,0' : '255,255,255'},0.05)`;
      ctx.fillRect(x0 + rand() * (x1 - x0), z0 + rand() * (z1 - z0), 0.05 + rand() * 0.2, 0.05 + rand() * 0.2);
    }
    ctx.fillStyle = hospital ? '#9c9fa2' : '#b9b3a8';
    ctx.fillRect(x0, layout.park - 0.18, x1 - x0, 0.36);
    ctx.strokeStyle = '#f5d000';
    ctx.lineWidth = 0.025;
    for (let i = 0; i < 5; i++) {
      const sx = layout.slot0 + i * layout.gap;
      ctx.strokeRect(sx - 0.31, layout.park - 0.15, 0.62, 0.3);
    }
    const px = (x) => ((x - x0) / (x1 - x0)) * W;
    const pz = (z) => ((z - z0) / (z1 - z0)) * H;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#f5d000';
    ctx.font = `bold ${Math.round((0.07 * W) / (x1 - x0))}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < 5; i++) ctx.fillText('AMBULANCE', px(layout.slot0 + i * layout.gap), pz(layout.park));
    ctx.restore();
    ctx.fillStyle = '#ffffff';
    for (let x = layout.xa + 0.4; x < layout.xb - 0.3; x += 0.9) {
      ctx.beginPath();
      ctx.moveTo(x, layout.lane - 0.04);
      ctx.lineTo(x + 0.18, layout.lane - 0.04);
      ctx.lineTo(x + 0.18, layout.lane - 0.08);
      ctx.lineTo(x + 0.3, layout.lane);
      ctx.lineTo(x + 0.18, layout.lane + 0.08);
      ctx.lineTo(x + 0.18, layout.lane + 0.04);
      ctx.lineTo(x, layout.lane + 0.04);
      ctx.closePath();
      ctx.fill();
    }
  });
}

const facadeCache = new Map();

function facadeTexture(width, floors, kind, accent) {
  const cols = Math.max(2, Math.round(width / (kind === 'clinic' ? 0.3 : 0.24)));
  const key = `${kind}-${cols}-${floors}-${accent}`;
  if (facadeCache.has(key)) return facadeCache.get(key);
  const W = 256, H = Math.min(512, 96 * floors);
  const rand = mulberry32(cols * 131 + floors * 17 + kind.length);
  const lit = Array.from({ length: cols * floors }, () => rand() < 0.6);
  const accentCols = Array.from({ length: cols }, () => rand() < 0.35);
  const paint = (night) => (ctx) => {
    const cw = W / cols, rh = H / floors;
    ctx.fillStyle = night ? '#000000' : kind === 'clinic' ? '#efe6cf' : '#f3f3ef';
    ctx.fillRect(0, 0, W, H);
    if (kind === 'glass') {
      ctx.fillStyle = night ? '#ffdca0' : '#2b465b';
      ctx.fillRect(0, H * 0.14, W, H * 0.86);
      if (!night) {
        const g = ctx.createLinearGradient(0, 0, W, H);
        g.addColorStop(0, 'rgba(255,255,255,0.2)');
        g.addColorStop(0.5, 'rgba(255,255,255,0)');
        g.addColorStop(1, 'rgba(255,255,255,0.12)');
        ctx.fillStyle = g;
        ctx.fillRect(0, H * 0.14, W, H * 0.86);
      }
      ctx.fillStyle = night ? '#000000' : '#d7dadc';
      for (let c = 0; c <= cols * 2; c++) ctx.fillRect((c * W) / (cols * 2) - 1.5, H * 0.14, 3, H);
      ctx.fillRect(0, H * 0.55, W, 3);
      return;
    }
    if (kind === 'clinic' && !night) {
      ctx.fillStyle = accent;
      ctx.fillRect(0, H * 0.74, W, H * 0.26);
    }
    for (let r = 0; r < floors; r++) {
      if (kind === 'tower' && !night) {
        ctx.fillStyle = 'rgba(0,0,0,0.06)';
        ctx.fillRect(0, r * rh, W, 3);
      }
      for (let c = 0; c < cols; c++) {
        const x = c * cw + cw * 0.18;
        const y = r * rh + rh * (kind === 'clinic' ? 0.2 : 0.16);
        const w = cw * 0.64;
        const h = rh * (kind === 'clinic' ? 0.42 : 0.58);
        if (night) {
          if (lit[r * cols + c]) {
            ctx.fillStyle = '#ffd68a';
            ctx.fillRect(x, y, w, h);
          }
          continue;
        }
        if (accent && (kind === 'clinic' || accentCols[c])) {
          ctx.fillStyle = accent;
          ctx.fillRect(x - cw * 0.07, y - rh * 0.05, w + cw * 0.14, h + rh * 0.1);
        }
        const g = ctx.createLinearGradient(x, y, x + w, y + h);
        g.addColorStop(0, '#4d6f8a');
        g.addColorStop(1, '#1a2a39');
        ctx.fillStyle = g;
        ctx.fillRect(x, y, w, h);
        ctx.fillStyle = 'rgba(255,255,255,0.35)';
        ctx.fillRect(x + w / 2 - 1, y, 2, h);
      }
    }
  };
  const result = { map: canvasTexture(W, H, paint(false)), emissiveMap: canvasTexture(W, H, paint(true)) };
  facadeCache.set(key, result);
  return result;
}

function jitter(geometry, amount, seed) {
  const rand = mulberry32(seed);
  const pos = geometry.attributes.position;
  const offsets = new Map();
  for (let i = 0; i < pos.count; i++) {
    const key = `${pos.getX(i).toFixed(3)},${pos.getY(i).toFixed(3)},${pos.getZ(i).toFixed(3)}`;
    if (!offsets.has(key)) offsets.set(key, 1 + (rand() - 0.5) * amount);
    const k = offsets.get(key);
    pos.setXYZ(i, pos.getX(i) * k, pos.getY(i) * k, pos.getZ(i) * k);
  }
  geometry.computeVertexNormals();
  return geometry;
}

function leafGeometry() {
  const geometry = new THREE.PlaneGeometry(0.075, 0.32, 1, 5);
  geometry.translate(0, 0.16, 0);
  const pos = geometry.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const l = pos.getY(i);
    pos.setXYZ(i, pos.getX(i), l * 0.9 - l * l * 2.2, l * 0.95);
  }
  geometry.computeVertexNormals();
  return geometry;
}

function polyGeometry(faces) {
  const positions = [], uvs = [];
  for (const { pts, uv } of faces) {
    for (let k = 1; k < pts.length - 1; k++) {
      for (const i of [0, k, k + 1]) {
        positions.push(...pts[i]);
        uvs.push(...uv[i]);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.computeVertexNormals();
  return geometry;
}

function hipRoofGeometry(ridge) {
  const A = [-0.5, 0, -0.5], B = [0.5, 0, -0.5], C = [0.5, 0, 0.5], D = [-0.5, 0, 0.5];
  const R1 = [-ridge, 1, 0], R2 = [ridge, 1, 0];
  const r = 3;
  return polyGeometry([
    { pts: [D, C, R2, R1], uv: [[0, 0], [r, 0], [(0.5 + ridge) * r, 1], [(0.5 - ridge) * r, 1]] },
    { pts: [B, A, R1, R2], uv: [[0, 0], [r, 0], [(0.5 + ridge) * r, 1], [(0.5 - ridge) * r, 1]] },
    { pts: [C, B, R2], uv: [[0, 0], [r, 0], [r / 2, 1]] },
    { pts: [A, D, R1], uv: [[0, 0], [r, 0], [r / 2, 1]] },
  ]);
}

function gableRoofGeometry() {
  const r = 3;
  return polyGeometry([
    { pts: [[-0.5, 0, 0.5], [0.5, 0, 0.5], [0.5, 1, 0], [-0.5, 1, 0]], uv: [[0, 0], [r, 0], [r, 1], [0, 1]] },
    { pts: [[0.5, 0, -0.5], [-0.5, 0, -0.5], [-0.5, 1, 0], [0.5, 1, 0]], uv: [[0, 0], [r, 0], [r, 1], [0, 1]] },
  ]);
}

function gableWallGeometry() {
  const uv = [[0.02, 0.55], [0.08, 0.55], [0.05, 0.6]];
  return polyGeometry([
    { pts: [[0.5, 0, 0.5], [0.5, 0, -0.5], [0.5, 1, 0]], uv },
    { pts: [[-0.5, 0, -0.5], [-0.5, 0, 0.5], [-0.5, 1, 0]], uv },
  ]);
}

function mergeGeometries(list) {
  const positions = [], normals = [], uvs = [], colours = [];
  for (const g of list) {
    const geometry = g.index ? g.toNonIndexed() : g;
    positions.push(...geometry.attributes.position.array);
    normals.push(...geometry.attributes.normal.array);
    uvs.push(...(geometry.attributes.uv ? geometry.attributes.uv.array : new Float32Array((geometry.attributes.position.count) * 2)));
    if (geometry.attributes.color) colours.push(...geometry.attributes.color.array);
  }
  const merged = new THREE.BufferGeometry();
  merged.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  merged.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  merged.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  if (colours.length) merged.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
  return merged;
}

function crownGeometry(seed, blobs, spread, squash, detail = 1) {
  const rand = mulberry32(seed);
  const parts = [];
  for (let k = 0; k < blobs; k++) {
    const r = 0.16 + rand() * 0.12;
    const g = jitter(new THREE.IcosahedronGeometry(r, detail), 0.22, seed + k * 13);
    const p = g.attributes.position, n = g.attributes.normal, v = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i).normalize();
      n.setXYZ(i, v.x, v.y, v.z);
    }
    const a = rand() * Math.PI * 2, d = k === 0 ? 0 : spread * (0.5 + rand() * 0.5);
    g.translate(Math.cos(a) * d, (rand() - 0.3) * 0.14, Math.sin(a) * d);
    g.scale(1, squash, 1);
    parts.push(g);
  }
  const merged = mergeGeometries(parts);
  const pos = merged.attributes.position;
  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < pos.count; i++) {
    minY = Math.min(minY, pos.getY(i));
    maxY = Math.max(maxY, pos.getY(i));
  }
  const colours = [];
  for (let i = 0; i < pos.count; i++) {
    const u = (pos.getY(i) - minY) / (maxY - minY);
    const shade = 0.5 + 0.55 * u;
    colours.push(shade, shade, shade);
  }
  merged.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
  return merged;
}

function grassTuftGeometry() {
  const rand = mulberry32(606);
  const positions = [], colours = [], indices = [];
  const base = new THREE.Color('#2f4a1c'), tip = new THREE.Color('#a7bd5e'), c = new THREE.Color();
  for (let b = 0; b < 5; b++) {
    const a = rand() * Math.PI * 2, lean = 0.25 + rand() * 0.5, h = 0.6 + rand() * 0.5, w = 0.07 + rand() * 0.04;
    const ox = (rand() - 0.5) * 0.25, oz = (rand() - 0.5) * 0.25;
    const dx = Math.cos(a), dz = Math.sin(a);
    const start = positions.length / 3;
    for (let s = 0; s <= 2; s++) {
      const u = s / 2;
      const bend = lean * u * u;
      const half = w * (1 - u * 0.9);
      const cx = ox + dx * bend * h, cz = oz + dz * bend * h, cy = h * u;
      positions.push(cx - dz * half, cy, cz + dx * half, cx + dz * half, cy, cz - dx * half);
      c.copy(base).lerp(tip, u);
      colours.push(c.r, c.g, c.b, c.r, c.g, c.b);
      if (s < 2) {
        const q = start + s * 2;
        indices.push(q, q + 1, q + 2, q + 1, q + 3, q + 2);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function leafClusterTexture() {
  const texture = canvasTexture(256, 256, (ctx, w, h) => {
    const rand = mulberry32(707);
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(70,52,30,0.9)';
    ctx.lineWidth = 2;
    for (let i = 0; i < 6; i++) {
      ctx.beginPath();
      ctx.moveTo(w / 2, h / 2);
      ctx.lineTo(w / 2 + (rand() - 0.5) * w * 0.8, h / 2 + (rand() - 0.5) * h * 0.8);
      ctx.stroke();
    }
    const tones = ['#e3f0c8', '#cfe2ad', '#b9d293', '#a3c27e', '#8fb06a', '#c6dc9e', '#dbe9bb'];
    for (let i = 0; i < 90; i++) {
      const r = Math.sqrt(rand()) * w * 0.42;
      const a = rand() * Math.PI * 2;
      ctx.save();
      ctx.translate(w / 2 + Math.cos(a) * r, h / 2 + Math.sin(a) * r);
      ctx.rotate(a + (rand() - 0.5) * 1.2);
      const len = 14 + rand() * 12, wid = 5 + rand() * 4;
      ctx.fillStyle = tones[Math.floor(rand() * tones.length)];
      ctx.beginPath();
      ctx.moveTo(-len / 2, 0);
      ctx.quadraticCurveTo(0, -wid, len / 2, 0);
      ctx.quadraticCurveTo(0, wid, -len / 2, 0);
      ctx.fill();
      ctx.strokeStyle = 'rgba(80,100,50,0.35)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(-len / 2, 0);
      ctx.lineTo(len / 2, 0);
      ctx.stroke();
      ctx.restore();
    }
  });
  texture.generateMipmaps = true;
  return texture;
}

function flowerTexture() {
  return canvasTexture(64, 64, (ctx, w, h) => {
    const rand = mulberry32(808);
    ctx.clearRect(0, 0, w, h);
    for (let i = 0; i < 5; i++) {
      const x = 8 + rand() * 48, top = 6 + rand() * 24;
      ctx.strokeStyle = '#4d6b2a';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x + (rand() - 0.5) * 6, h);
      ctx.lineTo(x, top);
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      for (let p = 0; p < 5; p++) {
        const a = (p / 5) * Math.PI * 2;
        ctx.beginPath();
        ctx.arc(x + Math.cos(a) * 2.6, top + Math.sin(a) * 2.6, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = '#f2c500';
      ctx.beginPath();
      ctx.arc(x, top, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

function foliageCards(crown, count, size, seed) {
  const rand = mulberry32(seed);
  const pos = crown.attributes.position, nrm = crown.attributes.normal, col = crown.attributes.color;
  const positions = [], normals = [], uvs = [], colours = [], indices = [];
  const p = new THREE.Vector3(), n = new THREE.Vector3(), f = new THREE.Vector3(), t1 = new THREE.Vector3(), t2 = new THREE.Vector3(), r = new THREE.Vector3();
  for (let k = 0; k < count; k++) {
    const i = Math.floor(rand() * pos.count);
    p.fromBufferAttribute(pos, i);
    n.fromBufferAttribute(nrm, i).normalize();
    f.copy(n).add(r.set(rand() - 0.5, rand() - 0.5, rand() - 0.5).multiplyScalar(1.4)).normalize();
    t1.crossVectors(f, r.set(rand() - 0.5, 1, rand() - 0.5)).normalize();
    t2.crossVectors(f, t1);
    const h = size * (0.7 + rand() * 0.6);
    p.addScaledVector(n, 0.015);
    const start = positions.length / 3;
    const shade = col ? col.getX(i) : 1;
    for (const [a, b, u, v] of [[-1, -1, 0, 0], [1, -1, 1, 0], [1, 1, 1, 1], [-1, 1, 0, 1]]) {
      positions.push(p.x + (t1.x * a + t2.x * b) * h, p.y + (t1.y * a + t2.y * b) * h, p.z + (t1.z * a + t2.z * b) * h);
      normals.push(n.x, n.y, n.z);
      uvs.push(u, v);
      colours.push(shade, shade, shade);
    }
    indices.push(start, start + 1, start + 2, start, start + 2, start + 3);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
  geometry.setIndex(indices);
  return geometry;
}

function branchGeometry(seed, count, height) {
  const rand = mulberry32(seed);
  const parts = [];
  for (let k = 0; k < count; k++) {
    const len = 0.14 + rand() * 0.1;
    const g = new THREE.CylinderGeometry(0.006, 0.012, len, 5).translate(0, len / 2, 0);
    g.rotateZ(0.6 + rand() * 0.5);
    g.rotateY((k / count) * Math.PI * 2 + rand() * 0.6);
    g.translate(0, height * (0.75 + rand() * 0.2), 0);
    parts.push(g);
  }
  return mergeGeometries(parts);
}

function flowerGeometry() {
  const a = new THREE.PlaneGeometry(1, 1).translate(0, 0.5, 0);
  const b = a.clone().rotateY(Math.PI / 2);
  return mergeGeometries([a, b]);
}

function personModel(dress, { pregnant = false, scarf = null, skin = 0x4a2f22 } = {}) {
  const group = new THREE.Group();
  const skinMat = new THREE.MeshStandardMaterial({ color: skin, roughness: 0.7 });
  const cloth = new THREE.MeshStandardMaterial({ color: dress, roughness: 0.85 });
  const add = (geometry, material, x, y, z) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x, y, z);
    mesh.castShadow = true;
    group.add(mesh);
    return mesh;
  };
  for (const x of [-0.09, 0.09]) add(new THREE.CylinderGeometry(0.06, 0.05, 0.8, 8), skinMat, x, 0.4, 0);
  add(new THREE.CylinderGeometry(0.17, 0.27, 0.75, 14), cloth, 0, 0.95, 0);
  add(new THREE.CapsuleGeometry(0.16, 0.3, 4, 12), cloth, 0, 1.3, 0);
  if (pregnant) add(new THREE.SphereGeometry(0.18, 16, 12), cloth, 0, 1.12, 0.14);
  for (const x of [-0.22, 0.22]) add(new THREE.CapsuleGeometry(0.045, 0.45, 4, 8), skinMat, x, 1.22, 0).rotation.z = x > 0 ? 0.12 : -0.12;
  add(new THREE.SphereGeometry(0.11, 16, 12), skinMat, 0, 1.66, 0);
  if (scarf) add(new THREE.SphereGeometry(0.125, 16, 10, 0, Math.PI * 2, 0, Math.PI * 0.55), new THREE.MeshStandardMaterial({ color: scarf, roughness: 0.8 }), 0, 1.68, -0.005);
  return group;
}

function stretcherModel() {
  const group = new THREE.Group();
  const metal = new THREE.MeshStandardMaterial({ color: 0xb8bec6, roughness: 0.35, metalness: 0.6 });
  const add = (geometry, material, x, y, z) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x, y, z);
    mesh.castShadow = true;
    group.add(mesh);
    return mesh;
  };
  add(new THREE.BoxGeometry(0.64, 0.1, 1.95), new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.6 }), 0, 0.8, 0);
  for (const x of [-0.26, 0.26]) {
    for (const z of [-0.8, 0.8]) {
      add(new THREE.CylinderGeometry(0.025, 0.025, 0.72, 6), metal, x, 0.4, z);
      add(new THREE.CylinderGeometry(0.06, 0.06, 0.05, 10), metal, x, 0.06, z).rotation.z = Math.PI / 2;
    }
  }
  const patient = personModel(0xe86fa0, { pregnant: true, scarf: 0xf2c500 });
  patient.rotation.x = -Math.PI / 2;
  patient.position.set(0, 1.02, 0.88);
  group.add(patient);
  add(new THREE.BoxGeometry(0.58, 0.05, 0.9), new THREE.MeshStandardMaterial({ color: 0x60a5fa, roughness: 0.9 }), 0, 1.1, 0.45);
  return group;
}

function addWind(material, uniform, amplitude, nearFade = 0) {
  material.customProgramCacheKey = () => `wind-${amplitude}-${nearFade}`;
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = uniform;
    if (nearFade > 0) {
      shader.vertexShader = `varying vec3 vFadePos;\n${shader.vertexShader}`.replace(
        '#include <project_vertex>',
        `#include <project_vertex>
        #ifdef USE_INSTANCING
        vFadePos = (modelMatrix * instanceMatrix * vec4(transformed, 1.0)).xyz;
        #else
        vFadePos = (modelMatrix * vec4(transformed, 1.0)).xyz;
        #endif`,
      );
      shader.fragmentShader = `varying vec3 vFadePos;\n${shader.fragmentShader}`.replace(
        '#include <clipping_planes_fragment>',
        `#include <clipping_planes_fragment>
        if (distance(vFadePos, cameraPosition) < ${nearFade.toFixed(3)}) discard;`,
      );
    }
    shader.vertexShader = `uniform float uTime;\n${shader.vertexShader}`.replace(
      '#include <begin_vertex>',
      `#include <begin_vertex>
      #ifdef USE_INSTANCING
      vec2 windSeed = vec2(instanceMatrix[3][0], instanceMatrix[3][2]);
      #else
      vec2 windSeed = vec2(0.0);
      #endif
      float windH = max(position.y, 0.0);
      float sway = sin(uTime * 1.6 + windSeed.x * 0.9 + windSeed.y * 0.7) + 0.45 * sin(uTime * 3.7 + windSeed.x * 2.3 + position.x * 4.0);
      transformed.x += sway * ${amplitude.toFixed(4)} * windH;
      transformed.z += cos(uTime * 1.25 + windSeed.y * 1.1) * ${(amplitude * 0.6).toFixed(4)} * windH;`,
    );
  };
  return material;
}

const DUST_VERTEX = `
attribute float aSize;
attribute float aAlpha;
uniform float uScale;
varying float vAlpha;
void main() {
  vAlpha = aAlpha;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = aSize * uScale / max(-mv.z, 0.05);
  gl_Position = projectionMatrix * mv;
}`;

const DUST_FRAGMENT = `
uniform vec3 uColor;
varying float vAlpha;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c);
  if (d > 0.5) discard;
  gl_FragColor = vec4(uColor, smoothstep(0.5, 0.0, d) * vAlpha);
  #include <colorspace_fragment>
}`;

const SKY_VERTEX = `
varying vec3 vDir;
void main() {
  vDir = normalize(position);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const SKY_FRAGMENT = `
uniform vec3 top;
uniform vec3 horizon;
varying vec3 vDir;
void main() {
  float h = clamp(vDir.y, 0.0, 1.0);
  gl_FragColor = vec4(mix(horizon, top, pow(h, 0.5)), 1.0);
  #include <colorspace_fragment>
}`;

export class World {
  constructor(container) {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.05, 1500);
    this.camera.position.copy(VIEWS.overview.pos);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.composer = new EffectComposer(this.renderer, new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, samples: 4 }));
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.2, 0.45, 0.9);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.pmrem = new THREE.PMREMGenerator(this.renderer);
    this.envScene = new THREE.Scene();
    this.envKey = '';
    this.envAt = -Infinity;
    this.wind = { value: 0 };
    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.setSize(window.innerWidth, window.innerHeight);
    Object.assign(this.labelRenderer.domElement.style, { position: 'absolute', top: '0', left: '0', pointerEvents: 'none' });
    container.appendChild(this.labelRenderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(VIEWS.overview.target);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = 1.42;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 120;
    this.scene.fog = new THREE.Fog(0xc3dcea, 40, 180);
    this.hemi = new THREE.HemisphereLight(0xdfeeff, 0x55683a, 0.9);
    this.sun = new THREE.DirectionalLight(0xfff1d6, 2.4);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.02;
    this.sun.shadow.camera.near = 1;
    this.sun.shadow.camera.far = 200;
    this.scene.add(this.hemi, this.sun, this.sun.target);
    this.sunDir = new THREE.Vector3(-0.5, 0.8, 0.35).normalize();
    this.shadowExtent = 0;
    this.tex = {
      livery: liveryTexture(),
      rear: rearTexture(),
      hub: hubTexture(),
      grille: grilleTexture(),
      glow: glowTexture(),
      wallFront: wallTexture(true, false),
      wallSide: wallTexture(false, false),
      brickFront: wallTexture(true, true),
      brickSide: wallTexture(false, true),
      roof: roofTexture(),
      shopSide: shopSideTexture(),
      asphalt: asphaltTexture(),
      murram: murramTexture(),
      shoulder: shoulderTexture(),
      leaf: leafTexture(),
      leafCluster: leafClusterTexture(),
      flower: flowerTexture(),
      detail: detailTexture(),
      helipad: helipadTexture(),
      flag: flagTexture(),
      star: starTexture(),
      hood: textTexture('AMBULANCE', { color: '#cc1626', mirror: true }),
      side: textTexture('AMBULANCE', { color: '#cc1626' }),
      emergency: textTexture('EMERGENCY', { color: '#ff2a2a', bg: '#1a1a1a', glow: '#ff0000' }),
    };
    this.ambulances = [];
    this.beacons = {};
    this.facilityLabels = [];
    this.hospitalLabels = [];
    this.windowMaterials = [];
    this.trips = [];
    this.info = [];
    this.movingCount = 0;
    this.cameraMode = 'cinematic';
    this.followIndex = 0;
    this.followDistance = 1.4;
    this.followHeight = 0.3;
    this.camYaw = 0;
    this.manualUntil = 0;
    this.cameraGoal = null;
    this.camOffset = new THREE.Vector3(0, 3, -6);
    this.lookOffset = new THREE.Vector3();
    this.transition = 0;
    this.shot = null;
    this.shotTime = 0;
    this.flight = null;
    this.onAmbulanceClick = null;
    this.onFollowChange = null;
    this.raycaster = new THREE.Raycaster();
    this.occluders = [];
    this.built = false;
    this.bindInput();
    window.addEventListener('resize', () => this.resize());
  }

  get tracking() {
    return this.cameraMode !== 'overview';
  }

  get shotName() {
    return this.cameraMode === 'cinematic' ? this.shot?.type ?? 'chase' : this.cameraMode;
  }

  bindInput() {
    const el = this.renderer.domElement;
    el.addEventListener('wheel', (e) => {
      if (!this.tracking) return;
      e.preventDefault();
      this.followDistance = clamp(this.followDistance * Math.exp(e.deltaY * 0.0012), 0.5, 12);
    }, { passive: false });
    el.addEventListener('pointerdown', (e) => {
      if (!this.tracking) return;
      this.dragging = true;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      this.camYaw -= e.movementX * 0.006;
      this.followHeight = clamp(this.followHeight + e.movementY * 0.003, 0.04, 1.5);
      this.manualUntil = performance.now() + 4000;
    });
    el.addEventListener('pointerup', () => {
      this.dragging = false;
    });
  }

  resize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.composer.setSize(window.innerWidth, window.innerHeight);
    this.labelRenderer.setSize(window.innerWidth, window.innerHeight);
    this.updateDustScale();
  }

  setCameraMode(mode) {
    if (mode === 'follow') mode = 'chase';
    this.cameraMode = mode;
    this.controls.enabled = mode === 'overview';
    this.transition = 0;
    this.shot = null;
    this.cameraGoal = mode === 'overview' ? { pos: VIEWS.overview.pos.clone(), target: VIEWS.overview.target.clone() } : null;
    const view = this.ambulances[this.followIndex];
    if (view) {
      this.camYaw = view.heading;
      this.camOffset.copy(this.camera.position).sub(view.pos);
      this.lookOffset.copy(this.controls.target).sub(view.pos);
    }
    this.flight = null;
  }

  follow(index) {
    this.pinnedUntil = performance.now() + 20000;
    if (index === this.followIndex && this.tracking) return;
    this.followIndex = index;
    const view = this.ambulances[index];
    if (view) this.camYaw = view.heading;
    if (this.cameraMode === 'overview') this.setCameraMode('chase');
    else if (this.cameraMode === 'cinematic') this.shot = null;
    this.onFollowChange?.(index);
  }

  build(state) {
    if (this.built) return;
    this.built = true;
    this.nodes = state.nodes;
    this.nodePos = this.nodes.map((n) => toWorld(n.x, n.y));
    this.hospitalNames = Object.fromEntries(state.hospitals.map((h) => [h.id, h.name]));
    this.buildCurves(state.edges);
    this.buildRoadGrid();
    this.layoutFacilities();
    this.placeScenery();
    this.buildTerrain();
    this.buildRoads();
    this.buildFacilities();
    this.buildScenery();
    this.buildAmbulances(state);
    this.buildDust();
    this.buildRain();
    this.buildSky();
  }

  buildCurves(edges) {
    this.curveMap = new Map();
    this.edgeData = edges.map((e, i) => {
      const a = this.nodePos[e.from], b = this.nodePos[e.to];
      const rand = mulberry32(1000 + i * 37);
      const len = a.distanceTo(b);
      const dir = b.clone().sub(a).normalize();
      const normal = new THREE.Vector3(-dir.z, 0, dir.x);
      const n = Math.max(2, Math.round(len / 2.2));
      const ctrl = [a.clone()];
      for (let k = 1; k < n; k++) {
        const f = k / n;
        const amp = Math.min(0.9, len * 0.07) * Math.sin(Math.PI * f);
        ctrl.push(a.clone().lerp(b, f).addScaledVector(normal, (rand() - 0.5) * 2 * amp));
      }
      ctrl.push(b.clone());
      const curve = new THREE.CatmullRomCurve3(ctrl, false, 'centripetal');
      const points = curve.getSpacedPoints(Math.ceil(curve.getLength() / 0.08));
      const km = len * ROAD_WINDING;
      this.curveMap.set(`${e.from},${e.to}`, { points, road: e.road, km });
      this.curveMap.set(`${e.to},${e.from}`, { points: points.slice().reverse(), road: e.road, km });
      return { ...e, points };
    });
  }

  cellIndex(x, z) {
    const i = Math.floor((x + INNER_W / 2) / CELL);
    const j = Math.floor((z + INNER_H / 2) / CELL);
    return i < 0 || j < 0 || i >= NX || j >= NZ ? -1 : j * NX + i;
  }

  buildRoadGrid() {
    this.clearance = new Float32Array(NX * NZ).fill(99);
    this.occ = new Uint8Array(NX * NZ);
    const R = 2.5;
    const stamp = (p, hw) => {
      const i0 = Math.floor((p.x - R + INNER_W / 2) / CELL), i1 = Math.floor((p.x + R + INNER_W / 2) / CELL);
      const j0 = Math.floor((p.z - R + INNER_H / 2) / CELL), j1 = Math.floor((p.z + R + INNER_H / 2) / CELL);
      for (let j = Math.max(0, j0); j <= Math.min(NZ - 1, j1); j++) {
        for (let i = Math.max(0, i0); i <= Math.min(NX - 1, i1); i++) {
          const x = (i + 0.5) * CELL - INNER_W / 2, z = (j + 0.5) * CELL - INNER_H / 2;
          const d = Math.hypot(x - p.x, z - p.z) - hw;
          const k = j * NX + i;
          if (d < this.clearance[k]) this.clearance[k] = d;
        }
      }
    };
    for (const e of this.edgeData) {
      const hw = ROAD_STYLE[e.road].width / 2;
      e.points.forEach((p, k) => {
        if (k % 2 === 0 || k === e.points.length - 1) stamp(p, hw);
      });
    }
  }

  roadClear(x, z) {
    const k = this.cellIndex(x, z);
    return k < 0 ? 99 : this.clearance[k];
  }

  occFree(x, z, r) {
    for (let dz = -r; dz <= r; dz += CELL) {
      for (let dx = -r; dx <= r; dx += CELL) {
        if (dx * dx + dz * dz > r * r) continue;
        const k = this.cellIndex(x + dx, z + dz);
        if (k < 0 || this.occ[k]) return false;
      }
    }
    return true;
  }

  occMark(x, z, r, value = 1) {
    for (let dz = -r; dz <= r; dz += CELL / 2) {
      for (let dx = -r; dx <= r; dx += CELL / 2) {
        if (dx * dx + dz * dz > r * r) continue;
        const k = this.cellIndex(x + dx, z + dz);
        if (k >= 0 && !this.occ[k]) this.occ[k] = value;
      }
    }
  }

  makeFacility(n, layout, dir) {
    const centre = this.nodePos[n.id];
    const side = new THREE.Vector3(-dir.z, 0, dir.x);
    const spot = centre.clone().addScaledVector(dir, layout.offset);
    const local = (lx, lz) => spot.clone().addScaledVector(side, lx).addScaledVector(dir, -lz);
    return { node: n.id, type: n.type, name: n.name, hospital: n.type === 'hospital', L: layout, centre, dir, side, spot, local, laneHeading: Math.atan2(side.x, side.z), paths: {} };
  }

  rectPoints(f, rect, step) {
    const [x0, x1, z0, z1] = rect;
    const pts = [];
    for (let x = x0; x <= x1 + 1e-6; x += step) for (let z = z0; z <= z1 + 1e-6; z += step) pts.push([x, z, f.local(x, z)]);
    return pts;
  }

  layoutFacilities() {
    PADS.length = 0;
    this.facilities = {};
    for (const n of this.nodes) {
      if (n.type === 'junction') continue;
      const layout = LAYOUT[n.type];
      let best = null;
      for (let k = 0; k < 48; k++) {
        const angle = (k / 48) * Math.PI * 2;
        const f = this.makeFacility(n, layout, new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle)));
        let score = Infinity;
        for (const [, , p] of this.rectPoints(f, layout.building, 0.25)) score = Math.min(score, this.roadClear(p.x, p.z));
        for (const [, z, p] of this.rectPoints(f, layout.yard, 0.25)) if (z < layout.offset - 0.6) score = Math.min(score, this.roadClear(p.x, p.z) + 0.3);
        if (!best || score > best.score) best = { score, f };
      }
      const f = best.f;
      const pc = f.local(layout.padX, layout.padZ);
      f.padH = (baseHeight(f.centre.x, f.centre.z) + baseHeight(pc.x, pc.z)) / 2;
      PADS.push({ x: pc.x, z: pc.z, inner: layout.inner, outer: layout.outer, h: f.padH });
      const [bx0, bx1, bz0] = layout.building;
      const [, , , yz1] = layout.yard;
      for (const [, , p] of this.rectPoints(f, [Math.min(bx0, layout.yard[0]) - 0.2, Math.max(bx1, layout.yard[1]) + 0.2, bz0 - 0.2, yz1], 0.06)) {
        const k = this.cellIndex(p.x, p.z);
        if (k >= 0) this.occ[k] = 3;
      }
      this.facilities[n.id] = f;
    }
  }

  sampleCurve(points) {
    const curve = new THREE.CatmullRomCurve3(points, false, 'centripetal');
    return curve.getSpacedPoints(Math.max(4, Math.ceil(curve.getLength() / 0.04)));
  }

  slotX(f, i) {
    return f.L.slot0 + i * f.L.gap;
  }

  slotPos(f, i) {
    return f.local(this.slotX(f, i), f.L.park);
  }

  entryPath(f, i) {
    const key = `in${i}`;
    if (f.paths[key]) return f.paths[key];
    const L = f.L, sx = this.slotX(f, i), ax = sx - L.approach;
    const pts = [[0, L.offset], [L.xa - 0.1, (L.lane + L.offset) / 2 + 0.1]];
    if (ax > L.xa + 0.35) pts.push([L.xa + 0.15, L.lane]);
    pts.push([ax, L.lane], [sx - 0.25, L.park], [sx, L.park]);
    f.paths[key] = this.sampleCurve(pts.map(([x, z]) => f.local(x, z)));
    return f.paths[key];
  }

  exitPath(f, i) {
    const key = `out${i}`;
    if (f.paths[key]) return f.paths[key];
    const L = f.L, sx = this.slotX(f, i), bx = sx + L.approach;
    const pts = [[sx, L.park], [sx + 0.25, L.park], [bx, L.lane]];
    if (bx < L.xb - 0.35) pts.push([L.xb - 0.15, L.lane]);
    pts.push([L.xb + 0.1, (L.lane + L.offset) / 2 + 0.1], [0, L.offset]);
    f.paths[key] = this.sampleCurve(pts.map(([x, z]) => f.local(x, z)));
    return f.paths[key];
  }

  placeScenery() {
    const rand = mulberry32(2024);
    const P = (this.placement = { houses: [], shops: [], bananas: [], trees: [], acacias: [], eucalyptus: [], tea: [], tanks: [], lights: [], bushes: [], grass: [] });
    const towns = this.nodes.map((n) => ({ p: this.nodePos[n.id], sigma: n.type === 'junction' ? 2.2 : n.type === 'hospital' ? 5 : 3.8 }));
    const density = (x, z) => towns.reduce((sum, t) => sum + Math.exp(-((x - t.p.x) ** 2 + (z - t.p.z) ** 2) / (t.sigma * t.sigma)), 0);
    const inMap = (x, z) => Math.abs(x) < INNER_W / 2 - 1.5 && Math.abs(z) < INNER_H / 2 - 1.5;
    const pick = (list) => list[Math.floor(rand() * list.length)];
    const grove = (x, z, yaw, s) => {
      const fx = Math.sin(yaw), fz = Math.cos(yaw);
      const cx = x - fx * 0.75 * s, cz = z - fz * 0.75 * s;
      const count = 3 + Math.floor(rand() * 6);
      for (let k = 0; k < count && P.bananas.length < 5000; k++) {
        const a = rand() * Math.PI * 2, r = rand() * 0.45;
        const px = cx + Math.cos(a) * r, pz = cz + Math.sin(a) * r;
        if (this.roadClear(px, pz) < 0.15 || !this.occFree(px, pz, 0.09)) continue;
        this.occMark(px, pz, 0.08);
        P.bananas.push({ x: px, z: pz, s: 0.8 + rand() * 0.5, yaw: rand() * Math.PI * 2 });
      }
    };
    const tryBuilding = (x, z, yaw, s, shop) => {
      const r = (shop ? 0.62 : 0.5) * s;
      if (!inMap(x, z) || this.roadClear(x, z) < r * 0.8 || !this.occFree(x, z, r)) return false;
      this.occMark(x, z, r);
      if (shop) P.shops.push({ x, z, yaw, s, colour: pick(SHOP_COLOURS), sign: Math.floor(rand() * SHOP_SIGNS.length) });
      else {
        const brick = rand() < 0.35;
        const w = 0.72 + rand() * 0.16;
        P.houses.push({ x, z, yaw, s, brick, gable: rand() < 0.4, wall: brick ? '#ffffff' : pick(WALL_COLOURS), roof: pick(ROOF_COLOURS), w, d: 0.52 + rand() * 0.14 });
        if (rand() < 0.3) {
          const side = rand() < 0.5 ? 1 : -1;
          const lx = side * (w / 2 + 0.09) * s, lz = -0.1 * s;
          const c = Math.cos(yaw), sn = Math.sin(yaw);
          P.tanks.push({ x: x + lx * c + lz * sn, z: z - lx * sn + lz * c, s, blue: rand() < 0.45 });
        }
        if (rand() < 0.65) grove(x, z, yaw, s);
      }
      return true;
    };

    let lamps = 0;
    for (const e of this.edgeData) {
      const pts = e.points;
      const hw = ROAD_STYLE[e.road].width / 2;
      let acc = 0;
      for (let k = 1; k < pts.length - 1; k++) {
        acc += pts[k].distanceTo(pts[k - 1]);
        if (acc < 0.5) continue;
        acc = 0;
        const t = pts[k + 1].clone().sub(pts[k - 1]).normalize();
        const nrm = new THREE.Vector3(-t.z, 0, t.x);
        for (const side of [1, -1]) {
          const d = density(pts[k].x, pts[k].z);
          if (rand() > 0.08 + 0.85 * Math.min(1, d)) continue;
          const shop = d > 0.55 && rand() < 0.3;
          const s = shop ? 0.9 + rand() * 0.3 : 0.8 + rand() * 0.5;
          const off = hw + (shop ? 0.4 : 0.34) * s + 0.1 + rand() * 0.2;
          const q = pts[k].clone().addScaledVector(nrm, side * off);
          const fx = -side * nrm.x, fz = -side * nrm.z;
          tryBuilding(q.x, q.z, Math.atan2(fx, fz), s, shop);
        }
        if (e.road === 'paved' && density(pts[k].x, pts[k].z) > 0.35 && ++lamps % 3 === 0) {
          const side = (lamps / 3) % 2 ? 1 : -1;
          const q = pts[k].clone().addScaledVector(nrm, side * (hw + 0.1));
          if (this.occFree(q.x, q.z, 0.05)) {
            this.occMark(q.x, q.z, 0.05);
            P.lights.push({ x: q.x, z: q.z, yaw: Math.atan2(-side * nrm.x, -side * nrm.z) });
          }
        }
        if (e.road === 'paved' && rand() < 0.22) {
          const side = rand() < 0.5 ? 1 : -1;
          const q = pts[k].clone().addScaledVector(nrm, side * (hw + 0.22));
          if (this.occFree(q.x, q.z, 0.18) && this.roadClear(q.x, q.z) > 0.15) {
            this.occMark(q.x, q.z, 0.16);
            (rand() < 0.5 ? P.eucalyptus : P.trees).push({ x: q.x, z: q.z, s: 0.75 + rand() * 0.4, yaw: rand() * 6 });
          }
        }
      }
    }

    for (const n of this.nodes) {
      const centre = this.nodePos[n.id];
      const tries = n.type === 'hospital' ? 220 : n.type === 'health_centre' ? 150 : 50;
      const radius = n.type === 'hospital' ? 6.5 : n.type === 'health_centre' ? 4.8 : 3;
      for (let k = 0; k < tries; k++) {
        const a = rand() * Math.PI * 2, r = 1.4 + Math.pow(rand(), 0.8) * radius;
        const x = centre.x + Math.cos(a) * r, z = centre.z + Math.sin(a) * r;
        if (rand() > density(x, z) * 0.9) continue;
        tryBuilding(x, z, Math.round((rand() * Math.PI * 2) / (Math.PI / 2)) * (Math.PI / 2) + a, 0.75 + rand() * 0.5, false);
      }
    }

    for (const f of Object.values(this.facilities)) {
      const spots = f.hospital ? [[2.15, 0.4], [-2.35, 0.3], [2.15, -0.6], [-2.35, -0.6]] : [[1.6, -0.9], [-1.6, -1.2]];
      for (const [lx, lz] of spots) {
        const p = f.local(lx, lz);
        P.trees.push({ x: p.x, z: p.z, s: 0.65, yaw: rand() * 6 });
      }
    }

    for (let k = 0; k < 9 && P.tea.length < 6; k++) {
      const x = (rand() - 0.5) * 50, z = (rand() - 0.5) * 38;
      if (this.roadClear(x, z) < 2 || !this.occFree(x, z, 1.2)) continue;
      const tea = { x, z, rx: 2 + rand() * 1.6, rz: 1.2 + rand() * 1.0, rot: rand() * Math.PI };
      P.tea.push(tea);
      const c = Math.cos(tea.rot), s = Math.sin(tea.rot);
      for (let dz = -tea.rx; dz <= tea.rx; dz += CELL) {
        for (let dx = -tea.rx; dx <= tea.rx; dx += CELL) {
          const u = (dx * c + dz * s) / tea.rx, v = (-dx * s + dz * c) / tea.rz;
          if (u * u + v * v > 1) continue;
          const idx = this.cellIndex(x + dx, z + dz);
          if (idx >= 0 && !this.occ[idx]) this.occ[idx] = 2;
        }
      }
    }

    for (let lot = 0; lot < 20; lot++) {
      const cx = (rand() - 0.5) * (INNER_W - 6), cz = (rand() - 0.5) * (INNER_H - 6);
      if (this.roadClear(cx, cz) < 1) continue;
      const radius = 0.9 + rand() * 1.5;
      for (let k = 0; k < 90 && P.eucalyptus.length < 1800; k++) {
        const a = rand() * Math.PI * 2, r = Math.sqrt(rand()) * radius;
        const x = cx + Math.cos(a) * r, z = cz + Math.sin(a) * r;
        if (!inMap(x, z) || this.roadClear(x, z) < 0.2 || !this.occFree(x, z, 0.14)) continue;
        this.occMark(x, z, 0.13);
        P.eucalyptus.push({ x, z, s: 0.8 + rand() * 0.6, yaw: rand() * 6 });
      }
    }

    for (let k = 0; k < 3200 && P.trees.length + P.acacias.length < 1300; k++) {
      const x = (rand() - 0.5) * (INNER_W - 2), z = (rand() - 0.5) * (INNER_H - 2);
      if (this.roadClear(x, z) < 0.25 || !this.occFree(x, z, 0.24)) continue;
      const s = 0.7 + rand() * 0.6;
      this.occMark(x, z, 0.22 * s);
      (rand() < 0.22 ? P.acacias : P.trees).push({ x, z, s, yaw: rand() * 6 });
    }

    for (let k = 0; k < 9000 && P.bushes.length < 2600; k++) {
      const x = (rand() - 0.5) * (INNER_W - 2), z = (rand() - 0.5) * (INNER_H - 2);
      const clear = this.roadClear(x, z);
      if (clear < 0.08 || clear > 2.5 + density(x, z) * 2 || !this.occFree(x, z, 0.07)) continue;
      this.occMark(x, z, 0.06);
      P.bushes.push({ x, z, s: 0.5 + rand() * 0.7, yaw: rand() * 6 });
    }

    for (const e of this.edgeData) {
      const hw = ROAD_STYLE[e.road].width / 2;
      const pts = e.points;
      for (let k = 1; k < pts.length - 1 && P.grass.length < 70000; k++) {
        const t = pts[k + 1].clone().sub(pts[k - 1]).normalize();
        for (let n = 0; n < 34; n++) {
          const side = rand() < 0.5 ? 1 : -1;
          const off = hw + 0.05 + Math.pow(rand(), 1.6) * 3.2;
          const along = (rand() - 0.5) * pts[k].distanceTo(pts[k - 1]);
          const x = pts[k].x - t.z * off * side + t.x * along, z = pts[k].z + t.x * off * side + t.z * along;
          if (this.roadClear(x, z) < 0.04) continue;
          const idx = this.cellIndex(x, z);
          if (idx < 0 || (this.occ[idx] && rand() < 0.85)) continue;
          P.grass.push({ x, z, s: 0.035 + rand() * 0.045, yaw: rand() * 6, dry: rand() });
        }
      }
    }
  }

  paintGround() {
    const P = this.placement;
    const SIZE = 4096;
    const texture = canvasTexture(SIZE, SIZE, (ctx) => {
      const sx = SIZE / INNER_W, sz = SIZE / INNER_H;
      ctx.setTransform(sx, 0, 0, sz, (INNER_W / 2) * sx, (INNER_H / 2) * sz);
      const rand = mulberry32(77);
      ctx.fillStyle = '#5f8c38';
      ctx.fillRect(-INNER_W / 2, -INNER_H / 2, INNER_W, INNER_H);
      for (let i = 0; i < 2500; i++) {
        ctx.globalAlpha = 0.18;
        ctx.fillStyle = rand() < 0.5 ? '#7da046' : '#4c7a30';
        ctx.beginPath();
        ctx.arc((rand() - 0.5) * INNER_W, (rand() - 0.5) * INNER_H, 0.8 + rand() * 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      for (let i = 0; i < 15000; i++) {
        const x = (rand() - 0.5) * INNER_W, z = (rand() - 0.5) * INNER_H;
        const gx = baseHeight(x + 0.1, z) - baseHeight(x - 0.1, z);
        const gz = baseHeight(x, z + 0.1) - baseHeight(x, z - 0.1);
        const angle = Math.atan2(gz, gx) + Math.PI / 2 + (rand() - 0.5) * 0.5;
        const w = 0.5 + rand() * 1.4, h = 0.18 + rand() * 0.45;
        ctx.save();
        ctx.translate(x, z);
        ctx.rotate(angle);
        ctx.globalAlpha = 0.6 + rand() * 0.35;
        ctx.fillStyle = FIELD_COLOURS[Math.floor(rand() * FIELD_COLOURS.length)];
        ctx.fillRect(-w / 2, -h / 2, w, h);
        if (rand() < 0.4) {
          ctx.globalAlpha = 0.25;
          ctx.strokeStyle = '#2c4a1c';
          ctx.lineWidth = 0.015;
          for (let k = -h / 2 + 0.05; k < h / 2; k += 0.06) {
            ctx.beginPath();
            ctx.moveTo(-w / 2, k);
            ctx.lineTo(w / 2, k);
            ctx.stroke();
          }
        }
        ctx.restore();
      }
      ctx.globalAlpha = 1;
      for (const t of P.tea) {
        ctx.save();
        ctx.translate(t.x, t.z);
        ctx.rotate(t.rot);
        ctx.beginPath();
        ctx.ellipse(0, 0, t.rx, t.rz, 0, 0, Math.PI * 2);
        ctx.clip();
        ctx.fillStyle = '#3f7d2c';
        ctx.fillRect(-t.rx, -t.rz, t.rx * 2, t.rz * 2);
        ctx.strokeStyle = '#2a5a20';
        ctx.lineWidth = 0.05;
        for (let k = -t.rz; k < t.rz; k += 0.11) {
          ctx.beginPath();
          for (let x = -t.rx; x <= t.rx; x += 0.2) {
            const y = k + Math.sin(x * 1.3 + k) * 0.08;
            if (x === -t.rx) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
        ctx.restore();
      }
      ctx.globalAlpha = 0.5;
      ctx.fillStyle = '#35602d';
      for (const t of P.eucalyptus) {
        ctx.beginPath();
        ctx.arc(t.x, t.z, 0.26, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = '#4c7a2a';
      for (const b of P.bananas) {
        ctx.beginPath();
        ctx.arc(b.x, b.z, 0.13, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 0.55;
      ctx.fillStyle = '#a68d66';
      for (const h of [...P.houses, ...P.shops]) {
        ctx.beginPath();
        ctx.ellipse(h.x, h.z, 0.6 * h.s, 0.5 * h.s, h.yaw, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 0.4;
      for (const f of Object.values(this.facilities)) {
        const p = f.local(f.L.padX, f.L.padZ);
        ctx.fillStyle = '#a99674';
        ctx.beginPath();
        ctx.arc(p.x, p.z, f.L.inner * 0.9, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      for (const e of this.edgeData) {
        const style = ROAD_STYLE[e.road];
        for (const [extra, colour, alpha] of [[0.9, style.verge, 0.25], [0.32, style.verge, 1]]) {
          ctx.globalAlpha = alpha;
          ctx.strokeStyle = colour;
          ctx.lineWidth = style.width + extra;
          ctx.beginPath();
          e.points.forEach((p, k) => (k ? ctx.lineTo(p.x, p.z) : ctx.moveTo(p.x, p.z)));
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
    });
    texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    return texture;
  }

  buildTerrain() {
    const inner = new THREE.PlaneGeometry(INNER_W, INNER_H, 256, 208);
    inner.rotateX(-Math.PI / 2);
    const pos = inner.attributes.position;
    const colours = [];
    for (let i = 0; i < pos.count; i++) {
      const h = terrainHeight(pos.getX(i), pos.getZ(i));
      pos.setY(i, h);
      const shade = 0.8 + 0.2 * smoothstep((h - 0.3) / 3.5);
      colours.push(shade, shade, shade);
    }
    inner.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
    inner.computeVertexNormals();
    const groundMaterial = new THREE.MeshStandardMaterial({ map: this.paintGround(), vertexColors: true, roughness: 0.96 });
    const detail = this.tex.detail;
    groundMaterial.onBeforeCompile = (shader) => {
      shader.uniforms.detailMap = { value: detail };
      shader.vertexShader = `varying vec2 vDetailUv;\n${shader.vertexShader}`.replace('#include <begin_vertex>', '#include <begin_vertex>\nvDetailUv = position.xz;');
      shader.fragmentShader = `uniform sampler2D detailMap;\nvarying vec2 vDetailUv;\n${shader.fragmentShader}`.replace(
        '#include <map_fragment>',
        `#include <map_fragment>
        float fine = texture2D(detailMap, vDetailUv * 2.3).r * 1.6;
        float broad = texture2D(detailMap, vDetailUv * 0.21 + 0.37).r * 1.6;
        diffuseColor.rgb *= mix(1.0, fine, 0.55) * mix(1.0, broad, 0.45);`,
      );
    };
    const ground = new THREE.Mesh(inner, groundMaterial);
    ground.receiveShadow = true;
    this.scene.add(ground);

    const outer = new THREE.PlaneGeometry(240, 220, 120, 110);
    outer.rotateX(-Math.PI / 2);
    const op = outer.attributes.position;
    const oc = [];
    const a = new THREE.Color('#567f36'), b = new THREE.Color('#3c6a2e'), c = new THREE.Color('#7c9a48'), tmp = new THREE.Color();
    for (let i = 0; i < op.count; i++) {
      const x = op.getX(i), z = op.getZ(i);
      const inside = Math.abs(x) < INNER_W / 2 - 1e-3 && Math.abs(z) < INNER_H / 2 - 1e-3;
      op.setY(i, baseHeight(x, z) - (inside ? 3 : 0));
      const n = 0.5 + 0.5 * Math.sin(x * 0.31 + Math.cos(z * 0.27) * 2) * Math.cos(z * 0.23 - x * 0.11);
      tmp.copy(a).lerp(b, n).lerp(c, Math.max(0, Math.sin(x * 0.7 + z * 0.5)) * 0.3);
      oc.push(tmp.r, tmp.g, tmp.b);
    }
    outer.setAttribute('color', new THREE.Float32BufferAttribute(oc, 3));
    outer.computeVertexNormals();
    this.scene.add(new THREE.Mesh(outer, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1 })));
  }

  ribbon(buffer, pts, left, right, lift, vScale = 1, u0 = 0, u1 = 1) {
    const start = buffer.positions.length / 3;
    buffer.uvs ??= [];
    let v = 0;
    for (let k = 0; k < pts.length; k++) {
      const a = pts[Math.max(0, k - 1)], b = pts[Math.min(pts.length - 1, k + 1)];
      let dx = b.x - a.x, dz = b.z - a.z;
      const n = Math.hypot(dx, dz) || 1;
      dx /= n;
      dz /= n;
      if (k) v += pts[k].distanceTo(pts[k - 1]) * vScale;
      for (const [off, u] of [[left, u0], [right, u1]]) {
        const x = pts[k].x - dz * off, z = pts[k].z + dx * off;
        buffer.positions.push(x, terrainHeight(x, z) + lift, z);
        buffer.uvs.push(u, v);
      }
      if (k < pts.length - 1) {
        const q = start + k * 2;
        buffer.indices.push(q, q + 1, q + 2, q + 1, q + 3, q + 2);
      }
    }
  }

  dashed(buffer, pts, offset, width, on, off, lift) {
    let acc = 0, run = [];
    for (let k = 0; k < pts.length; k++) {
      if (k) acc += pts[k].distanceTo(pts[k - 1]);
      if (acc % (on + off) < on) run.push(pts[k]);
      else {
        if (run.length > 1) this.ribbon(buffer, run, offset - width / 2, offset + width / 2, lift);
        run = [];
      }
    }
    if (run.length > 1) this.ribbon(buffer, run, offset - width / 2, offset + width / 2, lift);
  }

  meshFromBuffer(buffer, material, shadow = true) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(buffer.positions, 3));
    if (buffer.uvs?.length) geometry.setAttribute('uv', new THREE.Float32BufferAttribute(buffer.uvs, 2));
    geometry.setIndex(buffer.indices);
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry, material);
    mesh.receiveShadow = shadow;
    this.scene.add(mesh);
    return mesh;
  }

  groundDisc(x, z, r, lift, material, uvScale = 1) {
    const positions = [x, terrainHeight(x, z) + lift, z];
    const uvs = [x * uvScale, z * uvScale];
    const indices = [];
    const segments = 28;
    for (let k = 0; k <= segments; k++) {
      const a = (k / segments) * Math.PI * 2;
      const px = x + Math.cos(a) * r, pz = z + Math.sin(a) * r;
      positions.push(px, terrainHeight(px, pz) + lift, pz);
      uvs.push(px * uvScale, pz * uvScale);
      if (k < segments) indices.push(0, k + 2, k + 1);
    }
    return this.meshFromBuffer({ positions, uvs, indices }, material);
  }

  buildRoads() {
    const surface = (road, factor) => {
      const murram = road !== 'paved';
      return new THREE.MeshStandardMaterial({
        map: murram ? this.tex.murram : this.tex.asphalt,
        color: road === 'rough' ? 0xc2a68c : 0xffffff,
        roughness: murram ? 0.97 : 0.88,
        metalness: 0,
        polygonOffset: true,
        polygonOffsetFactor: factor,
        polygonOffsetUnits: factor,
        side: THREE.DoubleSide,
      });
    };
    this.roadMaterials = {};
    const materialFor = (road) => (this.roadMaterials[road] ??= surface(road, -2));
    const surfaces = {}, shoulders = {}, paint = { positions: [], indices: [] };
    const drains = { paved: { positions: [], indices: [] }, murram: { positions: [], indices: [] } };
    const markers = [];
    for (const e of this.edgeData) {
      const style = ROAD_STYLE[e.road];
      const hw = style.width / 2;
      const vScale = 1 / style.width;
      this.ribbon((surfaces[e.road] ??= { positions: [], indices: [] }), e.points, -hw, hw, 0.03, vScale);
      const sw = e.road === 'paved' ? 0.16 : 0.12;
      if (drains[e.road]) {
        this.ribbon(drains[e.road], e.points, -hw - sw - 0.05, -hw - sw + 0.01, 0.012);
        this.ribbon(drains[e.road], e.points, hw + sw - 0.01, hw + sw + 0.05, 0.012);
      }
      if (e.road !== 'rough') {
        let acc = 0, side = 1;
        for (let k = 1; k < e.points.length - 1; k++) {
          acc += e.points[k].distanceTo(e.points[k - 1]);
          if (acc < 1.4) continue;
          acc = 0;
          side = -side;
          const a = e.points[k - 1], b = e.points[k + 1];
          const dx = b.x - a.x, dz = b.z - a.z, n = Math.hypot(dx, dz) || 1;
          const off = side * (hw + 0.1);
          const x = e.points[k].x - (dz / n) * off, z = e.points[k].z + (dx / n) * off;
          markers.push([x, terrainHeight(x, z), z]);
        }
      }
      const sb = (shoulders[e.road] ??= { positions: [], indices: [] });
      this.ribbon(sb, e.points, -hw - sw, -hw + 0.01, 0.028, 1.5, 1, 0);
      this.ribbon(sb, e.points, hw - 0.01, hw + sw, 0.028, 1.5, 0, 1);
      if (e.road === 'paved') {
        this.dashed(paint, e.points, 0, 0.016, 0.3, 0.45, 0.034);
        this.ribbon(paint, e.points, -hw + 0.03, -hw + 0.045, 0.034);
        this.ribbon(paint, e.points, hw - 0.045, hw - 0.03, 0.034);
      }
    }
    for (const [road, buffer] of Object.entries(surfaces)) this.meshFromBuffer(buffer, materialFor(road));
    const shoulderColour = { paved: 0x8a7458, murram: 0xa8673f, rough: 0x8d6c4a };
    for (const [road, buffer] of Object.entries(shoulders)) {
      this.meshFromBuffer(buffer, new THREE.MeshStandardMaterial({ color: shoulderColour[road], alphaMap: this.tex.shoulder, transparent: true, depthWrite: false, roughness: 1, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }));
    }
    this.meshFromBuffer(drains.paved, new THREE.MeshStandardMaterial({ color: 0x8f8d86, roughness: 0.95, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }));
    this.meshFromBuffer(drains.murram, new THREE.MeshStandardMaterial({ color: 0x6e3a20, roughness: 1, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }));
    const markerGeo = new THREE.BoxGeometry(0.012, 0.07, 0.012, 1, 7, 1).translate(0, 0.035, 0);
    const colours = markerGeo.attributes.position.array.length / 3;
    const tint = [];
    for (let i = 0; i < colours; i++) {
      const y = markerGeo.attributes.position.getY(i);
      tint.push(...(y > 0.05 ? [0.85, 0.1, 0.1] : [0.95, 0.95, 0.92]));
    }
    markerGeo.setAttribute('color', new THREE.Float32BufferAttribute(tint, 3));
    const posts = new THREE.InstancedMesh(markerGeo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6 }), Math.max(1, markers.length));
    posts.count = markers.length;
    const mm = new THREE.Matrix4();
    markers.forEach(([x, y, z], k) => posts.setMatrixAt(k, mm.makeTranslation(x, y, z)));
    posts.castShadow = true;
    this.scene.add(posts);
    this.paintMaterial = new THREE.MeshStandardMaterial({ color: 0xe9e8df, roughness: 0.7, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
    this.meshFromBuffer(paint, this.paintMaterial);
    this.nodes.forEach((n) => {
      const incident = this.edgeData.filter((e) => e.from === n.id || e.to === n.id);
      const best = incident.reduce((a, e) => (ROAD_STYLE[e.road].rank > ROAD_STYLE[a.road].rank ? e : a), incident[0]);
      const r = Math.max(...incident.map((e) => ROAD_STYLE[e.road].width / 2)) + 0.03;
      const p = this.nodePos[n.id];
      this.groundDisc(p.x, p.z, r, 0.031, materialFor(best.road), 1 / ROAD_STYLE[best.road].width);
    });
    this.buildStreetLights();
  }

  buildStreetLights() {
    const L = this.placement.lights;
    const n = Math.max(1, L.length);
    const metal = new THREE.MeshStandardMaterial({ color: 0x6b7178, roughness: 0.45, metalness: 0.7 });
    const pole = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.006, 0.01, 0.8, 6).translate(0, 0.4, 0), metal, n);
    const arm = new THREE.InstancedMesh(new THREE.BoxGeometry(0.008, 0.008, 0.16).translate(0, 0.79, 0.075), metal, n);
    this.lampMaterial = new THREE.MeshStandardMaterial({ color: 0xfff4d6, emissive: 0xffd9a0, emissiveIntensity: 0, roughness: 0.3 });
    const lamp = new THREE.InstancedMesh(new THREE.BoxGeometry(0.04, 0.012, 0.07).translate(0, 0.78, 0.15), this.lampMaterial, n);
    const pools = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.7, 0.7).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ map: this.tex.glow, color: 0xffc77a, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -6, polygonOffsetUnits: -6 }), n);
    const glowPos = [];
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0), one = new THREE.Vector3(1, 1, 1);
    L.forEach((l, k) => {
      const y = terrainHeight(l.x, l.z) - 0.01;
      q.setFromAxisAngle(up, l.yaw);
      m.compose(new THREE.Vector3(l.x, y, l.z), q, one);
      pole.setMatrixAt(k, m);
      arm.setMatrixAt(k, m);
      lamp.setMatrixAt(k, m);
      const hx = l.x + Math.sin(l.yaw) * 0.15, hz = l.z + Math.cos(l.yaw) * 0.15;
      glowPos.push(hx, y + 0.77, hz);
      m.compose(new THREE.Vector3(hx + Math.sin(l.yaw) * 0.08, terrainHeight(hx, hz) + 0.036, hz + Math.cos(l.yaw) * 0.08), new THREE.Quaternion(), one);
      pools.setMatrixAt(k, m);
    });
    for (const mesh of [pole, arm, lamp, pools]) {
      mesh.count = L.length;
      this.scene.add(mesh);
    }
    pole.castShadow = arm.castShadow = true;
    pools.renderOrder = 2;
    this.lampPools = pools;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(glowPos, 3));
    this.lampGlow = new THREE.Points(geometry, new THREE.PointsMaterial({ map: this.tex.glow, color: 0xffd59a, size: 0.35, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.scene.add(this.lampGlow);
  }

  block(w, h, d, floors, kind, accent, roofColor = 0xc4c3bd) {
    const mat = (t) => {
      const m = new THREE.MeshStandardMaterial({ map: t.map, emissiveMap: t.emissiveMap, emissive: 0xffffff, emissiveIntensity: 0, roughness: kind === 'glass' ? 0.2 : 0.8, metalness: kind === 'glass' ? 0.4 : 0 });
      this.windowMaterials.push(m);
      return m;
    };
    const front = mat(facadeTexture(w, floors, kind, accent));
    const side = mat(facadeTexture(d, floors, kind, accent));
    const roof = new THREE.MeshStandardMaterial({ color: roofColor, roughness: 0.9 });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), [side, side, roof, roof, front, front]);
    mesh.castShadow = mesh.receiveShadow = true;
    return mesh;
  }

  gableRoof(len, span, rise, roofMat, wallMat) {
    const shape = new THREE.Shape();
    shape.moveTo(-span / 2, 0);
    shape.lineTo(span / 2, 0);
    shape.lineTo(0, rise);
    shape.closePath();
    const geometry = new THREE.ExtrudeGeometry(shape, { depth: len, bevelEnabled: false });
    geometry.translate(0, 0, -len / 2);
    geometry.rotateY(Math.PI / 2);
    const mesh = new THREE.Mesh(geometry, [wallMat, roofMat]);
    mesh.castShadow = mesh.receiveShadow = true;
    return mesh;
  }

  buildFacilities() {
    const concrete = new THREE.MeshStandardMaterial({ color: 0xe4e2dc, roughness: 0.8 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x2b2e33, roughness: 0.7 });
    this.flags = [];
    for (const f of Object.values(this.facilities)) {
      const group = new THREE.Group();
      group.position.set(f.spot.x, f.padH, f.spot.z);
      group.rotation.y = Math.atan2(-f.dir.x, -f.dir.z);
      const add = (mesh, x, y, z, shadow = true) => {
        mesh.position.set(x, y, z);
        mesh.castShadow = shadow;
        mesh.receiveShadow = true;
        group.add(mesh);
        return mesh;
      };
      const box = (w, h, d, mat) => new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
      const plane = (w, h, mat) => new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
      const [yx0, yx1, yz0, yz1] = f.L.yard;
      const yard = add(plane(yx1 - yx0, yz1 - yz0, new THREE.MeshStandardMaterial({ map: forecourtTexture(f.L, f.hospital), roughness: 0.95, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 })), (yx0 + yx1) / 2, 0.012, (yz0 + yz1) / 2, false);
      yard.rotation.x = -Math.PI / 2;
      if (f.hospital) {
        add(this.block(3.8, 0.45, 1.8, 1, 'glass', null), 0, 0.225, 0);
        add(this.block(2.4, 1.65, 1.3, 5, 'tower', '#2fa84f'), 0.5, 0.45 + 0.825, -0.2);
        add(this.block(1.25, 0.66, 1.2, 2, 'tower', '#d62839'), -1.2, 0.45 + 0.33, -0.25);
        const star = new THREE.MeshStandardMaterial({ map: this.tex.star, transparent: true, alphaTest: 0.4, emissive: 0xff2030, emissiveMap: this.tex.star, emissiveIntensity: 0.25 });
        add(plane(0.5, 0.5, star), 0.5, 1.8, 0.455, false);
        add(plane(0.32, 0.32, star), -1.2, 0.86, 0.355, false);
        add(plane(0.5, 0.5, star), 1.705, 1.8, -0.2, false).rotation.y = Math.PI / 2;
        const nameTex = textTexture(f.name.toUpperCase(), { color: '#1b3a6b', bg: '#ffffff', font: 'bold 64px Arial' });
        add(plane(2.0, 0.12, new THREE.MeshStandardMaterial({ map: nameTex, emissive: 0xffffff, emissiveMap: nameTex, emissiveIntensity: 0 })), 0.75, 0.39, 0.905, false);
        this.windowMaterials.push(group.children[group.children.length - 1].material);
        add(box(1.9, 0.035, 0.75, concrete), -1.15, 0.42, 1.27);
        const em = new THREE.MeshStandardMaterial({ map: this.tex.emergency, emissive: 0xffffff, emissiveMap: this.tex.emergency, emissiveIntensity: 1.2 });
        add(plane(1.1, 0.11, em), -1.15, 0.49, 1.646, false);
        add(box(0.8, 0.03, 0.35, concrete), 0.9, 0.36, 1.07);
        add(plane(0.84, 0.84, new THREE.MeshStandardMaterial({ map: this.tex.helipad, roughness: 0.8 })), 0.5, 2.106, -0.2, false).rotation.x = -Math.PI / 2;
        for (const [x, z] of [[-1.5, -0.45], [-1.0, -0.1], [-1.45, 0.1]]) add(box(0.14, 0.08, 0.12, new THREE.MeshStandardMaterial({ color: 0x9da3a8 })), x, 1.15, z);
        const label = makeLabel(f.name, 'label hospital-label');
        label.position.copy(f.local(0.5, -0.2)).setY(f.padH + 2.45);
        this.scene.add(label);
        this.facilityLabels.push(label);
        this.hospitalLabels.push({ node: f.node, label, name: f.name });
      } else {
        add(this.block(2.2, 0.36, 0.85, 1, 'clinic', '#2f6fae'), 0, 0.18, -0.1);
        const wall = new THREE.MeshStandardMaterial({ color: 0xefe6cf, roughness: 0.85 });
        const roof = new THREE.MeshStandardMaterial({ color: 0x8e3424, roughness: 0.55, metalness: 0.35, flatShading: true });
        const greenRoof = new THREE.MeshStandardMaterial({ color: 0x2e6b4a, roughness: 0.55, metalness: 0.35, flatShading: true });
        add(this.gableRoof(2.45, 1.25, 0.24, roof, wall), 0, 0.36, 0);
        const pillar = new THREE.MeshStandardMaterial({ color: 0xf4f1e8, roughness: 0.7 });
        for (let k = 0; k < 6; k++) add(new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, 0.36, 8), pillar), -1.0 + k * 0.4, 0.18, 0.56);
        add(box(2.3, 0.03, 0.32, concrete), 0, 0.015, 0.47);
        add(this.block(1.1, 0.32, 0.6, 1, 'clinic', '#1f8a4c'), 0.45, 0.16, -0.95);
        add(this.gableRoof(1.25, 0.78, 0.18, greenRoof, wall), 0.45, 0.32, -0.95);
        add(box(0.16, 0.2, 0.16, dark), -0.85, 0.1, -0.95);
        add(new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.2, 16), new THREE.MeshStandardMaterial({ color: 0x15171a, roughness: 0.5 })), -0.85, 0.3, -0.95);
        const signTex = textTexture(f.name.toUpperCase(), { color: '#ffffff', bg: '#1f6f43', font: 'bold 54px Arial', sub: 'IKIGO NDERABUZIMA', h: 200 });
        add(plane(0.62, 0.2, new THREE.MeshStandardMaterial({ map: signTex, side: THREE.DoubleSide })), -1.7, 0.32, 0.25, false);
        for (const dx of [-0.27, 0.27]) add(new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, 0.42, 6), dark), -1.7 + dx, 0.21, 0.245);
        add(new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.8, 6), pillar), -1.75, 0.4, -0.6);
        const flag = add(plane(0.3, 0.2, new THREE.MeshStandardMaterial({ map: this.tex.flag, side: THREE.DoubleSide })), -1.6, 0.68, -0.6, false);
        this.flags.push(flag);
        const label = makeLabel(`${f.name.replace(' Health Centre', '')} <span class="small">Health Centre</span>`, 'label hc-label');
        label.position.copy(f.local(0, 0)).setY(f.padH + 0.9);
        this.scene.add(label);
        this.facilityLabels.push(label);
        const beaconPos = f.local(1.6, 0.4).setY(f.padH);
        this.buildBeacon(f.node, beaconPos);
      }
      this.scene.add(group);
      this.occluders.push(...group.children.filter((c) => {
        if (!c.isMesh || c.geometry.type === 'PlaneGeometry' || c.geometry.type === 'CylinderGeometry') return false;
        c.geometry.computeBoundingBox();
        const size = c.geometry.boundingBox.getSize(new THREE.Vector3());
        return size.y > 0.15 && Math.min(size.x, size.z) > 0.15;
      }));
    }
  }

  buildBeacon(nodeId, position) {
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 2.6, 16, 1, true), new THREE.MeshBasicMaterial({ color: 0xff3b3b, transparent: true, opacity: 0.6, depthWrite: false, blending: THREE.AdditiveBlending }));
    beam.position.copy(position).add(new THREE.Vector3(0, 1.3, 0));
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.12, 0.2, 40), new THREE.MeshBasicMaterial({ color: 0xff3b3b, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.copy(position).add(new THREE.Vector3(0, 0.025, 0));
    const people = new THREE.Group();
    people.position.copy(position);
    people.scale.setScalar(VEHICLE_SCALE);
    const mother = personModel(0xe86fa0, { pregnant: true, scarf: 0xf2c500 });
    const nurse = personModel(0xf4f6f8, { scarf: 0xffffff });
    const relative = personModel(0x2563eb, { skin: 0x3b2518 });
    mother.position.set(0, 0, 0);
    nurse.position.set(0.75, 0, 0.2);
    nurse.rotation.y = -1.2;
    relative.position.set(-0.7, 0, -0.25);
    relative.rotation.y = 0.9;
    people.add(mother, nurse, relative);
    const label = makeLabel('', 'beacon-label');
    label.position.copy(position).add(new THREE.Vector3(0, 2.75, 0));
    const group = new THREE.Group();
    group.add(beam, ring, people);
    group.visible = false;
    label.visible = false;
    this.scene.add(group, label);
    this.beacons[nodeId] = { group, beam, ring, label, mother, position: position.clone() };
  }

  buildScenery() {
    const P = this.placement;
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), c = new THREE.Color();
    const up = new THREE.Vector3(0, 1, 0);
    const rand = mulberry32(5);
    const inst = (geometry, material, n, shadow = true) => {
      const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, n));
      mesh.count = n;
      mesh.castShadow = shadow;
      mesh.receiveShadow = true;
      this.scene.add(mesh);
      return mesh;
    };
    const set = (mesh, k, x, y, z, quat, sx, sy, sz, colour) => {
      m.compose(p.set(x, y, z), quat, s.set(sx, sy, sz));
      mesh.setMatrixAt(k, m);
      if (colour !== undefined) mesh.setColorAt(k, colour);
    };
    const footing = (x, z, r) => Math.min(terrainHeight(x - r, z - r), terrainHeight(x + r, z - r), terrainHeight(x - r, z + r), terrainHeight(x + r, z + r)) - 0.02;

    const unitBox = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);
    const wallMat = (map) => new THREE.MeshStandardMaterial({ map, roughness: 0.92 });
    const plinth = new THREE.MeshStandardMaterial({ color: 0x6f5a4a, roughness: 1 });
    const plasterMats = [wallMat(this.tex.wallSide), wallMat(this.tex.wallSide), plinth, plinth, wallMat(this.tex.wallFront), wallMat(this.tex.wallSide)];
    const brickMats = [wallMat(this.tex.brickSide), wallMat(this.tex.brickSide), plinth, plinth, wallMat(this.tex.brickFront), wallMat(this.tex.brickSide)];
    const roofMat = new THREE.MeshStandardMaterial({ map: this.tex.roof, roughness: 0.5, metalness: 0.25, side: THREE.DoubleSide });
    const groups = { plaster: [], brick: [], hip: [], gable: [] };
    for (const h of P.houses) {
      groups[h.brick ? 'brick' : 'plaster'].push(h);
      groups[h.gable ? 'gable' : 'hip'].push(h);
    }
    const WALL_H = 0.28, ROOF_H = 0.17;
    const houseBase = (h) => footing(h.x, h.z, 0.32 * h.s);
    for (const [key, mats] of [['plaster', plasterMats], ['brick', brickMats]]) {
      const list = groups[key];
      const mesh = inst(unitBox, mats, list.length);
      this.occluders.push(mesh);
      list.forEach((h, k) => {
        q.setFromAxisAngle(up, h.yaw);
        set(mesh, k, h.x, houseBase(h), h.z, q, h.w * h.s, WALL_H * h.s + 0.02, h.d * h.s, c.set(h.wall));
      });
    }
    const roofTop = (h) => houseBase(h) + WALL_H * h.s + 0.02;
    const hips = inst(hipRoofGeometry(0.22), roofMat, groups.hip.length);
    groups.hip.forEach((h, k) => {
      q.setFromAxisAngle(up, h.yaw);
      set(hips, k, h.x, roofTop(h) - 0.004, h.z, q, h.w * h.s + 0.1 * h.s, ROOF_H * h.s, h.d * h.s + 0.12 * h.s, c.set(h.roof));
    });
    const gables = inst(gableRoofGeometry(), roofMat, groups.gable.length);
    const gableWalls = inst(gableWallGeometry(), new THREE.MeshStandardMaterial({ roughness: 0.95 }), groups.gable.length);
    groups.gable.forEach((h, k) => {
      q.setFromAxisAngle(up, h.yaw + Math.PI / 2);
      set(gables, k, h.x, roofTop(h) - 0.004, h.z, q, h.d * h.s + 0.1 * h.s, ROOF_H * 1.15 * h.s, h.w * h.s + 0.12 * h.s, c.set(h.roof));
      set(gableWalls, k, h.x, roofTop(h) - 0.006, h.z, q, h.d * h.s, ROOF_H * 1.15 * h.s, h.w * h.s, c.set(h.brick ? '#a85a3c' : h.wall));
    });

    const local = (h, lx, lz) => [h.x + lx * Math.cos(h.yaw) + lz * Math.sin(h.yaw), h.z - lx * Math.sin(h.yaw) + lz * Math.cos(h.yaw)];
    const segs = { wall: [], gate: [], slab: [], post: [], shade: [] };
    const hr = mulberry32(77);
    for (const h of P.houses) {
      const A = (h.w * h.s) / 2, B = (h.d * h.s) / 2, base = houseBase(h);
      const tint = h.brick ? '#b8664a' : h.wall;
      if (hr() < 0.5) {
        const [sx, sz] = local(h, 0, B + 0.07 * h.s);
        segs.slab.push({ x: sx, y: base, z: sz, yaw: h.yaw, w: A * 1.5, h: 0.02 * h.s, d: 0.14 * h.s, c: '#9b938a' });
        for (const side of [-1, 1]) {
          const [px, pz] = local(h, side * A * 0.68, B + 0.125 * h.s);
          segs.post.push({ x: px, y: base, z: pz, yaw: h.yaw, w: 0.012 * h.s, h: WALL_H * h.s * 0.82, d: 0.012 * h.s, c: '#ece6da' });
        }
        const [rx, rz] = local(h, 0, B + 0.075 * h.s);
        segs.shade.push({ x: rx, y: base + WALL_H * h.s * 0.84, z: rz, yaw: h.yaw, w: A * 1.6, h: 0.006, d: 0.17 * h.s, c: h.roof });
      }
      if (hr() < 0.38) {
        const m = 0.1 * h.s, WA = A + m, WB = B + m + 0.08 * h.s, H = 0.1 * h.s, T = 0.014, gate = 0.16 * h.s;
        const add = (lx, lz, w, d, list = segs.wall, colour = tint, height = H) => {
          const [x, z] = local(h, lx, lz);
          list.push({ x, y: footing(x, z, 0.02), z, yaw: h.yaw, w, h: height, d, c: colour });
        };
        add(0, -WB, WA * 2, T);
        add(-WA, 0, T, WB * 2);
        add(WA, 0, T, WB * 2);
        const half = (WA * 2 - gate) / 2;
        add(-WA + half / 2, WB, half, T);
        add(WA - half / 2, WB, half, T);
        add(0, WB, gate, T * 0.6, segs.gate, '#2f3a44', H * 1.05);
      }
    }
    const segMat = { wall: new THREE.MeshStandardMaterial({ map: this.tex.wallSide, roughness: 0.95 }), gate: new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.6 }), slab: new THREE.MeshStandardMaterial({ roughness: 1 }), post: new THREE.MeshStandardMaterial({ roughness: 0.9 }), shade: new THREE.MeshStandardMaterial({ map: this.tex.roof, roughness: 0.5, metalness: 0.3 }) };
    for (const [key, list] of Object.entries(segs)) {
      const mesh = inst(unitBox, segMat[key], list.length);
      list.forEach((b, k) => {
        q.setFromAxisAngle(up, b.yaw);
        if (key === 'shade') q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.18));
        set(mesh, k, b.x, b.y, b.z, q, b.w, b.h, b.d, c.set(b.c));
      });
    }

    const tanks = inst(new THREE.CylinderGeometry(0.05, 0.05, 0.13, 14).translate(0, 0.065, 0), new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.1 }), P.tanks.length);
    const tankStands = inst(new THREE.BoxGeometry(0.11, 0.05, 0.11).translate(0, 0.025, 0), new THREE.MeshStandardMaterial({ color: 0x9a9890, roughness: 0.95 }), P.tanks.length);
    P.tanks.forEach((t, k) => {
      const y = terrainHeight(t.x, t.z) - 0.01;
      set(tankStands, k, t.x, y, t.z, q.identity(), t.s, t.s, t.s);
      set(tanks, k, t.x, y + 0.05 * t.s, t.z, q, t.s, t.s, t.s, c.set(t.blue ? '#1e4fb8' : '#1b1e22'));
    });

    const shopSide = wallMat(this.tex.shopSide);
    const shopTop = new THREE.MeshStandardMaterial({ color: 0x8d8c86, roughness: 0.9 });
    const awningMat = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.3, side: THREE.DoubleSide });
    const awnings = inst(new THREE.BoxGeometry(1, 0.012, 0.24).translate(0, 0, 0.12), awningMat, P.shops.length);
    const posts = inst(new THREE.CylinderGeometry(0.006, 0.006, 0.3, 5).translate(0, 0.15, 0), new THREE.MeshStandardMaterial({ color: 0x3b3f45, metalness: 0.5, roughness: 0.5 }), P.shops.length * 2);
    const SHOP_H = 0.42;
    SHOP_SIGNS.forEach((sign, si) => {
      const list = P.shops.filter((h) => h.sign === si);
      const front = wallMat(shopFrontTexture(sign));
      const mesh = inst(unitBox, [shopSide, shopSide, shopTop, plinth, front, shopSide], list.length);
      this.occluders.push(mesh);
      list.forEach((h, k) => {
        q.setFromAxisAngle(up, h.yaw);
        set(mesh, k, h.x, footing(h.x, h.z, 0.36 * h.s), h.z, q, h.s, SHOP_H * h.s, 0.7 * h.s);
      });
    });
    const tilt = new THREE.Quaternion(), fx = new THREE.Vector3(1, 0, 0);
    P.shops.forEach((h, k) => {
      const y = footing(h.x, h.z, 0.36 * h.s);
      const sx = Math.sin(h.yaw), sz = Math.cos(h.yaw);
      q.setFromAxisAngle(up, h.yaw).multiply(tilt.setFromAxisAngle(fx, 0.28));
      set(awnings, k, h.x + sx * 0.35 * h.s, y + 0.3 * h.s, h.z + sz * 0.35 * h.s, q, h.s * 0.96, h.s, h.s, c.set(SHOP_SIGNS[h.sign].awning));
      for (const side of [-1, 1]) {
        const ox = Math.cos(h.yaw) * side * 0.45 * h.s, oz = -Math.sin(h.yaw) * side * 0.45 * h.s;
        set(posts, k * 2 + (side + 1) / 2, h.x + sx * 0.57 * h.s + ox, y, h.z + sz * 0.57 * h.s + oz, q.identity(), h.s, h.s * 0.85, h.s);
      }
    });

    const stems = inst(new THREE.CylinderGeometry(0.018, 0.028, 0.24, 5).translate(0, 0.12, 0), new THREE.MeshStandardMaterial({ color: 0x7d8f3a, roughness: 0.9 }), P.bananas.length);
    const leaves = inst(leafGeometry(), addWind(new THREE.MeshStandardMaterial({ color: 0xffffff, side: THREE.DoubleSide, roughness: 0.7 }), this.wind, 0.03, 0.9), P.bananas.length * 6);
    const e = new THREE.Euler();
    P.bananas.forEach((b, k) => {
      const y = terrainHeight(b.x, b.z) - 0.01;
      set(stems, k, b.x, y, b.z, q.identity(), b.s, b.s, b.s);
      for (let l = 0; l < 6; l++) {
        e.set(-0.15 + rand() * 0.3, b.yaw + (l / 6) * Math.PI * 2 + rand() * 0.4, 0, 'YXZ');
        q.setFromEuler(e);
        set(leaves, k * 6 + l, b.x, y + 0.22 * b.s, b.z, q, b.s, b.s, b.s, c.setHSL(0.24 + rand() * 0.05, 0.55, 0.36 + rand() * 0.1));
      }
    });

    const foliage = (amplitude) => addWind(new THREE.MeshStandardMaterial({ map: this.tex.leaf, vertexColors: true, roughness: 0.82 }), this.wind, amplitude, 0.9);
    const cardMat = (amplitude) => addWind(new THREE.MeshStandardMaterial({ map: this.tex.leafCluster, vertexColors: true, roughness: 0.78, alphaTest: 0.42, side: THREE.DoubleSide }), this.wind, amplitude, 0.9);
    const cardDepth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: this.tex.leafCluster, alphaTest: 0.42 });
    const crownPair = (geometry, cards, size, seed, amplitude, n, shadow = true) => {
      const inner = inst(geometry, foliage(amplitude), n, shadow);
      const outer = inst(foliageCards(geometry, cards, size, seed), cardMat(amplitude), n, shadow);
      outer.customDepthMaterial = cardDepth;
      return [inner, outer];
    };
    const setPair = (pair, k, x, y, z, quat, sx, sy, sz, colour) => {
      set(pair[0], k, x, y, z, quat, sx * 0.9, sy * 0.9, sz * 0.9, colour.clone().multiplyScalar(0.7));
      set(pair[1], k, x, y, z, quat, sx, sy, sz, colour);
    };
    const bark = new THREE.MeshStandardMaterial({ color: 0x5a4130, roughness: 1 });
    const trunkGeo = new THREE.CylinderGeometry(0.02, 0.04, 0.5, 7).translate(0, 0.25, 0);
    const trunks = inst(trunkGeo, bark, P.trees.length);
    const branches = inst(branchGeometry(17, 4, 0.5), bark, P.trees.length);
    const variants = [crownGeometry(11, 5, 0.2, 0.82), crownGeometry(29, 4, 0.16, 0.9), crownGeometry(47, 5, 0.22, 0.75)];
    const byVariant = variants.map(() => []);
    P.trees.forEach((t, k) => {
      const y = terrainHeight(t.x, t.z) - 0.02;
      set(trunks, k, t.x, y, t.z, q.identity(), t.s, t.s, t.s);
      q.setFromAxisAngle(up, t.yaw);
      set(branches, k, t.x, y, t.z, q, t.s, t.s, t.s);
      byVariant[k % variants.length].push({ ...t, y });
    });
    variants.forEach((geometry, v) => {
      const list = byVariant[v];
      const pair = crownPair(geometry, 150, 0.075, 900 + v, 0.05, list.length);
      list.forEach((t, k) => {
        q.setFromAxisAngle(up, t.yaw);
        setPair(pair, k, t.x, t.y + 0.56 * t.s, t.z, q, t.s * 1.15, t.s * 1.1, t.s * 1.15, c.setHSL(0.22 + rand() * 0.08, 0.38 + rand() * 0.18, 0.3 + rand() * 0.1));
      });
    });

    const aTrunks = inst(new THREE.CylinderGeometry(0.012, 0.026, 0.55, 6).translate(0, 0.275, 0), bark, P.acacias.length);
    const aBranches = inst(branchGeometry(41, 5, 0.55), bark, P.acacias.length);
    const aPair = crownPair(crownGeometry(83, 8, 0.32, 0.32), 180, 0.07, 950, 0.035, P.acacias.length);
    P.acacias.forEach((t, k) => {
      const y = terrainHeight(t.x, t.z) - 0.02;
      q.setFromAxisAngle(up, t.yaw).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.08));
      set(aTrunks, k, t.x, y, t.z, q, t.s, t.s, t.s);
      set(aBranches, k, t.x, y, t.z, q, t.s, t.s, t.s);
      q.setFromAxisAngle(up, t.yaw);
      setPair(aPair, k, t.x, y + 0.58 * t.s, t.z, q, t.s * 1.3, t.s, t.s * 1.3, c.setHSL(0.2 + rand() * 0.04, 0.4, 0.3 + rand() * 0.06));
    });

    const eTrunks = inst(new THREE.CylinderGeometry(0.011, 0.026, 1.3, 6).translate(0, 0.65, 0), new THREE.MeshStandardMaterial({ color: 0xd6cbb5, roughness: 0.85 }), P.eucalyptus.length);
    const ePair = crownPair(crownGeometry(23, 3, 0.07, 1.9), 110, 0.06, 970, 0.06, P.eucalyptus.length);
    P.eucalyptus.forEach((t, k) => {
      const y = terrainHeight(t.x, t.z) - 0.02;
      set(eTrunks, k, t.x, y, t.z, q.identity(), t.s, t.s, t.s);
      q.setFromAxisAngle(up, t.yaw);
      setPair(ePair, k, t.x, y + 1.18 * t.s, t.z, q, t.s * 0.75, t.s * 0.75, t.s * 0.75, c.setHSL(0.28 + rand() * 0.05, 0.22, 0.4 + rand() * 0.08));
    });

    const bPair = crownPair(crownGeometry(131, 3, 0.12, 0.7, 0), 60, 0.09, 990, 0.02, P.bushes.length, false);
    P.bushes.forEach((b, k) => {
      q.setFromAxisAngle(up, b.yaw);
      setPair(bPair, k, b.x, terrainHeight(b.x, b.z) + 0.02 * b.s, b.z, q, b.s * 0.3, b.s * 0.26, b.s * 0.3, c.setHSL(0.22 + rand() * 0.08, 0.45, 0.26 + rand() * 0.1));
    });

    const flowerList = P.grass.filter((g, k) => k % 9 === 0);
    const flowers = inst(flowerGeometry(), new THREE.MeshStandardMaterial({ map: this.tex.flower, alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.8 }), flowerList.length, false);
    const petals = ['#ffffff', '#f6d32d', '#c084fc', '#f97316', '#fda4af'];
    flowerList.forEach((g, k) => {
      q.setFromAxisAngle(up, g.yaw);
      set(flowers, k, g.x + 0.02, terrainHeight(g.x, g.z) - 0.003, g.z - 0.02, q, g.s * 0.5, g.s * 0.55, g.s * 0.5, c.set(petals[k % petals.length]));
    });
    this.flowers = flowers;

    const tuft = grassTuftGeometry();
    const grassMat = addWind(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, side: THREE.DoubleSide }), this.wind, 0.18);
    const dry = new THREE.Color('#c9b26a'), green = new THREE.Color('#ffffff');
    const chunks = new Map();
    for (const g of P.grass) {
      const key = `${Math.floor(g.x / GRASS_CHUNK)},${Math.floor(g.z / GRASS_CHUNK)}`;
      if (!chunks.has(key)) chunks.set(key, []);
      chunks.get(key).push(g);
    }
    this.grassChunks = [];
    for (const list of chunks.values()) {
      const mesh = inst(tuft, grassMat, list.length, false);
      const centre = new THREE.Vector3();
      list.forEach((g, k) => {
        q.setFromAxisAngle(up, g.yaw);
        const y = terrainHeight(g.x, g.z) - 0.004;
        set(mesh, k, g.x, y, g.z, q, g.s, g.s, g.s, c.copy(green).lerp(dry, g.dry * g.dry * 0.8).multiplyScalar(0.82 + rand() * 0.3));
        centre.add(p.set(g.x, y, g.z));
      });
      mesh.computeBoundingSphere();
      this.grassChunks.push({ mesh, centre: centre.multiplyScalar(1 / list.length) });
    }
  }

  updateGrass() {
    const eye = this.camera.position;
    const reach = this.cameraMode === 'overview' ? 0 : this.cameraMode === 'top' ? 9 : 13;
    for (const chunk of this.grassChunks) chunk.mesh.visible = chunk.centre.distanceTo(eye) < reach;
    if (this.flowers) this.flowers.visible = reach > 0 && this.camera.position.y - terrainHeight(eye.x, eye.z) < 2.5;
  }

  buildAmbulances(state) {
    const shared = {
      paint: new THREE.MeshPhysicalMaterial({ color: 0xf5f5f0, roughness: 0.35, metalness: 0.05, clearcoat: 1, clearcoatRoughness: 0.08 }),
      livery: new THREE.MeshPhysicalMaterial({ map: this.tex.livery, roughness: 0.35, metalness: 0.05, clearcoat: 1, clearcoatRoughness: 0.08 }),
      contact: new THREE.MeshBasicMaterial({ map: this.tex.glow, color: 0x000000, transparent: true, opacity: 0.6, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -8, polygonOffsetUnits: -8 }),
      dark: new THREE.MeshStandardMaterial({ color: 0x24272b, roughness: 0.8 }),
      glass: new THREE.MeshStandardMaterial({ color: 0x16202a, roughness: 0.08, metalness: 0.7, side: THREE.DoubleSide }),
      tyre: new THREE.MeshStandardMaterial({ color: 0x161616, roughness: 0.92 }),
      hub: new THREE.MeshStandardMaterial({ map: this.tex.hub, roughness: 0.4, metalness: 0.6 }),
      rear: new THREE.MeshStandardMaterial({ map: this.tex.rear, roughness: 0.35, metalness: 0.2 }),
      grille: new THREE.MeshStandardMaterial({ map: this.tex.grille, roughness: 0.5, metalness: 0.4 }),
      hood: new THREE.MeshStandardMaterial({ map: this.tex.hood, transparent: true, alphaTest: 0.3 }),
      side: new THREE.MeshStandardMaterial({ map: this.tex.side, transparent: true, alphaTest: 0.3 }),
    };
    const shape = new THREE.Shape();
    shape.moveTo(2.97, 0.45);
    shape.lineTo(2.45, 0.45);
    shape.absarc(1.95, 0.45, 0.5, 0, Math.PI, false);
    shape.lineTo(-1.4, 0.45);
    shape.absarc(-1.9, 0.45, 0.5, 0, Math.PI, false);
    shape.lineTo(-2.95, 0.45);
    shape.lineTo(-2.95, 2.55);
    shape.lineTo(0.95, 2.55);
    shape.quadraticCurveTo(1.3, 2.55, 1.48, 2.22);
    shape.lineTo(2.18, 1.36);
    shape.quadraticCurveTo(2.85, 1.2, 2.95, 0.98);
    shape.lineTo(2.97, 0.45);
    const shellGeo = new THREE.ExtrudeGeometry(shape, { depth: 1.84, bevelEnabled: true, bevelThickness: 0.08, bevelSize: 0.08, bevelSegments: 3, curveSegments: 12 });
    shellGeo.translate(0, 0, -0.92);
    shellGeo.rotateY(-Math.PI / 2);
    const tyreGeo = new THREE.CylinderGeometry(0.4, 0.4, 0.28, 22);
    tyreGeo.rotateZ(Math.PI / 2);
    for (const amb of state.ambulances) {
      const view = this.ambulanceModel(shared, shellGeo, tyreGeo);
      view.label.element.textContent = `A${amb.id + 1}`;
      view.label.element.addEventListener('click', () => this.onAmbulanceClick?.(amb.id));
      this.scene.add(view.root, view.halo);
      this.ambulances.push(view);
    }
  }

  ambulanceModel(shared, shellGeo, tyreGeo) {
    const root = new THREE.Group();
    root.rotation.order = 'YXZ';
    const model = new THREE.Group();
    model.scale.setScalar(VEHICLE_SCALE);
    root.add(model);
    const body = new THREE.Group();
    model.add(body);
    const add = (parent, geometry, material, x, y, z, shadow = true) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(x, y, z);
      mesh.castShadow = shadow;
      parent.add(mesh);
      return mesh;
    };
    const box = (w, h, d) => new THREE.BoxGeometry(w, h, d);
    add(body, shellGeo, [shared.livery, shared.paint], 0, 0, 0);
    add(body, box(1.6, 0.3, 5.2), shared.dark, 0, 0.5, 0, false);
    add(body, new THREE.PlaneGeometry(1.75, 1.0), shared.glass, 0, 1.844, 1.896, false).rotation.x = -0.683;
    add(body, new THREE.PlaneGeometry(1.5, 0.45), shared.hood, 0, 1.25, 2.61, false).rotation.x = -1.112;
    const cabWindow = new THREE.Shape();
    cabWindow.moveTo(0.5, 1.55);
    cabWindow.lineTo(1.78, 1.55);
    cabWindow.lineTo(1.34, 2.25);
    cabWindow.lineTo(0.5, 2.25);
    cabWindow.lineTo(0.5, 1.55);
    const cabGeo = new THREE.ShapeGeometry(cabWindow);
    for (const sideSign of [1, -1]) {
      const w = add(body, cabGeo, shared.glass, sideSign * 1.008, 0, 0, false);
      w.rotation.y = (sideSign * Math.PI) / 2;
      w.scale.x = -sideSign;
      add(body, box(0.02, 0.75, 0.08), shared.dark, sideSign * 1.01, 1.9, 0.48, false);
      add(body, box(0.3, 0.42, 0.02), shared.dark, sideSign * 0.88, 0.36, -2.42, false);
      add(body, box(0.3, 0.42, 0.02), shared.dark, sideSign * 0.88, 0.36, 1.42, false);
      add(body, new THREE.PlaneGeometry(2.3, 0.32), shared.side, sideSign * 1.006, 2.22, -0.75, false).rotation.y = (sideSign * Math.PI) / 2;
      add(body, box(0.25, 0.04, 0.04), shared.dark, sideSign * 1.1, 1.75, 1.55);
      add(body, box(0.06, 0.28, 0.2), shared.dark, sideSign * 1.22, 1.72, 1.55);
    }
    add(body, new THREE.PlaneGeometry(1.1, 0.32), shared.grille, 0, 0.72, 3.056, false);
    add(body, box(2.05, 0.24, 0.22), shared.dark, 0, 0.5, 3.0);
    add(body, box(2.05, 0.22, 0.2), shared.dark, 0, 0.48, -3.0);
    add(body, new THREE.PlaneGeometry(1.85, 1.95), shared.rear, 0, 1.55, -3.036, false).rotation.y = Math.PI;
    add(body, box(0.9, 0.18, 0.9), shared.paint, 0, 2.72, -1.2);
    add(body, box(1.7, 0.1, 0.38), shared.dark, 0, 2.67, 1.0);
    const red = new THREE.MeshStandardMaterial({ color: 0xff2a2a, emissive: 0xff1010, emissiveIntensity: 0.2, roughness: 0.3 });
    const blue = new THREE.MeshStandardMaterial({ color: 0x2a5cff, emissive: 0x1040ff, emissiveIntensity: 0.2, roughness: 0.3 });
    const brake = new THREE.MeshStandardMaterial({ color: 0xaa1010, emissive: 0xff1a1a, emissiveIntensity: 0.15 });
    const head = new THREE.MeshStandardMaterial({ color: 0xfff6d8, emissive: 0xfff1c4, emissiveIntensity: 0.4 });
    add(body, box(0.72, 0.15, 0.32), red, 0.44, 2.79, 1.0, false);
    add(body, box(0.72, 0.15, 0.32), blue, -0.44, 2.79, 1.0, false);
    add(body, box(0.22, 0.12, 0.14), red, 0.8, 2.65, -2.85, false);
    add(body, box(0.22, 0.12, 0.14), blue, -0.8, 2.65, -2.85, false);
    add(body, box(0.16, 0.07, 0.04), blue, 0.35, 0.67, 3.07, false);
    add(body, box(0.16, 0.07, 0.04), blue, -0.35, 0.67, 3.07, false);
    add(body, box(0.42, 0.2, 0.08), head, 0.68, 0.86, 3.0, false);
    add(body, box(0.42, 0.2, 0.08), head, -0.68, 0.86, 3.0, false);
    add(body, box(0.14, 0.55, 0.05), brake, 0.92, 1.05, -3.04, false);
    add(body, box(0.14, 0.55, 0.05), brake, -0.92, 1.05, -3.04, false);
    const glow = (colour, x, y, z, size) => {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.tex.glow, color: colour, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
      sprite.position.set(x, y, z);
      sprite.scale.set(size, size, size);
      body.add(sprite);
      return sprite;
    };
    const glows = {
      red: [glow(0xff2020, 0.44, 2.85, 1.0, 2.6), glow(0xff2020, 0.8, 2.7, -2.9, 1.4)],
      blue: [glow(0x2050ff, -0.44, 2.85, 1.0, 2.6), glow(0x2050ff, -0.8, 2.7, -2.9, 1.4), glow(0x2050ff, 0.35, 0.67, 3.15, 1.2), glow(0x2050ff, -0.35, 0.67, 3.15, 1.2)],
    };
    const wheels = [];
    for (const [x, z, front] of [[0.88, 1.95, true], [-0.88, 1.95, true], [0.88, -1.9, false], [-0.88, -1.9, false]]) {
      const steer = new THREE.Group();
      steer.position.set(x, 0.4, z);
      const spin = new THREE.Group();
      const tyre = new THREE.Mesh(tyreGeo, [shared.tyre, shared.hub, shared.hub]);
      tyre.castShadow = true;
      spin.add(tyre);
      steer.add(spin);
      model.add(steer);
      wheels.push({ steer, spin, front });
    }
    const poolMat = new THREE.MeshBasicMaterial({ map: this.tex.glow, color: 0xfff1c4, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false });
    const pool = new THREE.Mesh(new THREE.PlaneGeometry(4.5, 9), poolMat);
    pool.rotation.x = -Math.PI / 2;
    pool.position.set(0, 0.12, 7.5);
    model.add(pool);
    const contact = new THREE.Mesh(new THREE.PlaneGeometry(3.4, 7.6), shared.contact);
    contact.rotation.x = -Math.PI / 2;
    contact.position.y = 0.04;
    model.add(contact);
    const care = new THREE.Group();
    const stretcher = stretcherModel();
    stretcher.position.set(0, 0, -4.4);
    const medic = personModel(0xf4f6f8, { scarf: 0xffffff });
    medic.position.set(1.0, 0, -4.1);
    medic.rotation.y = -1.4;
    const medic2 = personModel(0x15803d, { skin: 0x3b2518 });
    medic2.position.set(-1.0, 0, -4.6);
    medic2.rotation.y = 1.4;
    care.add(stretcher, medic, medic2);
    care.visible = false;
    model.add(care);
    const label = makeLabel('', 'amb-label');
    label.position.set(0, 0.42, 0);
    root.add(label);
    const halo = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.42, 36), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7, side: THREE.DoubleSide, depthWrite: false, depthTest: false }));
    halo.rotation.x = -Math.PI / 2;
    halo.renderOrder = 5;
    return { root, body, care, wheels, red, blue, brake, head, glows, pool, label, halo, heading: 0, pos: new THREE.Vector3(), wheelAngle: 0, steer: 0, pitch: 0, dustAcc: 0, init: false, moving: false };
  }

  curve(u, v) {
    return this.curveMap.get(`${u},${v}`);
  }

  buildLeg(index, wps) {
    const firstNode = wps[0][1], lastNode = wps[wps.length - 1][1];
    const startFac = this.facilities[firstNode], endFac = this.facilities[lastNode];
    const pts = [];
    const nodeIdx = [];
    if (startFac) pts.push(...this.exitPath(startFac, index).map((p) => p.clone()));
    else pts.push(this.nodePos[firstNode].clone());
    nodeIdx.push(pts.length - 1);
    const segInfo = [];
    for (let k = 0; k < wps.length - 1; k++) {
      const edge = this.curve(wps[k][1], wps[k + 1][1]);
      if (edge) for (let m = 1; m < edge.points.length; m++) pts.push(edge.points[m].clone());
      else pts.push(this.nodePos[wps[k + 1][1]].clone());
      nodeIdx.push(pts.length - 1);
      const minutes = Math.max(1e-3, wps[k + 1][0] - wps[k][0]);
      segInfo.push({ road: edge ? edge.road : 'paved', kmh: edge ? edge.km / (minutes / 60) : 30 });
    }
    if (endFac) pts.push(...this.entryPath(endFac, index).slice(1).map((p) => p.clone()));
    const cum = new Float64Array(pts.length);
    for (let k = 1; k < pts.length; k++) cum[k] = cum[k - 1] + pts[k].distanceTo(pts[k - 1]);
    const L = cum[pts.length - 1];
    const turn = new Float64Array(pts.length);
    for (let k = 1; k < pts.length - 1; k++) {
      if (cum[k] - cum[k - 1] < 1e-5 || cum[k + 1] - cum[k] < 1e-5) continue;
      const a1 = Math.atan2(pts[k].x - pts[k - 1].x, pts[k].z - pts[k - 1].z);
      const a2 = Math.atan2(pts[k + 1].x - pts[k].x, pts[k + 1].z - pts[k].z);
      turn[k] = Math.abs(Math.atan2(Math.sin(a2 - a1), Math.cos(a2 - a1)));
    }
    const R = 0.4;
    const density = new Float64Array(pts.length);
    let lo = 0, hi = 0, sum = 0;
    for (let k = 0; k < pts.length; k++) {
      while (hi < pts.length && cum[hi] <= cum[k] + R) sum += turn[hi++];
      while (cum[lo] < cum[k] - R) sum -= turn[lo++];
      density[k] = 1 + Math.min(4, (sum / (2 * R)) * 1.3);
    }
    const effort = new Float64Array(pts.length);
    for (let k = 1; k < pts.length; k++) effort[k] = effort[k - 1] + ((density[k] + density[k - 1]) / 2) * (cum[k] - cum[k - 1]);
    const t0 = wps[0][0], tN = wps[wps.length - 1][0], T = tN - t0;
    const eS = startFac ? Math.min(T * 0.12, 1.5) : 0;
    const eE = endFac ? Math.min(T * 0.12, 1.5) : 0;
    const remap = (t) => t0 + eS + ((t - t0) / T) * (T - eS - eE);
    const knots = [];
    if (startFac) knots.push({ t: t0, s: 0 });
    wps.forEach(([wt], k) => knots.push({ t: remap(wt), s: cum[nodeIdx[k]] }));
    if (endFac) knots.push({ t: tN, s: L });
    const segs = [];
    if (startFac) segs.push({ s0: 0, s1: cum[nodeIdx[0]], road: 'driveway', kmh: 15 });
    segInfo.forEach((info, k) => segs.push({ s0: cum[nodeIdx[k]], s1: cum[nodeIdx[k + 1]], ...info }));
    if (endFac) segs.push({ s0: cum[nodeIdx[nodeIdx.length - 1]], s1: L, road: 'driveway', kmh: 15 });
    return { t0, t1: tN, pts, cum, L, knots, segs, density, effort, meanDensity: effort[pts.length - 1] / (L || 1), sRoad0: cum[nodeIdx[0]], sRoad1: cum[nodeIdx[nodeIdx.length - 1]], endNode: lastNode };
  }

  tripFor(index, waypoints) {
    const key = `${waypoints[0][0]}|${waypoints.length}|${waypoints[waypoints.length - 1][0]}`;
    const cached = this.trips[index];
    if (cached && cached.key === key) return cached;
    if (cached) for (const leg of cached.legs) if (leg.ribbon) {
      this.scene.remove(leg.ribbon);
      leg.ribbon.geometry.dispose();
      leg.ribbon.material.dispose();
    }
    const legs = [];
    let i = 0;
    while (i < waypoints.length - 1) {
      if (waypoints[i + 1][1] === waypoints[i][1]) {
        i++;
        continue;
      }
      let j = i;
      while (j < waypoints.length - 1 && waypoints[j + 1][1] !== waypoints[j][1]) j++;
      if (waypoints[j][0] > waypoints[i][0]) legs.push(this.buildLeg(index, waypoints.slice(i, j + 1)));
      i = j;
    }
    this.trips[index] = { key, legs };
    return this.trips[index];
  }

  pointAt(leg, s) {
    const cum = leg.cum;
    if (s <= 0) return leg.pts[0].clone();
    if (s >= leg.L) return leg.pts[leg.pts.length - 1].clone();
    let lo = 0, hi = cum.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] <= s) lo = mid;
      else hi = mid;
    }
    const span = cum[hi] - cum[lo] || 1;
    return leg.pts[lo].clone().lerp(leg.pts[hi], (s - cum[lo]) / span);
  }

  smoothPoint(leg, s) {
    const r = Math.max(0, Math.min(0.35, s, leg.L - s));
    const out = new THREE.Vector3();
    for (let k = -2; k <= 2; k++) out.add(this.pointAt(leg, s + (k * r) / 2));
    return out.multiplyScalar(0.2);
  }

  lanePoint(leg, s) {
    const c = this.smoothPoint(leg, s);
    const taper = smoothstep((s - leg.sRoad0) / 0.6) * smoothstep((leg.sRoad1 - s) / 0.6);
    if (taper <= 0) return c;
    const a = this.smoothPoint(leg, Math.min(leg.L, s + 0.05)), b = this.smoothPoint(leg, Math.max(0, s - 0.05));
    const dx = a.x - b.x, dz = a.z - b.z, n = Math.hypot(dx, dz) || 1;
    c.x += (-dz / n) * LANE_OFFSET * taper;
    c.z += (dx / n) * LANE_OFFSET * taper;
    return c;
  }

  legIndex(leg, s) {
    const cum = leg.cum;
    let lo = 0, hi = cum.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] <= s) lo = mid;
      else hi = mid;
    }
    return [lo, hi, (s - cum[lo]) / (cum[hi] - cum[lo] || 1)];
  }

  effortAt(leg, s) {
    const [lo, hi, f] = this.legIndex(leg, clamp(s, 0, leg.L));
    return leg.effort[lo] + (leg.effort[hi] - leg.effort[lo]) * f;
  }

  densityAt(leg, s) {
    const [lo, hi, f] = this.legIndex(leg, clamp(s, 0, leg.L));
    return leg.density[lo] + (leg.density[hi] - leg.density[lo]) * f;
  }

  sAtEffort(leg, e) {
    const E = leg.effort;
    let lo = 0, hi = E.length - 1;
    if (e <= 0) return 0;
    if (e >= E[hi]) return leg.L;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (E[mid] <= e) lo = mid;
      else hi = mid;
    }
    return leg.cum[lo] + (leg.cum[hi] - leg.cum[lo]) * ((e - E[lo]) / (E[hi] - E[lo] || 1));
  }

  sLinear(leg, t) {
    const k = leg.knots;
    if (t <= k[0].t) return k[0].s;
    for (let i = 0; i < k.length - 1; i++) {
      if (t <= k[i + 1].t) {
        const f = (t - k[i].t) / Math.max(1e-6, k[i + 1].t - k[i].t);
        const e0 = this.effortAt(leg, k[i].s), e1 = this.effortAt(leg, k[i + 1].s);
        return this.sAtEffort(leg, e0 + f * (e1 - e0));
      }
    }
    return k[k.length - 1].s;
  }

  legS(leg, tw) {
    const d = Math.max(0, Math.min(0.35, (tw - leg.t0) / 2, (leg.t1 - tw) / 2));
    let sum = 0;
    for (let k = -2; k <= 2; k++) sum += this.sLinear(leg, tw + k * d);
    return sum / 5;
  }

  legPose(leg, t) {
    const T = leg.t1 - leg.t0;
    const u = clamp((t - leg.t0) / T, 0, 1);
    const a = Math.min(0.3, 2.0 / T);
    const vmax = 1 / (1 - a);
    let w, v;
    if (u < a) {
      w = (vmax * u * u) / (2 * a);
      v = (vmax * u) / a;
    } else if (u > 1 - a) {
      const r = 1 - u;
      w = 1 - (vmax * r * r) / (2 * a);
      v = (vmax * r) / a;
    } else {
      w = vmax * (u - a / 2);
      v = vmax;
    }
    const s = this.legS(leg, leg.t0 + w * T);
    const pos = this.lanePoint(leg, s);
    const ahead = this.lanePoint(leg, Math.min(leg.L, s + 0.06));
    const behind = this.lanePoint(leg, Math.max(0, s - 0.06));
    const heading = Math.atan2(ahead.x - behind.x, ahead.z - behind.z);
    const far = this.lanePoint(leg, Math.min(leg.L, s + 0.3));
    const turn = Math.atan2(Math.sin(Math.atan2(far.x - ahead.x, far.z - ahead.z) - heading), Math.cos(Math.atan2(far.x - ahead.x, far.z - ahead.z) - heading));
    const steer = clamp(Math.atan((0.385 * turn) / 0.24), -0.55, 0.55);
    const seg = leg.segs.find((sg) => s >= sg.s0 && s <= sg.s1) || leg.segs[leg.segs.length - 1];
    const here = this.densityAt(leg, s);
    const speed = clamp((v / vmax) * (leg.meanDensity / here), 0, 1.4);
    const braking = u > 1 - a || this.densityAt(leg, s + 0.25) > here * 1.25;
    return { pos, heading, moving: true, speed, kmh: seg.kmh * speed, road: seg.road, steer, braking, leg, s };
  }

  poseAt(index, trip, waypoints, t) {
    for (const leg of trip.legs) if (t >= leg.t0 && t < leg.t1) return this.legPose(leg, t);
    let node = waypoints[0][1];
    for (const [wt, n] of waypoints) if (wt <= t) node = n;
    const f = this.facilities[node];
    const view = this.ambulances[index];
    if (f) return { pos: this.slotPos(f, index), heading: f.laneHeading, moving: false, speed: 0, kmh: 0, road: null, steer: 0 };
    return { pos: this.nodePos[node].clone(), heading: view.heading, moving: false, speed: 0, kmh: 0, road: null, steer: 0 };
  }

  legRibbon(leg, colour) {
    if (!leg.ribbon) {
      const step = 0.08;
      const n = Math.max(1, Math.ceil(leg.L / step));
      const positions = [], indices = [];
      for (let k = 0; k <= n; k++) {
        const sv = Math.min(leg.L, k * step);
        const c = this.lanePoint(leg, sv);
        const a = this.lanePoint(leg, Math.min(leg.L, sv + 0.04)), b = this.lanePoint(leg, Math.max(0, sv - 0.04));
        let dx = a.x - b.x, dz = a.z - b.z;
        const len = Math.hypot(dx, dz) || 1;
        dx /= len;
        dz /= len;
        for (const off of [-0.022, 0.022]) {
          const x = c.x - dz * off, z = c.z + dx * off;
          positions.push(x, terrainHeight(x, z) + 0.05, z);
        }
        if (k < n) {
          const q = k * 2;
          indices.push(q, q + 1, q + 2, q + 1, q + 3, q + 2);
        }
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.setIndex(indices);
      leg.ribbon = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.7, depthWrite: false, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -6, polygonOffsetUnits: -6 }));
      leg.ribbonStep = step;
      leg.ribbonCount = n * 6;
      this.scene.add(leg.ribbon);
    }
    leg.ribbon.material.color.set(colour);
    return leg.ribbon;
  }

  buildDust() {
    const N = 700;
    this.dust = { n: N, next: 0, pos: new Float32Array(N * 3), vel: new Float32Array(N * 3), age: new Float32Array(N).fill(99), life: new Float32Array(N).fill(1), size: new Float32Array(N), alpha: new Float32Array(N) };
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.dust.pos, 3));
    geometry.setAttribute('aSize', new THREE.BufferAttribute(this.dust.size, 1));
    geometry.setAttribute('aAlpha', new THREE.BufferAttribute(this.dust.alpha, 1));
    this.dustMaterial = new THREE.ShaderMaterial({ uniforms: { uColor: { value: new THREE.Color(0xc49a72) }, uScale: { value: 600 } }, vertexShader: DUST_VERTEX, fragmentShader: DUST_FRAGMENT, transparent: true, depthWrite: false });
    this.dustPoints = new THREE.Points(geometry, this.dustMaterial);
    this.dustPoints.frustumCulled = false;
    this.scene.add(this.dustPoints);
    this.updateDustScale();
  }

  updateDustScale() {
    if (!this.dustMaterial) return;
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.dustMaterial.uniforms.uScale.value = size.y / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
  }

  emitDust(x, y, z, vx, vy, vz) {
    const d = this.dust, k = d.next;
    d.next = (k + 1) % d.n;
    d.pos.set([x, y, z], k * 3);
    d.vel.set([vx, vy, vz], k * 3);
    d.age[k] = 0;
    d.life[k] = 1.3 + Math.random() * 0.9;
  }

  updateDust(dt) {
    const d = this.dust;
    for (let k = 0; k < d.n; k++) {
      if (d.age[k] >= d.life[k]) {
        d.alpha[k] = 0;
        continue;
      }
      d.age[k] += dt;
      for (let a = 0; a < 3; a++) d.pos[k * 3 + a] += d.vel[k * 3 + a] * dt;
      d.vel[k * 3] *= 0.97;
      d.vel[k * 3 + 2] *= 0.97;
      const f = d.age[k] / d.life[k];
      d.size[k] = 0.06 + d.age[k] * 0.22;
      d.alpha[k] = 0.4 * (1 - f) * Math.min(1, d.age[k] * 8);
    }
    const attrs = this.dustPoints.geometry.attributes;
    attrs.position.needsUpdate = attrs.aSize.needsUpdate = attrs.aAlpha.needsUpdate = true;
  }

  buildRain() {
    const N = 2600;
    this.rainSize = { box: 30, height: 14 };
    const positions = new Float32Array(N * 6);
    for (let i = 0; i < N; i++) {
      const x = (Math.random() - 0.5) * 30, y = Math.random() * 14, z = (Math.random() - 0.5) * 30;
      positions.set([x, y, z, x + 0.02, y + 0.2, z], i * 6);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.rain = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0xaec6dc, transparent: true, opacity: 0.5 }));
    this.rain.frustumCulled = false;
    this.rain.visible = false;
    this.scene.add(this.rain);
  }

  updateRain(dt) {
    const arr = this.rain.geometry.attributes.position.array;
    const fall = 9 * dt;
    for (let i = 0; i < arr.length; i += 6) {
      arr[i + 1] -= fall;
      arr[i + 4] -= fall;
      if (arr[i + 1] < 0) {
        arr[i + 1] += 14;
        arr[i + 4] += 14;
      }
    }
    this.rain.geometry.attributes.position.needsUpdate = true;
    const k = this.tracking && this.cameraMode !== 'top' ? 0.6 : 2.4;
    this.rain.scale.set(k, 1, k);
    this.rain.position.set(this.controls.target.x, this.controls.target.y - 3, this.controls.target.z);
  }

  buildSky() {
    this.skyUniforms = { top: { value: new THREE.Color() }, horizon: { value: new THREE.Color() } };
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(600, 32, 16), new THREE.ShaderMaterial({ side: THREE.BackSide, depthWrite: false, fog: false, uniforms: this.skyUniforms, vertexShader: SKY_VERTEX, fragmentShader: SKY_FRAGMENT }));
    this.sky.frustumCulled = false;
    this.sky.renderOrder = -1;
    this.scene.add(this.sky);
    const rand = mulberry32(9);
    const puffs = 6;
    this.clouds = Array.from({ length: 32 }, () => ({ x: (rand() - 0.5) * 260, z: (rand() - 0.5) * 220, y: 30 + rand() * 14, s: 3 + rand() * 4, offsets: Array.from({ length: puffs }, () => [(rand() - 0.5) * 2.2, (rand() - 0.5) * 0.5, (rand() - 0.5) * 1.2, 0.6 + rand() * 0.6]) }));
    this.cloudMesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2), new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0.92 }), this.clouds.length * puffs);
    this.cloudMesh.frustumCulled = false;
    this.scene.add(this.cloudMesh);
    this.updateClouds(0);
  }

  updateClouds(dt) {
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3();
    let k = 0;
    for (const c of this.clouds) {
      c.x += dt * 0.7;
      if (c.x > 130) c.x -= 260;
      for (const [ox, oy, oz, os] of c.offsets) {
        m.compose(p.set(c.x + ox * c.s, c.y + oy * c.s, c.z + oz * c.s), q, s.set(os * c.s, os * c.s * 0.45, os * c.s * 0.8));
        this.cloudMesh.setMatrixAt(k++, m);
      }
    }
    this.cloudMesh.instanceMatrix.needsUpdate = true;
  }

  updateEnvironment(state, t) {
    const hour = (t % 1440) / 60;
    const daylight = clamp(Math.sin(((hour - 6) / 12) * Math.PI) * 1.6 + 0.1, 0, 1);
    const dusk = Math.min(1, Math.max(0, 1 - Math.abs(hour - 18.2) / 1.3) + Math.max(0, 1 - Math.abs(hour - 6) / 1.3));
    const rain = state.raining;
    const top = this.skyUniforms.top.value, horizon = this.skyUniforms.horizon.value;
    top.copy(SKY.nightTop).lerp(SKY.dayTop, daylight);
    horizon.copy(SKY.nightHorizon).lerp(SKY.dayHorizon, daylight).lerp(SKY.dusk, dusk * 0.55 * (rain ? 0.3 : 1));
    if (rain) {
      top.lerp(SKY.rainTop, 0.75 * daylight + 0.15);
      horizon.lerp(SKY.rainHorizon, 0.75 * daylight + 0.15);
    }
    this.scene.fog.color.copy(horizon);
    const near = this.tracking && this.cameraMode !== 'top';
    this.scene.fog.near = (near ? 16 : 45) * (rain ? 0.55 : 1);
    this.scene.fog.far = (near ? 90 : 190) * (rain ? 0.6 : 1);
    const angle = ((hour - 6) / 12) * Math.PI;
    if (daylight > 0.02) this.sunDir.set(-Math.cos(angle) * 0.8, Math.max(0.18, Math.sin(angle)), 0.45).normalize();
    else this.sunDir.set(0.3, 0.85, -0.4).normalize();
    this.sun.color.copy(SKY.moon).lerp(SKY.sun, daylight).lerp(SKY.sunset, dusk * 0.6 * daylight);
    this.sun.intensity = 0.4 * (1 - daylight) + (rain ? 0.9 : 2.4) * daylight;
    this.hemi.intensity = 0.25 + 0.35 * daylight;
    this.hemi.color.copy(SKY.moon).lerp(new THREE.Color(0xdfeeff), daylight);
    for (const material of this.windowMaterials) material.emissiveIntensity = (1 - daylight) * 1.1;
    this.rain.visible = rain;
    const night = 1 - daylight;
    this.wind.value = performance.now() / 1000;
    this.bloom.strength = 0.12 + 0.55 * night;
    this.bloom.threshold = 0.92 - 0.25 * night;
    this.renderer.toneMappingExposure = 1.0 + 0.15 * night;
    const wet = rain ? 1 : 0;
    this.roadMaterials.paved.roughness = 0.88 - 0.5 * wet;
    this.roadMaterials.paved.color.setScalar(1 - 0.25 * wet);
    for (const road of ['murram', 'rough']) if (this.roadMaterials[road]) this.roadMaterials[road].roughness = 0.97 - 0.35 * wet;
    const lit = smoothstep((night - 0.6) / 0.25);
    this.lampMaterial.emissiveIntensity = lit * 4;
    this.lampGlow.material.opacity = lit * 0.9;
    this.lampPools.material.opacity = lit * 0.55;
    this.updateEnvMap(top, horizon, daylight);
    for (const flag of this.flags) flag.rotation.y = Math.sin(performance.now() * 0.003 + flag.position.x) * 0.25;
    return { daylight, night: 1 - daylight, raining: rain };
  }

  updateEnvMap(top, horizon, daylight) {
    const key = [top.r, top.g, top.b, horizon.r, horizon.g, horizon.b].map((v) => Math.round(v * 40)).join(',');
    const now = performance.now();
    if (key === this.envKey || now - this.envAt < 1500) return;
    this.envKey = key;
    this.envAt = now;
    if (!this.envSky) {
      this.envUniforms = { top: { value: new THREE.Color() }, horizon: { value: new THREE.Color() } };
      this.envSky = new THREE.Mesh(new THREE.SphereGeometry(50, 32, 16), new THREE.ShaderMaterial({ side: THREE.BackSide, depthWrite: false, uniforms: this.envUniforms, vertexShader: SKY_VERTEX, fragmentShader: SKY_FRAGMENT }));
      this.envGround = new THREE.Mesh(new THREE.CircleGeometry(48, 32).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x3d4a2c }));
      this.envGround.position.y = -2;
      this.envScene.add(this.envSky, this.envGround);
    }
    this.envUniforms.top.value.copy(top);
    this.envUniforms.horizon.value.copy(horizon);
    this.envGround.material.color.setHex(0x3d4a2c).multiplyScalar(0.15 + 0.85 * daylight);
    const target = this.pmrem.fromScene(this.envScene, 0, 0.1, 100);
    this.envTarget?.dispose();
    this.envTarget = target;
    this.scene.environment = target.texture;
  }

  updateAmbulance(i, amb, referrals, t, dt, elapsed, env) {
    const view = this.ambulances[i];
    const trip = this.tripFor(i, amb.waypoints);
    const pose = this.poseAt(i, trip, amb.waypoints, t);
    const wps = amb.waypoints;
    const active = t >= wps[0][0] && t < wps[wps.length - 1][0];
    const referral = amb.referral !== null ? referrals[amb.referral] : null;
    const task = active ? (amb.referral !== null ? 'referral' : 'general_duty') : 'available';
    const siren = task === 'referral' && referral && referral.hospital_arrival !== null && t < referral.hospital_arrival;
    const travelled = Math.hypot(pose.pos.x - view.pos.x, pose.pos.z - view.pos.z);
    const jump = !view.init || travelled > 4;
    if (jump) {
      view.heading = pose.heading;
      view.init = true;
    } else view.heading = lerpAngle(view.heading, pose.heading, 1 - Math.exp(-dt * 7));
    view.pos.copy(pose.pos);
    view.moving = pose.moving;
    if (pose.moving) this.movingCount++;
    const p = pose.pos;
    const fx = Math.sin(view.heading), fz = Math.cos(view.heading);
    const hF = terrainHeight(p.x + fx * 0.2, p.z + fz * 0.2), hB = terrainHeight(p.x - fx * 0.2, p.z - fz * 0.2);
    const hL = terrainHeight(p.x + fz * 0.1, p.z - fx * 0.1), hR = terrainHeight(p.x - fz * 0.1, p.z + fx * 0.1);
    const y = (hF + hB + hL + hR) / 4;
    view.pos.y = y;
    view.root.position.set(p.x, y, p.z);
    view.root.rotation.set(-Math.atan2(hF - hB, 0.4), view.heading, Math.atan2(hL - hR, 0.2));
    view.steer += ((pose.moving ? pose.steer : 0) - view.steer) * (1 - Math.exp(-dt * 8));
    view.body.rotation.z = clamp(view.steer * 0.08 * pose.speed, -0.05, 0.05);
    view.pitch += ((pose.moving ? (pose.braking ? 0.012 : pose.speed < 0.95 ? -0.008 : 0) : 0) - view.pitch) * (1 - Math.exp(-dt * 5));
    view.body.rotation.x = view.pitch;
    const bump = pose.moving ? (ROAD_STYLE[pose.road]?.bump ?? 0.1) * pose.speed : 0;
    view.body.position.y = (Math.sin(elapsed * 21 + i * 3) * 0.025 + Math.sin(elapsed * 13.7 + i) * 0.015) * bump;
    if (!jump) view.wheelAngle += travelled / (0.4 * VEHICLE_SCALE);
    for (const w of view.wheels) {
      w.spin.rotation.x = view.wheelAngle;
      if (w.front) w.steer.rotation.y = view.steer;
    }
    const cycle = (elapsed * 1.5 + i * 0.13) % 1;
    const redOn = siren && (cycle < 0.1 || (cycle > 0.16 && cycle < 0.26));
    const blueOn = siren && ((cycle > 0.5 && cycle < 0.6) || (cycle > 0.66 && cycle < 0.76));
    view.red.emissiveIntensity = redOn ? 6 : 0.2;
    view.blue.emissiveIntensity = blueOn ? 6 : 0.2;
    for (const g of view.glows.red) g.material.opacity = redOn ? 0.95 : 0;
    for (const g of view.glows.blue) g.material.opacity = blueOn ? 0.95 : 0;
    view.brake.emissiveIntensity = pose.moving && pose.braking ? 3 : 0.15 + env.night;
    view.head.emissiveIntensity = 0.4 + env.night * 3;
    view.pool.material.opacity = env.night * (pose.moving || siren ? 0.5 : 0.2);
    const withReferral = task === 'referral' && referral && referral.pickup_time !== null;
    const loading = withReferral && !pose.moving && t >= referral.pickup_time && t < referral.hospital_arrival;
    const handover = withReferral && !pose.moving && t >= referral.hospital_arrival;
    const onboard = withReferral && t >= referral.pickup_time && t < referral.hospital_arrival;
    view.care.visible = Boolean(loading || handover);
    view.label.element.className = `amb-label ${task}${siren ? ' siren' : ''}`;
    const text = `A${i + 1}${onboard ? ' 🤰' : ''}`;
    if (view.label.element.textContent !== text) view.label.element.textContent = text;
    view.label.visible = !(i === this.followIndex && this.cameraMode === 'cinematic');
    view.halo.visible = this.cameraMode === 'overview' || this.cameraMode === 'top';
    if (view.halo.visible) {
      const k = clamp(this.camera.position.distanceTo(p) / 22, 1, 3.5);
      view.halo.scale.set(k, k, k);
      view.halo.position.set(p.x, y + 0.04, p.z);
      view.halo.material.color.set(task === 'referral' ? 0xff4d6d : task === 'general_duty' ? 0xffd23f : 0xffffff);
    }
    if (pose.moving && !env.raining && (pose.road === 'murram' || pose.road === 'rough') && pose.speed > 0.15) {
      view.dustAcc += dt * 45 * pose.speed;
      while (view.dustAcc >= 1) {
        view.dustAcc -= 1;
        const side = Math.random() < 0.5 ? 1 : -1;
        this.emitDust(p.x - fx * 0.3 + fz * 0.09 * side, y + 0.03, p.z - fz * 0.3 - fx * 0.09 * side, -fx * 0.08 + (Math.random() - 0.5) * 0.08, 0.04 + Math.random() * 0.06, -fz * 0.08 + (Math.random() - 0.5) * 0.08);
      }
    }
    for (const leg of trip.legs) if (leg.ribbon) leg.ribbon.visible = false;
    if (siren && pose.moving && pose.leg) {
      const ribbon = this.legRibbon(pose.leg, PRIORITY_COLORS[referral.priority]);
      const start = Math.min(pose.leg.ribbonCount, Math.floor(pose.s / pose.leg.ribbonStep) * 6);
      ribbon.geometry.setDrawRange(start, pose.leg.ribbonCount - start);
      ribbon.visible = true;
    }
    let phase = 'Parked at base, available';
    if (task === 'general_duty') phase = 'On general duty (non-referral task)';
    else if (task === 'referral' && referral) {
      const hospital = this.hospitalNames[referral.hospital] ?? 'hospital';
      const origin = referral.origin_name.replace(' Health Centre', ' HC');
      if (t < referral.pickup_time) phase = `Driving to ${origin} for patient #${referral.id}`;
      else if (!pose.moving && t < referral.hospital_arrival) phase = `Loading patient #${referral.id} at ${origin}`;
      else if (t < referral.hospital_arrival) phase = `Patient #${referral.id} on board → ${hospital}`;
      else if (!pose.moving) phase = `Handing over patient #${referral.id} at ${hospital}`;
      else phase = 'Returning to base';
    }
    this.info[i] = { moving: pose.moving, kmh: Math.round(pose.kmh), road: pose.road, phase, siren, task, loading, handover, onboard, referral: referral?.id ?? null, priority: referral?.priority ?? null };
  }

  update(state, t, elapsed, dt) {
    if (!this.built) return;
    const referrals = Object.fromEntries(state.referrals.map((r) => [r.id, r]));
    const env = this.updateEnvironment(state, t);
    this.movingCount = 0;
    state.ambulances.forEach((amb, i) => this.updateAmbulance(i, amb, referrals, t, dt, elapsed, env));

    const waiting = {};
    for (const r of state.referrals) {
      const resolvedAt = r.pickup_time ?? (r.status === 'unresolved' ? r.arrival + 480 : null);
      if (r.arrival <= t && (resolvedAt === null || resolvedAt > t)) (waiting[r.origin] ??= []).push(r);
    }
    for (const [nodeId, beacon] of Object.entries(this.beacons)) {
      const list = waiting[nodeId] || [];
      beacon.group.visible = beacon.label.visible = list.length > 0;
      if (!list.length) continue;
      beacon.mother.rotation.y = Math.sin(elapsed * 0.8) * 0.25;
      const top = Math.min(...list.map((r) => r.priority));
      const colour = PRIORITY_COLORS[top];
      beacon.beam.material.color.set(colour);
      beacon.ring.material.color.set(colour);
      beacon.beam.material.opacity = 0.3 + 0.35 * (0.5 + 0.5 * Math.sin(elapsed * 5));
      const k = 1 + 0.35 * Math.sin(elapsed * 5);
      beacon.ring.scale.set(k, k, k);
      beacon.label.element.style.background = `#${colour.toString(16).padStart(6, '0')}`;
      beacon.label.element.style.color = top === 1 ? 'white' : '#111';
      beacon.label.element.textContent = `🚨 ${list.map((r) => `#${r.id} ${r.priority_name} · ${Math.round(t - r.arrival)} min`).join('  |  ')}`;
    }
    for (const h of this.hospitalLabels) {
      const info = state.hospitals.find((x) => x.node === h.node);
      h.label.element.innerHTML = `${h.name}<br><span class="small">🛏 ${info.free_beds}/${info.capacity} free · surgery ${info.surgery_ready ? '✓' : '✗'} · neonatal ${info.neonatal_ready ? '✓' : '✗'}</span>`;
    }
    this.dustMaterial.uniforms.uColor.value.setHex(0xc49a72).multiplyScalar(1 - 0.82 * env.night);
    this.updateDust(dt);
    if (env.raining) this.updateRain(dt);
    this.updateClouds(dt);
    this.updateCamera(dt);
    this.updateGrass();
    this.updateShadows();
    this.fadeLabels();
  }

  updateShadows() {
    const extent = { chase: Math.round(6 + this.followDistance * 1.5), cinematic: 9, top: Math.round(8 + this.followDistance * 2) }[this.cameraMode] ?? 38;
    if (extent !== this.shadowExtent) {
      this.shadowExtent = extent;
      Object.assign(this.sun.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent });
      this.sun.shadow.camera.updateProjectionMatrix();
    }
    const focus = this.controls.target;
    this.sun.target.position.copy(focus);
    this.sun.position.copy(focus).addScaledVector(this.sunDir, 80);
  }

  fadeLabels() {
    const mode = this.cameraMode;
    for (const label of this.facilityLabels) {
      const d = this.camera.position.distanceTo(label.position);
      const o = mode === 'overview' ? 1 : 1 - smoothstep((d - 9) / 14);
      label.element.style.opacity = o.toFixed(2);
      label.visible = o > 0.02;
    }
  }

  nearestFacility(p) {
    let best = null, bestD = 4;
    for (const f of Object.values(this.facilities)) {
      const d = Math.hypot(f.spot.x - p.x, f.spot.z - p.z);
      if (d < bestD) {
        bestD = d;
        best = f;
      }
    }
    return best;
  }

  pickSubject() {
    if (performance.now() < (this.pinnedUntil ?? 0)) return this.followIndex;
    let best = this.followIndex, bestScore = -Infinity;
    this.ambulances.forEach((view, i) => {
      const info = this.info[i] ?? {};
      const score = (info.loading || info.handover ? 5 : 0) + (info.siren ? 4 : 0) + (view.moving ? 2.5 : 0) + (info.task === 'general_duty' ? 0.6 : 0) + (i === this.followIndex ? 3 : 0);
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    });
    return best;
  }

  nextShot() {
    const subject = this.pickSubject();
    if (subject !== this.followIndex) {
      this.followIndex = subject;
      this.onFollowChange?.(subject);
    }
    const view = this.ambulances[subject];
    const info = this.info[subject] ?? {};
    const type = info.loading || info.handover ? 'loading' : view.moving ? 'chase' : 'orbit';
    this.shot = { type, t: 0, dur: type === 'chase' ? 40 : 20, side: 1, angle: Math.atan2(this.camOffset.x, this.camOffset.z) };
  }

  updateCamera(dt) {
    const mode = this.cameraMode;
    if (mode === 'overview') {
      if (this.cameraGoal) {
        const k = 1 - Math.exp(-dt * 3);
        this.camera.position.lerp(this.cameraGoal.pos, k);
        this.controls.target.lerp(this.cameraGoal.target, k);
        if (this.camera.position.distanceTo(this.cameraGoal.pos) < 0.1) this.cameraGoal = null;
      }
      this.setFov(50, dt);
      return;
    }
    let view = this.ambulances[this.followIndex];
    if (!view || !view.init) return;
    if (!this.settled) {
      this.settled = true;
      this.cut = true;
    }
    if (mode === 'cinematic') {
      if (!this.shot || this.shot.t > this.shot.dur) this.nextShot();
      view = this.ambulances[this.followIndex];
      this.shot.t += dt;
      const info = this.info[this.followIndex] ?? {};
      const want = info.loading || info.handover ? 'loading' : view.moving ? 'chase' : 'orbit';
      if (want !== this.shot.type && this.shot.t > 2) this.shot.t = this.shot.dur + 1;
      if (view.moving) this.camYaw = lerpAngle(this.camYaw, view.heading, 1 - Math.exp(-dt * 1.4));
    }
    const style = mode;
    const anchor = view.pos;
    if (this.anchorIndex !== undefined && this.anchorIndex !== this.followIndex && !this.cut) {
      const span = this.camera.position.distanceTo(anchor);
      this.flight = { from: this.camera.position.clone(), look: this.controls.target.clone(), t: 0, dur: clamp(2.5 + span * 0.12, 3, 6.5), span };
      if (view.moving) this.camYaw = view.heading;
    }
    this.anchorIndex = this.followIndex;
    const fwd = new THREE.Vector3(Math.sin(view.heading), 0, Math.cos(view.heading));
    const manual = performance.now() < this.manualUntil;
    const speed = clamp((this.info[this.followIndex]?.kmh ?? 0) / 80, 0, 1);

    const wantCam = new THREE.Vector3(), wantLook = new THREE.Vector3(0, 0.1, 0);
    let fov = 50, stiff = 4.5, lookStiff = 8;
    if (style === 'chase') {
      if (!manual) {
        if (view.moving) this.camYaw = lerpAngle(this.camYaw, view.heading, 1 - Math.exp(-dt * 2.4));
        else {
          const f = this.nearestFacility(anchor);
          const open = f ? Math.atan2(f.dir.x, f.dir.z) + Math.sin(performance.now() / 6000) * 0.7 : this.camYaw + dt * 0.1;
          this.camYaw = lerpAngle(this.camYaw, open, 1 - Math.exp(-dt * 1.2));
        }
      }
      const dist = this.followDistance * (1 + 0.15 * speed);
      wantCam.set(-Math.sin(this.camYaw) * dist, 0.08 + dist * this.followHeight, -Math.cos(this.camYaw) * dist);
      wantLook.addScaledVector(fwd, view.moving ? 0.45 : 0.1);
      fov = 52 + 10 * speed;
      stiff = 5;
    } else if (mode === 'top') {
      const h = 2.2 + this.followDistance * 3;
      wantCam.set(0, h, h * 0.18);
      wantLook.set(0, 0, 0);
      fov = 45;
      stiff = 3;
    } else {
      const s = this.shot;
      lookStiff = 4;
      stiff = 2.2;
      if (s.type === 'chase') {
        const a = this.camYaw + 0.4;
        const dist = 1.7 + 0.3 * speed;
        wantCam.set(-Math.sin(a) * dist, 0.85, -Math.cos(a) * dist);
        wantLook.addScaledVector(fwd, 0.35);
        fov = 48 + 4 * speed;
      } else {
        const f = this.nearestFacility(anchor);
        const open = f ? Math.atan2(-f.dir.x, -f.dir.z) : s.angle;
        if (s.type === 'orbit') {
          const a = open + Math.sin(s.t * 0.09) * 0.75;
          wantCam.set(Math.sin(a) * 2.1, 1.0, Math.cos(a) * 2.1);
          fov = 44;
        } else {
          const a = open + 0.45;
          wantCam.set(Math.sin(a) * 1.7, 0.95, Math.cos(a) * 1.7);
          wantLook.copy(fwd).multiplyScalar(-0.42).setY(0.07);
          fov = 40;
        }
      }
    }

    if (this.flight && !this.cut) {
      const fl = this.flight;
      fl.t += dt;
      const u = clamp(fl.t / fl.dur, 0, 1);
      const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      const goal = anchor.clone().add(wantCam);
      const pos = fl.from.clone().lerp(goal, e);
      pos.y += Math.sin(Math.PI * e) * Math.min(1 + fl.span * 0.28, 9);
      pos.y = Math.max(pos.y, terrainHeight(pos.x, pos.z) + 0.3);
      const k = clamp(u * 1.35, 0, 1);
      const look = fl.look.clone().lerp(anchor.clone().add(wantLook), k * k * (3 - 2 * k));
      this.camera.position.copy(pos);
      this.controls.target.copy(look);
      this.camera.up.set(0, 1, 0);
      this.camera.lookAt(look);
      this.camOffset.copy(pos).sub(anchor);
      this.lookOffset.copy(look).sub(anchor);
      this.setFov(fov, dt);
      if (u >= 1) {
        this.flight = null;
        this.camOffset.copy(wantCam);
        this.lookOffset.copy(wantLook);
        this.armLength = undefined;
      }
      return;
    }
    this.flight = null;
    const far = this.camOffset.distanceTo(wantCam);
    if (far > 3 && !this.cut) wantCam.y += Math.min(10, far * 0.3);
    if (this.cut) this.camOffset.copy(wantCam);
    else this.camOffset.lerp(wantCam, 1 - Math.exp(-dt * (far > 3 ? 1.8 : stiff)));
    if (this.cut) this.lookOffset.copy(wantLook);
    else this.lookOffset.lerp(wantLook, 1 - Math.exp(-dt * lookStiff));
    const pos = anchor.clone().add(this.camOffset);
    const mid = anchor.clone().addScaledVector(this.camOffset, 0.5);
    const ground = Math.max(terrainHeight(pos.x, pos.z), terrainHeight(mid.x, mid.z) - 0.05);
    pos.y = Math.max(pos.y, ground + 0.05);
    this.controls.target.copy(anchor).add(this.lookOffset);
    if ((style === 'chase' || style === 'cinematic') && this.camOffset.y < 0.9) {
      const from = anchor.clone().setY(anchor.y + 0.12);
      const dir = pos.clone().sub(from);
      const length = dir.length();
      this.raycaster.set(from, dir.normalize());
      this.raycaster.far = length;
      const hit = this.raycaster.intersectObjects(this.occluders, false)[0];
      const arm = hit ? Math.max(0.45, hit.distance - 0.06) : length;
      this.armLength = Math.min(arm, (this.armLength ?? length) + dt * 1.5 * length);
      if (this.armLength < length - 1e-3) pos.copy(from).addScaledVector(dir, this.armLength);
    } else this.armLength = undefined;
    this.camera.position.copy(pos);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(this.controls.target);
    this.setFov(fov, dt, this.cut);
    this.cut = false;
  }

  setFov(fov, dt, snap = false) {
    const next = snap ? fov : this.camera.fov + (fov - this.camera.fov) * (1 - Math.exp(-dt * 3));
    if (Math.abs(next - this.camera.fov) < 0.01) return;
    this.camera.fov = next;
    this.camera.updateProjectionMatrix();
    this.updateDustScale();
  }

  render() {
    if (this.cameraMode === 'overview') this.controls.update();
    if (this.sky) this.sky.position.copy(this.camera.position);
    this.composer.render();
    this.labelRenderer.render(this.scene, this.camera);
  }
}
