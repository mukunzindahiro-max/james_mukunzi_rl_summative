import { World, PRIORITY_COLORS } from './scene.js';

const $ = (id) => document.getElementById(id);
const PRIORITY_LABEL = { 1: 'critical', 2: 'urgent', 3: 'moderate' };
const HOSPITAL_SHORT = ['District', 'Eastern Rural', 'Provincial'];
const DECISION_HOLD_MS = 2600;
const QUIET_SPEED = 180;
const ROAD_NAMES = { paved: 'Paved road', murram: 'Murram road', rough: 'Rough road', driveway: 'Facility driveway' };
const SHOT_NAMES = {
  chase: 'Following ambulance',
  orbit: 'Ambulance at base',
  loading: 'Loading patient',
  overview: 'Map view',
  top: 'Top view',
};

const world = new World($('scene'));
let state = null;
let displayT = 0;
let playing = true;
let requesting = false;
let speed = 2;
let rate = 2;
let lastFrame = performance.now();
let lastPanelUpdate = 0;
let elapsed = 0;
let holdUntil = 0;
let announcedDone = false;
let lastDecisionStep = -1;
let offline = false;

async function api(path, body) {
  let response;
  try {
    response = await fetch(path, body === undefined ? undefined : {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error('offline');
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

function goOffline() {
  if (offline) return;
  offline = true;
  showBanner('⚠ Simulation server is not running. Start it with <b>uv run main.py</b> · reconnecting…', 0);
  reconnect();
}

async function reconnect() {
  try {
    const fresh = await api('/api/state');
    offline = false;
    applyState(fresh, true);
    showBanner('✅ Reconnected to the simulation server');
  } catch {
    setTimeout(reconnect, 2000);
  }
}

function clockText(t) {
  const day = Math.floor(t / 1440) + 1;
  const minute = Math.floor(t % 1440);
  return `Day ${day} ${String(Math.floor(minute / 60)).padStart(2, '0')}:${String(minute % 60).padStart(2, '0')}`;
}

function needsText(r) {
  const needs = [];
  if (r.needs_surgery) needs.push('surgery');
  if (r.needs_neonatal) needs.push('neonatal');
  return needs.length ? needs.join(' + ') : 'basic care';
}

function showBanner(text, ms = 3500) {
  const banner = $('banner');
  banner.innerHTML = text;
  banner.classList.remove('hidden');
  clearTimeout(showBanner.timer);
  if (ms) showBanner.timer = setTimeout(() => banner.classList.add('hidden'), ms);
}

function speedText(value) {
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} min/s`;
}

function setSpeedFromSlider() {
  speed = 0.5 * Math.pow(10, Number($('speed').value) / 40);
  $('speed-label').textContent = speedText(speed);
}

function setCamera(mode) {
  world.setCameraMode(mode);
  updateCameraPill();
}

function updateCameraPill() {
  const auto = world.cameraMode === 'cinematic';
  $('camera-name').textContent = auto ? `Auto camera · ${SHOT_NAMES[world.shotName] ?? ''}` : SHOT_NAMES[world.cameraMode] ?? world.cameraMode;
}

function followAmbulance(index) {
  world.follow(index);
}

function desiredRate(t, target) {
  if (world.movingCount > 0) return speed;
  const busy = state.ambulances.some((a) => a.referral !== null && a.waypoints.length && t >= a.waypoints[0][0] && t < a.waypoints[a.waypoints.length - 1][0]);
  const waiting = state.referrals.some((r) => r.arrival <= t && (r.pickup_time === null || r.pickup_time > t) && r.status !== 'unresolved');
  const ceiling = busy || waiting ? speed * 4 : Math.max(speed, QUIET_SPEED);
  return Math.min(ceiling, Math.max(speed, (target - t) * 1.2));
}

function updateFollowCard() {
  const info = world.info[world.followIndex];
  const card = $('follow-card');
  if (!world.tracking || !info) {
    card.classList.add('hidden');
    return;
  }
  card.classList.remove('hidden');
  $('follow-speed').textContent = info.moving ? info.kmh : 0;
  card.querySelector('.speedo').style.borderColor = info.siren ? `#${PRIORITY_COLORS[info.priority].toString(16).padStart(6, '0')}` : info.task === 'general_duty' ? '#ffd23f' : '#9fb3c8';
  $('follow-title').textContent = `🚑 Ambulance ${world.followIndex + 1}${info.siren ? ' · 🚨 emergency' : ''}`;
  $('follow-phase').textContent = info.phase;
  $('follow-road').textContent = info.moving ? ROAD_NAMES[info.road] || '' : 'Stopped';
}

function showAction(d) {
  const hud = $('action-hud');
  const r = d.referral;
  const dispatch = d.outcome === 'dispatch';
  hud.className = dispatch ? '' : d.outcome === 'wait' ? 'wait' : 'invalid';
  if (dispatch && r) hud.style.borderBottomColor = `#${PRIORITY_COLORS[r.priority].toString(16).padStart(6, '0')}`;
  else hud.style.borderBottomColor = '';
  $('action-text').textContent = dispatch ? `🚑 ${d.description.replace('Dispatch ', '').replace('->', '→')}` : d.outcome === 'wait' ? '⏳ WAIT' : `⛔ ${d.description} (${d.outcome.replaceAll('_', ' ')})`;
  $('action-sub').textContent = r
    ? `${r.priority_name.toUpperCase()} patient #${r.id} at ${r.origin_name} · needs ${needsText(r)} · reward ${d.reward >= 0 ? '+' : ''}${d.reward.toFixed(1)}`
    : '';
  clearTimeout(showAction.timer);
  showAction.timer = setTimeout(() => hud.classList.add('hidden'), dispatch ? 3600 : 1400);
}

function onNewDecision() {
  const d = state.decision;
  if (!d || d.step === lastDecisionStep) return;
  lastDecisionStep = d.step;
  showAction(d);
  $('reward-step').textContent = `${d.reward >= 0 ? '+' : ''}${d.reward.toFixed(1)}`;
  $('reward-step').className = `big ${d.reward >= 0 ? 'pos' : 'neg'}`;
  if (d.outcome === 'dispatch') {
    holdUntil = performance.now() + DECISION_HOLD_MS;
    if (world.tracking) followAmbulance(Math.floor((d.action - 1) / 3));
  }
}

function applyState(next, resetClock = false) {
  state = next;
  world.build(state);
  if (resetClock) {
    displayT = Math.max(0, state.time_minutes - 60);
    announcedDone = false;
    lastDecisionStep = -1;
    $('action-hud').classList.add('hidden');
    $('reward-step').textContent = '+0.0';
  }
  $('agent-label').textContent = state.policy_label;
  $('chip-agent').textContent = state.policy === 'manual' ? 'Manual' : state.policy_label.split(' ')[0];
  $('reward-total').textContent = state.total_reward.toFixed(1);
  onNewDecision();
  updatePanels();
}

async function requestStep(action) {
  if (requesting || offline || !state || state.done) return;
  requesting = true;
  try {
    applyState(await api('/api/step', action === undefined ? {} : { action }));
  } catch (err) {
    if (err.message === 'offline') goOffline();
    else showBanner(`⚠ ${err.message}`);
  } finally {
    requesting = false;
  }
}

async function reset() {
  try {
    applyState(await api('/api/reset', {}), true);
    showBanner(`New episode · ${state.policy_label} · seed ${state.seed}`);
  } catch (err) {
    if (err.message === 'offline') goOffline();
    else showBanner(`⚠ ${err.message}`);
  }
}

function frame(now) {
  const dt = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  elapsed += dt;
  if (state) {
    const target = state.time_minutes;
    const goal = now >= holdUntil && playing ? desiredRate(displayT, target) : 0;
    rate += (goal - rate) * (1 - Math.exp(-dt * (goal > rate ? 2.5 : 6)));
    if (now >= holdUntil && playing) {
      if (displayT < target) {
        displayT = Math.min(target, displayT + rate * dt);
      } else if (state.policy !== 'manual' && !state.done) {
        requestStep();
      }
    }
    if (state.done && displayT >= target && !announcedDone) {
      announcedDone = true;
      const m = state.metrics;
      showBanner(
        `<b>Episode complete</b> · return ${state.total_reward.toFixed(1)} · ${m.completed}/50 delivered · ` +
        `${(m.on_time_rate * 100).toFixed(1)}% on time · ${m.unresolved} unresolved`, 0);
    }
    world.update(state, displayT, elapsed, dt);
    if (now - lastPanelUpdate > 200) {
      lastPanelUpdate = now;
      updatePanels();
      updateFollowCard();
      updateCameraPill();
    }
  }
  world.render();
  requestAnimationFrame(frame);
}

function updatePanels() {
  if (!state) return;
  const t = displayT;
  $('clock').textContent = clockText(t);
  const hour = (t % 1440) / 60;
  const night = hour >= 20 || hour < 6;
  $('conditions').textContent = `${state.raining ? '🌧 Rain · dirt roads slowed' : '☀ Dry roads'} · ${night ? '🌙 Night' : '🌤 Day'}`;
  const m = state.metrics;
  $('chip-step').textContent = state.step;
  $('chip-done').textContent = `${m.completed}/50`;
  $('chip-ontime').textContent = m.completed ? `${(m.on_time_rate * 100).toFixed(0)}%` : '—';

  const waiting = state.referrals.filter((r) => r.arrival <= t && (r.dispatch_time === null || r.dispatch_time > t) && !(r.status === 'unresolved' && r.arrival + 480 <= t));
  waiting.sort((a, b) => a.priority - b.priority || a.arrival - b.arrival);
  const transit = state.referrals.filter((r) => r.dispatch_time !== null && r.dispatch_time <= t && r.hospital_arrival > t);
  $('patients-count').textContent = `${waiting.length} waiting · ${transit.length} en route`;
  $('mission-list').innerHTML = waiting.map((r) => `
      <div class="card p${r.priority}">
        <div class="top"><span>⏳ #${r.id} · ${r.origin_name.replace(' Health Centre', ' HC')}</span><span class="badge b${r.priority}">${PRIORITY_LABEL[r.priority]}</span></div>
        <div class="sub">needs ${needsText(r)} · waiting ${Math.round(t - r.arrival)} min</div>
      </div>`).join('');
  $('transit-list').innerHTML = transit.map((r) => `
      <div class="card p${r.priority}">
        <div class="top"><span>🚑 #${r.id} · A${r.ambulance + 1}</span><span>→ ${HOSPITAL_SHORT[r.hospital]}</span></div>
        <div class="sub">${r.pickup_time > t ? 'driving to pick up' : '🤰 on board'} · ETA ${clockText(r.hospital_arrival).split(' ')[2]}</div>
      </div>`).join('') || (waiting.length ? '' : '<div class="empty">No patients right now</div>');

  renderDecision();
  renderManual();
}

function renderDecision() {
  const d = state.decision;
  if (!d) {
    $('decision').innerHTML = '<div class="empty">Waiting for the first referral…</div>';
    return;
  }
  const key = `${d.step}`;
  if ($('decision').dataset.key === key) return;
  $('decision').dataset.key = key;
  let scores = '';
  if (d.scores && d.scores.length) {
    const values = d.scores.map((s) => s.score);
    const min = Math.min(...values), max = Math.max(...values);
    scores = `<h3>${d.score_type === 'Q-value' ? 'Q-values of valid actions' : 'Policy probabilities'}</h3>` + d.scores.slice(0, 4).map((s) => {
      const width = d.score_type === 'probability' ? s.score * 100 : max === min ? 100 : 15 + 85 * (s.score - min) / (max - min);
      const value = d.score_type === 'probability' ? `${(s.score * 100).toFixed(1)}%` : s.score.toFixed(2);
      return `<div class="score-row ${s.action === d.action ? 'chosen' : ''}"><span class="lbl">${s.description.replace('Dispatch ', '')}</span><div class="bar-wrap"><div class="bar" style="width:${width}%"></div></div><span class="val">${value}</span></div>`;
    }).join('');
  }
  const parts = Object.entries(d.reward_parts || {}).map(([k, v]) => `<span class="${v >= 0 ? 'pos' : 'neg'}">${k.replaceAll('_', ' ')} ${v >= 0 ? '+' : ''}${v.toFixed(1)}</span>`).join('');
  $('decision').innerHTML = `
    <div class="action">${d.description}</div>
    <div class="outcome">${d.clock} · <b>${d.outcome}</b> · reward <b class="${d.reward >= 0 ? 'pos' : 'neg'}">${d.reward >= 0 ? '+' : ''}${d.reward.toFixed(1)}</b></div>
    ${scores}
    <div class="reward-parts">${parts}</div>`;
}

function renderManual() {
  const manual = state.policy === 'manual';
  $('manual-actions').classList.toggle('hidden', !manual);
  if (!manual) return;
  const ready = displayT >= state.time_minutes && !state.done && !requesting;
  const current = state.current_referral_detail;
  const key = `${state.step}-${ready}`;
  if ($('manual-buttons').dataset.key === key) return;
  $('manual-buttons').dataset.key = key;
  if (!current || !ready) {
    $('manual-buttons').innerHTML = `<div class="empty">${state.done ? 'Episode complete' : 'Simulating…'}</div>`;
    return;
  }
  $('manual-buttons').innerHTML = `<div class="muted">Patient #${current.id} (${current.priority_name}) at ${current.origin_name}</div>` +
    state.valid_actions.map((a) => `<button data-action="${a.action}">${a.description}</button>`).join('');
  $('manual-buttons').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => requestStep(Number(b.dataset.action))));
}

function togglePanels() {
  document.body.classList.toggle('clean');
}

function togglePlay() {
  playing = !playing;
  $('play').textContent = playing ? '⏸' : '▶';
  $('play').classList.toggle('primary', playing);
}

function toggleCamera() {
  setCamera(world.cameraMode === 'cinematic' ? 'overview' : 'cinematic');
}

async function init() {
  let initial;
  try {
    initial = await api('/api/state');
  } catch (err) {
    if (err.message !== 'offline') throw err;
    showBanner('⚠ Simulation server is not running. Start it with <b>uv run main.py</b> · reconnecting…', 0);
    while (!initial) {
      await new Promise((r) => setTimeout(r, 2000));
      initial = await api('/api/state').catch(() => null);
    }
    showBanner('✅ Connected to the simulation server');
  }
  world.onAmbulanceClick = followAmbulance;
  document.querySelectorAll('.collapsible > h2').forEach((h) => h.addEventListener('click', () => h.parentElement.classList.toggle('collapsed')));
  applyState(initial, true);
  setSpeedFromSlider();
  setCamera('cinematic');
  $('speed').addEventListener('input', setSpeedFromSlider);
  $('reset').addEventListener('click', reset);
  $('play').addEventListener('click', togglePlay);
  $('camera-pill').addEventListener('click', toggleCamera);
  $('step').addEventListener('click', () => {
    holdUntil = 0;
    if (state.policy !== 'manual' && displayT >= state.time_minutes) requestStep();
    else displayT = state.time_minutes;
  });
  window.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT'].includes(document.activeElement.tagName)) return;
    const key = e.key.toLowerCase();
    if (key === 'c') toggleCamera();
    else if (key === 'n') followAmbulance((world.followIndex + 1) % state.ambulances.length);
    else if (key === 'h') togglePanels();
    else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  });
  requestAnimationFrame(frame);
}

init().catch((err) => showBanner(`⚠ ${err.message}`, 0));
