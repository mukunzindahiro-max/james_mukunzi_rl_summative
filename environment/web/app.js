import { World, PRIORITY_COLORS } from './scene.js';

const $ = (id) => document.getElementById(id);
const PRIORITY_LABEL = { 1: 'critical', 2: 'urgent', 3: 'moderate' };
const HOSPITAL_SHORT = ['District', 'Eastern Rural', 'Provincial'];
const DECISION_HOLD_MS = 1800;
const QUIET_SPEED = 180;
const ROAD_NAMES = { paved: 'Paved road', murram: 'Murram road', rough: 'Rough road', driveway: 'Facility driveway' };

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

async function api(path, body) {
  const response = await fetch(path, body === undefined ? undefined : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
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
  document.querySelectorAll('#camera-bar [data-view]').forEach((b) => b.classList.toggle('active', b.dataset.view === mode));
  document.querySelectorAll('#follow-buttons button').forEach((b) => b.classList.toggle('active', mode === 'follow' && Number(b.dataset.amb) === world.followIndex));
}

function followAmbulance(index) {
  world.follow(index);
  setCamera('follow');
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
  if (world.cameraMode !== 'follow' || !info) {
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
  if (dispatch && r) hud.style.borderColor = `#${PRIORITY_COLORS[r.priority].toString(16).padStart(6, '0')}`;
  else hud.style.borderColor = '';
  $('action-text').textContent = dispatch ? `🚑 ${d.description.replace('Dispatch ', '').replace('->', '→')}` : d.outcome === 'wait' ? '⏳ WAIT' : `⛔ ${d.description} (${d.outcome.replaceAll('_', ' ')})`;
  $('action-sub').textContent = r
    ? `${r.priority_name.toUpperCase()} referral #${r.id} at ${r.origin_name} · needs ${needsText(r)} · reward ${d.reward >= 0 ? '+' : ''}${d.reward.toFixed(1)}`
    : '';
  clearTimeout(showAction.timer);
  showAction.timer = setTimeout(() => hud.classList.add('hidden'), dispatch ? 3200 : 1400);
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
    if (world.cameraMode === 'follow') followAmbulance(Math.floor((d.action - 1) / 3));
  }
}

function applyState(next, resetClock = false) {
  state = next;
  world.build(state);
  if (resetClock) {
    displayT = Math.max(0, state.time_minutes - 60);
    announcedDone = false;
    lastDecisionStep = -1;
    $('banner').classList.add('hidden');
    $('action-hud').classList.add('hidden');
    $('reward-step').textContent = '+0.0';
  }
  $('mode').value = state.policy === 'manual' ? 'manual' : 'ai';
  $('policy').value = state.policy;
  $('scenario').value = state.scenario;
  $('seed').value = state.seed;
  $('agent-label').textContent = state.policy_label;
  $('chip-agent').textContent = state.policy === 'manual' ? 'Manual' : state.policy_label.split(' ')[0];
  $('reward-total').textContent = state.total_reward.toFixed(1);
  onNewDecision();
  updatePanels();
}

async function requestStep(action) {
  if (requesting || !state || state.done) return;
  requesting = true;
  try {
    applyState(await api('/api/step', action === undefined ? {} : { action }));
  } catch (err) {
    showBanner(`⚠ ${err.message}`);
  } finally {
    requesting = false;
  }
}

async function reset() {
  const policy = $('mode').value === 'manual' ? 'manual' : $('policy').value === 'manual' ? 'dqn' : $('policy').value;
  try {
    applyState(await api('/api/reset', { policy, scenario: $('scenario').value, seed: Number($('seed').value) }), true);
    showBanner(`New episode · ${state.policy_label} · ${state.scenario.replace('_', ' ')} · seed ${state.seed}`);
  } catch (err) {
    showBanner(`⚠ ${err.message}`);
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
        `<b>Episode complete</b> · return ${state.total_reward.toFixed(1)} · ${m.completed}/50 completed · ` +
        `${(m.on_time_rate * 100).toFixed(1)}% on time · ${m.unresolved} unresolved`, 0);
    }
    world.update(state, displayT, elapsed, dt);
    if (now - lastPanelUpdate > 200) {
      lastPanelUpdate = now;
      updatePanels();
      updateFollowCard();
      $('reward-speed').textContent = playing ? speedText(Math.max(rate, 0)) : 'paused';
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
  $('conditions').textContent = `${state.raining ? '🌧 Rain · murram & rough roads slowed' : '☀ Dry roads'} · ${night ? '🌙 Night' : '🌤 Day'} · ${state.scenario.replace('_', ' ')} · seed ${state.seed}`;
  const m = state.metrics;
  $('chip-step').textContent = state.step;
  $('chip-done').textContent = `${m.completed}/50`;
  $('chip-ontime').textContent = m.completed ? `${(m.on_time_rate * 100).toFixed(0)}%` : '—';

  const waiting = state.referrals.filter((r) => r.arrival <= t && (r.dispatch_time === null || r.dispatch_time > t) && !(r.status === 'unresolved' && r.arrival + 480 <= t));
  waiting.sort((a, b) => a.priority - b.priority || a.arrival - b.arrival);
  $('mission-list').innerHTML = waiting.length
    ? waiting.map((r) => `
      <div class="card p${r.priority}">
        <div class="top"><span>#${r.id} · ${r.origin_name.replace(' Health Centre', ' HC')}</span><span class="badge b${r.priority}">${PRIORITY_LABEL[r.priority]}</span></div>
        <div class="sub">needs ${needsText(r)} · waiting ${Math.round(t - r.arrival)} min</div>
      </div>`).join('')
    : '<div class="empty">No referrals waiting</div>';

  const transit = state.referrals.filter((r) => r.dispatch_time !== null && r.dispatch_time <= t && r.hospital_arrival > t);
  $('transit-list').innerHTML = transit.length
    ? transit.map((r) => `
      <div class="card p${r.priority}">
        <div class="top"><span>#${r.id} · Ambulance ${r.ambulance + 1}</span><span>→ ${HOSPITAL_SHORT[r.hospital]}</span></div>
        <div class="sub">${r.pickup_time > t ? '🚑 driving to pick up' : '🤰 patient on board'} · ETA ${clockText(r.hospital_arrival).split(' ')[2]}</div>
      </div>`).join('')
    : '<div class="empty">No ambulances on referral</div>';

  $('hospital-list').innerHTML = state.hospitals.map((h) => `
    <div class="hospital">
      <div class="name"><span>${h.name}</span><span>${h.free_beds}/${h.capacity} free</span></div>
      <div class="beds">${Array.from({ length: h.capacity }, (_, i) => `<div class="bed ${i < h.capacity - h.free_beds ? 'used' : ''}"></div>`).join('')}</div>
      <span class="svc ${h.surgery_ready ? 'ok' : 'down'}">${h.surgery_ready ? '✓' : '✗'} surgery</span>
      <span class="svc ${h.neonatal_ready ? 'ok' : 'down'}">${h.neonatal_ready ? '✓' : '✗'} neonatal care</span>
    </div>`).join('');

  renderDecision();
  renderManual();

  const stats = [
    ['Completed', `${m.completed}/50`],
    ['On time', `${(m.on_time_rate * 100).toFixed(1)}%`],
    ['Unresolved', m.unresolved],
    ['Mean wait', `${m.mean_wait_minutes.toFixed(1)} min`],
    ['Mean referral', `${m.mean_referral_minutes.toFixed(0)} min`],
    ['Utilization', `${(m.ambulance_utilization * 100).toFixed(1)}%`],
    ['Coverage', `${(m.coverage * 100).toFixed(0)}%`],
    ['Distance', `${m.total_km.toFixed(0)} km`],
  ];
  $('stats').innerHTML = stats.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');

  const events = state.events.filter((e) => e.time <= t).slice(-20).reverse();
  $('log').innerHTML = events.map((e) => `<div><span class="t">${e.clock}</span>${e.message}</div>`).join('');
}

function renderDecision() {
  const d = state.decision;
  if (!d) {
    $('decision').innerHTML = '<div class="empty">Waiting for the first referral…</div>';
    return;
  }
  const r = d.referral;
  const referral = r
    ? `<div class="card p${r.priority}"><div class="top"><span>#${r.id} · ${r.origin_name}</span><span class="badge b${r.priority}">${r.priority_name}</span></div><div class="sub">needs ${needsText(r)} · decided at ${d.clock}</div></div>`
    : '';
  let scores = '';
  if (d.scores && d.scores.length) {
    const values = d.scores.map((s) => s.score);
    const min = Math.min(...values), max = Math.max(...values);
    scores = `<h3>${d.score_type === 'Q-value' ? 'Q-values (valid actions)' : 'Policy probabilities'}</h3>` + d.scores.map((s) => {
      const width = d.score_type === 'probability' ? s.score * 100 : max === min ? 100 : 15 + 85 * (s.score - min) / (max - min);
      const value = d.score_type === 'probability' ? `${(s.score * 100).toFixed(1)}%` : s.score.toFixed(2);
      return `<div class="score-row ${s.action === d.action ? 'chosen' : ''}"><span class="lbl">${s.description.replace('Dispatch ', '')}</span><div class="bar-wrap"><div class="bar" style="width:${width}%"></div></div><span class="val">${value}</span></div>`;
    }).join('');
  }
  const parts = Object.entries(d.reward_parts || {}).map(([k, v]) => `<span class="${v >= 0 ? 'pos' : 'neg'}">${k.replaceAll('_', ' ')} ${v >= 0 ? '+' : ''}${v.toFixed(1)}</span>`).join('');
  $('decision').innerHTML = `
    ${referral}
    <div class="action">${d.description}</div>
    <div class="outcome">Outcome: <b>${d.outcome}</b> · step reward <b class="${d.reward >= 0 ? 'pos' : 'neg'}">${d.reward >= 0 ? '+' : ''}${d.reward.toFixed(1)}</b></div>
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
  $('manual-buttons').innerHTML = `<div class="muted">Referral #${current.id} (${current.priority_name}) at ${current.origin_name}</div>` +
    state.valid_actions.map((a) => `<button data-action="${a.action}">${a.description}</button>`).join('');
  $('manual-buttons').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => requestStep(Number(b.dataset.action))));
}

async function openAnalytics() {
  $('analytics').classList.remove('hidden');
  const [comparison, figures] = await Promise.all([api('/api/comparison'), api('/api/figures')]);
  if (comparison.algorithms) {
    const rows = [];
    for (const [name, b] of Object.entries(comparison.baselines)) {
      rows.push(`<tr><td>${name} baseline</td><td>—</td><td>${b.standard.return_mean.toFixed(1)} ± ${b.standard.return_std.toFixed(1)}</td><td>${(b.standard.on_time_rate * 100).toFixed(1)}%</td><td>${b.standard.unresolved.toFixed(2)}</td><td>${b.rainy_season.return_mean.toFixed(1)}</td><td>—</td></tr>`);
    }
    for (const [key, a] of Object.entries(comparison.algorithms)) {
      const s = a.evaluation.standard, rainy = a.evaluation.rainy_season, seeds = a.seed_robustness;
      rows.push(`<tr class="${key === comparison.overall_best ? 'best' : ''}"><td>${a.label}</td><td>${a.best_experiment}</td><td>${s.return_mean.toFixed(1)} ± ${s.return_std.toFixed(1)}</td><td>${(s.on_time_rate * 100).toFixed(1)}%</td><td>${s.unresolved.toFixed(2)}</td><td>${rainy.return_mean.toFixed(1)}</td><td>${seeds.eval_return_mean.toFixed(1)} ± ${seeds.eval_return_std.toFixed(1)}</td></tr>`);
    }
    $('comparison').innerHTML = `<table><tr><th>Policy</th><th>Best experiment</th><th>Eval return (20 held-out seeds)</th><th>On time</th><th>Unresolved</th><th>Rainy season</th><th>3 training seeds</th></tr>${rows.join('')}</table>`;
  } else {
    $('comparison').innerHTML = '<div class="empty">No comparison found. Run `uv run main.py compare`.</div>';
  }
  $('figures').innerHTML = figures.length
    ? figures.map((f) => `<figure><img src="/assets/${f}" alt="${f}"><figcaption>${f.replace('.png', '').replaceAll('_', ' ')}</figcaption></figure>`).join('')
    : '<div class="empty">No figures generated yet.</div>';
}

function togglePanels() {
  document.body.classList.toggle('clean');
  $('toggle-panels').textContent = document.body.classList.contains('clean') ? 'Show panels' : 'Hide panels';
}

function togglePlay() {
  playing = !playing;
  $('play').textContent = playing ? '⏸ Pause' : '▶ Play';
}

async function init() {
  const [policies, scenarios] = await Promise.all([api('/api/policies'), api('/api/scenarios')]);
  $('policy').innerHTML = policies.filter((p) => p.key !== 'manual').map((p) => `<option value="${p.key}" ${p.available ? '' : 'disabled'}>${p.label}</option>`).join('') + '<option value="manual" hidden>Manual control</option>';
  $('scenario').innerHTML = scenarios.map((s) => `<option value="${s}">${s.replace('_', ' ')}</option>`).join('');
  const initial = await api('/api/state');
  $('follow-buttons').innerHTML = initial.ambulances.map((a) => `<button data-amb="${a.id}" title="Follow ambulance ${a.id + 1}">A${a.id + 1}</button>`).join('');
  world.onAmbulanceClick = followAmbulance;
  document.querySelectorAll('.collapsible > h2').forEach((h) => h.addEventListener('click', () => h.parentElement.classList.toggle('collapsed')));
  document.querySelectorAll('#follow-buttons button').forEach((b) => b.addEventListener('click', () => followAmbulance(Number(b.dataset.amb))));
  document.querySelectorAll('#camera-bar [data-view]').forEach((b) => b.addEventListener('click', () => setCamera(b.dataset.view)));
  applyState(initial, true);
  setSpeedFromSlider();
  setCamera('follow');
  $('speed').addEventListener('input', setSpeedFromSlider);
  $('reset').addEventListener('click', reset);
  $('mode').addEventListener('change', reset);
  $('policy').addEventListener('change', reset);
  $('scenario').addEventListener('change', reset);
  $('play').addEventListener('click', togglePlay);
  $('step').addEventListener('click', () => {
    holdUntil = 0;
    if (state.policy !== 'manual' && displayT >= state.time_minutes) requestStep();
    else displayT = state.time_minutes;
  });
  $('toggle-panels').addEventListener('click', togglePanels);
  $('analytics-btn').addEventListener('click', openAnalytics);
  $('analytics-close').addEventListener('click', () => $('analytics').classList.add('hidden'));
  window.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT'].includes(document.activeElement.tagName)) return;
    if (e.key === '1') setCamera('overview');
    else if (e.key === '2') setCamera('follow');
    else if (e.key === '3') setCamera('top');
    else if (e.key.toLowerCase() === 'h') togglePanels();
    else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  });
  requestAnimationFrame(frame);
}

init().catch((err) => showBanner(`⚠ ${err.message}`, 0));
