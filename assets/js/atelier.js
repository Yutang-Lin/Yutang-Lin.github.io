/* Publication filter: keep one bibliography so abstract and author controls
   retain unique IDs in both views. */
(() => {
  const publications = document.getElementById('home-publications');
  if (!publications) return;
  const buttons = document.querySelectorAll('[data-publication-view]');
  buttons.forEach(button => button.addEventListener('click', () => {
    publications.dataset.view = button.dataset.publicationView;
    buttons.forEach(other => other.setAttribute('aria-pressed', String(other === button)));
  }));
})();

/* Logarithmic experience axis: event dates are positioned on the scale;
   editorial cards are kept evenly spaced so their labels never collide. */
(() => {
  const section = document.getElementById('research-experience');
  if (!section) return;
  const panels = Array.from(section.querySelectorAll('.experience-panel'));
  const buttons = Array.from(section.querySelectorAll('[data-experience-view]'));
  let hoveredCard = null, focusedCard = null;
  let svg;
  const clock = section.querySelector('[data-live-clock]');
  const ns = 'http://www.w3.org/2000/svg';
  if (!panels.length || !clock) return;
  const parse = value => new Date(value + 'T00:00:00+08:00').getTime();
  const day = 86400000;
  function highlight(panel) {
    const cards = Array.from(panel.querySelectorAll('[data-experience-start]'));
    const active = cards.includes(hoveredCard) ? hoveredCard : cards.includes(focusedCard) ? focusedCard : null;
    cards.forEach(card => card.classList.toggle('is-highlighted', card === active));
    let band = panel.rangeBand;
    if (!band) {
      band = document.createElement('div'); band.className = 'experience-range';
      band.setAttribute('aria-hidden', 'true');
      panel.querySelector('.experience-axis').appendChild(band); panel.rangeBand = band;
    }
    if (!active || !panel.timeGeometry) { band.style.opacity = '0'; band.style.transform = 'scaleX(.94)'; return; }
    const { x, right } = panel.timeGeometry;
    const start = x(parse(active.dataset.experienceStart));
    const end = active.dataset.experienceEnd ? x(parse(active.dataset.experienceEnd)) : active.dataset.experienceOngoing ? right : start;
    band.style.left = `${start - 4}px`; band.style.width = `${Math.max(8, end - start + 8)}px`;
    band.style.opacity = '1'; band.style.transform = 'scaleX(1)';
  }
  function shape(tag, attributes) {
    const node = document.createElementNS(ns, tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    svg.appendChild(node); return node;
  }
  function render() {
    const panel = panels.find(panel => !panel.hidden);
    if (!panel) return;
    const plot = panel.querySelector('.experience-plot');
    svg = panel.querySelector('.experience-lines');
    const ticks = panel.querySelector('.experience-ticks');
    const cards = Array.from(panel.querySelectorAll('[data-experience-start]'));
    const earliest = Math.min(...cards.map(card => parse(card.dataset.experienceStart)));
    const now = Date.now(), width = plot.clientWidth, left = 16, right = width - 16;
    const maxAge = Math.max(1, (now - earliest) / day);
    // A smooth mixture of logarithms preserves readable historical spacing
    // and a live-time lens, without a piecewise boundary or derivative jump.
    const timeScale = 730;
    const liveWeight = .32;
    const positionAge = age => {
      age = Math.max(0, age);
      const live = Math.log1p(age / 1000) / Math.log1p(maxAge * day / 1000);
      const history = Math.log1p(age / day / timeScale) / Math.log1p(maxAge / timeScale);
      return right - (right - left) * (liveWeight * live + (1 - liveWeight) * history);
    };
    const x = time => positionAge(now - time);
    svg.replaceChildren(); ticks.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${width} 88`);
    shape('line', { x1: left, y1: 55, x2: right, y2: 55, stroke: 'var(--global-divider-color)', 'stroke-width': 1 });
    panel.timeGeometry = { left, right, maxAge, timeScale, width, x, positionAge };
    highlight(panel);
    const month = new Intl.DateTimeFormat('en-GB', { timeZone: 'Asia/Shanghai', month: 'short', year: 'numeric' });
    // Calendar-quarter ticks reveal the non-linear spacing. Only label ticks
    // with enough room, keeping the first date and Now legible at any width.
    const first = new Date(earliest + 8 * 3600000);
    const marks = [[earliest, month.format(earliest)]];
    let year = first.getUTCFullYear(), quarter = Math.floor(first.getUTCMonth() / 3) * 3;
    let date = Date.UTC(year, quarter, 1) - 8 * 3600000;
    while (date < now) {
      if (date > earliest) marks.push([date, month.format(date)]);
      quarter += 3;
      date = Date.UTC(year, quarter, 1) - 8 * 3600000;
    }
    [[30 * day, '1mo'], [7 * day, '1w'], [day, '1d'], [3600000, '1h'], [60000, '1m']].forEach(([age,label]) => marks.push([now - age, label]));
    marks.push([now, 'Now']);
    marks.sort((a,b) => a[0] - b[0]);
    let lastLabel = -Infinity;
    marks.forEach(([date,label], index) => {
      const at = x(date);
      const endpoint = index === 0 || index === marks.length - 1;
      const labeled = endpoint || (at - lastLabel >= (date > now - 31 * day ? 65 : 110) && right - at >= 60);
      shape('line', { x1: at, y1: labeled ? 48 : 51, x2: at, y2: labeled ? 62 : 59, stroke: 'var(--global-divider-color)', 'stroke-width': labeled ? 1.5 : 1 });
      if (labeled) {
        const tick = document.createElement('span'); tick.className = 'experience-tick'; tick.textContent = label; tick.style.left = `${at}px`; ticks.appendChild(tick);
        lastLabel = at;
      }
    });
    cards.forEach(card => {
      const start = x(parse(card.dataset.experienceStart));
      shape('circle',{cx:start,cy:55,r:4,fill:'var(--global-theme-color)',stroke:'var(--global-bg-color)','stroke-width':2});
    });
    shape('circle',{cx:right,cy:55,r:9,fill:'none',stroke:'var(--global-theme-color)','stroke-opacity':.4,class:'time-now-ring'});
    shape('circle',{cx:right,cy:55,r:3,fill:'var(--global-theme-color)'});
  }
  let today = '';
  const format = new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Shanghai',day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
  function update() {
    if (document.hidden) return;
    const now = new Date(); clock.textContent = format.format(now); clock.dateTime = now.toISOString();
    const date = now.toISOString().slice(0,10);
    if (date !== today) { today=date; render(); }
  }
  buttons.forEach(button => button.addEventListener('click', () => {
    hoveredCard = focusedCard = null;
    panels.forEach(panel => { panel.hidden = panel.id !== `experience-${button.dataset.experienceView}`; });
    buttons.forEach(other => other.setAttribute('aria-pressed', String(other === button)));
    render();
  }));
  panels.forEach(panel => panel.querySelectorAll('[data-experience-start]').forEach(card => {
    card.addEventListener('pointerenter', () => { hoveredCard = card; highlight(panel); });
    card.addEventListener('pointerleave', () => { if (hoveredCard === card) hoveredCard = null; highlight(panel); });
    card.addEventListener('focusin', () => { focusedCard = card; highlight(panel); });
    card.addEventListener('focusout', event => { if (!card.contains(event.relatedTarget)) { focusedCard = null; highlight(panel); } });
  }));
  const observer = new ResizeObserver(render);
  panels.forEach(panel => observer.observe(panel.querySelector('.experience-plot')));
  setInterval(update,1000); document.addEventListener('visibilitychange',update);
  update();
})();

/* A real-time logarithmic zoom integrated into the main axis: each stroke is a second
   boundary, positioned by its measured age, not a decorative particle. */
(() => {
  const section = document.getElementById('research-experience');
  if (!section) return;
  const flows = Array.from(section.querySelectorAll('.experience-panel')).map(panel => {
    const canvas = panel.querySelector('.experience-live-ticks');
    return { panel, canvas, ctx: canvas?.getContext?.('2d') };
  }).filter(flow => flow.ctx);
  if (!flows.length) return;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let visible = false, frame = 0;
  function draw() {
    frame = 0;
    if (!visible || document.hidden) return;
    const flow = flows.find(flow => !flow.panel.hidden);
    if (!flow?.panel.timeGeometry) return;
    const { canvas, ctx } = flow;
    const { width, right, maxAge, positionAge } = flow.panel.timeGeometry;
    const ratio = Math.min(devicePixelRatio || 1, 1.5);
    if (canvas.width !== Math.round(width * ratio)) { canvas.width = Math.round(width * ratio); canvas.height = Math.round(88 * ratio); }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, 88);
    const color = getComputedStyle(section).getPropertyValue('--global-theme-color').trim() || '#d99c6f';
    ctx.strokeStyle = ctx.fillStyle = color;
    const now = Date.now();
    let lastLabel = -Infinity;
    // Overlapping timestamp lattices cross-fade from seconds to weeks. Every
    // tick shares the very same smooth age-to-position function as the cards.
    const bands = [
      [1000, 0, 0, 60000, 120000],
      [60000, 60000, 120000, 3600000, 10800000],
      [3600000, 3600000, 10800000, 86400000, 259200000],
      [86400000, 86400000, 259200000, 2592000000, 7776000000],
      [604800000, 2592000000, 7776000000, maxAge * 86400000, maxAge * 86400000]
    ];
    bands.forEach(([step, enter, full, fade, end]) => {
    for (let stamp = Math.ceil((now - Math.min(end, maxAge * 86400000)) / step) * step; stamp <= now - enter; stamp += step) {
      const age = now - stamp;
      const at = positionAge(age);
      const ease = value => { value = Math.min(1, Math.max(0, value)); return value * value * (3 - 2 * value); };
      const strength = (full > enter ? ease((age - enter) / (full - enter)) : 1) * (end > fade ? 1 - ease((age - fade) / (end - fade)) : 1);
      const gap = Math.abs(positionAge(age + step) - at);
      const density = Math.min(1, gap / 2);
      const fall = Math.min(1, age / 1000);
      const y = reduced.matches ? 55 : 55 - 40 * (1 - fall) ** 2;
      const major = stamp % (step * (step === 1000 ? 10 : 5)) === 0;
      ctx.globalAlpha = strength * density * (major ? .65 : .38);
      ctx.lineWidth = major ? 1.5 : 1;
      ctx.beginPath(); ctx.moveTo(at, y - (major ? 8 : 4)); ctx.lineTo(at, y + 4); ctx.stroke();
      if (step === 1000 && major && strength > .5 && at - lastLabel > 46 && right - at > 36) {
        ctx.font = '9px monospace'; ctx.textAlign = 'center'; ctx.globalAlpha = .65;
        ctx.fillText(new Date(stamp + 8 * 3600000).toISOString().slice(14,19), at, 79);
        lastLabel = at;
      }
    }
    });
    ctx.globalAlpha = 1;
    if (!reduced.matches) frame = requestAnimationFrame(draw);
  }
  function start() { if (!frame && visible && !document.hidden) frame = requestAnimationFrame(draw); }
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; start(); }).observe(section);
  document.addEventListener('visibilitychange', start);
  reduced.addEventListener('change', start);
  setInterval(() => { if (reduced.matches) start(); }, 1000);
})();

/* A gently moving point landscape. No libraries or external assets. */
(() => {
  const canvas = document.querySelector('.cover-landscape');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const cover = canvas.closest('section');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let width = 0, height = 0, frame = 0, visible = true, last = 0;
  let pointerX = 0, pointerY = 0, x = 0, y = 0;
  let cursorX = 0, cursorY = 0, hovering = false, influence = 0;
  const ripples = [];
  function resize() {
    width = cover.clientWidth; height = cover.clientHeight;
    const ratio = Math.min(devicePixelRatio || 1, 1.5);
    canvas.width = width * ratio; canvas.height = height * ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (reduced.matches) draw(0);
  }
  function draw(time, delta = 16.67) {
    ctx.clearRect(0, 0, width, height);
    const light = document.documentElement.dataset.theme === 'light';
    const t = reduced.matches ? 0 : time * .00013;
    const ease = 1 - Math.exp(-delta / 180);
    x += (pointerX - x) * ease; y += (pointerY - y) * ease;
    influence += ((hovering && !reduced.matches ? 1 : 0) - influence) * (1 - Math.exp(-delta / 100));
    while (ripples.length && time - ripples[0].time > 1800) ripples.shift();
    const columns = width < 600 ? 72 : 120;
    // Batch equal-color dots into 32 paths instead of thousands of fill calls.
    const paths = Array.from({ length: 32 }, () => new Path2D());
    for (let row = 0; row < 55; row++) {
      const depth = row / 54;
      for (let col = 0; col < columns; col++) {
        const u = col / (columns - 1);
        const hill = Math.sin(u * 9 + depth * 4 + t) * .55 + Math.cos(u * 15 - depth * 7 + t * .65) * .25;
        let px = width * (.24 + u * .98) + (depth - .5) * 100 + x * depth * 22;
        let py = height * (.43 + depth * .62) - hill * height * .18 * Math.sin(depth * Math.PI) + y * depth * 15;
        const dx = px - cursorX, dy = py - cursorY;
        const distance = Math.hypot(dx, dy);
        const proximity = Math.max(0, 1 - distance / 180) * influence;
        const push = proximity * proximity * 32;
        px += dx / Math.max(1, distance) * push;
        py += dy / Math.max(1, distance) * push;
        if (!reduced.matches) ripples.forEach(ripple => {
          const age = Math.max(0, (time - ripple.time) / 1800);
          const rx = px - ripple.x, ry = py - ripple.y, radius = Math.hypot(rx, ry);
          const wave = Math.exp(-Math.pow((radius - age * 650) / 35, 2)) * (1 - age) * 20;
          px += rx / Math.max(1, radius) * wave;
          py += ry / Math.max(1, radius) * wave;
        });
        const fade = Math.min(1, Math.max(0, (u - .02) * 3));
        const alpha = Math.min(.85, (light ? .3 : .12 + depth * .42) * fade + proximity * .22);
        const path = paths[(depth > .5 ? 16 : 0) + Math.min(15, Math.floor(alpha * 16))];
        const radius = .5 + depth * 1.05;
        path.moveTo(px + radius, py); path.arc(px, py, radius, 0, Math.PI * 2);
      }
    }
    paths.forEach((path, index) => {
      ctx.fillStyle = `rgba(${index >= 16 ? '165,181,166' : '217,156,111'},${(index % 16 + .5) / 16})`;
      ctx.fill(path);
    });
  }
  function tick(time) {
    frame = 0;
    if (!visible || document.hidden || reduced.matches) return;
    draw(time, Math.min(64, last ? time - last : 16.67)); last = time;
    frame = requestAnimationFrame(tick);
  }
  function start() {
    if (!frame && visible && !document.hidden && !reduced.matches) frame = requestAnimationFrame(tick);
  }
  cover.addEventListener('pointermove', e => {
    const rect = cover.getBoundingClientRect();
    pointerX = (e.clientX - rect.left) / rect.width - .5;
    pointerY = (e.clientY - rect.top) / rect.height - .5;
    cursorX = e.clientX - rect.left; cursorY = e.clientY - rect.top;
    hovering = true;
  }, { passive: true });
  cover.addEventListener('pointerleave', () => { pointerX = pointerY = 0; hovering = false; });
  cover.addEventListener('pointerdown', e => {
    if (reduced.matches || e.target.closest('a, button, input')) return;
    const rect = cover.getBoundingClientRect();
    if (ripples.length >= 4) ripples.shift();
    ripples.push({ x: e.clientX - rect.left, y: e.clientY - rect.top, time: performance.now() });
  }, { passive: true });
  cover.addEventListener('pointerup', e => { if (e.pointerType !== 'mouse') hovering = false; }, { passive: true });
  cover.addEventListener('pointercancel', () => { hovering = false; });
  new ResizeObserver(resize).observe(cover);
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; start(); }).observe(cover);
  document.addEventListener('visibilitychange', start);
  reduced.addEventListener('change', () => { draw(0); start(); });
  resize(); draw(0); start();
})();
