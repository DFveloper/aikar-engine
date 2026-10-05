const $ = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("en-US");
const windowSelect = $("ema-window");
let selectedEma = Number(localStorage.getItem("lumen-ema-window")) || 100;

if (![25, 50, 100, 250, 500, 1000].includes(selectedEma)) selectedEma = 100;
windowSelect.value = String(selectedEma);
windowSelect.addEventListener("change", () => {
  selectedEma = Number(windowSelect.value);
  localStorage.setItem("lumen-ema-window", String(selectedEma));
  refresh();
});
const combineToggle = $("combine-charts");
combineToggle.checked = localStorage.getItem("lumen-combine-charts") === "true";
document.querySelector(".shell").classList.toggle("combined", combineToggle.checked);
combineToggle.addEventListener("change", () => {
  localStorage.setItem("lumen-combine-charts", String(combineToggle.checked));
  document.querySelector(".shell").classList.toggle("combined", combineToggle.checked);
});

function value(id, content) {
  $(id).textContent = content;
}

function fixed(input, digits = 4) {
  return Number.isFinite(input) ? input.toFixed(digits) : "--";
}

function renderTrend(id, trend, metric, digits) {
  const element = $(id);
  if (!trend || !Number.isFinite(trend.delta)) {
    element.className = "trend neutral";
    element.textContent = "비교 데이터 대기";
    return;
  }
  const delta = trend.delta;
  const direction = metric === "accuracy" ? delta : -delta;
  const state = Math.abs(delta) < 10 ** -digits ? "neutral" : direction > 0 ? "good" : "bad";
  const arrow = direction > 0 ? "▲" : direction < 0 ? "▼" : "─";
  element.className = `trend ${state}`;
  element.textContent = `${trend.windows}W ${arrow}${Math.abs(delta).toFixed(digits)}`;
}

function drawChart(id, rangeId, history, key, color, domain = null) {
  const svg = $(id);
  const pointsByStep = new Map(history.filter((point) => Number.isFinite(point[key]) && Number.isFinite(point.step)
    && (!domain || (point.step >= domain.start && point.step <= domain.end))).map((point) => [point.step, point]));
  history = [...pointsByStep.values()].sort((a, b) => a.step - b.step);
  const values = history.map((point) => point[key]);
  if (values.length < 1) {
    svg.innerHTML = '<text x="300" y="78" text-anchor="middle" fill="#65717d" font-size="12">추세 데이터 대기 중</text>';
    value(rangeId, "");
    return;
  }

  const width = 600;
  const height = 150;
  const top = 10;
  const bottom = 140;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = Math.max(max - min, Math.abs(max) * 0.002, 0.001);
  const minStep = domain ? domain.start : history[0].step;
  const maxStep = domain ? domain.end : history.at(-1).step;
  const stepSpread = Math.max(Number.EPSILON, maxStep - minStep);
  const x = (index) => ((history[index].step - minStep) / stepSpread) * width;
  const y = (item) => max === min ? (top + bottom) / 2 : bottom - ((item - min) / spread) * (bottom - top);
  const points = values.map((item, index) => `${x(index).toFixed(1)},${y(item).toFixed(1)}`);
  const line = `M${points.join(" L")}`;
  const area = `${line} L${x(values.length - 1)},${height} L${x(0)},${height} Z`;
  const lastX = x(values.length - 1);
  const lastY = y(values.at(-1));
  svg.style.setProperty("--accent", color);
  svg.innerHTML = `
    <line class="grid" x1="0" y1="20" x2="600" y2="20"></line>
    <line class="grid" x1="0" y1="80" x2="600" y2="80"></line>
    <line class="grid" x1="0" y1="140" x2="600" y2="140"></line>
    ${values.length > 1 ? `<path class="area" d="${area}"></path><path class="line" d="${line}"></path>` : ""}
    <circle class="point" cx="${lastX}" cy="${lastY}" r="4"></circle>`;
  $(rangeId).innerHTML = `<span>최저 ${min.toFixed(key === "accuracy" ? 2 : 4)}</span><span>최고 ${max.toFixed(key === "accuracy" ? 2 : 4)}</span>`;
}

function drawComparisonChart(id, rangeId, trainHistory, validationHistory, key, trainColor, validationColor, domain = null) {
  const svg = $(id);
  const series = [
    { points: trainHistory.filter((point) => Number.isFinite(point[key]) && Number.isFinite(point.step)
      && (!domain || (point.step >= domain.start && point.step <= domain.end))).sort((a, b) => a.step - b.step), color: trainColor },
    { points: validationHistory.filter((point) => Number.isFinite(point[key]) && Number.isFinite(point.step)
      && (!domain || (point.step >= domain.start && point.step <= domain.end))).sort((a, b) => a.step - b.step), color: validationColor },
  ];
  const values = series.flatMap((item) => item.points.map((point) => point[key]));
  if (!values.length) {
    svg.innerHTML = '<text x="300" y="78" text-anchor="middle" fill="#65717d" font-size="12">검증 평가 데이터를 기다리는 중</text>';
    value(rangeId, "");
    return;
  }
  const steps = series.flatMap((item) => item.points.map((point) => point.step));
  const min = Math.min(...values), max = Math.max(...values);
  const minStep = domain ? domain.start : Math.min(...steps);
  const maxStep = domain ? domain.end : Math.max(...steps);
  const spread = Math.max(max - min, Math.abs(max) * 0.002, 0.001);
  const stepSpread = Math.max(Number.EPSILON, maxStep - minStep);
  const x = (step) => ((step - minStep) / stepSpread) * 600;
  const y = (item) => max === min ? 75 : 140 - ((item - min) / spread) * 130;
  const path = (points) => points.length > 1 ? `M${points.map((point) => `${x(point.step).toFixed(1)},${y(point[key]).toFixed(1)}`).join(" L")}` : "";
  const markers = (points, className) => points.map((point) =>
    `<circle class="comparison-point ${className}" cx="${x(point.step).toFixed(1)}" cy="${y(point[key]).toFixed(1)}" r="3"></circle>`
  ).join("");
  svg.style.setProperty("--train-color", trainColor);
  svg.style.setProperty("--validation-color", validationColor);
  svg.innerHTML = `
    <line class="grid" x1="0" y1="20" x2="600" y2="20"></line>
    <line class="grid" x1="0" y1="80" x2="600" y2="80"></line>
    <line class="grid" x1="0" y1="140" x2="600" y2="140"></line>
    <path class="line train-line" d="${path(series[0].points)}"></path>
    <path class="line validation-line" d="${path(series[1].points)}"></path>
    ${markers(series[0].points.slice(-1), "train-point")}
    ${markers(series[1].points, "validation-point")}`;
  $(rangeId).innerHTML = `<span>Train <i style="color:${trainColor}">━</i> · Val <i style="color:${validationColor}">━</i></span><span>step ${minStep.toLocaleString()}–${maxStep.toLocaleString()} · 최저 ${min.toFixed(key === "accuracy" ? 2 : 4)} · 최고 ${max.toFixed(key === "accuracy" ? 2 : 4)}</span>`;
}

function render(data) {
  const domain = {start: data.chart_start ?? 0, end: data.chart_current ?? data.current};
  const emaLabel = `EMA${data.ema_n}`;
  value("loss-label", `가중 Loss · ${emaLabel}`);
  value("accuracy-label", `정확도 (ACC) · ${emaLabel}`);
  value("unweighted-nll-label", `비가중 NLL · ${emaLabel}`);
  value("critical-label", `토큰 가중치 진단 · ${emaLabel}`);
  value("eta-label", "예상 남은 시간");
  value("rate-label", "평균 처리 속도");
  const percent = data.progress * 100;
  value("percent", fixed(percent, 2));
  value("steps", `${number.format(data.current)} / ${number.format(data.total)} Window (epoch ${data.epoch || 1})`);
  $("progress-bar").style.width = `${Math.min(100, percent)}%`;
  value("elapsed", data.elapsed);
  value("eta", data.eta);
  value("rate", data.rate ? `${data.rate.toFixed(2)} Window/초` : "--");
  value("loss", fixed(data.loss, 5));
  value("accuracy", fixed(data.accuracy, 2));
  value("unweighted-nll", fixed(data.unweighted_nll, 5));
  renderTrend("loss-trend", data.trends.loss, "loss", 4);
  renderTrend("accuracy-trend", data.trends.accuracy, "accuracy", 2);
  renderTrend("unweighted-nll-trend", data.trends.unweighted_nll, "unweighted_nll", 4);
  value("accuracy-scope", data.accuracy_scope === "legacy_all_positions"
    ? "구형 전체 위치 기준"
    : "Supervised token top-1");
  value("source", data.source === "log" ? "train.log" : "tmux 실시간");

  const status = $("status");
  status.className = `status ${data.state}`;
  const stateLabels = { running: "학습 중", stalled: "응답 지연", idle: "대기" };
  status.innerHTML = `<i></i>${stateLabels[data.state] || data.state}`;

  drawChart("loss-chart", "loss-range", data.history, "loss", "#55d7e6", domain);
  drawChart("accuracy-chart", "accuracy-range", data.history, "accuracy", "#f7b955", domain);
  drawChart("unweighted-nll-chart", "unweighted-nll-range", data.nll_history, "nll", "#71d6a0", domain);
  const validation = data.validation;
  value("validation-loss", fixed(validation?.loss, 5));
  value("validation-accuracy", fixed(validation?.accuracy, 2));
  drawChart("validation-loss-chart", "validation-loss-range", data.validation_history, "loss", "#b89cff", domain);
  drawChart("validation-accuracy-chart", "validation-accuracy-range", data.validation_history, "accuracy", "#b89cff", domain);
  // Validation loss is an unweighted NLL.  Keep the weighted train loss as
  // its own series and compare validation only with the matching NLL metric.
  drawChart("combined-weighted-loss-chart", "combined-weighted-loss-range", data.history, "loss", "#55d7e6", domain);
  const validationNllHistory = data.validation_history.map((point) => ({ ...point, nll: point.loss }));
  drawComparisonChart("combined-nll-chart", "combined-nll-range", data.nll_history, validationNllHistory, "nll", "#71d6a0", "#b89cff", domain);
  drawComparisonChart("combined-accuracy-chart", "combined-accuracy-range", data.history, data.validation_history, "accuracy", "#f7b955", "#b89cff", domain);

  const critical = data.critical;
  if (critical && Number.isFinite(critical.fraction)) {
    const criticalPercent = critical.fraction * 100;
    value("critical-fraction", `${criticalPercent.toFixed(1)}% 선택`);
    $("critical-bar").style.width = `${Math.min(100, criticalPercent)}%`;
    value("active-tokens", number.format(Math.round(critical.active)));
    value("critical-tokens", number.format(Math.round(critical.critical)));
    value("critical-nll", fixed(critical.nll, 4));
    value("weighted-loss", fixed(critical.weighted_loss, 4));
    value("mean-weight", fixed(critical.mean_weight, 4));
    value("max-weight", fixed(critical.max_weight, 2));
  } else {
    for (const id of ["critical-fraction", "active-tokens", "critical-tokens", "critical-nll", "weighted-loss", "mean-weight", "max-weight"]) value(id, "--");
    $("critical-bar").style.width = "0%";
  }

  const checkpoints = data.checkpoints;
  value("last-checkpoint", checkpoints.last_saved_step ? number.format(checkpoints.last_saved_step) : "--");
  value("saved-count", number.format(checkpoints.successful));
  value("failed-count", number.format(checkpoints.failed));
  const health = $("checkpoint-health");
  if (checkpoints.last_failed_step > (checkpoints.last_saved_step || 0)) {
    health.textContent = "저장 오류";
    health.className = "pill bad";
  } else if (checkpoints.successful > 0) {
    health.textContent = "정상";
    health.className = "pill good";
  } else {
    health.textContent = "기록 없음";
    health.className = "pill neutral";
  }

  const events = $("events");
  if (!data.events.length) {
    events.innerHTML = '<li class="empty">로그 이벤트를 기다리는 중입니다.</li>';
  } else {
    events.innerHTML = data.events.map((event, index) => {
      const kind = /failed|cannot open/i.test(event) ? "error" : /saved/i.test(event) ? "success" : "";
      return `<li class="${kind}" data-index="${String(index + 1).padStart(2, "0")}"></li>`;
    }).join("");
    [...events.children].forEach((item, index) => item.append(document.createTextNode(data.events[index])));
  }
  value("updated", `업데이트 ${new Date(data.updated_at * 1000).toLocaleTimeString("ko-KR", { hour12: false })}`);
}

async function refresh() {
  try {
    const response = await fetch(`/api/status?ema=${selectedEma}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    const status = $("status");
    status.className = "status stalled";
    status.innerHTML = "<i></i>연결 끊김";
  }
}

refresh();
setInterval(refresh, 2000);
