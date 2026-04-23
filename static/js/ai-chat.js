// static/js/ai-chat.js
// ════════════════════════════════════════════════════════
// GROQ AI CHAT ASSISTANT
// ════════════════════════════════════════════════════════

let _groqApiKey = null;
let riskExplainAbort = null;

async function _getGroqKey() {
  if (_groqApiKey) return _groqApiKey;
  const res = await fetch(`${API_BASE}/api/config`);
  if (!res.ok) throw new Error('Could not load runtime config from server');
  const cfg = await res.json();
  if (!cfg.groq_api_key) throw new Error('GROQ_API_KEY not configured on server');
  _groqApiKey = cfg.groq_api_key;
  return _groqApiKey;
}

async function _groqChat(systemPrompt, messages, maxTokens = 1000, signal) {
  const res = await fetch(`${API_BASE}/api/simulation/ai/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ system: systemPrompt, max_tokens: maxTokens, messages }),
    signal,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail || `AI proxy error ${res.status}`);
  }
  const data = await res.json();
  return data.text || 'No response.';
}

function _riskContextPayload() {
  const rr = currentAnalytics?.risk_report || {};
  return {
    overall_risk_level: rr.overall_risk_level || 'UNKNOWN',
    threshold_metrics: rr.threshold_metrics || {},
    n_risk_windows: rr.n_risk_windows || 0,
    cumulative_load: rr.cumulative_load || {},
    conclusions: (document.getElementById('mission-risk-conclusions')?.innerText || '').trim(),
  };
}

async function explainRisk() {
  markUserInteraction();
  const box = document.getElementById('risk-explain-wrap');
  if (!box) return;
  if (!currentAnalytics?.risk_report) {
    box.style.display = 'block';
    box.innerHTML = 'RISK ANALYTICS NOT AVAILABLE YET.';
    return;
  }
  
  if (riskExplainAbort) riskExplainAbort.abort();
  riskExplainAbort = new AbortController();
  _setRiskExplainLoading(true);
  box.style.display = 'block';
  box.innerHTML = '';
  
  const payload = _riskContextPayload();
  const userContent = `Analyze this mission risk data and explain what drove risk in 2-3 plain-English paragraphs, then provide exactly 3 ranked mitigation suggestions.\n\nRisk data:\n${JSON.stringify(payload, null, 2)}`;
  
  try {
    const text = await _groqChat(
      'You are a mission risk analyst for an Astronaut Digital Twin simulator. Be concise, technical, and actionable.',
      [{ role: 'user', content: userContent }],
      1000,
      riskExplainAbort.signal,
    );
    
    const msgDiv = document.createElement('div');
    msgDiv.style.cssText = 'margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid var(--border);white-space:pre-wrap';
    msgDiv.textContent = text;
    box.appendChild(msgDiv);
    
    const initHistory = [
      { role: 'user', content: userContent },
      { role: 'assistant', content: text },
    ];
    _attachChatInput(box,
      'You are a mission risk analyst for an Astronaut Digital Twin simulator. Be concise, technical, and actionable.',
      initHistory);
  } catch (e) {
    if (e.name !== 'AbortError') {
      box.textContent = `EXPLAINER ERROR: ${e.message}`;
    }
  } finally {
    _setRiskExplainLoading(false);
  }
}

async function explainRiskMC() {
  const box = document.getElementById('mc-explain-wrap');
  const spinner = document.getElementById('mc-risk-spinner');
  if (!box) return;
  box.style.display = 'block';
  box.innerHTML = '';
  if (spinner) spinner.style.display = 'inline-block';
  
  const statsEl = document.getElementById('mc-stats');
  const concEl = document.getElementById('mc-conclusions');
  const context = [
    statsEl ? 'MC STATS: ' + statsEl.innerText.replace(/\s+/g, ' ') : '',
    concEl ? 'CONCLUSIONS: ' + concEl.innerText.replace(/\s+/g, ' ') : '',
  ].filter(Boolean).join('\n');
  
  const systemPrompt = 'You are an aerospace medicine AI assistant for a mission digital twin simulator. Be concise, technical, and actionable.';
  const userContent = `Analyze this Monte Carlo simulation risk data for an astronaut mission and explain what it means for crew health and mission safety in clear, actionable terms.\n\n${context || 'No MC data available yet — run Monte Carlo first.'}`;
  
  try {
    const text = await _groqChat(systemPrompt, [{ role: 'user', content: userContent }], 1000);
    const msgDiv = document.createElement('div');
    msgDiv.style.cssText = 'margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid var(--border);white-space:pre-wrap';
    msgDiv.textContent = text;
    box.appendChild(msgDiv);
    
    const initHistory = [
      { role: 'user', content: userContent },
      { role: 'assistant', content: text },
    ];
    _attachChatInput(box, systemPrompt, initHistory);
  } catch(e) {
    box.textContent = `ERROR: ${e.message}`;
  } finally {
    if (spinner) spinner.style.display = 'none';
  }
}

function _attachChatInput(box, systemPrompt, history) {
  const old = box.querySelector('.ai-chat-input-row');
  if (old) old.remove();
  
  const row = document.createElement('div');
  row.className = 'ai-chat-input-row';
  row.style.cssText = 'display:flex;gap:6px;margin-top:8px;position:sticky;bottom:0;background:rgba(5,15,30,.97);padding:4px 0';
  
  const inp = document.createElement('input');
  inp.type = 'text';
  inp.placeholder = 'ASK A FOLLOW-UP...';
  inp.style.cssText = 'flex:1;background:rgba(0,212,255,.05);border:1px solid var(--border2);color:var(--textbright);font:9px var(--mono);padding:5px 8px;border-radius:2px;outline:none;letter-spacing:1px';
  inp.onfocus = () => inp.style.borderColor = 'var(--cyan)';
  inp.onblur = () => inp.style.borderColor = 'var(--border2)';
  
  const sendBtn = document.createElement('button');
  sendBtn.type = 'button';
  sendBtn.textContent = 'SEND';
  sendBtn.className = 'btn';
  sendBtn.style.cssText = 'width:auto;margin:0;padding:5px 10px;border-color:var(--cyan);color:var(--cyan)';
  
  const sendMsg = async () => {
    const q = inp.value.trim();
    if (!q) return;
    inp.value = '';
    sendBtn.disabled = true;
    
    const userDiv = document.createElement('div');
    userDiv.style.cssText = 'color:var(--cyan);margin:6px 0 2px;font-size:8px;letter-spacing:1px';
    userDiv.textContent = '▶ YOU: ' + q;
    box.insertBefore(userDiv, row);
    
    const thinkDiv = document.createElement('div');
    thinkDiv.style.cssText = 'color:var(--text);margin-bottom:6px;font-size:9px;white-space:pre-wrap';
    thinkDiv.textContent = '⟳ THINKING...';
    box.insertBefore(thinkDiv, row);
    box.scrollTop = box.scrollHeight;
    
    history.push({ role: 'user', content: q });
    
    try {
      const answer = await _groqChat(systemPrompt, history, 800);
      history.push({ role: 'assistant', content: answer });
      thinkDiv.style.cssText = 'color:var(--textbright);margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid var(--border);white-space:pre-wrap';
      thinkDiv.textContent = answer;
    } catch(e) {
      history.pop();
      thinkDiv.style.color = 'var(--red)';
      thinkDiv.textContent = 'ERROR: ' + e.message;
    } finally {
      sendBtn.disabled = false;
      inp.focus();
      box.scrollTop = box.scrollHeight;
    }
  };
  
  sendBtn.onclick = sendMsg;
  inp.onkeydown = e => { if (e.key === 'Enter') sendMsg(); };
  row.appendChild(inp);
  row.appendChild(sendBtn);
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

function _setRiskExplainLoading(loading) {
  const sp = document.getElementById('risk-spinner');
  const btn = document.getElementById('btn-explain-risk');
  if (sp) sp.style.display = loading ? 'block' : 'none';
  if (btn) {
    btn.disabled = !!loading;
    btn.textContent = loading ? 'EXPLAINING...' : 'EXPLAIN RISK';
  }
}