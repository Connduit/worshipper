import { useState } from 'react'
import './App.css'

type Run = { id: string; goal: string; status: 'Queued' | 'Completed' | 'Blocked'; project: string; time: string }
const seed: Run[] = [
  { id: 'RUN-0042', goal: 'Add retry handling to the inference client', status: 'Completed', project: 'Worshipper', time: 'Demo · 12 min ago' },
  { id: 'RUN-0041', goal: 'Review sandbox permission boundaries', status: 'Blocked', project: 'Worshipper', time: 'Demo · 38 min ago' },
  { id: 'RUN-0040', goal: 'Build context compaction regression tests', status: 'Completed', project: 'Worshipper', time: 'Demo · 1 hour ago' },
]
function read<T,>(key: string, fallback: T): T {
  try { return JSON.parse(localStorage.getItem(key) || 'null') ?? fallback } catch { return fallback }
}
const nav = ['Overview', 'Agent runs', 'Models', 'Memory', 'Settings']
const icons = ['◈', '↗', '▧', '▤', '⚙']
function App() {
  const [page, setPage] = useState('Overview')
  const [runs, setRuns] = useState<Run[]>(() => read('worshipper.runs', seed))
  const [settings, setSettings] = useState(() => read('worshipper.settings', { url: 'http://127.0.0.1:8080/v1', model: 'gpt-oss-20b', steps: 25 }))
  const [modal, setModal] = useState(false)
  const [goal, setGoal] = useState('')
  const [filter, setFilter] = useState('All runs')
  const [selected, setSelected] = useState<Run | null>(null)
  const [notice, setNotice] = useState('')
  function saveRun() {
    if (!goal.trim()) return
    const next: Run[] = [{ id: `RUN-${crypto.randomUUID().slice(0, 8).toUpperCase()}`, goal: goal.trim(), status: 'Queued', project: 'Worshipper', time: 'Local draft · just now' }, ...runs]
    setRuns(next); localStorage.setItem('worshipper.runs', JSON.stringify(next)); setModal(false); setGoal(''); setNotice('Run draft saved locally. Connect a backend to execute it.')
  }
  const visible = runs.filter(r => filter === 'All runs' || r.status === filter)
  return <div className="shell">
    <aside><a className="brand" href="#" onClick={() => setPage('Overview')}><span className="logo">w</span> worshipper<span className="beta">LAB</span></a>
      <div className="workspace"><span className="avatar">C</span><div>Personal workspace<small>Local AI infrastructure</small></div><span>⌄</span></div>
      <div className="nav-label">WORKSPACE</div><nav>{nav.map((item, i) => <button key={item} aria-label={item} className={page === item ? 'active' : ''} onClick={() => { setPage(item); setNotice('') }}><span>{icons[i]}</span>{item}{item === 'Agent runs' && <em>{runs.length}</em>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="local-card"><span className="dot amber"/> Demo workspace<p>Your control plane, taking shape.</p><small>Backend connection pending</small></div><div className="profile"><span className="avatar">C</span><div>Connduit<small>Workspace owner</small></div></div></div>
    </aside>
    <div className="main"><header><div className="breadcrumb">Workspace <span>/</span> <strong>{page}</strong></div><div className="header-right"><span className="mode">◉ &nbsp; Demo mode</span><span className="avatar small">C</span></div></header>
      <main><div className="page-heading"><div><div className="eyebrow">YOUR AI CONTROL PLANE</div><h1>{page === 'Overview' ? 'Everything in orbit.' : page}</h1><p>{page === 'Overview' ? 'One place for your agents, models, and the work they do.' : 'Manage your local AI workspace.'}</p></div><button className="primary" onClick={() => setModal(true)}>＋ New run</button></div>
      {notice && <div className="notice" role="status">{notice}<button aria-label="Dismiss notification" onClick={() => setNotice('')}>×</button></div>}
      {(page === 'Overview' || page === 'Agent runs') && <>
        {page === 'Overview' && <><div className="stats"><Stat label="RUN DRAFTS" value={String(runs.filter(r => r.status === 'Queued').length)} note="Waiting for a connected runner" accent/><Stat label="COMPLETED" value={String(runs.filter(r => r.status === 'Completed').length)} note="Sample results · demo data"/><Stat label="MODEL ENDPOINTS" value="1" note="Configured · connection unverified"/><Stat label="AUTONOMY TARGET" value="Level 2" note="Bounded tasks. Verified outcomes."/></div>
        <div className="feature-grid"><section className="mission"><div className="eyebrow green">BUILT TO WORK WITH YOU</div><h2>Give your next idea<br/>a place to start.</h2><p>Plan a task, choose your local model, and keep the outcome in view.</p><button className="text-button" onClick={() => setModal(true)}>Create a run draft <span>↗</span></button><div className="orbit"><div/><span>✳</span><i/><b/></div></section><section className="panel model-card"><div className="section-title"><h3>Inference endpoint</h3><span className="pill">Not connected</span></div><div className="model-icon">▧</div><h2>{settings.model}</h2><p>llama.cpp · OpenAI-compatible</p><div className="model-meta"><span>Deployment<strong>Local server</strong></span><span>Health<strong>Unverified</strong></span></div><button className="secondary wide" onClick={() => setPage('Models')}>Manage endpoint <span>→</span></button></section></div></>}
        <section className="panel run-panel"><div className="section-title"><div><h3>{page === 'Overview' ? 'Recent activity' : 'All agent runs'}</h3><p>Run drafts and illustrative results. No live execution.</p></div><select aria-label="Filter runs" value={filter} onChange={e => setFilter(e.target.value)}>{['All runs', 'Queued', 'Completed', 'Blocked'].map(s => <option key={s}>{s}</option>)}</select></div><div className="table-scroll"><table><thead><tr><th>TASK</th><th>PROJECT</th><th>STATUS</th><th>CREATED</th><th/></tr></thead><tbody>{visible.map(run => <tr key={run.id}><td><button className="task" onClick={() => setSelected(run)}><span className="task-icon">↗</span><span>{run.goal}<small>{run.id}</small></span></button></td><td>{run.project}</td><td><span className={`status ${run.status.toLowerCase()}`}><span className="dot"/>{run.status}</span></td><td>{run.time}</td><td><button className="row-open" aria-label={`View ${run.goal}`} onClick={() => setSelected(run)}>↗</button></td></tr>)}</tbody></table>{visible.length === 0 && <p className="empty">No runs match this filter.</p>}</div></section>
      </>}
      {(page === 'Models' || page === 'Settings') && <section className="panel settings"><div className="section-title"><div><h3>{page === 'Models' ? 'Local inference configuration' : 'Workspace preferences'}</h3><p>Saved in this browser. These settings do not change your server.</p></div><span className="pill">Not connected</span></div><form onSubmit={e => { e.preventDefault(); localStorage.setItem('worshipper.settings', JSON.stringify(settings)); setNotice('Preferences saved in this browser.') }}><label>OpenAI-compatible base URL<input type="url" required value={settings.url} onChange={e => setSettings({ ...settings, url: e.target.value })}/></label><label>Model identifier<input required value={settings.model} onChange={e => setSettings({ ...settings, model: e.target.value })}/></label><label>Step budget<input type="number" min="1" max="1000" required value={settings.steps} onChange={e => setSettings({ ...settings, steps: Number(e.target.value) })}/></label><div className="callout">A backend adapter is required to check model health, start runs, or change server settings. Your local endpoint has not been contacted.</div><button className="primary" type="submit">Save preferences</button></form></section>}
      {page === 'Memory' && <section className="panel empty-memory"><div className="model-icon">▤</div><h2>A little context goes a long way.</h2><p>Project memory will live here once a backend is connected.</p><div className="memory-types"><span>Project decisions</span><span>Verified outcomes</span><span>Conversation archives</span></div><button className="secondary" onClick={() => setPage('Settings')}>View connection settings →</button></section>}
      <footer><span><span className="dot amber"/> Local-first by design</span><span>Worshipper / Control plane v0.1</span></footer></main>
    </div>
    {modal && <div className="overlay" onClick={() => setModal(false)}><section className="dialog" role="dialog" aria-modal="true" aria-labelledby="new-title" onClick={e => e.stopPropagation()}><button className="close" aria-label="Close" onClick={() => setModal(false)}>×</button><div className="eyebrow green">START SOMETHING</div><h2 id="new-title">New run draft</h2><p>Define a bounded task and what a successful result should look like.</p><form onSubmit={e => { e.preventDefault(); saveRun() }}><label>Task and acceptance criteria<textarea autoFocus required rows={5} value={goal} onChange={e => setGoal(e.target.value)} placeholder="Add bounded retries to the inference client. Include regression tests and report the results."/></label><div className="callout">Demo mode: this saves a local draft. No agent or shell commands will execute.</div><button className="primary wide" type="submit">Save run draft ↗</button></form></section></div>}
    {selected && <div className="overlay" onClick={() => setSelected(null)}><section className="dialog" role="dialog" aria-modal="true" aria-labelledby="detail-title" onClick={e => e.stopPropagation()}><button className="close" aria-label="Close" onClick={() => setSelected(null)}>×</button><div className="eyebrow">{selected.id}</div><h2 id="detail-title">{selected.goal}</h2><span className={`status ${selected.status.toLowerCase()}`}>{selected.status}</span><p>{selected.time}</p><div className="callout">{selected.status === 'Queued' ? 'Draft stored in this browser. Waiting for backend integration.' : 'Illustrative demo result. No execution evidence is attached.'}</div><button className="secondary wide" onClick={() => setSelected(null)}>Back to workspace</button></section></div>}
  </div>
}
function Stat({ label, value, note, accent = false }: { label: string; value: string; note: string; accent?: boolean }) { return <section className="stat"><div className="stat-label">{label}<span>↗</span></div><div className={accent ? 'stat-value green' : 'stat-value'}>{value}</div><small>{note}</small></section> }
export default App
