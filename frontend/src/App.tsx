import { useEffect, useRef, useState } from 'react'
import './App.css'

// Shapes of the FastAPI responses (backend/app.py).
type Source = { chunk_id: string; faq_no: number; section: string; text: string; score: number }
type AskResponse = {
  answer: string
  abstained: boolean
  gated: boolean
  sources: Source[]
  timings_ms: { retrieval: number; generation: number; backend_total: number }
  tokens: { input: number; output: number }
  model: string
  adapter: string | null
  mode: string
}
type Health = { status: string; model: string; adapter: string | null; mode: string; gpu: string | null }
type Turn = { question: string; result?: AskResponse; error?: string; clientMs?: number }

// Two in-domain questions, then two outside the knowledge base (the assistant must refuse those).
const EXAMPLES = [
  'What are typical approaches to tune expensive SQL statements?',
  'How can I capture bind values of prepared SQL statements?',
  'How do I renew an expired SAP HANA SSL certificate?',
  'What is the capital of France?',
]

function seconds(ms: number) {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

// The memorable element: a trace strip that splits the response time into its stages,
// like an execution trace in a query-plan tool.
function TraceStrip({ result, clientMs }: { result: AskResponse; clientMs: number }) {
  const t = result.timings_ms
  const network = Math.max(clientMs - t.backend_total, 0)
  const parts = [
    { key: 'retrieval', label: 'Retrieval', ms: t.retrieval },
    { key: 'generation', label: 'Generation', ms: t.generation },
    { key: 'network', label: 'Browser and network', ms: network },
  ]
  return (
    <figure className="trace" aria-label="Response time breakdown">
      <div className="trace-bar">
        {parts.map((p) => (
          <span
            key={p.key}
            className={`trace-seg trace-${p.key}`}
            style={{ flexGrow: Math.max(p.ms, 1) }}
            title={`${p.label}: ${seconds(p.ms)}`}
          />
        ))}
      </div>
      <figcaption>
        <strong>{seconds(clientMs)}</strong> total
        {parts.map((p) => (
          <span key={p.key} className="trace-legend">
            <i className={`trace-dot trace-${p.key}`} />
            {p.label} {seconds(p.ms)}
          </span>
        ))}
        <span className="trace-legend">{result.tokens.output} tokens generated</span>
      </figcaption>
    </figure>
  )
}

function Answer({ turn }: { turn: Turn }) {
  if (turn.error) return <p className="error">{turn.error}</p>
  if (!turn.result) return <p className="pending">Searching the knowledge base and writing an answer…</p>
  const r = turn.result
  return (
    <>
      {r.abstained ? (
        <p className="verdict verdict-missing">
          {r.gated ? 'Not covered: no similar section in the knowledge base' : 'Not covered by the knowledge base'}
        </p>
      ) : (
        <p className="verdict verdict-grounded">
          {r.mode === 'rag' ? `Answer drawn from ${r.sources.length} excerpts` : 'Answer from the model alone'}
        </p>
      )}
      <div className="answer-text">{r.answer}</div>
      <TraceStrip result={r} clientMs={turn.clientMs ?? r.timings_ms.backend_total} />
    </>
  )
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [question, setQuestion] = useState('')
  const [turns, setTurns] = useState<Turn[]>([])
  const [busy, setBusy] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    let stop = false
    const poll = async () => {
      try {
        const res = await fetch('/api/health')
        const h: Health = await res.json()
        if (!stop) setHealth(h)
        if (h.status !== 'ok' && !stop) setTimeout(poll, 2000)
      } catch {
        if (!stop) {
          setHealth(null)
          setTimeout(poll, 3000)
        }
      }
    }
    poll()
    return () => {
      stop = true
    }
  }, [])

  async function ask(q: string) {
    const text = q.trim()
    if (text.length < 3 || busy) return
    setBusy(true)
    setQuestion('')
    setTurns((t) => [{ question: text }, ...t])
    const started = performance.now()
    let update: Partial<Turn>
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text }),
      })
      if (!res.ok) throw new Error(`The server answered ${res.status}. Check that the backend is running and the model has loaded.`)
      update = { result: await res.json(), clientMs: performance.now() - started }
    } catch (e) {
      update = { error: e instanceof Error ? e.message : 'The request failed. Check that the backend is running.' }
    }
    setTurns((t) => [{ ...t[0], ...update }, ...t.slice(1)])
    setBusy(false)
    inputRef.current?.focus()
  }

  const latest = turns[0]
  const ready = health?.status === 'ok'

  return (
    <div className="page">
      <header className="masthead">
        <h1>HANA SQL Tuning Assistant</h1>
        <p className={`status ${ready ? 'status-ready' : 'status-wait'}`} role="status">
          {health === null
            ? 'Backend not reachable on port 8000'
            : ready
              ? `${health.model.split('/').pop()}${health.adapter ? ' + fine-tuned adapter' : ''}, ${health.mode === 'rag' ? 'retrieval on' : 'closed book'}`
              : 'Loading the model…'}
        </p>
      </header>

      <p className="scope-note">
        This assistant only knows one SAP HANA knowledge base article about SQL performance. Questions outside that
        domain are refused on purpose: it answers "not covered" instead of guessing.
      </p>

      <main className="layout">
        <section className="ask-column" aria-label="Question and answer">
          <form
            className="ask"
            onSubmit={(e) => {
              e.preventDefault()
              ask(question)
            }}
          >
            <label htmlFor="q">Ask about SQL performance in SAP HANA</label>
            <textarea
              id="q"
              ref={inputRef}
              rows={3}
              value={question}
              placeholder="For example: how can I identify the client coding responsible for a database request?"
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  ask(question)
                }
              }}
            />
            <div className="ask-row">
              <button type="submit" disabled={!ready || busy || question.trim().length < 3}>
                {busy ? 'Answering…' : 'Ask'}
              </button>
              <span className="hint">Enter to send, Shift+Enter for a new line</span>
            </div>
          </form>

          {turns.length === 0 && (
            <div className="examples">
              <p>Try one of these. The last two are outside the knowledge base on purpose.</p>
              <ul>
                {EXAMPLES.map((ex) => (
                  <li key={ex}>
                    <button type="button" className="example" disabled={!ready || busy} onClick={() => ask(ex)}>
                      {ex}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {turns.map((turn, i) => (
            <article key={turns.length - i} className={`turn ${i === 0 ? 'turn-latest' : 'turn-old'}`}>
              <h2 className="question">{turn.question}</h2>
              <Answer turn={turn} />
            </article>
          ))}
        </section>

        <aside className="sources" aria-label="Sources used for the latest answer">
          <h2>Sources</h2>
          {!latest?.result && <p className="muted">The knowledge base excerpts behind each answer appear here.</p>}
          {latest?.result && latest.result.sources.length === 0 && (
            <p className="muted">
              {latest.result.gated
                ? 'Out of scope: no knowledge base section is close enough to this question, so the model was not called.'
                : 'This mode answers without retrieval, so there are no excerpts to show.'}
            </p>
          )}
          {latest?.result?.sources.map((s, i) => (
            <details key={s.chunk_id} open={i === 0}>
              <summary>
                <span className="src-rank">{i + 1}</span>
                <span className="src-section">
                  FAQ {s.faq_no}: {s.section}
                </span>
              </summary>
              <pre className="src-text">{s.text}</pre>
              <p className="src-id">{s.chunk_id}</p>
            </details>
          ))}
        </aside>
      </main>
    </div>
  )
}
