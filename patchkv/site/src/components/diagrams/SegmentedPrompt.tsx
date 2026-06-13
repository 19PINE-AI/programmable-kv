import { Fragment, useState } from 'react'

export interface Segment {
  name: string
  role: 'ctx' | 'field' | 'rule' | 'filler' | 'decision' | string
  start: number
  end: number
}

const ROLE_CLASS: Record<string, string> = {
  field: 'hl-field',
  rule: 'hl-rule',
  erratum: 'hl-erratum',
}

/**
 * Renders a verbatim prompt with role-colored segments; long "filler" segments
 * are collapsed behind a toggle so the load-bearing text stays visible.
 */
export function SegmentedPrompt({
  text,
  segments,
  extra,
  maxHeight = 460,
}: {
  text: string
  segments: Segment[]
  /** optional appended block (e.g. an erratum) rendered highlighted at the end */
  extra?: { text: string; role: string } | null
  maxHeight?: number
}) {
  const [open, setOpen] = useState<Record<number, boolean>>({})

  // fill gaps between segments so no text is dropped
  const parts: { seg: Segment | null; text: string; idx: number }[] = []
  let cursor = 0
  segments
    .slice()
    .sort((a, b) => a.start - b.start)
    .forEach((seg, i) => {
      if (seg.start > cursor) parts.push({ seg: null, text: text.slice(cursor, seg.start), idx: -i - 1 })
      parts.push({ seg, text: text.slice(seg.start, seg.end), idx: i })
      cursor = Math.max(cursor, seg.end)
    })
  if (cursor < text.length) parts.push({ seg: null, text: text.slice(cursor), idx: -999 })

  return (
    <div className="prompt-box" style={{ maxHeight }}>
      {parts.map(({ seg, text: t, idx }) => {
        if (!seg) return <Fragment key={idx}>{t}</Fragment>
        if (seg.role === 'filler' && t.length > 220 && !open[idx]) {
          const lines = t.split('\n').filter((l) => l.trim())
          return (
            <Fragment key={idx}>
              <span className="dim">{lines[0]}</span>
              {'\n'}
              <button
                onClick={() => setOpen((o) => ({ ...o, [idx]: true }))}
                style={{
                  fontFamily: 'var(--sans)',
                  fontSize: 11,
                  color: 'var(--ink-faint)',
                  background: 'var(--bg-code)',
                  border: '1px solid var(--rule)',
                  borderRadius: 4,
                  padding: '1px 8px',
                  cursor: 'pointer',
                  margin: '2px 0',
                }}
              >
                ⌄ {lines.length - 1} more neutral filler rules (collapsed — click to expand)
              </button>
              {'\n'}
            </Fragment>
          )
        }
        const cls = ROLE_CLASS[seg.role]
        return cls ? (
          <span key={idx} className={cls} title={seg.name}>
            {t}
          </span>
        ) : (
          <Fragment key={idx}>{t}</Fragment>
        )
      })}
      {extra && (
        <>
          {'\n'}
          <span className={ROLE_CLASS[extra.role] ?? 'hl-erratum'}>{extra.text}</span>
        </>
      )}
    </div>
  )
}
