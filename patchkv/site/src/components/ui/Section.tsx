import { ReactNode } from 'react'

export interface SectionMeta {
  id: string
  num: string
  title: string
  tocTitle?: string
}

export function Section({ meta, children }: { meta: SectionMeta; children: ReactNode }) {
  return (
    <section id={meta.id}>
      <div className="prose">
        <h2>
          <span className="sec-num">§ {meta.num}</span>
          {meta.title}
        </h2>
      </div>
      {children}
    </section>
  )
}

export function P({ children }: { children: ReactNode }) {
  return (
    <div className="prose">
      <p>{children}</p>
    </div>
  )
}

export function H3({ children }: { children: ReactNode }) {
  return (
    <div className="prose">
      <h3>{children}</h3>
    </div>
  )
}

export function Aside({ children, warn = false }: { children: ReactNode; warn?: boolean }) {
  return (
    <div className="prose">
      <div className={`aside${warn ? ' warn' : ''}`}>{children}</div>
    </div>
  )
}

/** Marker for numbers that exist only in the paper text, not in result records. */
export function PaperConst({ src }: { src: string }) {
  return (
    <span className="paper-const" title={`This value is stated in ${src}; it is not present in the released result records.`}>
      ⊙ from paper text · {src}
    </span>
  )
}
