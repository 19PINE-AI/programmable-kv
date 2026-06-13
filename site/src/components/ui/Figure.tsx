import { ReactNode } from 'react'

export function Figure({
  label,
  caption,
  title,
  sub,
  narrow = false,
  children,
}: {
  label?: string
  caption?: ReactNode
  title?: string
  sub?: string
  narrow?: boolean
  children: ReactNode
}) {
  return (
    <figure className={`figure${narrow ? ' figure-narrow' : ''}`} style={{ margin: '36px auto' }}>
      <div className="figure-frame">
        {title && <div className="fig-title">{title}</div>}
        {sub && <div className="fig-sub">{sub}</div>}
        {children}
      </div>
      {caption && (
        <figcaption className="figure-caption">
          {label && <span className="fig-label">{label} </span>}
          {caption}
        </figcaption>
      )}
    </figure>
  )
}
