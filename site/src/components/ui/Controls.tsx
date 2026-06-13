import { ReactNode } from 'react'

export function Controls({ children }: { children: ReactNode }) {
  return <div className="controls">{children}</div>
}

export function Seg<T extends string>({
  options,
  value,
  onChange,
  labels,
  accent,
}: {
  options: readonly T[]
  value: T
  onChange: (v: T) => void
  labels?: Partial<Record<T, string>>
  accent?: 'orange' | 'blue'
}) {
  return (
    <span className="seg">
      {options.map((o) => (
        <button
          key={o}
          className={`${value === o ? 'on' : ''}${value === o && accent ? ` accent-${accent}` : ''}`}
          onClick={() => onChange(o)}
        >
          {labels?.[o] ?? o}
        </button>
      ))}
    </span>
  )
}

export function ControlGroup({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      <span className="control-label">{label}</span>
      {children}
    </span>
  )
}

export function ModelPicker({
  models,
  value,
  onChange,
}: {
  models: { id: string; label: string }[]
  value: string
  onChange: (id: string) => void
}) {
  return (
    <select className="model-picker" value={value} onChange={(e) => onChange(e.target.value)}>
      {models.map((m) => (
        <option key={m.id} value={m.id}>
          {m.label}
        </option>
      ))}
    </select>
  )
}
