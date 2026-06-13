import { useScrollSpy } from './lib/useScrollSpy'
import { Hero } from './sections/Hero'
import { Puzzle } from './sections/Puzzle'
import { Mechanism } from './sections/Mechanism'
import { Attention } from './sections/Attention'
import { DeepControls } from './sections/DeepControls'
import { Circuit } from './sections/Circuit'
import { Editable } from './sections/Editable'
import { Composable } from './sections/Composable'
import { Keystone } from './sections/Keystone'
import { Reach } from './sections/Reach'
import { Systems } from './sections/Systems'
import { Memory } from './sections/Memory'
import { Horizon } from './sections/Horizon'
import { Explorer } from './sections/Explorer'
import { Boundaries } from './sections/Boundaries'

const TOC = [
  { id: 'top', num: '', title: 'Models Take Notes at Prefill' },
  { id: 'puzzle', num: '1', title: 'The puzzle' },
  { id: 'mechanism', num: '2', title: 'The discovery: four probes' },
  { id: 'attention', num: '3', title: 'How attention reads the notes' },
  { id: 'controls', num: '4', title: 'Stress-testing the account' },
  { id: 'circuit', num: '5', title: 'The circuit' },
  { id: 'editable', num: '6', title: 'Consequence I: editable' },
  { id: 'composable', num: '7', title: 'Consequence II: composable' },
  { id: 'keystone', num: '8', title: 'One substrate' },
  { id: 'reach', num: '9', title: 'Reach & the frontier' },
  { id: 'systems', num: '10', title: 'Systems payoff' },
  { id: 'memory', num: '11', title: 'Application: user memory' },
  { id: 'horizon', num: '12', title: 'No compounding error' },
  { id: 'explorer', num: '13', title: 'Prompts & test cases' },
  { id: 'boundaries', num: '14', title: 'Boundaries & colophon' },
]

export default function App() {
  const active = useScrollSpy(TOC.map((t) => t.id))

  return (
    <div className="page">
      <nav className="toc">
        <div className="toc-title">Contents</div>
        {TOC.map((t) => (
          <a key={t.id} href={`#${t.id}`} className={active === t.id ? 'active' : ''}>
            {t.num && <span className="toc-num">{t.num}</span>}
            {t.id === 'top' ? 'Top' : t.title}
          </a>
        ))}
      </nav>
      <main className="article" id="top">
        <Hero />
        <Puzzle />
        <Mechanism />
        <Attention />
        <DeepControls />
        <Circuit />
        <Editable />
        <Composable />
        <Keystone />
        <Reach />
        <Systems />
        <Memory />
        <Horizon />
        <Explorer />
        <Boundaries />
      </main>
    </div>
  )
}
