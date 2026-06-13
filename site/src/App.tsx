import { useScrollSpy } from './lib/useScrollSpy'
import { Hero } from './sections/Hero'
import { Challenges } from './sections/Challenges'
import { Composable } from './sections/Composable'
import { Editable } from './sections/Editable'
import { Memory } from './sections/Memory'
import { Systems } from './sections/Systems'
import { Puzzle } from './sections/Puzzle'
import { Mechanism } from './sections/Mechanism'
import { Attention } from './sections/Attention'
import { Keystone } from './sections/Keystone'
import { Horizon } from './sections/Horizon'
import { Reach } from './sections/Reach'
import { DeepControls } from './sections/DeepControls'
import { Circuit } from './sections/Circuit'
import { Explorer } from './sections/Explorer'
import { Boundaries } from './sections/Boundaries'

const TOC = [
  { id: 'top', num: '', title: 'Programmable KV Cache' },
  { id: 'challenge', num: '1', title: 'The challenge' },
  { id: 'composable', num: '2', title: 'Load a skill once' },
  { id: 'editable', num: '3', title: 'Mutate in place' },
  { id: 'memory', num: '4', title: 'User memory' },
  { id: 'systems', num: '5', title: 'Systems payoff' },
  { id: 'puzzle', num: '6', title: 'Why it works' },
  { id: 'mechanism', num: '7', title: 'Models take notes' },
  { id: 'attention', num: '8', title: 'How attention reads' },
  { id: 'keystone', num: '9', title: 'One substrate' },
  { id: 'horizon', num: '10', title: 'No compounding error' },
  { id: 'reach', num: '11', title: 'Reach' },
  { id: 'controls', num: '12', title: 'Under the hood: controls' },
  { id: 'circuit', num: '13', title: 'Under the hood: circuit' },
  { id: 'explorer', num: '14', title: 'Prompts & test cases' },
  { id: 'boundaries', num: '15', title: 'Boundaries' },
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
        {/* Part I — what it does (results & applications) */}
        <Hero />
        <Challenges />
        <Composable />
        <Editable />
        <Memory />
        <Systems />
        {/* Part II — why it works (the principle) */}
        <Puzzle />
        <Mechanism />
        <Attention />
        <Keystone />
        <Horizon />
        <Reach />
        {/* Part III — under the hood (deep interpretability, optional) */}
        <DeepControls />
        <Circuit />
        <Explorer />
        <Boundaries />
      </main>
    </div>
  )
}
