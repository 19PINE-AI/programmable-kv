// Headline metrics derived once from the released result records, shared by the
// Hero band and the Challenges section so the two never drift.
import composing from '../data/composing.json'
import systems from '../data/systems.json'
import keystone from '../data/keystone.json'
import memory from '../data/memory.json'

const sc = (composing.scaling as any[]).find((s) => s.tag === 'qwen3_8b')
export const ttft32: number = sc?.points.find((p: any) => p.L === 32000)?.speedup

const sat = (systems.rows as any[])[(systems.rows as any[]).length - 1]
export const servingThroughput: number = sat.throughput_speedup
export const apcErratum: number = sat.erratum.prefix_hit_rate
export const apcBaseline: number = sat.baseline.prefix_hit_rate

const e5 = Object.values(memory.e5.by_model as any).map((v: any) => v.cum_speedup_vs_end as number)
export const memTtftLo: number = Math.min(...e5)
export const memTtftHi: number = Math.max(...e5)

const agr = (keystone.agent as any[]).map((a) => a.agreement as number)
export const agrLo: number = Math.min(...agr)
export const agrHi: number = Math.max(...agr)
