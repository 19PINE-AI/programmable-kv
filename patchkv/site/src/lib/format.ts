export function fmt(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  return x.toFixed(digits)
}

export function fmtPct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  return `${(100 * x).toFixed(digits)}%`
}

export function fmtX(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  return `${x.toFixed(digits)}×`
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return '—'
  if (ms >= 10_000) return `${(ms / 1000).toFixed(1)} s`
  if (ms >= 1_000) return `${(ms / 1000).toFixed(2)} s`
  return `${ms.toFixed(0)} ms`
}

export function fmtTokens(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}k`
  return String(n)
}

/** Pretty model label from a record's model id or filename tag. */
export function modelLabel(id: string): string {
  const MAP: Record<string, string> = {
    qwen3_0p6b: 'Qwen3-0.6B', qwen3_1p7b: 'Qwen3-1.7B', qwen3_4b: 'Qwen3-4B',
    qwen3_8b: 'Qwen3-8B', qwen3_14b: 'Qwen3-14B', qwen3_32b: 'Qwen3-32B-FP8',
    qwen3_30a3b: 'Qwen3-30B-A3B', qwen3_32b_fp8: 'Qwen3-32B-FP8',
    llama31_8b: 'Llama-3.1-8B', llama31_70b_4bit: 'Llama-3.1-70B (4-bit)',
    mistral7b: 'Mistral-7B', mistral_7b: 'Mistral-7B',
    gemma2_2b: 'Gemma-2-2B', gemma2_9b: 'Gemma-2-9B', gemma2_27b: 'Gemma-2-27B',
    gemma3_4b: 'Gemma-3-4B', gemma3_27b: 'Gemma-3-27B', gemma3_27b_bf16: 'Gemma-3-27B',
    dsr1_llama8b: 'R1-Distill-Llama-8B', dsr1llama8b: 'R1-Distill-Llama-8B',
    smollm2_1p7b: 'SmolLM2-1.7B',
    qwen25vl_3b: 'Qwen2.5-VL-3B', qwen25vl_7b: 'Qwen2.5-VL-7B', qwen25vl_32b: 'Qwen2.5-VL-32B',
    qwen2vl_7b: 'Qwen2-VL-7B', qwen3vl_8b: 'Qwen3-VL-8B', qwen3vl_30a3b: 'Qwen3-VL-30B-A3B',
    dscoderv2_mla: 'DeepSeek-Coder-V2-Lite', dsv2lite_mla: 'DeepSeek-V2-Lite',
  }
  if (MAP[id]) return MAP[id]
  // HF-style ids: keep the part after the slash
  const tail = id.includes('/') ? id.split('/').pop()! : id
  return tail
    .replace('Meta-Llama-3.1-', 'Llama-3.1-')
    .replace('-Instruct-bnb-4bit', ' (4-bit)')
    .replace('-Instruct', '')
    .replace('-it', '')
    .replace('unsloth/', '')
}

/** Stable ordering for model families: Qwen3 by size, then Llama, Mistral, Gemma, etc. */
export function modelSortKey(label: string): string {
  const order = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-30B-A3B',
    'Qwen3-32B', 'Llama-3.1-8B', 'Llama-3.1-70B', 'Mistral-7B', 'Gemma-2-2B', 'Gemma-2-9B',
    'Gemma-2-27B', 'Gemma-3-4B', 'Gemma-3-27B', 'R1-Distill']
  const i = order.findIndex((p) => label.startsWith(p))
  return `${i === -1 ? 99 : String(i).padStart(2, '0')}_${label}`
}
