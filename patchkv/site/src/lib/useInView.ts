import { useEffect, useRef, useState } from 'react'

/** Returns [ref, inView]; inView flips true once the element enters the viewport. */
export function useInView<T extends HTMLElement = HTMLDivElement>(threshold = 0.35) {
  const ref = useRef<T | null>(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) setInView(true)
      },
      { threshold },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [threshold])

  return [ref, inView] as const
}
