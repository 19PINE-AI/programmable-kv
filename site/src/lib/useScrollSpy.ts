import { useEffect, useState } from 'react'

/** Tracks which section id is currently active while scrolling. */
export function useScrollSpy(ids: string[]): string {
  const [active, setActive] = useState(ids[0] ?? '')

  useEffect(() => {
    function onScroll() {
      const probe = window.scrollY + window.innerHeight * 0.25
      let current = ids[0] ?? ''
      for (const id of ids) {
        const el = document.getElementById(id)
        if (el && el.offsetTop <= probe) current = id
      }
      setActive(current)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [ids.join(',')])

  return active
}
