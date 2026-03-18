import { useEffect, useMemo, useRef, useState } from "react"
import { Keyboard } from "lucide-react"
import type {
  AxisInputBinding,
  HoldInputBinding,
  ManifestInput,
  PressInputBinding,
  SessionState,
} from "@/lib/lucid"

function keyLabel(code: string): string {
  const map: Record<string, string> = {
    Space: "Space",
    ArrowUp: "↑",
    ArrowDown: "↓",
    ArrowLeft: "←",
    ArrowRight: "→",
    ShiftLeft: "⇧",
    ShiftRight: "⇧",
    ControlLeft: "Ctrl",
    ControlRight: "Ctrl",
    AltLeft: "Alt",
    AltRight: "Alt",
    Escape: "Esc",
    Enter: "↵",
    Backspace: "⌫",
    Tab: "⇥",
    CapsLock: "Caps",
  }
  if (code in map) return map[code]
  if (code.startsWith("Key")) return code.slice(3)
  if (code.startsWith("Digit")) return code.slice(5)
  if (code.startsWith("Numpad")) return code.slice(6)
  return code
}

function mouseButtonLabel(button: number): string {
  if (button === 0) return "LMB"
  if (button === 1) return "MMB"
  if (button === 2) return "RMB"
  return `M${button}`
}

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false
  return (
    target.tagName === "TEXTAREA" ||
    target.tagName === "INPUT" ||
    target.isContentEditable
  )
}

function inputDisplayName(name: string): string {
  return name.replace(/_/g, " ").toUpperCase()
}

// Keys that define WASD movement, in natural display order W→A→S→D
const WASD_PRIORITY: Record<string, number> = {
  KeyW: 0,
  ArrowUp: 0,
  KeyA: 1,
  ArrowLeft: 1,
  KeyS: 2,
  ArrowDown: 2,
  KeyD: 3,
  ArrowRight: 3,
}

const MODIFIER_KEYS = new Set([
  "ShiftLeft", "ShiftRight",
  "ControlLeft", "ControlRight",
  "AltLeft", "AltRight",
])

function inputSortScore(input: ManifestInput): number {
  const b = input.binding
  if (!b) return 100

  if (b.kind === "hold") {
    // WASD/arrow movement cluster comes first
    const wasdScore = Math.min(...b.keys.map((k) => WASD_PRIORITY[k] ?? Infinity))
    if (wasdScore !== Infinity) return wasdScore          // 0–3
    if (b.mouse_buttons.length > 0) return 12             // mouse-hold (e.g. fire)
    if (b.keys.some((k) => MODIFIER_KEYS.has(k))) return 40  // Ctrl/Shift/Alt
    return 30                                             // other held keys
  }

  if (b.kind === "pointer") return 10                     // look (mouse drag) after movement
  if (b.kind === "axis") return 20                        // analog axes

  if (b.kind === "press") {
    if (b.keys.includes("Space")) return 50               // jump
    if (b.keys.some((k) => MODIFIER_KEYS.has(k))) return 55
    if (b.mouse_buttons.length > 0) return 52             // mouse-click action
    return 60                                             // other press actions
  }

  if (b.kind === "wheel") return 70                       // zoom / scroll last

  return 90
}

function KeyCap({ code, active }: { code: string; active: boolean }) {
  return (
    <span className={`io-key${active ? " io-key-active" : ""}`}>
      {keyLabel(code)}
    </span>
  )
}

function MouseBtn({ button, active }: { button: number; active: boolean }) {
  return (
    <span className={`io-key${active ? " io-key-active" : ""}`}>
      {mouseButtonLabel(button)}
    </span>
  )
}

interface RowProps {
  input: ManifestInput
  pressedKeys: ReadonlySet<string>
  pressedButtons: ReadonlySet<number>
  flashedKeys: ReadonlySet<string>
}

function InputRow({ input, pressedKeys, pressedButtons, flashedKeys }: RowProps) {
  const { binding } = input
  if (!binding) return null

  const label = inputDisplayName(input.name)

  return (
    <div className="io-row">
      <span className="io-label">{label}</span>
      <span className="io-keys">
        {(binding.kind === "hold" || binding.kind === "press") && (() => {
          const b = binding as HoldInputBinding | PressInputBinding
          const isPress = binding.kind === "press"
          return (
            <>
              {b.keys.map((code) => (
                <KeyCap
                  key={code}
                  code={code}
                  active={
                    pressedKeys.has(code) || (isPress && flashedKeys.has(code))
                  }
                />
              ))}
              {b.mouse_buttons.map((btn) => (
                <MouseBtn key={btn} button={btn} active={pressedButtons.has(btn)} />
              ))}
            </>
          )
        })()}

        {binding.kind === "axis" && (() => {
          const b = binding as AxisInputBinding
          return (
            <>
              {b.negative_keys.map((code) => (
                <KeyCap key={code} code={code} active={pressedKeys.has(code)} />
              ))}
              <span className="io-axis-sep">↔</span>
              {b.positive_keys.map((code) => (
                <KeyCap key={code} code={code} active={pressedKeys.has(code)} />
              ))}
            </>
          )
        })()}

        {binding.kind === "pointer" && (
          <span className="io-mouse-hint">
            {binding.pointer_lock ? "mouse drag" : "mouse move"}
          </span>
        )}

        {binding.kind === "wheel" && (
          <span className="io-mouse-hint">mouse scroll</span>
        )}
      </span>
    </div>
  )
}

interface InputOverlayProps {
  inputs: ManifestInput[]
  sessionState: SessionState
}

export function InputOverlay({ inputs, sessionState }: InputOverlayProps) {
  const [collapsed, setCollapsed] = useState(false)
  const [pressedKeys, setPressedKeys] = useState<ReadonlySet<string>>(new Set())
  const [pressedButtons, setPressedButtons] = useState<ReadonlySet<number>>(new Set())
  const [flashedKeys, setFlashedKeys] = useState<ReadonlySet<string>>(new Set())
  const flashTimersRef = useRef(new Map<string, ReturnType<typeof setTimeout>>())

  const sessionActive = sessionState === "RUNNING" || sessionState === "PAUSED"
  const boundInputs = useMemo(
    () =>
      inputs
        .filter((i) => i.binding)
        .slice()
        .sort((a, b) => inputSortScore(a) - inputSortScore(b)),
    [inputs],
  )
  const pressKeyCodes = useMemo(
    () =>
      new Set(
        boundInputs
          .filter((i) => i.binding?.kind === "press")
          .flatMap((i) => (i.binding as PressInputBinding).keys),
      ),
    [boundInputs],
  )

  useEffect(() => {
    if (!sessionActive) {
      setPressedKeys(new Set())
      setPressedButtons(new Set())
      return
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (isEditableTarget(e.target)) return
      setPressedKeys((prev) => {
        if (prev.has(e.code)) return prev
        const next = new Set(prev)
        next.add(e.code)
        return next
      })
      if (pressKeyCodes.has(e.code)) {
        setFlashedKeys((prev) => new Set([...prev, e.code]))
        const existing = flashTimersRef.current.get(e.code)
        if (existing) clearTimeout(existing)
        flashTimersRef.current.set(
          e.code,
          setTimeout(() => {
            setFlashedKeys((prev) => {
              const next = new Set(prev)
              next.delete(e.code)
              return next
            })
            flashTimersRef.current.delete(e.code)
          }, 220),
        )
      }
    }

    const handleKeyUp = (e: KeyboardEvent) => {
      setPressedKeys((prev) => {
        if (!prev.has(e.code)) return prev
        const next = new Set(prev)
        next.delete(e.code)
        return next
      })
    }

    const handleMouseDown = (e: MouseEvent) => {
      setPressedButtons((prev) => new Set([...prev, e.button]))
    }

    const handleMouseUp = (e: MouseEvent) => {
      setPressedButtons((prev) => {
        const next = new Set(prev)
        next.delete(e.button)
        return next
      })
    }

    window.addEventListener("keydown", handleKeyDown)
    window.addEventListener("keyup", handleKeyUp)
    document.addEventListener("mousedown", handleMouseDown)
    document.addEventListener("mouseup", handleMouseUp)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
      window.removeEventListener("keyup", handleKeyUp)
      document.removeEventListener("mousedown", handleMouseDown)
      document.removeEventListener("mouseup", handleMouseUp)
    }
  }, [sessionActive, pressKeyCodes])

  useEffect(() => {
    return () => {
      for (const timer of flashTimersRef.current.values()) clearTimeout(timer)
    }
  }, [])

  if (!boundInputs.length) return null

  return (
    <div className={`io-overlay${!sessionActive ? " io-overlay-dim" : ""}`}>
      {!collapsed && (
        <div className="io-panel">
          {boundInputs.map((input) => (
            <InputRow
              key={input.name}
              input={input}
              pressedKeys={pressedKeys}
              pressedButtons={pressedButtons}
              flashedKeys={flashedKeys}
            />
          ))}
        </div>
      )}
      <button
        type="button"
        className="io-toggle"
        onClick={() => setCollapsed((c) => !c)}
        aria-label={collapsed ? "Show controls" : "Hide controls"}
      >
        <Keyboard size={11} />
      </button>
    </div>
  )
}
