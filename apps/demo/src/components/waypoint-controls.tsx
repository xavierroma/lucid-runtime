import { useRef } from "react"
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Crosshair, MousePointer2 } from "lucide-react"

export interface WaypointControlState {
  forward: boolean
  backward: boolean
  left: boolean
  right: boolean
  jump: boolean
  sprint: boolean
  crouch: boolean
  primary_fire: boolean
  secondary_fire: boolean
  mouse_x: number
  mouse_y: number
  scroll_wheel: number
}

export type WaypointHoldControl =
  | "forward"
  | "backward"
  | "left"
  | "right"
  | "jump"
  | "sprint"
  | "crouch"
  | "primary_fire"
  | "secondary_fire"

interface WaypointControlsProps {
  value: WaypointControlState
  disabled: boolean
  onHoldChange: (control: WaypointHoldControl, pressed: boolean) => void
  onLookChange: (mouseX: number, mouseY: number) => void
  onLookReset: () => void
  onScrollNudge: (direction: -1 | 1) => void
}

interface HoldButtonProps {
  label: string
  icon?: React.ReactNode
  active: boolean
  disabled: boolean
  onPressedChange: (pressed: boolean) => void
  className?: string
}

function HoldButton({
  label,
  icon,
  active,
  disabled,
  onPressedChange,
  className,
}: HoldButtonProps) {
  return (
    <button
      type="button"
      className={`control-button ${active ? "control-button-active" : ""} ${className ?? ""}`}
      disabled={disabled}
      onContextMenu={(event) => event.preventDefault()}
      onPointerDown={(event) => {
        if (disabled) {
          return
        }
        event.preventDefault()
        event.currentTarget.setPointerCapture(event.pointerId)
        onPressedChange(true)
      }}
      onPointerUp={(event) => {
        event.preventDefault()
        onPressedChange(false)
      }}
      onPointerCancel={() => onPressedChange(false)}
      onPointerLeave={(event) => {
        if (event.buttons === 0) {
          onPressedChange(false)
        }
      }}
    >
      {icon ? <span className="control-button-icon">{icon}</span> : null}
      <span>{label}</span>
    </button>
  )
}

export function WaypointControls({
  value,
  disabled,
  onHoldChange,
  onLookChange,
  onLookReset,
  onScrollNudge,
}: WaypointControlsProps) {
  const lookPadRef = useRef<HTMLDivElement | null>(null)
  const lastPointerRef = useRef<{ x: number; y: number } | null>(null)

  return (
    <section
      className={`control-deck ${disabled ? "control-deck-disabled" : ""}`}
      aria-label="Waypoint controls"
    >
      <div className="control-deck-header">
        <div>
          <p className="control-deck-kicker">Waypoint Controls</p>
          <p className="control-deck-copy">
            Hold movement and action buttons, then drag the look pad to steer.
          </p>
        </div>
        <div className="control-scroll">
          <button
            type="button"
            className="scroll-chip"
            disabled={disabled}
            onClick={() => onScrollNudge(-1)}
          >
            -
          </button>
          <span className="scroll-readout">
            WHEEL {value.scroll_wheel > 0 ? "+" : ""}
            {value.scroll_wheel}
          </span>
          <button
            type="button"
            className="scroll-chip"
            disabled={disabled}
            onClick={() => onScrollNudge(1)}
          >
            +
          </button>
        </div>
      </div>

      <div className="control-deck-grid">
        <div className="dpad-grid" aria-label="Movement pad">
          <div />
          <HoldButton
            label="Forward"
            icon={<ArrowUp className="size-4" />}
            active={value.forward}
            disabled={disabled}
            className="dpad-button"
            onPressedChange={(pressed) => onHoldChange("forward", pressed)}
          />
          <div />
          <HoldButton
            label="Left"
            icon={<ArrowLeft className="size-4" />}
            active={value.left}
            disabled={disabled}
            className="dpad-button"
            onPressedChange={(pressed) => onHoldChange("left", pressed)}
          />
          <div className="dpad-center" aria-hidden="true" />
          <HoldButton
            label="Right"
            icon={<ArrowRight className="size-4" />}
            active={value.right}
            disabled={disabled}
            className="dpad-button"
            onPressedChange={(pressed) => onHoldChange("right", pressed)}
          />
          <div />
          <HoldButton
            label="Back"
            icon={<ArrowDown className="size-4" />}
            active={value.backward}
            disabled={disabled}
            className="dpad-button"
            onPressedChange={(pressed) => onHoldChange("backward", pressed)}
          />
          <div />
        </div>

        <div
          ref={lookPadRef}
          className="look-pad"
          aria-label="Look pad"
          onContextMenu={(event) => event.preventDefault()}
          onWheel={(event) => {
            if (disabled || event.deltaY === 0) {
              return
            }
            event.preventDefault()
            onScrollNudge(event.deltaY < 0 ? 1 : -1)
          }}
          onPointerDown={(event) => {
            if (disabled || !lookPadRef.current) {
              return
            }
            event.preventDefault()
            event.currentTarget.setPointerCapture(event.pointerId)
            lastPointerRef.current = {
              x: event.clientX,
              y: event.clientY,
            }
          }}
          onPointerMove={(event) => {
            if (disabled || !lookPadRef.current || event.buttons === 0) {
              return
            }
            const lastPointer = lastPointerRef.current
            lastPointerRef.current = {
              x: event.clientX,
              y: event.clientY,
            }
            if (!lastPointer) {
              return
            }
            const deltaX = event.clientX - lastPointer.x
            const deltaY = event.clientY - lastPointer.y
            if (deltaX === 0 && deltaY === 0) {
              return
            }
            onLookChange(deltaX, deltaY)
          }}
          onPointerUp={() => {
            lastPointerRef.current = null
            onLookReset()
          }}
          onPointerCancel={() => {
            lastPointerRef.current = null
            onLookReset()
          }}
          onPointerLeave={(event) => {
            if (event.buttons === 0) {
              lastPointerRef.current = null
              onLookReset()
            }
          }}
        >
          <div className="look-pad-grid" aria-hidden="true" />
          <div className="look-pad-label">
            <MousePointer2 className="size-4" />
            LOOK
          </div>
          <div
            className="look-pad-thumb"
            style={{
              left: `${(value.mouse_x + 1) * 50}%`,
              top: `${(value.mouse_y + 1) * 50}%`,
            }}
          >
            <Crosshair className="size-4" />
          </div>
        </div>

        <div className="action-grid" aria-label="Action buttons">
          <HoldButton
            label="Jump"
            active={value.jump}
            disabled={disabled}
            onPressedChange={(pressed) => onHoldChange("jump", pressed)}
          />
          <HoldButton
            label="Sprint"
            active={value.sprint}
            disabled={disabled}
            onPressedChange={(pressed) => onHoldChange("sprint", pressed)}
          />
          <HoldButton
            label="Crouch"
            active={value.crouch}
            disabled={disabled}
            onPressedChange={(pressed) => onHoldChange("crouch", pressed)}
          />
          <HoldButton
            label="Fire"
            active={value.primary_fire}
            disabled={disabled}
            onPressedChange={(pressed) => onHoldChange("primary_fire", pressed)}
          />
          <HoldButton
            label="Alt"
            active={value.secondary_fire}
            disabled={disabled}
            onPressedChange={(pressed) => onHoldChange("secondary_fire", pressed)}
          />
        </div>
      </div>
    </section>
  )
}
