import type {
  WaypointControlState,
  WaypointHoldControl,
} from "@/components/waypoint-controls"

export const WAYPOINT_MOUSE_WHEEL_STEP = 120

const LOOK_PREVIEW_SCALE_PX = 40
const VK_LBUTTON = 0x01
const VK_RBUTTON = 0x02
const VK_SPACE = 0x20
const VK_A = 0x41
const VK_D = 0x44
const VK_S = 0x53
const VK_W = 0x57
const VK_LSHIFT = 0xa0
const VK_RSHIFT = 0xa1
const VK_LCONTROL = 0xa2
const VK_RCONTROL = 0xa3

const HOLD_CONTROL_TO_BUTTON_ID: Record<WaypointHoldControl, number> = {
  forward: VK_W,
  backward: VK_S,
  left: VK_A,
  right: VK_D,
  jump: VK_SPACE,
  sprint: VK_LSHIFT,
  crouch: VK_LCONTROL,
  primary_fire: VK_LBUTTON,
  secondary_fire: VK_RBUTTON,
}

const ACTIVE_BUTTON_GROUPS: Record<WaypointHoldControl, readonly number[]> = {
  forward: [VK_W],
  backward: [VK_S],
  left: [VK_A],
  right: [VK_D],
  jump: [VK_SPACE],
  sprint: [VK_LSHIFT, VK_RSHIFT],
  crouch: [VK_LCONTROL, VK_RCONTROL],
  primary_fire: [VK_LBUTTON],
  secondary_fire: [VK_RBUTTON],
}

const KEYBOARD_CODE_TO_BUTTON_ID: Record<string, number> = {
  KeyW: VK_W,
  KeyA: VK_A,
  KeyS: VK_S,
  KeyD: VK_D,
  Space: VK_SPACE,
  ShiftLeft: VK_LSHIFT,
  ShiftRight: VK_RSHIFT,
  ControlLeft: VK_LCONTROL,
  ControlRight: VK_RCONTROL,
  KeyJ: VK_LBUTTON,
  KeyK: VK_RBUTTON,
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

export function waypointButtonIdForHoldControl(control: WaypointHoldControl) {
  return HOLD_CONTROL_TO_BUTTON_ID[control]
}

export function waypointButtonIdForKeyboardCode(code: string) {
  return KEYBOARD_CODE_TO_BUTTON_ID[code] ?? null
}

export function waypointButtonIdForPointerButton(button: number) {
  if (button === 0) {
    return VK_LBUTTON
  }
  if (button === 2) {
    return VK_RBUTTON
  }
  return null
}

export function sortUniqueButtonIds(buttonIds: Iterable<number>) {
  return Array.from(new Set(buttonIds)).sort((left, right) => left - right)
}

export function toWaypointLookPreview(dx: number, dy: number) {
  return {
    mouse_x: clamp(dx / LOOK_PREVIEW_SCALE_PX, -1, 1),
    mouse_y: clamp(dy / LOOK_PREVIEW_SCALE_PX, -1, 1),
  }
}

export function buildWaypointControlState(args: {
  buttonIds: readonly number[]
  mouseX: number
  mouseY: number
  scrollAmount: number
}): WaypointControlState {
  const activeButtons = new Set(args.buttonIds)
  const isActive = (control: WaypointHoldControl) =>
    ACTIVE_BUTTON_GROUPS[control].some((buttonId) => activeButtons.has(buttonId))

  return {
    forward: isActive("forward"),
    backward: isActive("backward"),
    left: isActive("left"),
    right: isActive("right"),
    jump: isActive("jump"),
    sprint: isActive("sprint"),
    crouch: isActive("crouch"),
    primary_fire: isActive("primary_fire"),
    secondary_fire: isActive("secondary_fire"),
    mouse_x: args.mouseX,
    mouse_y: args.mouseY,
    scroll_wheel: args.scrollAmount,
  }
}
