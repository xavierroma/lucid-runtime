export function isTerminalSessionState(state) {
    return state === "ENDED" || state === "FAILED";
}
