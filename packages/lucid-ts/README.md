# lucid-ts

Slim TypeScript SDK for Lucid clients.

It covers the client-side Lucid surface without taking a runtime dependency on
`livekit-client` or any other transport package:

- Coordinator HTTP client
- Lucid manifest and capability types
- Control, action, and status wire codecs
- Manifest helpers for prompt and uploaded-file inputs
- A small session client that works with any transport adapter
- A structural LiveKit adapter for `localParticipant`

## Example

```ts
import {
  CoordinatorClient,
  LucidSessionClient,
  createLiveKitTransport,
} from "lucid-ts"

const coordinator = new CoordinatorClient({
  baseUrl: "https://coordinator.example.com",
  apiKey: process.env.LUCID_API_KEY,
})

const sessionResponse = await coordinator.createSession("yume")

const lucid = new LucidSessionClient({
  sessionId: sessionResponse.session.session_id,
  capabilities: sessionResponse.capabilities,
  transport: createLiveKitTransport(room.localParticipant),
})

await lucid.sendInput("set_prompt", { prompt: "sunlit canyon" })
await lucid.resume()
```

For uploaded file inputs, call `lucid.handleStatusMessage(...)` from your status
channel listener and use `lucid.sendUploadedInput(...)`.
