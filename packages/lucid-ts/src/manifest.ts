import type {
  InputBinding,
  LucidManifest,
  ManifestInput,
  UploadFieldConfig,
  UploadInputSpec,
} from "./types.js"

export function manifestForModel(
  modelId: string | null,
  capabilitiesManifest: LucidManifest | null,
  staticManifests: Record<string, LucidManifest>,
) {
  if (capabilitiesManifest) {
    return capabilitiesManifest
  }
  if (!modelId) {
    return null
  }
  return staticManifests[modelId] ?? null
}

export function findInputByName(
  inputs: ManifestInput[] | readonly ManifestInput[],
  name: string,
) {
  return inputs.find((input) => input.name === name) ?? null
}

export function findInputsByBinding<TKind extends InputBinding["kind"]>(
  inputs: ManifestInput[] | readonly ManifestInput[],
  kind: TKind,
): Array<ManifestInput & { binding: Extract<InputBinding, { kind: TKind }> }> {
  return Array.from(inputs).filter(
    (input): input is ManifestInput & { binding: Extract<InputBinding, { kind: TKind }> } =>
      input.binding?.kind === kind,
  )
}

export function findPromptInput(inputs: ManifestInput[] | readonly ManifestInput[]) {
  const named = inputs.find((input) => input.name === "set_prompt" && !input.binding)
  if (named) {
    return named
  }

  return (
    inputs.find((input) => {
      if (input.binding) {
        return false
      }
      const properties = inputProperties(input)
      const promptSchema = properties.prompt
      return Object.keys(properties).length === 1 && promptSchema?.type === "string"
    }) ?? null
  )
}

export function findUploadedFileInput(inputs: ManifestInput[] | readonly ManifestInput[]) {
  return findUploadedFileInputs(inputs)[0] ?? null
}

export function findUploadedFileInputs(inputs: ManifestInput[] | readonly ManifestInput[]) {
  const uploaded: UploadInputSpec[] = []
  for (const input of inputs) {
    if (input.binding) {
      continue
    }
    const properties = inputProperties(input)
    const entries = Object.entries(properties)
    if (entries.length !== 1) {
      continue
    }
    const [argName, propertySchema] = entries[0]
    if (!propertySchema) {
      continue
    }
    if (!isStringWireField(propertySchema)) {
      continue
    }
    const upload = parseUploadField(propertySchema["x-lucid-upload"])
    if (!upload) {
      continue
    }
    uploaded.push({ name: input.name, argName, upload })
  }
  return uploaded
}

export function uploadAcceptAttribute(upload: UploadFieldConfig) {
  return upload.mime_types.join(",")
}

export function inputProperties(input: ManifestInput) {
  const schema = input.args_schema as {
    properties?: Record<string, Record<string, unknown> | undefined>
  }
  return schema.properties ?? {}
}

export function isStringWireField(schema: Record<string, unknown> | undefined) {
  if (!schema) {
    return false
  }
  const schemaType = schema.type
  if (schemaType === "string") {
    return true
  }
  return Array.isArray(schemaType) && schemaType.includes("string")
}

export function parseUploadField(value: unknown): UploadFieldConfig | null {
  if (!value || typeof value !== "object") {
    return null
  }
  const upload = value as Record<string, unknown>
  const kind = typeof upload.kind === "string" ? upload.kind : null
  const maxBytes =
    typeof upload.max_bytes === "number" ? upload.max_bytes : Number(upload.max_bytes)
  if (!kind || !Number.isFinite(maxBytes) || maxBytes <= 0) {
    return null
  }
  const rawMimeTypes = Array.isArray(upload.mime_types) ? upload.mime_types : []
  const mimeTypes = rawMimeTypes
    .map((item) => (typeof item === "string" ? item.trim() : ""))
    .filter(Boolean)
  if (!mimeTypes.length) {
    return null
  }
  const targetWidth =
    typeof upload.target_width === "number"
      ? upload.target_width
      : Number(upload.target_width)
  const targetHeight =
    typeof upload.target_height === "number"
      ? upload.target_height
      : Number(upload.target_height)
  const parsed: UploadFieldConfig = {
    kind,
    mime_types: mimeTypes,
    max_bytes: Math.trunc(maxBytes),
  }
  if (Number.isFinite(targetWidth) && targetWidth > 0) {
    parsed.target_width = Math.trunc(targetWidth)
  }
  if (Number.isFinite(targetHeight) && targetHeight > 0) {
    parsed.target_height = Math.trunc(targetHeight)
  }
  return parsed
}
