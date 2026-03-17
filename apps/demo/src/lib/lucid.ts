import {
  findInputByName,
  findUploadedFileInput,
  type ManifestInput,
} from "../../../../packages/lucid-ts/src/index.ts"

export * from "../../../../packages/lucid-ts/src/index.ts"

export function findInitialFrameInput(
  inputs: ManifestInput[] | readonly ManifestInput[],
) {
  const named = findInputByName(inputs, "set_initial_frame")
  if (named) {
    const upload = findUploadedFileInput([named])
    if (upload?.upload.kind === "image") {
      return upload
    }
  }

  const upload = findUploadedFileInput(inputs)
  if (upload?.upload.kind === "image") {
    return upload
  }
  return null
}
