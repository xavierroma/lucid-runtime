function fileExtensionFromMimeType(mimeType: string) {
  const normalized = mimeType.split(";")[0]?.trim().toLowerCase() ?? ""
  if (!normalized.includes("/")) {
    return "bin"
  }
  return normalized.split("/")[1]?.split("+")[0] || "bin"
}

function sanitizeBaseName(value: string) {
  const sanitized = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
  return sanitized || "environment"
}

export function dataUrlToFile(
  dataUrl: string,
  baseName: string,
  lastModified = 0,
) {
  const [header, payload] = dataUrl.split(",", 2)
  if (!header || payload === undefined || !header.startsWith("data:")) {
    return null
  }

  const mimeType = header.slice(5).split(";")[0] || "application/octet-stream"
  const fileName = `${sanitizeBaseName(baseName)}.${fileExtensionFromMimeType(mimeType)}`
  const isBase64 = header.includes(";base64")

  if (isBase64) {
    const binary = atob(payload)
    const bytes = new Uint8Array(binary.length)
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index)
    }
    return new File([bytes], fileName, {
      type: mimeType,
      lastModified,
    })
  }

  const decoded = decodeURIComponent(payload)
  return new File([decoded], fileName, {
    type: mimeType,
    lastModified,
  })
}

export function fileToDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => {
      reject(new Error("failed to read file"))
    }
    reader.onload = () => {
      const result = reader.result
      if (typeof result !== "string") {
        reject(new Error("failed to read file"))
        return
      }
      resolve(result)
    }
    reader.readAsDataURL(file)
  })
}
