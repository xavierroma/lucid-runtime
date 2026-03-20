import { useMemo, useState, type DragEvent, type ChangeEvent, type FormEvent } from "react"
import { ImageIcon, Plus, Trash2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import type { SavedEnvironment } from "@/lib/environments"
import { fileToDataUrl } from "@/lib/input-files"

export interface SaveEnvironmentInput {
  environmentId?: string | null
  name: string
  prompt: string
  seedImageDataUrl: string
}

interface EnvironmentStudioProps {
  environments: SavedEnvironment[]
  initialEditingId?: string | null
  onSaveEnvironment: (input: SaveEnvironmentInput) => SavedEnvironment
  onDeleteEnvironment: (environmentId: string) => void
}

export function EnvironmentStudio({
  environments,
  initialEditingId = null,
  onSaveEnvironment,
  onDeleteEnvironment,
}: EnvironmentStudioProps) {
  const initEnv = useMemo(
    () => environments.find((e) => e.id === initialEditingId) ?? null,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

  const [editingId, setEditingId] = useState<string | null>(initEnv?.id ?? null)
  const [draftName, setDraftName] = useState(initEnv?.name ?? "")
  const [draftPrompt, setDraftPrompt] = useState(initEnv?.prompt ?? "")
  const [draftSeedImageDataUrl, setDraftSeedImageDataUrl] = useState<string | null>(
    initEnv?.seedImageDataUrl ?? null,
  )
  const [draftError, setDraftError] = useState<string | null>(null)
  const [fileInputKey, setFileInputKey] = useState(0)
  const [isDragging, setIsDragging] = useState(false)

  const editingEnvironment = useMemo(
    () => environments.find((e) => e.id === editingId) ?? null,
    [editingId, environments],
  )

  const resetDraft = () => {
    setEditingId(null)
    setDraftName("")
    setDraftPrompt("")
    setDraftSeedImageDataUrl(null)
    setDraftError(null)
    setFileInputKey((k) => k + 1)
  }

  const applyFile = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setDraftError("Please upload an image file.")
      return
    }
    try {
      const dataUrl = await fileToDataUrl(file)
      setDraftSeedImageDataUrl(dataUrl)
      setDraftError(null)
    } catch (error) {
      setDraftError(error instanceof Error ? error.message : "Failed to read image.")
    }
  }

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0] ?? null
    event.currentTarget.value = ""
    if (file) void applyFile(file)
  }

  const handleDragOver = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: DragEvent<HTMLLabelElement>) => {
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragging(false)
    }
  }

  const handleDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) void applyFile(file)
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    const name = draftName.trim()
    const prompt = draftPrompt.trim()

    if (!name) { setDraftError("Add a title."); return }
    if (!prompt) { setDraftError("Add a prompt."); return }
    if (!draftSeedImageDataUrl) { setDraftError("Add a first frame."); return }

    const saved = onSaveEnvironment({ environmentId: editingId, name, prompt, seedImageDataUrl: draftSeedImageDataUrl })
    setEditingId(saved.id)
    setDraftName(saved.name)
    setDraftPrompt(saved.prompt)
    setDraftSeedImageDataUrl(saved.seedImageDataUrl)
    setDraftError(null)
  }

  const handleDelete = () => {
    if (!editingEnvironment) return
    onDeleteEnvironment(editingEnvironment.id)
    resetDraft()
  }

  return (
    <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="env-title">Title</Label>
        <Input
          id="env-title"
          value={draftName}
          onChange={(e) => setDraftName(e.target.value)}
          placeholder="Low orbit dunes"
          maxLength={80}
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="env-prompt">Prompt</Label>
        <Textarea
          id="env-prompt"
          value={draftPrompt}
          onChange={(e) => setDraftPrompt(e.target.value)}
          placeholder="Windswept dunes at sunrise."
          rows={5}
          spellCheck={false}
          className="resize-none"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label>First frame</Label>
        <label
          className={cn(
            "relative flex min-h-36 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-lg border-2 border-dashed transition-colors",
            isDragging
              ? "border-ring bg-muted/50"
              : "border-border hover:border-ring/60 hover:bg-muted/30",
            draftSeedImageDataUrl && !isDragging && "border-solid border-border/60",
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            key={fileInputKey}
            type="file"
            accept="image/*"
            className="sr-only"
            onChange={handleFileChange}
          />
          {draftSeedImageDataUrl ? (
            <>
              <img
                src={draftSeedImageDataUrl}
                alt="First frame preview"
                className="absolute inset-0 h-full w-full object-cover"
              />
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 transition-opacity hover:opacity-100">
                <span className="text-sm font-medium text-white">Change image</span>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center gap-2 px-4 py-6 text-muted-foreground">
              <ImageIcon className="size-7" />
              <div className="text-center">
                <p className="text-sm font-medium">Drop an image here</p>
                <p className="text-xs text-muted-foreground/70">or click to browse</p>
              </div>
            </div>
          )}
        </label>
      </div>

      {draftError ? (
        <p className="text-sm font-medium text-destructive" role="alert">
          {draftError}
        </p>
      ) : null}

      <div className="flex flex-wrap gap-2">
        <Button type="submit">
          {editingEnvironment ? "Save" : "Create"}
        </Button>
        {editingEnvironment ? (
          <>
            <Button type="button" variant="outline" onClick={resetDraft}>
              <Plus className="size-4" />
              New
            </Button>
            <Button type="button" variant="destructive" onClick={handleDelete}>
              <Trash2 className="size-4" />
              Delete
            </Button>
          </>
        ) : null}
      </div>
    </form>
  )
}
