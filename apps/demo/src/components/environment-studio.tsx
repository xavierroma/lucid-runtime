import { useMemo, useState, type ChangeEvent, type FormEvent } from "react"
import { ArrowLeft, Plus, Trash2 } from "lucide-react"

import { Card, CardHeader, CardTitle } from "@/components/ui/card"
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
  selectedEnvironmentId: string | null
  onSelectEnvironment: (environmentId: string) => void
  onSaveEnvironment: (input: SaveEnvironmentInput) => SavedEnvironment
  onDeleteEnvironment: (environmentId: string) => void
  onNavigateHome: () => void
}

function environmentCardStyle(seedImageDataUrl: string | null | undefined) {
  if (!seedImageDataUrl) {
    return undefined
  }

  return {
    backgroundImage: `url(${seedImageDataUrl})`,
  }
}

export function EnvironmentStudio({
  environments,
  selectedEnvironmentId,
  onSelectEnvironment,
  onSaveEnvironment,
  onDeleteEnvironment,
  onNavigateHome,
}: EnvironmentStudioProps) {
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draftName, setDraftName] = useState("")
  const [draftPrompt, setDraftPrompt] = useState("")
  const [draftSeedImageDataUrl, setDraftSeedImageDataUrl] = useState<string | null>(null)
  const [draftError, setDraftError] = useState<string | null>(null)
  const [fileInputKey, setFileInputKey] = useState(0)

  const editingEnvironment = useMemo(
    () => environments.find((environment) => environment.id === editingId) ?? null,
    [editingId, environments],
  )

  const resetDraft = () => {
    setEditingId(null)
    setDraftName("")
    setDraftPrompt("")
    setDraftSeedImageDataUrl(null)
    setDraftError(null)
    setFileInputKey((current) => current + 1)
  }

  const startEditing = (environment: SavedEnvironment) => {
    setEditingId(environment.id)
    setDraftName(environment.name)
    setDraftPrompt(environment.prompt)
    setDraftSeedImageDataUrl(environment.seedImageDataUrl)
    setDraftError(null)
    setFileInputKey((current) => current + 1)
  }

  const handleImageChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0] ?? null
    event.currentTarget.value = ""

    if (!file) {
      return
    }

    try {
      const dataUrl = await fileToDataUrl(file)
      setDraftSeedImageDataUrl(dataUrl)
      setDraftError(null)
    } catch (error) {
      setDraftError(
        error instanceof Error ? error.message : "failed to read image",
      )
    }
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    const name = draftName.trim()
    const prompt = draftPrompt.trim()

    if (!name) {
      setDraftError("Add a title.")
      return
    }

    if (!prompt) {
      setDraftError("Add a prompt.")
      return
    }

    if (!draftSeedImageDataUrl) {
      setDraftError("Add a first frame.")
      return
    }

    const saved = onSaveEnvironment({
      environmentId: editingId,
      name,
      prompt,
      seedImageDataUrl: draftSeedImageDataUrl,
    })
    setEditingId(saved.id)
    setDraftName(saved.name)
    setDraftPrompt(saved.prompt)
    setDraftSeedImageDataUrl(saved.seedImageDataUrl)
    setDraftError(null)
  }

  const handleDelete = () => {
    if (!editingEnvironment) {
      return
    }
    onDeleteEnvironment(editingEnvironment.id)
    resetDraft()
  }

  return (
    <main className="environment-stage">
      <section className="environment-shell environment-shell-simple">
        <div className="environment-toolbar">
          <button
            type="button"
            className="environment-nav-button"
            onClick={onNavigateHome}
          >
            <ArrowLeft className="size-4" />
            Back
          </button>
          {editingEnvironment ? (
            <button
              type="button"
              className="environment-secondary-button"
              onClick={resetDraft}
            >
              <Plus className="size-4" />
              New
            </button>
          ) : null}
        </div>

        <form className="environment-editor-simple" onSubmit={handleSubmit}>
          <label className="environment-field">
            <span>Title</span>
            <input
              value={draftName}
              onChange={(event) => setDraftName(event.target.value)}
              placeholder="Low orbit dunes"
              maxLength={80}
            />
          </label>

          <label className="environment-field">
            <span>Prompt</span>
            <textarea
              value={draftPrompt}
              onChange={(event) => setDraftPrompt(event.target.value)}
              placeholder="Windswept dunes at sunrise."
              rows={6}
              spellCheck={false}
            />
          </label>

          <div className="environment-field">
            <span>First frame</span>
            <label className="environment-upload-label">
              <input
                key={fileInputKey}
                type="file"
                accept="image/*"
                className="sr-only"
                onChange={(event) => {
                  void handleImageChange(event)
                }}
              />
              <Card className="environment-card-surface environment-card-preview" size="sm">
                <div
                  className="environment-card-media"
                  style={environmentCardStyle(draftSeedImageDataUrl)}
                />
                <div className="environment-card-scrim" />
                <CardHeader className="environment-card-content">
                  <CardTitle className="environment-card-title">
                    {draftName.trim() || "Upload image"}
                  </CardTitle>
                </CardHeader>
              </Card>
            </label>
          </div>

          {draftError ? (
            <p className="environment-form-error" role="alert">
              {draftError}
            </p>
          ) : null}

          <div className="environment-form-actions">
            <button type="submit" className="environment-primary-button">
              {editingEnvironment ? "Save" : "Create"}
            </button>
            {editingEnvironment ? (
              <button
                type="button"
                className="environment-danger-button"
                onClick={handleDelete}
              >
                <Trash2 className="size-4" />
                Delete
              </button>
            ) : null}
          </div>
        </form>

        {environments.length ? (
          <section className="environment-card-grid" aria-label="Saved environments">
            {environments.map((environment) => {
              const isSelected = environment.id === selectedEnvironmentId
              return (
                <button
                  key={environment.id}
                  type="button"
                  className={`environment-card-button ${
                    isSelected ? "environment-card-button-selected" : ""
                  }`}
                  onClick={() => {
                    onSelectEnvironment(environment.id)
                    startEditing(environment)
                  }}
                >
                  <Card className="environment-card-surface" size="sm">
                    <div
                      className="environment-card-media"
                      style={environmentCardStyle(environment.seedImageDataUrl)}
                    />
                    <div className="environment-card-scrim" />
                    <CardHeader className="environment-card-content">
                      <CardTitle className="environment-card-title">
                        {environment.name}
                      </CardTitle>
                    </CardHeader>
                  </Card>
                </button>
              )
            })}
          </section>
        ) : (
          <div className="environment-empty-state">No environments</div>
        )}
      </section>
    </main>
  )
}
