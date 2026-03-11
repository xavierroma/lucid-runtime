import { useMemo, useState, type FormEvent } from "react"
import { ArrowLeft, Check, PencilLine, Plus, Trash2 } from "lucide-react"

import type { SavedEnvironment } from "@/lib/environments"

export interface SaveEnvironmentInput {
  environmentId?: string | null
  name: string
  prompt: string
}

interface EnvironmentStudioProps {
  environments: SavedEnvironment[]
  selectedEnvironmentId: string | null
  onSelectEnvironment: (environmentId: string) => void
  onSaveEnvironment: (input: SaveEnvironmentInput) => SavedEnvironment
  onDeleteEnvironment: (environmentId: string) => void
  onNavigateHome: () => void
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
  const [draftError, setDraftError] = useState<string | null>(null)

  const editingEnvironment = useMemo(
    () => environments.find((environment) => environment.id === editingId) ?? null,
    [editingId, environments],
  )

  const resetDraft = () => {
    setEditingId(null)
    setDraftName("")
    setDraftPrompt("")
    setDraftError(null)
  }

  const startEditing = (environment: SavedEnvironment) => {
    setEditingId(environment.id)
    setDraftName(environment.name)
    setDraftPrompt(environment.prompt)
    setDraftError(null)
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    const name = draftName.trim()
    const prompt = draftPrompt.trim()

    if (!name) {
      setDraftError("Name the environment before saving it.")
      return
    }

    if (!prompt) {
      setDraftError("Add a world prompt before saving it.")
      return
    }

    const saved = onSaveEnvironment({
      environmentId: editingId,
      name,
      prompt,
    })
    setEditingId(saved.id)
    setDraftName(saved.name)
    setDraftPrompt(saved.prompt)
    setDraftError(null)
  }

  const handleDelete = (environmentId: string) => {
    if (editingId === environmentId) {
      resetDraft()
    }
    onDeleteEnvironment(environmentId)
  }

  return (
    <main className="environment-stage">
      <section className="environment-shell">
        <header className="environment-header">
          <button
            type="button"
            className="environment-nav-button"
            onClick={onNavigateHome}
          >
            <ArrowLeft className="size-4" />
            Back to Console
          </button>
          <div className="environment-header-copy">
            <p className="environment-kicker">Environment Studio</p>
            <h1>Author reusable world starts.</h1>
            <p>
              Save text-conditioned environments here, then launch Lucid from the
              console with one selected.
            </p>
          </div>
        </header>

        <div className="environment-grid">
          <section className="environment-editor">
            <div className="environment-panel-heading">
              <div>
                <p className="environment-panel-kicker">
                  {editingEnvironment ? "Editing world" : "New world"}
                </p>
                <h2>{editingEnvironment ? editingEnvironment.name : "Create environment"}</h2>
              </div>
              <button
                type="button"
                className="environment-secondary-button"
                onClick={resetDraft}
              >
                <Plus className="size-4" />
                New
              </button>
            </div>

            <form className="environment-form" onSubmit={handleSubmit}>
              <label className="environment-field">
                <span>Name</span>
                <input
                  value={draftName}
                  onChange={(event) => setDraftName(event.target.value)}
                  placeholder="Low orbit dunes"
                  maxLength={80}
                />
              </label>

              <label className="environment-field">
                <span>World prompt</span>
                <textarea
                  value={draftPrompt}
                  onChange={(event) => setDraftPrompt(event.target.value)}
                  placeholder="Windswept dunes at sunrise, abandoned relay towers on the ridge, dust drifting across the valley floor."
                  rows={8}
                  spellCheck={false}
                />
              </label>

              {draftError ? (
                <p className="environment-form-error" role="alert">
                  {draftError}
                </p>
              ) : null}

              <div className="environment-form-actions">
                <button type="submit" className="environment-primary-button">
                  <Check className="size-4" />
                  {editingEnvironment ? "Save Changes" : "Save Environment"}
                </button>
                {editingEnvironment ? (
                  <button
                    type="button"
                    className="environment-secondary-button"
                    onClick={() => {
                      onSelectEnvironment(editingEnvironment.id)
                      onNavigateHome()
                    }}
                  >
                    Use on Console
                  </button>
                ) : null}
              </div>
            </form>
          </section>

          <section className="environment-library">
            <div className="environment-panel-heading">
              <div>
                <p className="environment-panel-kicker">Saved worlds</p>
                <h2>{environments.length} stored</h2>
              </div>
            </div>

            {environments.length ? (
              <div className="environment-card-list">
                {environments.map((environment) => {
                  const isSelected = environment.id === selectedEnvironmentId
                  const isEditing = environment.id === editingId
                  return (
                    <article
                      key={environment.id}
                      className={`environment-card ${isSelected ? "environment-card-selected" : ""}`}
                    >
                      <div className="environment-card-header">
                        <div>
                          <h3>{environment.name}</h3>
                          <p>
                            {isSelected ? "Selected on console" : "Saved environment"}
                            {isEditing ? " · editing" : ""}
                          </p>
                        </div>
                        {isSelected ? (
                          <span className="environment-card-badge">Active</span>
                        ) : null}
                      </div>

                      <p className="environment-card-prompt">{environment.prompt}</p>

                      <div className="environment-card-actions">
                        <button
                          type="button"
                          className="environment-primary-button"
                          onClick={() => {
                            onSelectEnvironment(environment.id)
                            onNavigateHome()
                          }}
                        >
                          Use on Console
                        </button>
                        <button
                          type="button"
                          className="environment-secondary-button"
                          onClick={() => startEditing(environment)}
                        >
                          <PencilLine className="size-4" />
                          Edit
                        </button>
                        <button
                          type="button"
                          className="environment-danger-button"
                          onClick={() => handleDelete(environment.id)}
                        >
                          <Trash2 className="size-4" />
                          Delete
                        </button>
                      </div>
                    </article>
                  )
                })}
              </div>
            ) : (
              <div className="environment-empty-state">
                <p className="environment-panel-kicker">No worlds yet</p>
                <h2>Save your first environment.</h2>
                <p>
                  Environments persist in this browser so you can reuse them across
                  sessions.
                </p>
              </div>
            )}
          </section>
        </div>
      </section>
    </main>
  )
}
