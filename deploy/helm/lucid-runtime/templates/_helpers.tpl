{{- define "lucid-runtime.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "lucid-runtime.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "lucid-runtime.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "lucid-runtime.coordinator.fullname" -}}
{{- printf "%s-coordinator" (include "lucid-runtime.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "lucid-runtime.worker.fullname" -}}
{{- printf "%s-worker" (include "lucid-runtime.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "lucid-runtime.labels" -}}
app.kubernetes.io/name: {{ include "lucid-runtime.name" . }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
