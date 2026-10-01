import { useCallback, useId, useRef, useState, type DragEvent, type ChangeEvent } from 'react'

const ACCEPTED_TYPES = new Set([
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/webp',
])

const ACCEPTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']

interface ImageUploaderProps {
  onImageSelected: (file: File) => void
  disabled?: boolean
}

function isAcceptedImage(file: File): boolean {
  if (ACCEPTED_TYPES.has(file.type)) return true
  const name = file.name.toLowerCase()
  return ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext))
}

export function ImageUploader({ onImageSelected, disabled = false }: ImageUploaderProps) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFile = useCallback(
    (file: File | undefined | null) => {
      if (!file) return

      if (!isAcceptedImage(file)) {
        setError('Unsupported file type. Please upload a JPG, JPEG, PNG, or WEBP image.')
        return
      }

      setError(null)
      onImageSelected(file)
    },
    [onImageSelected],
  )

  const onChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    handleFile(file)
    event.target.value = ''
  }

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    setIsDragging(false)
    if (disabled) return
    handleFile(event.dataTransfer.files?.[0])
  }

  const onDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    if (!disabled) setIsDragging(true)
  }

  const onDragLeave = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    setIsDragging(false)
  }

  return (
    <div className="uploader">
      <label
        htmlFor={inputId}
        className={`uploader__dropzone${isDragging ? ' uploader__dropzone--active' : ''}${disabled ? ' uploader__dropzone--disabled' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
      >
        <div className="uploader__icon" aria-hidden="true">
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
            <rect x="4" y="8" width="32" height="24" rx="3" stroke="currentColor" strokeWidth="1.75" />
            <circle cx="14" cy="16" r="2.5" fill="currentColor" />
            <path
              d="M4 26l8.5-7.5L18 23l6-6.5L36 26"
              stroke="currentColor"
              strokeWidth="1.75"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <p className="uploader__title">Drop an aerial image here</p>
        <p className="uploader__or">or</p>
        <span className="uploader__browse">Browse files</span>
        <p className="uploader__hint">Supported: JPG, JPEG, PNG, WEBP</p>
      </label>

      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept="image/jpeg,image/jpg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
        className="visually-hidden"
        onChange={onChange}
        disabled={disabled}
      />

      {error && (
        <p className="uploader__error" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}
