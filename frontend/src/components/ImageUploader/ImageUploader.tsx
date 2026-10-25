import { useRef } from 'react'
import styles from './ImageUploader.module.css'

interface ImageUploaderProps {
  onFilesSelected: (files: File[]) => void
  error: string | null
}

export function ImageUploader({ onFilesSelected, error }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  function handleChange() {
    const input = inputRef.current
    if (!input?.files) return

    onFilesSelected(Array.from(input.files))
    // so the same file can be picked again
    input.value = ''
  }

  return (
    <div className={styles.field}>
      <span className={styles.label}>Upload photos of the plant</span>

      <label className={styles.dropArea} htmlFor="images">
        <span className={styles.icon} aria-hidden="true">
          +
        </span>
        <span className={styles.mainText}>Choose photos</span>
        <span className={styles.hintText}>
          Up to 5 photos - flower, leaf, whole plant. JPEG, JPG or PNG, up to 10 MB each.
        </span>
        <input
          id="images"
          ref={inputRef}
          className={styles.input}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          multiple
          onChange={handleChange}
        />
      </label>

      {error !== null && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
    </div>
  )
}
