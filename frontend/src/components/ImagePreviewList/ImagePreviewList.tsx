import type { UploadedImage } from '../../types/identification'
import styles from './ImagePreviewList.module.css'

interface ImagePreviewListProps {
  images: UploadedImage[]
  onRemove: (id: string) => void
}

function formatSize(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function ImagePreviewList({ images, onRemove }: ImagePreviewListProps) {
  if (images.length === 0) return null

  return (
    <div className={styles.wrapper}>
      <span className={styles.count}>
        {images.length} {images.length === 1 ? 'photo' : 'photos'} attached
      </span>

      <ul className={styles.grid}>
        {images.map((image) => (
          <li key={image.id} className={styles.item}>
            <img className={styles.thumb} src={image.previewUrl} alt={image.file.name} />
            <span className={styles.name} title={image.file.name}>
              {image.file.name}
            </span>
            <span className={styles.size}>{formatSize(image.file.size)}</span>
            <button
              type="button"
              className={styles.remove}
              onClick={() => onRemove(image.id)}
              aria-label={`Remove ${image.file.name}`}
            >
              &times;
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
