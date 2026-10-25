import { useEffect, useRef, useState } from 'react'
import {
  ALLOWED_IMAGE_TYPES,
  MAX_DESCRIPTION_LENGTH,
  MAX_FILE_SIZE,
  MAX_IMAGES,
  MIN_IMAGE_SIDE,
  identifyPlant,
} from '../../api/identify'
import { DescriptionInput } from '../../components/DescriptionInput/DescriptionInput'
import { ImagePreviewList } from '../../components/ImagePreviewList/ImagePreviewList'
import { ImageUploader } from '../../components/ImageUploader/ImageUploader'
import { ResultPanel } from '../../components/ResultPanel/ResultPanel'
import { SearchButton } from '../../components/SearchButton/SearchButton'
import type {
  IdentificationResult,
  ResultStatus,
  UploadedImage,
} from '../../types/identification'
import styles from './PlantIdentificationPage.module.css'

function isSameFile(a: File, b: File): boolean {
  return a.name === b.name && a.size === b.size && a.lastModified === b.lastModified
}

async function shorterSide(file: File): Promise<number | null> {
  try {
    const bitmap = await createImageBitmap(file)
    const side = Math.min(bitmap.width, bitmap.height)
    bitmap.close()
    return side
  } catch {
    return null
  }
}

function statusOf(result: IdentificationResult): ResultStatus {
  if (result.tip === 'siguran') return 'confident'
  if (result.tip === 'nesiguran') return 'ambiguous'
  return 'none'
}

export function PlantIdentificationPage() {
  const [description, setDescription] = useState('')
  const [images, setImages] = useState<UploadedImage[]>([])
  const [status, setStatus] = useState<ResultStatus>('idle')
  const [result, setResult] = useState<IdentificationResult | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [checkingPhotos, setCheckingPhotos] = useState(false)

  const trimmedDescription = description.trim()
  const descriptionTooLong = trimmedDescription.length > MAX_DESCRIPTION_LENGTH

  // wait until picked photos are checked
  const canSearch =
    (trimmedDescription.length > 0 || images.length > 0) &&
    !descriptionTooLong &&
    !checkingPhotos

  // revoke object URLs on unmount (ref so StrictMode doesn't revoke them early)
  const imagesRef = useRef(images)

  useEffect(() => {
    imagesRef.current = images
  }, [images])

  useEffect(() => {
    return () => {
      for (const image of imagesRef.current) {
        URL.revokeObjectURL(image.previewUrl)
      }
    }
  }, [])

  // attach only photos the backend will accept
  async function addFiles(files: File[]) {
    const rejected: string[] = []
    const accepted: UploadedImage[] = []
    const current = imagesRef.current
    setCheckingPhotos(true)

    for (const file of files) {
      if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
        rejected.push(`${file.name} has an unsupported format - use JPEG, PNG or WebP`)
        continue
      }
      if (file.size > MAX_FILE_SIZE) {
        rejected.push(`${file.name} is larger than 10 MB`)
        continue
      }
      const alreadyAdded =
        current.some((image) => isSameFile(image.file, file)) ||
        accepted.some((image) => isSameFile(image.file, file))
      if (alreadyAdded) continue

      if (current.length + accepted.length >= MAX_IMAGES) {
        rejected.push(`${file.name} was not added - at most ${MAX_IMAGES} photos are allowed`)
        continue
      }

      const side = await shorterSide(file)
      if (side === null) {
        rejected.push(`${file.name} could not be opened as an image`)
        continue
      }
      if (side < MIN_IMAGE_SIDE) {
        rejected.push(
          `${file.name} is too small - the shorter side must be at least ${MIN_IMAGE_SIDE} px`,
        )
        continue
      }

      accepted.push({
        id: crypto.randomUUID(),
        file,
        previewUrl: URL.createObjectURL(file),
      })
    }

    if (accepted.length > 0) {
      setImages((now) => [...now, ...accepted])
    }
    setUploadError(rejected.length > 0 ? rejected.join('; ') : null)
    setCheckingPhotos(false)
  }

  function removeImage(id: string) {
    setImages((current) => {
      const removed = current.find((image) => image.id === id)
      if (removed) URL.revokeObjectURL(removed.previewUrl)
      return current.filter((image) => image.id !== id)
    })
    setUploadError(null)
  }

  async function handleSearch() {
    setStatus('loading')
    try {
      const answer = await identifyPlant(trimmedDescription, images)
      setResult(answer)
      setStatus(statusOf(answer))
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : null)
      setStatus('error')
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.heading}>Plant Identification</h1>
        <p className={styles.subheading}>
          Describe the plant, upload photos of it, or do both
        </p>
      </header>

      <div className={styles.layout}>
        <section className={styles.formColumn}>
          <DescriptionInput
            value={description}
            maxLength={MAX_DESCRIPTION_LENGTH}
            onChange={setDescription}
          />
          <ImageUploader onFilesSelected={addFiles} error={uploadError} />
          <ImagePreviewList images={images} onRemove={removeImage} />
          <SearchButton
            disabled={!canSearch}
            loading={status === 'loading'}
            onClick={handleSearch}
          />
        </section>

        <aside className={styles.resultColumn}>
          <ResultPanel status={status} result={result} errorMessage={errorMessage} />
        </aside>
      </div>
    </main>
  )
}
