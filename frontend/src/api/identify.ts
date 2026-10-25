import type { IdentificationResult, UploadedImage } from '../types/identification'

export const API_URL = 'http://127.0.0.1:8000'

// same limits as in server.py
export const MAX_IMAGES = 5
export const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10 MB
export const MIN_IMAGE_SIDE = 224 // px
export const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp']
export const MAX_DESCRIPTION_LENGTH = 500

export async function identifyPlant(
  description: string,
  images: UploadedImage[],
): Promise<IdentificationResult> {
  const form = new FormData()
  form.append('opis', description)
  for (const image of images) {
    form.append('slike', image.file)
  }

  let response: Response
  try {
    response = await fetch(`${API_URL}/identify`, { method: 'POST', body: form })
  } catch {
    throw new Error('The server cannot be reached. Is the backend running?')
  }

  if (!response.ok) {
    // { detail: message }
    const body: unknown = await response.json().catch(() => null)
    const detail = (body as { detail?: unknown } | null)?.detail
    throw new Error(
      typeof detail === 'string' ? detail : 'The search could not be completed. Please try again.',
    )
  }

  return (await response.json()) as IdentificationResult
}
