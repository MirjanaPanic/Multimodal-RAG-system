export type ResultStatus =
  | 'idle' // nothing searched yet
  | 'loading' // search in progress
  | 'confident' // one species identified
  | 'ambiguous' // list of candidates
  | 'none' // no match found
  | 'error' // request failed

export interface UploadedImage {
  id: string
  file: File
  previewUrl: string
}

export type BranchOutcome = 'no_result' | 'abstain' | 'siguran' | 'nesiguran'

// field names as returned by server.py
export interface SpeciesResult {
  specie_id: string
  naziv: string // scientific name
  opis: string // description written from the knowledge base
  izvor: string // source url
  slike: string[] // reference photos
}

// POST /identify
export interface IdentificationResult {
  tip: 'siguran' | 'nesiguran' | 'no_result'
  uvod: string // intro sentence
  vrste: SpeciesResult[]
  grane: { slika: BranchOutcome; tekst: BranchOutcome }
}
