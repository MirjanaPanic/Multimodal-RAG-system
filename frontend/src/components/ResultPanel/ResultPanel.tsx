import { API_URL } from '../../api/identify'
import type {
  BranchOutcome,
  IdentificationResult,
  ResultStatus,
  SpeciesResult,
} from '../../types/identification'
import styles from './ResultPanel.module.css'

const TITLES: Record<ResultStatus, string> = {
  idle: 'No search yet',
  loading: 'Searching...',
  confident: 'Plant identified',
  ambiguous: 'Not certain - possible species',
  none: 'No result',
  error: 'Something went wrong',
}

const BRANCH_LABELS: Record<BranchOutcome, string> = {
  no_result: 'not provided',
  abstain: 'no answer',
  siguran: 'confident',
  nesiguran: 'not certain',
}

interface ResultPanelProps {
  status: ResultStatus
  result: IdentificationResult | null
  errorMessage: string | null
}

export function ResultPanel({ status, result, errorMessage }: ResultPanelProps) {
  const hasAnswer = status === 'confident' || status === 'ambiguous' || status === 'none'

  return (
    <section className={`${styles.panel} ${styles[status]}`} aria-live="polite">
      <h2 className={styles.title}>{TITLES[status]}</h2>

      {status === 'idle' && (
        <p className={styles.body}>Describe the plant or upload a photo, then press Identify.</p>
      )}
      {status === 'loading' && (
        <p className={styles.body}>Comparing your query against the species collection.</p>
      )}
      {status === 'error' && (
        <p className={styles.body}>
          {errorMessage ?? 'The search could not be completed. Please try again.'}
        </p>
      )}

      {hasAnswer && result !== null && (
        <>
          <p className={styles.body}>{result.uvod}</p>

          {result.vrste.length > 0 && (
            <ol className={styles.speciesList}>
              {result.vrste.map((species, index) => (
                <SpeciesCard
                  key={species.specie_id}
                  species={species}
                  position={result.vrste.length > 1 ? index + 1 : null}
                />
              ))}
            </ol>
          )}

          <p className={styles.details}>
            Photo: {BRANCH_LABELS[result.grane.slika]} · Description:{' '}
            {BRANCH_LABELS[result.grane.tekst]}          </p>
        </>
      )}
    </section>
  )
}

interface SpeciesCardProps {
  species: SpeciesResult
  position: number | null
}

function SpeciesCard({ species, position }: SpeciesCardProps) {
  return (
    <li className={styles.species}>
      <h3 className={styles.speciesName}>
        {position !== null && <span className={styles.position}>{position}.</span>}
        <i>{species.naziv}</i>
      </h3>
      <p className={styles.speciesText}>{species.opis}</p>

      <div className={styles.photos}>
        {species.slike.map((path) => (
          <img
            key={path}
            className={styles.photo}
            src={`${API_URL}${path}`}
            alt={`${species.naziv}, photo from the collection`}
            loading="lazy"
          />
        ))}
      </div>

      {species.izvor.startsWith('http') && (
        <a className={styles.source} href={species.izvor} target="_blank" rel="noreferrer">
          Source of the description
        </a>
      )}
    </li>
  )
}
