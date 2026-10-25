import styles from './DescriptionInput.module.css'

const PLACEHOLDER =
  'e.g. yellow flower with five petals, hairy stem, leaves divided into narrow segments, growing in a dry meadow'

interface DescriptionInputProps {
  value: string
  maxLength: number
  onChange: (value: string) => void
}

export function DescriptionInput({ value, maxLength, onChange }: DescriptionInputProps) {
  const length = value.trim().length
  const tooLong = length > maxLength

  return (
    <div className={styles.field}>
      <label className={styles.label} htmlFor="description">
        Describe the plant
      </label>
      <textarea
        id="description"
        className={tooLong ? `${styles.textarea} ${styles.textareaTooLong}` : styles.textarea}
        rows={6}
        value={value}
        placeholder={PLACEHOLDER}
        aria-describedby="description-info"
        onChange={(event) => onChange(event.target.value)}
      />
      <div className={styles.footer}>
        <span id="description-info" className={styles.info} role="status">
          {tooLong
            ? `The description is too long. Shorten it to ${maxLength} characters (about 5 sentences).`
            : ''}
        </span>
        <span className={tooLong ? `${styles.counter} ${styles.counterTooLong}` : styles.counter}>
          {length} / {maxLength}
        </span>
      </div>
    </div>
  )
}
