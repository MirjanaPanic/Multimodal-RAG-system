import styles from './SearchButton.module.css'

interface SearchButtonProps {
  disabled: boolean
  loading: boolean
  onClick: () => void
}

export function SearchButton({ disabled, loading, onClick }: SearchButtonProps) {
  return (
    <button
      type="button"
      className={styles.button}
      disabled={disabled || loading}
      onClick={onClick}
    >
      {loading ? 'Searching...' : 'Identify'}
    </button>
  )
}
