import { useEffect } from 'react';

export default function useKeyboardShortcuts(handlers) {
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Ctrl/Cmd + K - Focus search
      if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
        event.preventDefault();
        handlers.onFocusSearch?.();
      }

      // Escape - Clear or close
      if (event.key === 'Escape') {
        handlers.onEscape?.();
      }

      // Ctrl/Cmd + Enter - Submit
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        handlers.onSubmit?.();
      }

      // Ctrl/Cmd + Shift + C - Clear chat
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'C') {
        event.preventDefault();
        handlers.onClearChat?.();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handlers]);
}
