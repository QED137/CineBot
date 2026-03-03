export function MovieCardSkeleton() {
  return (
    <div className="bg-slate-800/50 rounded-xl overflow-hidden border border-slate-700 animate-pulse">
      <div className="bg-slate-700/50 w-full h-64"></div>
      <div className="p-4 space-y-3">
        <div className="h-6 bg-slate-700/50 rounded w-3/4"></div>
        <div className="h-4 bg-slate-700/50 rounded w-full"></div>
        <div className="h-4 bg-slate-700/50 rounded w-5/6"></div>
        <div className="h-3 bg-slate-700/50 rounded w-1/2"></div>
      </div>
    </div>
  );
}

export function MessageSkeleton() {
  return (
    <div className="flex gap-3 fade-in">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-primary to-purple-600"></div>
      <div className="flex-1 bg-slate-800/50 rounded-2xl p-4 space-y-3 animate-pulse">
        <div className="h-4 bg-slate-700/50 rounded w-3/4"></div>
        <div className="h-4 bg-slate-700/50 rounded w-full"></div>
        <div className="h-4 bg-slate-700/50 rounded w-5/6"></div>
      </div>
    </div>
  );
}

export function TypingIndicator() {
  return (
    <div className="flex gap-3 fade-in">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center">
        <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
          <path d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6zM10 18a3 3 0 01-3-3h6a3 3 0 01-3 3z" />
        </svg>
      </div>
      <div className="bg-slate-800/50 rounded-2xl px-6 py-4">
        <div className="flex gap-1.5">
          <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </div>
  );
}
