import { useState } from 'react';
import { chatAPI } from '../services/api';
import MovieCard from './MovieCard';

export default function PosterTab() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [movies, setMovies] = useState([]);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setError('');
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
      setError('Please upload a poster image first.');
      return;
    }

    setIsLoading(true);
    setError('');
    setMessage('');
    setMovies([]);

    try {
      const data = await chatAPI.sendPosterImage(selectedFile);
      
      setMessage(data.llm_response_text || '');
      setMovies(data.context_movies || []);
      
    } catch (err) {
      console.error('Error uploading poster:', err);
      setError(err.response?.data?.error || 'Failed to get recommendations. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreview(null);
    setMovies([]);
    setMessage('');
    setError('');
  };

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-2">Find Movies by Poster</h2>
          <p className="text-slate-400">Upload a movie poster to discover similar films</p>
        </div>

        {/* Upload Form */}
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="bg-slate-800/50 rounded-xl border-2 border-dashed border-slate-600 hover:border-primary/50 transition-all p-8">
            {!preview ? (
              <div className="text-center">
                <label htmlFor="poster-upload" className="cursor-pointer block">
                  <div className="mx-auto w-16 h-16 bg-slate-700 rounded-full flex items-center justify-center mb-4">
                    <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  <p className="text-lg font-medium mb-1">Click to upload poster</p>
                  <p className="text-sm text-slate-400">PNG, JPG up to 16MB</p>
                </label>
                <input
                  id="poster-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </div>
            ) : (
              <div className="flex items-start gap-4">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-48 h-72 object-cover rounded-lg border border-slate-600"
                />
                <div className="flex-1">
                  <p className="text-sm text-slate-300 mb-2">
                    <strong>File:</strong> {selectedFile?.name}
                  </p>
                  <p className="text-sm text-slate-300 mb-4">
                    <strong>Size:</strong> {(selectedFile?.size / 1024).toFixed(2)} KB
                  </p>
                  <button
                    type="button"
                    onClick={clearSelection}
                    className="text-sm text-red-400 hover:text-red-300 transition-colors"
                  >
                    Remove
                  </button>
                </div>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 bg-red-500/10 border border-red-500/50 rounded-lg p-4">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          <button
            type="submit"
            disabled={!selectedFile || isLoading}
            className="mt-4 w-full bg-secondary hover:bg-secondary/80 disabled:bg-slate-700 disabled:cursor-not-allowed text-white font-medium rounded-xl px-6 py-3 transition-all flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
                Analyzing Poster...
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                Get Recommendations
              </>
            )}
          </button>
        </form>

        {/* Results */}
        {message && (
          <div className="mb-6 bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <p className="text-slate-200 whitespace-pre-wrap">{message}</p>
          </div>
        )}

        {movies.length > 0 && (
          <div>
            <h3 className="text-xl font-semibold mb-4">Recommended Movies</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {movies.map((movie, index) => (
                <MovieCard key={index} movie={movie} index={index} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
