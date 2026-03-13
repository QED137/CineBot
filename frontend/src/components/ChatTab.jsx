import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { chatAPI } from '../services/api';
import MovieCard from './MovieCard';
import ReactMarkdown from 'react-markdown';
import EmptyState from './EmptyState';
import { TypingIndicator, MovieCardSkeleton } from './SkeletonLoader';
import { useToast } from './Toast';
import CopyButton from './CopyButton';
import ShareButton from './ShareButton';
import useKeyboardShortcuts from '../hooks/useKeyboardShortcuts';

export default function ChatTab() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [placeholder, setPlaceholder] = useState('Ask for a movie or upload a poster...');
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [showScrollTopButton, setShowScrollTopButton] = useState(false);
  const [recentQueries, setRecentQueries] = useState([]);
  const [activeQueryIndex, setActiveQueryIndex] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const abortControllerRef = useRef(null);
  const { addToast } = useToast();

  // Load chat history from localStorage on mount
  useEffect(() => {
    try {
      const savedHistory = localStorage.getItem('chatHistory');
      if (savedHistory) {
        const parsed = JSON.parse(savedHistory);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setMessages(parsed);
        }
      }
    } catch (e) {
      console.error('Error loading chat history:', e);
    }
  }, []);

  // Save chat history to localStorage whenever messages change
  useEffect(() => {
    if (messages.length > 0) {
      try {
        localStorage.setItem('chatHistory', JSON.stringify(messages));
      } catch (e) {
        console.error('Error saving chat history:', e);
      }
    }
  }, [messages]);

  // Helper functions - define before keyboard shortcuts
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const scrollToTop = () => {
    messagesContainerRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const clearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const clearChat = () => {
    setMessages([]);
    setInputValue('');
    clearImage();
    localStorage.removeItem('chatHistory'); // Clear saved history
    addToast('Chat cleared', 'info');
  };

  const stopQuery = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsLoading(false);
      addToast('Query stopped', 'info');
    }
  };

  const fetchSuggestion = async () => {
    try {
      const data = await chatAPI.getSuggestion();
      if (data.suggestion) {
        setPlaceholder(data.suggestion);
        addToast('Got a new suggestion!', 'success');
      }
    } catch (error) {
      console.error('Failed to fetch suggestion:', error);
      addToast('Failed to fetch suggestion', 'error');
    }
  };

  // Keyboard shortcuts
  useKeyboardShortcuts({
    onFocusSearch: () => textareaRef.current?.focus(),
    onEscape: () => {
      if (isLoading) stopQuery();
      else if (selectedImage) clearImage();
      else if (inputValue) setInputValue('');
    },
    onSubmit: () => {
      if (!isLoading && (inputValue.trim() || selectedImage)) {
        const event = new Event('submit', { cancelable: true, bubbles: true });
        document.querySelector('form')?.dispatchEvent(event);
      }
    },
    onClearChat: clearChat,
  });

  // Detect if user has scrolled away from top or bottom
  const handleScroll = () => {
    if (!messagesContainerRef.current) return;
    
    const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
    const isNearTop = scrollTop < 100;
    
    setShowScrollButton(!isNearBottom && messages.length > 0);
    setShowScrollTopButton(!isNearTop && messages.length > 0);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [inputValue]);

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        addToast('Image size must be less than 10MB', 'error');
        return;
      }
      if (!file.type.startsWith('image/')) {
        addToast('Please select an image file', 'error');
        return;
      }
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
      addToast('Poster uploaded! Click send to analyze', 'success');
    }
  };

  // Drag and drop handlers with better visual feedback
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragIn = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setDragActive(true);
    }
  };

  const handleDragOut = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        if (file.size > 10 * 1024 * 1024) {
          addToast('Image size must be less than 10MB', 'error');
          return;
        }
        setSelectedImage(file);
        const reader = new FileReader();
        reader.onloadend = () => {
          setImagePreview(reader.result);
        };
        reader.readAsDataURL(file);
        addToast('Poster uploaded! Click send to analyze', 'success');
      } else {
        addToast('Please drop an image file', 'error');
      }
      e.dataTransfer.clearData();
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Require either text or image
    if ((!inputValue.trim() && !selectedImage) || isLoading) return;

    const userMessage = inputValue.trim() || '(Uploaded a poster)';
    const userMessageContent = inputValue.trim();
    const hasImage = !!selectedImage;
    const imageFile = selectedImage;
    
    setInputValue('');
    clearImage();
    setPlaceholder('');
    
    // Add user message to chat (with image preview if present)
    const newUserMessage = { 
      role: 'user', 
      content: userMessageContent || 'Uploaded a movie poster',
      imagePreview: hasImage ? imagePreview : null
    };
    setMessages(prev => [...prev, newUserMessage]);
    
    // Create abort controller for this request
    abortControllerRef.current = new AbortController();
    setIsLoading(true);

    try {
      // Build chat history with context for API call
      const chatHistory = messages.map(msg => ({
        role: msg.role,
        content: msg.content,
        ...(msg.movies && msg.movies.length > 0 ? { context: msg.movies } : {})
      }));
      
      // Add current user message (without image data in history)
      chatHistory.push({ role: 'user', content: userMessageContent || '(Uploaded a poster)' });
      
      let data;
      if (hasImage) {
        // Send image with chat history
        data = await chatAPI.sendPosterImage(imageFile, chatHistory, abortControllerRef.current?.signal);
      } else {
        // Send text query
        data = await chatAPI.sendTextQuery(userMessageContent, chatHistory, abortControllerRef.current?.signal);
      }
      
      // Add assistant response
      const assistantMessage = {
        role: 'assistant',
        content: data.message,
        movies: data.movies || [],
        metadata: {
          response_type: data.response_type,
          source: data.source,
          input_mode: data.input_mode
        }
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      
      // Save recommended movies for Articles tab
      if (data.movies && data.movies.length > 0) {
        try {
          localStorage.setItem('recommendedMovies', JSON.stringify(data.movies));
          localStorage.setItem('recommendedMoviesTimestamp', Date.now().toString());
        } catch (e) {
          console.error('Error saving recommended movies:', e);
        }
      }
      
      // Show success toast
      if (data.meta?.has_movies && data.meta?.count > 0) {
        addToast(`Found ${data.meta.count} movies for you!`, 'success');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Check if request was cancelled
      if (error.code === 'ERR_CANCELED' || error.name === 'CanceledError') {
        const cancelMessage = {
          role: 'assistant',
          content: 'Query stopped by user.',
          isError: false,
        };
        setMessages(prev => [...prev, cancelMessage]);
      } else {
        const errorMessage = {
          role: 'assistant',
          content: `I apologize, but I encountered an error: ${error.message}. Please try again.`,
          isError: true,
        };
        setMessages(prev => [...prev, errorMessage]);
        addToast('Failed to get recommendations. Please try again.', 'error');
      }
    } finally {
      abortControllerRef.current = null;
      setIsLoading(false);
      setPlaceholder('Ask for a movie or upload a poster...');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex flex-col h-full relative">
      {/* Drag and Drop Overlay */}
      {dragActive && (
        <div 
          className="absolute inset-0 z-50 bg-primary/20 backdrop-blur-sm border-4 border-dashed border-primary rounded-lg flex items-center justify-center"
          aria-live="polite"
          aria-label="Drop your poster image here"
        >
          <div className="bg-slate-900/90 rounded-2xl p-8 text-center">
            <svg className="w-16 h-16 mx-auto mb-4 text-primary animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <p className="text-xl font-semibold text-white">Drop your poster here!</p>
            <p className="text-sm text-slate-400 mt-2">We'll find similar movies for you</p>
          </div>
        </div>
      )}

      {/* Query Results Container - scrollable */}
      <div 
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-4 md:p-6 pb-8"
        onDragEnter={handleDragIn}
        onDragLeave={handleDragOut}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onScroll={handleScroll}
        role="log"
        aria-live="polite"
        aria-label="Query results"
      >
        {/* Clear button - top right when messages exist */}
        {messages.length > 0 && (
          <div className="flex items-center justify-end gap-2 mb-6">
            <ShareButton messages={messages} addToast={addToast} />
            <button
              onClick={clearChat}
              className="p-2 rounded-lg hover:bg-slate-700/50 transition-colors text-slate-400 hover:text-red-400"
              aria-label="Clear all queries"
              title="Clear all"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        )}
        
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          (() => {
            // Group messages into query-response pairs
            const pairs = messages.reduce((acc, message, index) => {
              if (message.role === 'user') {
                acc.push({ user: message, assistant: messages[index + 1] || null, index });
              }
              return acc;
            }, []);

            return (
              <div className="space-y-6">
                {pairs.map((pair, pairIndex) => (
                  <motion.div
                    key={pair.index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: pairIndex * 0.05 }}
                    className="bg-gradient-to-br from-slate-800/40 via-slate-800/30 to-slate-900/40 backdrop-blur-xl rounded-3xl border border-slate-700/40 overflow-hidden hover:border-primary/40 transition-all duration-500 shadow-2xl hover:shadow-primary/10 group"
                  >
                    {/* Editorial Query Header */}
                    <div className="relative bg-gradient-to-r from-primary/10 via-purple-600/10 to-pink-600/10 border-b border-slate-700/30 px-6 py-5">
                      <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                      <div className="relative flex items-start justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-3 mb-3">
                            <span className="px-3 py-1 bg-primary/20 border border-primary/30 rounded-full text-xs font-semibold text-primary-200 tracking-wide uppercase">
                              Query #{pairIndex + 1}
                            </span>
                            <div className="h-px flex-1 bg-gradient-to-r from-slate-600/50 to-transparent" />
                          </div>
                          {pair.user.imagePreview ? (
                            <div className="mt-4">
                              <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 border border-slate-600/50 rounded-lg mb-4">
                                <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                <span className="text-xs font-medium text-purple-300">Visual Search</span>
                              </div>
                              <img 
                                src={pair.user.imagePreview} 
                                alt="Uploaded poster" 
                                className="rounded-xl max-w-xs max-h-52 object-cover border border-slate-600/50 shadow-lg"
                                loading="lazy"
                              />
                            </div>
                          ) : (
                            <h3 className="text-white font-semibold text-xl leading-relaxed tracking-tight">{pair.user.content}</h3>
                          )}
                        </div>
                        {pair.assistant && (
                          <div className="flex-shrink-0">
                            <CopyButton text={pair.assistant.content} />
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Editorial Response Body */}
                    {pair.assistant ? (
                      <div className="px-6 py-6">
                        <article className={`prose prose-lg prose-invert max-w-none ${pair.assistant.isError ? 'text-red-400' : ''}`}>
                          <div className="text-slate-200 leading-relaxed space-y-4">
                            <ReactMarkdown
                              components={{
                                img: ({ node, ...props }) => (
                                  <img
                                    {...props}
                                    className="rounded-xl my-4 max-w-full h-auto shadow-lg border border-slate-700/50"
                                    loading="lazy"
                                    alt={props.alt || 'Movie poster'}
                                  />
                                ),
                                a: ({ node, ...props }) => (
                                  <a
                                    {...props}
                                    className="text-primary hover:text-primary/80 underline decoration-primary/30 hover:decoration-primary/60 transition-colors"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  />
                                ),
                                p: ({ node, ...props }) => (
                                  <p {...props} className="text-base leading-relaxed text-slate-300" />
                                ),
                                strong: ({ node, ...props }) => (
                                  <strong {...props} className="font-semibold text-white" />
                                ),
                              }}
                            >
                              {pair.assistant.content}
                            </ReactMarkdown>
                          </div>
                        </article>

                        {/* Editorial Movie Showcase */}
                        {pair.assistant.movies && pair.assistant.movies.length > 0 && (
                          <div className="mt-8 pt-8 border-t border-gradient-to-r from-transparent via-slate-700/50 to-transparent">
                            <div className="flex items-center justify-between mb-6">
                              <div className="flex items-center gap-3">
                                <div className="w-1 h-8 bg-gradient-to-b from-primary to-purple-600 rounded-full" />
                                <div>
                                  <h4 className="text-lg font-bold text-white tracking-tight">Curated Recommendations</h4>
                                  <p className="text-xs text-slate-400 mt-0.5">{pair.assistant.movies.length} {pair.assistant.movies.length === 1 ? 'film' : 'films'} selected for you</p>
                                </div>
                              </div>
                              <svg className="w-6 h-6 text-primary/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
                              </svg>
                            </div>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                              {pair.assistant.movies.slice(0, 6).map((movie, idx) => (
                                <MovieCard key={idx} movie={movie} />
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      isLoading && pairIndex === pairs.length - 1 && (
                        <div className="p-8 flex items-center justify-center">
                          <div className="flex flex-col items-center gap-3">
                            <TypingIndicator />
                            <p className="text-slate-400 text-sm">Processing your query...</p>
                          </div>
                        </div>
                      )
                    )}
                  </motion.div>
                ))}
              </div>
            );
          })()
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Scroll to Top Button */}
      {showScrollTopButton && (
        <button
          onClick={scrollToTop}
          className="absolute bottom-40 right-6 bg-primary hover:bg-primary/80 text-white rounded-full p-3 shadow-lg hover:shadow-xl transition-all transform hover:scale-110 active:scale-95 z-10 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
          aria-label="Scroll to top"
          title="Scroll to first message"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
          </svg>
        </button>
      )}

      {/* Scroll to Bottom Button */}
      {showScrollButton && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-24 right-6 bg-primary hover:bg-primary/80 text-white rounded-full p-3 shadow-lg hover:shadow-xl transition-all transform hover:scale-110 active:scale-95 z-10 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
          aria-label="Scroll to bottom"
          title="Scroll to latest message"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </button>
      )}

      {/* Unified Composer */}
      <div className="relative border-t border-slate-700/50 bg-gradient-to-b from-slate-800/30 via-slate-800/50 to-slate-900/70 backdrop-blur-2xl pb-[env(safe-area-inset-bottom)]">
        {/* Ambient Glow */}
        <div className="absolute inset-0 bg-gradient-to-t from-primary/5 via-transparent to-transparent pointer-events-none" />
        
        <div className="relative p-5 sm:p-6">
          {/* Image Preview */}
          {imagePreview && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mb-4 inline-flex items-start gap-3 p-3 bg-slate-800/50 border border-slate-600/50 rounded-2xl backdrop-blur-sm"
            >
              <img 
                src={imagePreview} 
                alt="Movie poster preview" 
                className="h-20 sm:h-24 rounded-lg border border-slate-600/50 shadow-xl"
              />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-slate-400 mb-1">Visual Search</p>
                <p className="text-sm text-slate-300">Poster ready to analyze</p>
              </div>
              <button
                type="button"
                onClick={clearImage}
                className="flex-shrink-0 bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 text-red-400 hover:text-red-300 rounded-lg p-2 transition-all hover:scale-110"
                title="Remove image"
                aria-label="Remove selected poster image"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </motion.div>
          )}
          
          {/* Unified Input Form */}
          <form onSubmit={handleSubmit} className="relative" role="search" aria-label="Movie search form">
            <div className="relative bg-slate-800/50 border border-slate-600/50 hover:border-slate-500/50 focus-within:border-primary/50 rounded-2xl transition-all duration-300 shadow-xl backdrop-blur-sm group">
              {/* Glow effect on focus */}
              <div className="absolute inset-0 bg-primary/5 rounded-2xl opacity-0 group-focus-within:opacity-100 transition-opacity duration-300 pointer-events-none" />
              
              <div className="relative flex items-end gap-2 p-3">
                {/* Image Upload */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                  id="image-upload"
                  aria-label="Upload movie poster image"
                />
                <label
                  htmlFor="image-upload"
                  className="flex-shrink-0 min-h-11 min-w-11 bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 hover:text-white rounded-xl p-3 transition-all hover:scale-105 cursor-pointer border border-slate-600/30 hover:border-primary/30 group/upload"
                  title="Upload poster image"
                  tabIndex={0}
                >
                  <svg className="w-5 h-5 transition-transform group-hover/upload:scale-110" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </label>
                
                {/* Text Input */}
                <div className="flex-1 relative">
                  <textarea
                    ref={textareaRef}
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={placeholder}
                    rows={1}
                    className="w-full bg-transparent text-white px-4 py-3 pr-12 resize-none focus:outline-none max-h-32 text-base placeholder:text-slate-500"
                    disabled={isLoading}
                    aria-label="Enter your movie query"
                    aria-describedby="input-help-text"
                  />
                  <button
                    type="button"
                    onClick={fetchSuggestion}
                    className="absolute right-2 top-1/2 -translate-y-1/2 min-h-10 min-w-10 p-2 text-slate-400 hover:text-primary transition-all hover:scale-110 hover:rotate-12 rounded-lg"
                    title="Get inspiration"
                    aria-label="Get query suggestion"
                    disabled={isLoading}
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </button>
                </div>
                
                {/* Action Button */}
                {isLoading ? (
                  <button
                    type="button"
                    onClick={stopQuery}
                    className="flex-shrink-0 min-h-11 bg-gradient-to-br from-red-600 to-red-700 hover:from-red-500 hover:to-red-600 text-white rounded-xl px-5 py-3 transition-all hover:scale-105 active:scale-95 flex items-center gap-2 shadow-lg shadow-red-600/30 font-medium border border-red-500/30"
                    aria-label="Stop query"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                    </svg>
                    <span className="hidden sm:inline">Stop</span>
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!inputValue.trim() && !selectedImage}
                    className="flex-shrink-0 min-h-11 bg-gradient-to-br from-primary via-purple-600 to-purple-700 hover:from-primary/90 hover:via-purple-600/90 hover:to-purple-700/90 disabled:from-slate-700 disabled:via-slate-750 disabled:to-slate-800 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl px-5 py-3 transition-all hover:scale-105 active:scale-95 flex items-center gap-2 shadow-lg shadow-primary/30 hover:shadow-primary/40 font-medium border border-primary/20"
                    aria-label="Send message"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                    <span className="hidden sm:inline">Send</span>
                  </button>
                )}
              </div>
            </div>
            
            <span id="input-help-text" className="sr-only">
              Type a movie-related question or upload a poster image to find similar movies
            </span>
          </form>
          
          {/* Keyboard Shortcuts */}
          <div className="hidden sm:flex items-center justify-center gap-4 mt-4 text-xs text-slate-500">
            <div className="flex items-center gap-1.5">
              <kbd className="px-2 py-1 bg-slate-800/50 rounded-md border border-slate-700/50 font-mono text-slate-400">Enter</kbd>
              <span>Send</span>
            </div>
            <div className="w-px h-3 bg-slate-700/50" />
            <div className="flex items-center gap-1.5">
              <kbd className="px-2 py-1 bg-slate-800/50 rounded-md border border-slate-700/50 font-mono text-slate-400">Shift</kbd>
              <span>+</span>
              <kbd className="px-2 py-1 bg-slate-800/50 rounded-md border border-slate-700/50 font-mono text-slate-400">Enter</kbd>
              <span>New line</span>
            </div>
            <div className="w-px h-3 bg-slate-700/50" />
            <div className="flex items-center gap-1.5">
              <kbd className="px-2 py-1 bg-slate-800/50 rounded-md border border-slate-700/50 font-mono text-slate-400">Ctrl</kbd>
              <span>+</span>
              <kbd className="px-2 py-1 bg-slate-800/50 rounded-md border border-slate-700/50 font-mono text-slate-400">K</kbd>
              <span>Clear</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
