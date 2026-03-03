import { useState, useRef, useEffect } from 'react';
import { chatAPI } from '../services/api';
import MovieCard from './MovieCard';
import ReactMarkdown from 'react-markdown';
import EmptyState from './EmptyState';
import { TypingIndicator, MovieCardSkeleton } from './SkeletonLoader';
import { useToast } from './Toast';
import CopyButton from './CopyButton';
import ShareButton from './ShareButton';

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
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const { addToast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const scrollToTop = () => {
    messagesContainerRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  };

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

  const fetchSuggestion = async () => {
    try {
      const data = await chatAPI.getSuggestion();
      if (data.suggestion) {
        setPlaceholder(data.suggestion);
        addToast('Got a new suggestion! ✨', 'success');
      }
    } catch (error) {
      console.error('Failed to fetch suggestion:', error);
      addToast('Failed to fetch suggestion', 'error');
    }
  };

  const clearChat = () => {
    setMessages([]);
    setInputValue('');
    clearImage();
    addToast('Chat cleared', 'info');
  };

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

  const clearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
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
        data = await chatAPI.sendPosterImage(imageFile, chatHistory);
      } else {
        // Send text query
        data = await chatAPI.sendTextQuery(userMessageContent, chatHistory);
      }
      
      // Add assistant response
      const assistantMessage = {
        role: 'assistant',
        content: data.llm_response_text || data.response,
        movies: data.context_movies || data.movies || [],
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      
      // Show success toast
      if (data.context_movies?.length > 0) {
        addToast(`Found ${data.context_movies.length} movies for you! 🎬`, 'success');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: `I apologize, but I encountered an error: ${error.message}. Please try again.`,
        isError: true,
      };
      setMessages(prev => [...prev, errorMessage]);
      addToast('Failed to get recommendations. Please try again.', 'error');
    } finally {
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

      {/* Messages Container */}
      <div 
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4 md:space-y-6"
        onDragEnter={handleDragIn}
        onDragLeave={handleDragOut}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onScroll={handleScroll}
        role="log"
        aria-live="polite"
        aria-label="Chat messages"
      >
        {/* Share/Clear buttons - top right when messages exist */}
        {messages.length > 0 && (
          <div className="flex items-center justify-end gap-2 mb-4">
            <ShareButton messages={messages} addToast={addToast} />
            <button
              onClick={clearChat}
              className="p-2 rounded-lg hover:bg-slate-700/50 transition-colors text-slate-400 hover:text-red-400"
              aria-label="Clear chat history"
              title="Clear chat"
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
          <>
            {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn mb-4`}
            >
              <div
                className={`max-w-[90%] sm:max-w-3xl relative group ${
                  message.role === 'user'
                    ? 'bg-primary text-white rounded-2xl rounded-tr-sm'
                    : 'bg-slate-800/50 text-slate-100 rounded-2xl rounded-tl-sm'
                } px-3 sm:px-4 py-3 shadow-lg hover:shadow-xl transition-shadow ${message.isError ? 'border border-red-500' : ''}`}
                role="article"
                aria-label={`${message.role === 'user' ? 'Your' : 'AI'} message`}
              >
                {/* Copy button for assistant messages */}
                {message.role === 'assistant' && (
                  <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                    <CopyButton text={message.content} />
                  </div>
                )}
                
                {/* Display uploaded image if present */}
                {message.imagePreview && (
                  <div className="mb-3">
                    <img 
                      src={message.imagePreview} 
                      alt="Uploaded poster" 
                      className="rounded-lg max-w-full sm:max-w-xs max-h-64 sm:max-h-96 object-cover"
                      loading="lazy"
                    />
                  </div>
                )}
                
                <div className="prose prose-invert max-w-none">
                  <ReactMarkdown
                    components={{
                      img: ({ node, ...props }) => (
                        <img
                          {...props}
                          className="rounded-lg my-2 max-w-full h-auto"
                          loading="lazy"
                          alt={props.alt || 'Movie poster'}
                        />
                      ),
                      a: ({ node, ...props }) => (
                        <a
                          {...props}
                          className="text-primary hover:text-primary/80 underline"
                          target="_blank"
                          rel="noopener noreferrer"
                        />
                      ),
                      p: ({ node, ...props }) => (
                        <p {...props} className="mb-2 last:mb-0" />
                      ),
                      ul: ({ node, ...props }) => (
                        <ul {...props} className="list-disc ml-4 mb-2" />
                      ),
                      ol: ({ node, ...props }) => (
                        <ol {...props} className="list-decimal ml-4 mb-2" />
                      ),
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
                
                {/* Display movie cards if available */}
                {message.movies && message.movies.length > 0 && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3 sm:gap-4 mt-4">
                    {message.movies.slice(0, 3).map((movie, idx) => (
                      <MovieCard key={idx} movie={movie} index={idx} />
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          </>
        )}
        
        {isLoading && <TypingIndicator />}
        
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

      {/* Input Area */}
      <div className="border-t border-slate-700 bg-slate-800/30 backdrop-blur-sm p-3 sm:p-4">
        {/* Image Preview */}
        {imagePreview && (
          <div className="mb-3 relative inline-block">
            <img 
              src={imagePreview} 
              alt="Movie poster preview" 
              className="h-20 sm:h-24 rounded-lg border-2 border-primary"
            />
            <button
              type="button"
              onClick={clearImage}
              className="absolute -top-2 -right-2 bg-red-500 hover:bg-red-600 text-white rounded-full p-1 transition-all hover:scale-110 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
              title="Remove image"
              aria-label="Remove selected poster image"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="flex gap-2" role="search" aria-label="Movie search form">
          <div className="flex gap-2">
            {/* Image Upload Button */}
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
              className="bg-slate-700/50 hover:bg-slate-700 text-slate-300 hover:text-white rounded-xl px-3 sm:px-4 py-3 transition-all hover:scale-105 cursor-pointer flex items-center gap-2 focus-within:ring-2 focus-within:ring-primary focus-within:ring-offset-2 focus-within:ring-offset-slate-900"
              title="Upload poster image"
              tabIndex={0}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <span className="sr-only">Upload poster</span>
            </label>
          </div>
          
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              rows={1}
              className="w-full bg-slate-700/50 text-white rounded-xl px-3 sm:px-4 py-3 pr-10 sm:pr-12 resize-none focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-slate-900 max-h-32 text-sm sm:text-base"
              disabled={isLoading}
              aria-label="Enter your movie query"
              aria-describedby="input-help-text"
            />
            <span id="input-help-text" className="sr-only">
              Type a movie-related question or upload a poster image to find similar movies
            </span>
            <button
              type="button"
              onClick={fetchSuggestion}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 text-slate-400 hover:text-primary transition-all hover:scale-110 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary rounded"
              title="Get inspiration"
              aria-label="Get query suggestion"
              disabled={isLoading}
            >
              <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </button>
          </div>
          
          <button
            type="submit"
            disabled={isLoading || (!inputValue.trim() && !selectedImage)}
            className="bg-primary hover:bg-primary/80 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-xl px-4 sm:px-6 py-3 transition-all hover:scale-105 active:scale-95 flex items-center gap-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
            aria-label={isLoading ? (selectedImage ? 'Analyzing poster' : 'Sending message') : 'Send message'}
          >
            {isLoading ? (
              <>
                <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" aria-hidden="true"></div>
                <span className="hidden sm:inline">{selectedImage ? 'Analyzing...' : 'Sending...'}</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
                <span className="hidden sm:inline">Send</span>
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
