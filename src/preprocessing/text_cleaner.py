"""
Text Cleaner Module

Cleans and normalizes subtitle text.
"""

import re
import html
from typing import List


class TextCleaner:
    """Cleans and normalizes subtitle text."""
    
    # Pattern for sound/music tags: [Music], [Applause], etc.
    SOUND_TAG_PATTERN = re.compile(r'\[(?:Music|Applause|Laughter|Crowd|Crowd\s+cheering|Laughing|Singing|Sighing|Breathing|Noise|Silence|.*?)\](?:\s|$)', re.IGNORECASE)
    
    # Pattern for speaker tags: <v SpeakerName>
    SPEAKER_TAG_PATTERN = re.compile(r'<v\s+[^>]+>', re.IGNORECASE)
    
    # Pattern for other XML/HTML-like tags
    XML_TAG_PATTERN = re.compile(r'<[^>]+>')
    
    # Pattern for multiple spaces
    MULTIPLE_SPACES_PATTERN = re.compile(r'\s+')
    
    # Pattern for multiple newlines
    MULTIPLE_NEWLINES_PATTERN = re.compile(r'\n\s*\n+')
    
    def __init__(self):
        """Initialize text cleaner."""
        pass
    
    def clean(self, text: str, preserve_sentences: bool = True) -> str:
        """
        Clean subtitle text.
        
        Args:
            text: Raw subtitle text
            preserve_sentences: If True, preserve sentence boundaries
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML entities
        text = html.unescape(text)
        
        # Remove sound/music tags
        text = self.SOUND_TAG_PATTERN.sub('', text)
        
        # Remove speaker tags
        text = self.SPEAKER_TAG_PATTERN.sub('', text)
        
        # Remove other XML/HTML tags
        text = self.XML_TAG_PATTERN.sub('', text)
        
        # Remove SRT formatting artifacts (italics, bold, etc.)
        text = self._remove_srt_formatting(text)
        
        # Normalize whitespace
        if preserve_sentences:
            # Preserve sentence boundaries (periods, exclamation, question marks)
            text = self._normalize_whitespace_preserve_sentences(text)
        else:
            # Simple normalization
            text = self.MULTIPLE_SPACES_PATTERN.sub(' ', text)
            text = text.strip()
        
        # Remove leading/trailing punctuation artifacts
        text = self._clean_punctuation_artifacts(text)
        
        return text.strip()
    
    def _remove_srt_formatting(self, text: str) -> str:
        """
        Remove SRT formatting tags.
        
        Args:
            text: Text with SRT formatting
        
        Returns:
            Text without formatting tags
        """
        # Remove italics: <i>text</i> or {i}text{/i}
        text = re.sub(r'<i>|</i>|{i}|{/i}', '', text, flags=re.IGNORECASE)
        
        # Remove bold: <b>text</b> or {b}text{/b}
        text = re.sub(r'<b>|</b>|{b}|{/b}', '', text, flags=re.IGNORECASE)
        
        # Remove underline: <u>text</u> or {u}text{/u}
        text = re.sub(r'<u>|</u>|{u}|{/u}', '', text, flags=re.IGNORECASE)
        
        # Remove font tags: <font ...>text</font>
        text = re.sub(r'<font[^>]*>|</font>', '', text, flags=re.IGNORECASE)
        
        # Remove color tags: {c:color} or <font color="...">
        text = re.sub(r'{c:[^}]+}', '', text)
        
        return text
    
    def _normalize_whitespace_preserve_sentences(self, text: str) -> str:
        """
        Normalize whitespace while preserving sentence boundaries.
        
        Args:
            text: Text to normalize
        
        Returns:
            Normalized text
        """
        # Replace multiple spaces with single space
        text = self.MULTIPLE_SPACES_PATTERN.sub(' ', text)
        
        # Replace multiple newlines with single newline
        text = self.MULTIPLE_NEWLINES_PATTERN.sub('\n', text)
        
        # Ensure space after sentence endings (if not already present)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Remove spaces after opening parentheses/brackets
        text = re.sub(r'([\(\[\{])\s+', r'\1', text)
        
        # Remove spaces before closing parentheses/brackets
        text = re.sub(r'\s+([\)\]\}])', r'\1', text)
        
        return text
    
    def _clean_punctuation_artifacts(self, text: str) -> str:
        """
        Remove punctuation artifacts from cleaning.
        
        Args:
            text: Text to clean
        
        Returns:
            Cleaned text
        """
        # Remove leading punctuation (except quotes)
        text = re.sub(r'^[^\w\s"\']+', '', text)
        
        # Remove trailing punctuation artifacts (keep sentence endings)
        text = re.sub(r'([^\s])[^\w\s"\'.!?]+$', r'\1', text)
        
        # Fix multiple consecutive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        return text
    
    def clean_batch(self, texts: List[str], preserve_sentences: bool = True) -> List[str]:
        """
        Clean multiple texts.
        
        Args:
            texts: List of texts to clean
            preserve_sentences: If True, preserve sentence boundaries
        
        Returns:
            List of cleaned texts
        """
        return [self.clean(text, preserve_sentences) for text in texts]
    
    def clean_subtitle_entries(self, entries: List) -> List[str]:
        """
        Clean text from subtitle entries.
        
        Args:
            entries: List of subtitle entry objects with 'text' attribute
        
        Returns:
            List of cleaned text strings
        """
        texts = [entry.text for entry in entries]
        return self.clean_batch(texts, preserve_sentences=True)
    
    def join_cleaned_text(self, entries: List, separator: str = ' ') -> str:
        """
        Clean and join subtitle entries into single text.
        
        Args:
            entries: List of subtitle entry objects
            separator: String to join entries with
        
        Returns:
            Joined cleaned text
        """
        cleaned_texts = self.clean_subtitle_entries(entries)
        return separator.join(cleaned_texts)
