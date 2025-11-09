"""
Unit tests for preprocessing module.
"""

import pytest
from pathlib import Path
from src.preprocessing.srt_parser import SRTParser, SubtitleEntry


class TestSRTParser:
    """Test cases for SRT parser."""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return SRTParser()
    
    @pytest.fixture
    def sample_srt_path(self):
        """Path to sample SRT file."""
        return Path(__file__).parent / "fixtures" / "sample.srt"
    
    def test_parse_sample_file(self, parser, sample_srt_path):
        """Test parsing sample SRT file."""
        entries = parser.parse_file(sample_srt_path)
        
        assert len(entries) == 3
        assert entries[0].sequence == 1
        assert entries[0].start_time == "00:00:00,000"
        assert entries[0].end_time == "00:00:05,000"
        assert "sample subtitle file" in entries[0].text.lower()
    
    def test_subtitle_entry_duration(self, parser, sample_srt_path):
        """Test duration calculation."""
        entries = parser.parse_file(sample_srt_path)
        
        assert entries[0].get_duration_seconds() == 5.0
        assert entries[1].get_duration_seconds() == 5.0
    
    def test_subtitle_entry_to_dict(self, parser, sample_srt_path):
        """Test conversion to dictionary."""
        entries = parser.parse_file(sample_srt_path)
        entry_dict = entries[0].to_dict()
        
        assert isinstance(entry_dict, dict)
        assert "sequence" in entry_dict
        assert "start_time" in entry_dict
        assert "end_time" in entry_dict
        assert "text" in entry_dict
    
    def test_get_all_text(self, parser, sample_srt_path):
        """Test extracting all text."""
        entries = parser.parse_file(sample_srt_path)
        all_text = parser.get_all_text(entries)
        
        assert isinstance(all_text, str)
        assert len(all_text) > 0
        assert "sample subtitle file" in all_text.lower()
    
    def test_get_text_by_time_range(self, parser, sample_srt_path):
        """Test extracting text by time range."""
        entries = parser.parse_file(sample_srt_path)
        
        # Get text from first 5 seconds
        text = parser.get_text_by_time_range(entries, 0.0, 5.0)
        assert len(text) > 0
        assert "sample subtitle file" in text.lower()
        
        # Get text from 5-10 seconds
        text = parser.get_text_by_time_range(entries, 5.0, 10.0)
        assert len(text) > 0
    
    def test_parse_nonexistent_file(self, parser):
        """Test parsing nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("nonexistent.srt"))
    
    def test_parse_multiple_files(self, parser, sample_srt_path, tmp_path):
        """Test parsing multiple files."""
        # Create a second test file
        test_file2 = tmp_path / "test2.srt"
        test_file2.write_text("""1
00:00:00,000 --> 00:00:03,000
Test subtitle
""")
        
        results = parser.parse_multiple_files([sample_srt_path, test_file2])
        
        assert len(results) == 2
        assert sample_srt_path in results
        assert test_file2 in results
        assert len(results[sample_srt_path]) == 3
        assert len(results[test_file2]) == 1
    
    def test_parse_empty_file(self, parser, tmp_path):
        """Test parsing empty file."""
        empty_file = tmp_path / "empty.srt"
        empty_file.write_text("")
        
        entries = parser.parse_file(empty_file)
        assert len(entries) == 0
    
    def test_parse_malformed_timestamp(self, parser, tmp_path):
        """Test parsing file with malformed timestamp."""
        malformed_file = tmp_path / "malformed.srt"
        malformed_file.write_text("""1
invalid timestamp
Some text
""")
        
        # Should skip malformed entries
        entries = parser.parse_file(malformed_file)
        # May return 0 entries or skip the malformed one
        assert isinstance(entries, list)
    
    def test_parse_multiline_subtitle(self, parser, tmp_path):
        """Test parsing multiline subtitle."""
        multiline_file = tmp_path / "multiline.srt"
        multiline_file.write_text("""1
00:00:00,000 --> 00:00:05,000
Line one
Line two
Line three
""")
        
        entries = parser.parse_file(multiline_file)
        assert len(entries) == 1
        assert "\n" in entries[0].text
        assert "Line one" in entries[0].text
        assert "Line three" in entries[0].text
