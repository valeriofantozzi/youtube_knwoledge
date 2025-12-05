import pytest
from unittest.mock import Mock, patch
from click.testing import CliRunner
from src.cli.commands.db import db_group

@pytest.fixture
def mock_manager():
    with patch('src.cli.commands.db.get_db_manager') as mock:
        manager = Mock()
        mock.return_value = manager
        yield manager

def test_create_db_switches_active(mock_manager):
    """Test that create_db creates and switches to the new database."""
    runner = CliRunner()
    mock_manager.create_database.return_value = True
    
    result = runner.invoke(db_group, ['create', 'new_db'])
    
    assert result.exit_code == 0
    assert "set as active" in result.output
    # Verify create was called
    mock_manager.create_database.assert_called_with('new_db')
    # Verify switch was called
    mock_manager.set_active_database.assert_called_with('new_db')

def test_remove_db_no_prompt(mock_manager):
    """Test that remove_db does not prompt."""
    runner = CliRunner()
    mock_manager.remove_database.return_value = True
    # Setup: Active DB is different from the one being removed
    mock_manager.get_active_database.return_value = 'other_db'
    
    result = runner.invoke(db_group, ['remove', 'to_remove'])
    
    assert result.exit_code == 0
    assert "removed successfully" in result.output
    # Should not have asked for input (CliRunner fails if input is requested but not provided)
    mock_manager.remove_database.assert_called_with('to_remove')

def test_remove_active_db_switches_default(mock_manager):
    """Test that removing the active DB switches to default first."""
    runner = CliRunner()
    mock_manager.remove_database.return_value = True
    # Setup: Active DB IS the one being removed
    mock_manager.get_active_database.return_value = 'active_db'
    
    result = runner.invoke(db_group, ['remove', 'active_db'])
    
    assert result.exit_code == 0
    assert "Switched to 'default'" in result.output
    
    # Verify sequence: switch then remove
    # We can check call args list or just that both were called
    mock_manager.set_active_database.assert_called_with('default')
    mock_manager.remove_database.assert_called_with('active_db')
