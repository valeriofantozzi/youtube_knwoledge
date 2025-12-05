import click
from src.utils.db_manager import get_db_manager
from src.cli.utils.output import print_success, print_error, print_info, console
from rich.table import Table

@click.group(name="db")
def db_group():
    """Manage vector databases."""
    pass

@db_group.command(name="list")
def list_dbs():
    """List available databases."""
    manager = get_db_manager()
    dbs = manager.list_databases()
    active = manager.get_active_database()
    
    table = Table(title="Available Databases")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    
    for db in dbs:
        status = "Active" if db == active else ""
        table.add_row(db, status)
        
    console.print(table)

@db_group.command(name="create")
@click.argument("name")
def create_db(name):
    """Create a new database."""
    manager = get_db_manager()
    try:
        if manager.create_database(name):
            manager.set_active_database(name)
            print_success(f"Database '{name}' created and set as active.")
        else:
            print_error(f"Database '{name}' already exists.")
    except ValueError as e:
        print_error(str(e))

@db_group.command(name="use")
@click.argument("name")
def use_db(name):
    """Switch to a different database."""
    manager = get_db_manager()
    if manager.set_active_database(name):
        print_success(f"Switched to database '{name}'.")
    else:
        print_error(f"Database '{name}' does not exist.")

@db_group.command(name="remove")
@click.argument("name")
def remove_db(name):
    """Remove a database."""
    manager = get_db_manager()
    try:
        # Check if we are removing the active database
        if manager.get_active_database() == name:
            manager.set_active_database("default")
            print_info(f"Switched to 'default' database before removing active database '{name}'.")
            
        if manager.remove_database(name):
            print_success(f"Database '{name}' removed successfully.")
        else:
            print_error(f"Database '{name}' does not exist.")
    except ValueError as e:
        print_error(str(e))

@db_group.command(name="current")
def current_db():
    """Show current active database."""
    manager = get_db_manager()
    print_info(f"Current database: {manager.get_active_database()}")
