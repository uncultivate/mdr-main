import numpy as np
import logging
import importlib.util
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_magic_square_and_navigate(grid, strategy='exploration', found_squares=None, window_coords=None, verbose_logging=False):
    """
    Combined function to detect magic squares and determine next direction.
    
    Args:
        grid: 2D numpy array representing the current visible grid (with NaN for unexplored areas)
        strategy: String indicating which search strategy to use
                 ('exploration', 'random_walk', 'spiral', 'pattern_detection', or custom strategy ID)
        found_squares: List of previously found magic square positions to avoid re-detection
        window_coords: Tuple of ((min_row, min_col), (max_row, max_col)) representing the current view window
        verbose_logging: Boolean flag to enable detailed logging
        
    Returns:
        tuple: (is_magic_square, next_direction, square_info)
               is_magic_square: Boolean indicating if any magic square is found
               next_direction: String - 'up', 'down', 'left', 'right'
               square_info: Dictionary with information about the found magic square, or None
    """
    if found_squares is None:
        found_squares = []
        
    # Convert None values to NaN if present
    grid = np.array(grid, dtype=float)
    grid = np.where(grid == None, np.nan, grid)
    
    # Log the grid shape
    logger.info(f"Grid shape received: {grid.shape}")
    
    # Always log the current window position (regardless of verbose_logging setting)
    if window_coords:
        (min_row, min_col), (max_row, max_col) = window_coords
        window_height = max_row - min_row + 1
        window_width = max_col - min_col + 1
        window_center_row = min_row + window_height // 2
        window_center_col = min_col + window_width // 2
        logger.info(f"WINDOW POSITION: Center=({window_center_row},{window_center_col}), " +
                   f"Bounds=({min_row},{min_col}) to ({max_row},{max_col}), " +
                   f"Size={window_height}x{window_width}")
    
    if verbose_logging:
        # Log the current grid state with NaN values
        logger.info(f"Original grid with NaNs:\n{np.array2string(grid, precision=1, separator=', ', threshold=100, edgeitems=3)}")
        
        # Log information about unexplored cells
        nan_count = np.sum(np.isnan(grid))
        total_cells = grid.size
        explored_percent = 100 * (1 - nan_count / total_cells)
        logger.info(f"Grid exploration status: {explored_percent:.1f}% explored ({total_cells - nan_count}/{total_cells} cells)")
        
        # Log window coordinates in detail
        if window_coords:
            (min_row, min_col), (max_row, max_col) = window_coords
            window_height = max_row - min_row + 1
            window_width = max_col - min_col + 1
            logger.info(f"Search window: {window_height}x{window_width} at ({min_row},{min_col}) to ({max_row},{max_col})")
            
            # Log the actual window content
            window_content = grid[min_row:max_row+1, min_col:max_col+1]
            logger.info(f"Window content:\n{np.array2string(window_content, precision=1, separator=', ')}")
    
    # Mark already found magic squares with inf to help search strategies avoid them
    marked_grid = grid.copy()
    
    if found_squares:
        logger.info(f"Attempting to mark {len(found_squares)} found squares in the grid")
        for idx, square in enumerate(found_squares):
            square_type = square.get('type')
            position = square.get('position')
            
            if verbose_logging:
                logger.info(f"Processing found square {idx+1}: type={square_type}, position={position}, full_square={square}")
            
            if square_type and position:
                try:
                    row_start, col_start = position
                    size = 3 if square_type == '3x3' else 5
                    
                    # Make sure position is valid
                    if not isinstance(row_start, int) or not isinstance(col_start, int):
                        logger.error(f"Invalid position format: {position}, expected two integers but got {type(row_start)} and {type(col_start)}")
                        continue
                    
                    # Check if this square is within our current grid bounds
                    if (row_start >= 0 and row_start + size <= marked_grid.shape[0] and 
                        col_start >= 0 and col_start + size <= marked_grid.shape[1]):
                        
                        # Mark the center of the found magic square with inf
                        center_row = row_start + size // 2
                        center_col = col_start + size // 2
                        
                        # Mark the entire magic square with inf, not just the center
                        for r in range(row_start, row_start + size):
                            for c in range(col_start, col_start + size):
                                marked_grid[r, c] = np.inf
                        
                        if verbose_logging:
                            logger.info(f"Marked entire {square_type} magic square at position ({row_start}, {col_start}) to ({row_start + size - 1}, {col_start + size - 1})")
                            # Extract the marked square to verify in logs
                            marked_region = marked_grid[row_start:row_start + size, col_start:col_start + size]
                            logger.info(f"Magic square region marked with inf:\n{np.array2string(marked_region, precision=1)}")
                    else:
                        if verbose_logging:
                            logger.warning(f"Square at position {position} with size {size} is out of grid bounds {marked_grid.shape}")
                except Exception as e:
                    logger.error(f"Error while marking square {square}: {str(e)}")
            else:
                if verbose_logging:
                    logger.warning(f"Invalid square format, missing type or position: {square}")
    
    # Add a visual grid representation when verbose logging is enabled (MOVED HERE after all squares are marked)
    if verbose_logging and window_coords:
        # Create a visual representation of the grid with window position and found squares
        rows, cols = grid.shape
        # Limit the visual grid to a reasonable size
        max_visual_size = 20
        if rows <= max_visual_size and cols <= max_visual_size:
            visual_grid = []
            for r in range(rows):
                row_chars = []
                for c in range(cols):
                    # Default character is '.' for unexplored
                    cell_char = '.'
                    # Check if this cell is in the current window
                    if (r >= min_row and r <= max_row and 
                        c >= min_col and c <= max_col):
                        cell_char = 'O'  # 'O' for visible cell
                    # Check if this is the center of the window
                    if r == window_center_row and c == window_center_col:
                        cell_char = 'C'  # 'C' for center
                    # Check if this cell is marked as inf (found magic square)
                    if np.isinf(marked_grid[r, c]):
                        cell_char = 'M'  # 'M' for magic square
                    row_chars.append(cell_char)
                visual_grid.append(''.join(row_chars))
            logger.info("Visual grid representation (. = unexplored, O = visible, C = center, M = magic square):\n" + 
                       '\n'.join(visual_grid))
    
    # Log the marked grid if any squares have been marked
    if verbose_logging and found_squares:
        logger.info(f"Marked grid with inf for found squares:\n{np.array2string(marked_grid, precision=1, separator=', ', threshold=100, edgeitems=3)}")
        
        # Count the number of inf values to verify marking worked
        inf_count = np.sum(np.isinf(marked_grid))
        if inf_count > 0:
            logger.info(f"Successfully marked {inf_count} found squares with inf values")
        else:
            logger.warning("No inf values were placed in the grid despite having found squares")
    
    # Check if this is a magic square (common across all strategies)
    is_magic, square_info = _is_magic_square(grid, found_squares, verbose_logging)
    
    # Choose the search strategy based on the parameter
    if strategy == 'random_walk':
        next_direction = _random_walk_strategy(marked_grid, window_coords, False)  # Pass False to disable function-level logging
    elif strategy == 'spiral':
        next_direction = _spiral_search_strategy(marked_grid, window_coords, False)
    elif strategy == 'pattern_detection':
        next_direction = _pattern_detection_strategy(marked_grid, window_coords, False)
    elif strategy.startswith('custom_'):
        try:
            next_direction = _execute_custom_strategy(strategy, marked_grid, window_coords, False)
        except Exception as e:
            logger.error(f"Error executing custom strategy {strategy}: {str(e)}")
            next_direction = _exploration_strategy(marked_grid, window_coords, False)  # Fallback to default
    else:  # Default to exploration strategy
        next_direction = _exploration_strategy(marked_grid, window_coords, False)
    
    if verbose_logging:
        logger.info(f"Using strategy '{strategy}', chose direction: {next_direction}")
        
    return is_magic, next_direction, square_info


def _execute_custom_strategy(strategy_id, grid, window_coords=None, verbose_logging=False):
    """
    Execute a custom search strategy.
    
    Args:
        strategy_id: The ID of the custom strategy to execute
        grid: 2D numpy array representing the current visible grid
        window_coords: Tuple of ((min_row, min_col), (max_row, max_col)) representing the current view window
        verbose_logging: Boolean flag to enable detailed logging
        
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    # This assumes the app.py file has already loaded the custom strategy
    # and created a temporary Python file
    temp_filename = f"temp_strategy_{strategy_id}.py"
    
    if not os.path.exists(temp_filename):
        logger.error(f"Custom strategy file not found: {temp_filename}")
        return _exploration_strategy(grid, window_coords, verbose_logging)  # Fallback
    
    try:
        # Dynamically import the temporary module if not already in sys.modules
        if f"temp_strategy_{strategy_id}" not in sys.modules:
            spec = importlib.util.spec_from_file_location(f"temp_strategy_{strategy_id}", temp_filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"temp_strategy_{strategy_id}"] = module
            spec.loader.exec_module(module)
        else:
            module = sys.modules[f"temp_strategy_{strategy_id}"]
        
        # Get the function name from the module
        function_name = None
        for name in dir(module):
            if name.startswith('_') or name == 'validate' or not callable(getattr(module, name)):
                continue
            if name != 'np' and name != 'validate':  # Skip imported modules and validation function
                function_name = name
                break
        
        if not function_name:
            logger.error(f"Could not find function in custom strategy module")
            return _exploration_strategy(grid, window_coords, verbose_logging)  # Fallback
            
        # Get the function
        strategy_func = getattr(module, function_name)
        
        if verbose_logging:
            logger.info(f"Executing custom strategy: {function_name} from {strategy_id}")
        
        # Check if the function accepts window_coords parameter
        try:
            # Call the function with window_coords if possible
            if window_coords:
                direction = strategy_func(grid, window_coords)
            else:
                direction = strategy_func(grid)
        except TypeError:
            # Fall back to original call if the custom function doesn't accept window_coords
            direction = strategy_func(grid)
        
        # Validate the result
        if direction not in ['up', 'down', 'left', 'right']:
            logger.error(f"Custom strategy returned invalid direction: {direction}")
            return _exploration_strategy(grid, window_coords, verbose_logging)  # Fallback
        
        if verbose_logging:
            logger.info(f"Custom strategy {function_name} decided to go: {direction}")
            
        return direction
        
    except Exception as e:
        logger.error(f"Error executing custom strategy: {str(e)}")
        return _exploration_strategy(grid, window_coords, verbose_logging)  # Fallback


def _is_magic_square(grid, found_squares, verbose_logging=False):
    """
    Check if the grid contains a magic square.
    
    Args:
        grid: 2D numpy array representing the current visible grid (with NaN for unexplored areas)
        found_squares: List of previously found magic square positions
        verbose_logging: Boolean flag to enable detailed logging
        
    Returns:
        tuple: (is_magic, square_info)
               is_magic: Boolean indicating if any magic square was found
               square_info: Dictionary with information about the found magic square, or None
    """
    # Get grid dimensions
    rows, cols = grid.shape
    logger.info(f"Scanning grid of shape {grid.shape} for magic squares")
    
    if verbose_logging:
        logger.info(f"Starting magic square scan on {rows}x{cols} grid")
        logger.info(f"Number of previously found squares: {len(found_squares)}")
    
    # Look for 3x3 magic squares
    for row_start in range(rows - 2):
        for col_start in range(cols - 2):
            # Create a unique identifier for this potential magic square
            square_id = {'type': '3x3', 'position': (row_start, col_start)}
            
            # Skip if we've already found this magic square
            if square_id in found_squares:
                continue
                
            sub_grid = grid[row_start:row_start+3, col_start:col_start+3]
            
            # Skip if there are any NaN values
            if np.any(np.isnan(sub_grid)):
                continue
            
            # Check if this 3x3 grid is a magic square
            if _check_3x3_magic(sub_grid, verbose_logging):
                logger.info(f"Found 3x3 magic square at position ({row_start}, {col_start})")
                if verbose_logging:
                    logger.info(f"Magic square contents:\n{sub_grid}")
                    logger.info(f"Returning square_id: {square_id} with type {type(square_id)} and position {square_id['position']} of type {type(square_id['position'])}")
                return True, square_id
    
    # Look for 5x5 magic squares (if grid is large enough)
    if rows >= 5 and cols >= 5:
        for row_start in range(rows - 4):
            for col_start in range(cols - 4):
                # Create a unique identifier for this potential magic square
                square_id = {'type': '5x5', 'position': (row_start, col_start)}
                
                # Skip if we've already found this magic square
                if square_id in found_squares:
                    continue
                    
                sub_grid = grid[row_start:row_start+5, col_start:col_start+5]
                
                # Skip if there are any NaN values
                if np.any(np.isnan(sub_grid)):
                    continue
                
                # Check if this 5x5 grid is a magic square
                if _check_5x5_magic(sub_grid, verbose_logging):
                    logger.info(f"Found 5x5 magic square at position ({row_start}, {col_start})")
                    if verbose_logging:
                        logger.info(f"Magic square contents:\n{sub_grid}")
                        logger.info(f"Returning square_id: {square_id} with type {type(square_id)} and position {square_id['position']} of type {type(square_id['position'])}")
                    return True, square_id
    
    return False, None


def _check_3x3_magic(grid, verbose_logging=False):
    """
    Check if a 3x3 grid is a magic square.
    
    Args:
        grid: 3x3 numpy array
        verbose_logging: Boolean flag to enable detailed logging
        
    Returns:
        bool: True if it's a magic square, False otherwise
    """
    # Magic square property: all rows, columns, and diagonals sum to the same value
    target_sum = np.sum(grid[0, :])  # Sum of first row
    
    # Check rows
    for i in range(3):
        row_sum = np.sum(grid[i, :])
        if row_sum != target_sum:
            return False
    
    # Check columns
    for i in range(3):
        col_sum = np.sum(grid[:, i])
        if col_sum != target_sum:
            return False
    
    # Check main diagonal
    diag_sum = np.sum(np.diag(grid))
    if diag_sum != target_sum:
        return False
    
    # Check other diagonal
    other_diag_sum = np.sum(np.diag(np.fliplr(grid)))
    if other_diag_sum != target_sum:
        return False
    
    # Magic square found
    logger.info(f"Valid 3x3 magic square found with target sum: {target_sum}")
    
    return True


def _check_5x5_magic(grid, verbose_logging=False):
    """
    Check if a 5x5 grid is a magic square.
    
    Args:
        grid: 5x5 numpy array
        verbose_logging: Boolean flag to enable detailed logging
        
    Returns:
        bool: True if it's a magic square, False otherwise
    """
    # Magic square property: all rows, columns, and diagonals sum to the same value
    target_sum = np.sum(grid[0, :])  # Sum of first row
    
    # Check rows
    for i in range(5):
        row_sum = np.sum(grid[i, :])
        if row_sum != target_sum:
            return False
    
    # Check columns
    for i in range(5):
        col_sum = np.sum(grid[:, i])
        if col_sum != target_sum:
            return False
    
    # Check main diagonal
    diag_sum = np.sum(np.diag(grid))
    if diag_sum != target_sum:
        return False
    
    # Check other diagonal
    other_diag_sum = np.sum(np.diag(np.fliplr(grid)))
    if other_diag_sum != target_sum:
        return False
    
    # Magic square found
    logger.info(f"Valid 5x5 magic square found with target sum: {target_sum}")
    
    return True


def _exploration_strategy(grid, window_coords=None, verbose_logging=False):
    """
    Strategy 1: Exploration priority - Seeks unexplored areas (areas with NaN values)
    
    Args:
        grid: 2D numpy array representing the current visible grid
        window_coords: Tuple of ((min_row, min_col), (max_row, max_col)) representing the current view window
        verbose_logging: Boolean flag to enable detailed logging
        
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    # For any grid size, we'll check the edges to determine direction
    rows, cols = grid.shape
    
    directions = ['up', 'right', 'down', 'left']
    scores = {dir: 0 for dir in directions}
    
    # Get edges based on current position
    top_edge = grid[0, :]
    bottom_edge = grid[rows-1, :]
    left_edge = grid[:, 0]
    right_edge = grid[:, cols-1]
    
    # Count NaN values in each direction
    none_count_up = np.sum(np.isnan(top_edge))
    none_count_down = np.sum(np.isnan(bottom_edge))
    none_count_left = np.sum(np.isnan(left_edge))
    none_count_right = np.sum(np.isnan(right_edge))
    
    # Assign scores based on unexplored cells
    scores['up'] = none_count_up * 3
    scores['down'] = none_count_down * 3
    scores['left'] = none_count_left * 3
    scores['right'] = none_count_right * 3
    
    # Penalize directions with no unexplored cells
    if none_count_up == 0:
        scores['up'] -= 5
    if none_count_down == 0:
        scores['down'] -= 5
    if none_count_left == 0:
        scores['left'] -= 5
    if none_count_right == 0:
        scores['right'] -= 5
    
    # Penalize directions with inf values (found magic squares)
    if np.any(np.isinf(top_edge)):
        scores['up'] -= 15
    if np.any(np.isinf(bottom_edge)):
        scores['down'] -= 15
    if np.any(np.isinf(left_edge)):
        scores['left'] -= 15
    if np.any(np.isinf(right_edge)):
        scores['right'] -= 15
    
    # NEW: Take window position into account - heavily penalize directions that would move off the grid
    if window_coords:
        (min_row, min_col), (max_row, max_col) = window_coords
        
        # Penalize up direction if already at top edge
        if min_row == 0:
            scores['up'] -= 50
            
        # Penalize down direction if already at bottom edge
        if max_row >= rows - 1:
            scores['down'] -= 50
            
        # Penalize left direction if already at left edge
        if min_col == 0:
            scores['left'] -= 50
            
        # Penalize right direction if already at right edge  
        if max_col >= cols - 1:
            scores['right'] -= 50
    
    # Add small randomness to avoid getting stuck
    for dir in directions:
        random_factor = np.random.randint(0, 3)
        scores[dir] += random_factor
    
    if verbose_logging:
        logger.info(f"Direction scores: {scores}")
        
    # Return direction with highest score
    best_direction = max(scores, key=scores.get)
    return best_direction


def _random_walk_strategy(grid, window_coords=None, verbose_logging=False):
    """
    Strategy 2: Random walk with bias - Uses mostly randomness with a bias 
    toward less explored areas
    
    Args:
        grid: 2D numpy array representing the current visible grid
        window_coords: Tuple of ((min_row, min_col), (max_row, max_col)) representing the current view window
        verbose_logging: Boolean flag to enable detailed logging
        
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    rows, cols = grid.shape
    directions = ['up', 'right', 'down', 'left']
    
    # Check boundaries to avoid going off the grid
    if not np.any(np.isnan(grid[0, :])):  # Top row is fully visible
        directions.remove('up') if 'up' in directions else None
    if not np.any(np.isnan(grid[rows-1, :])):  # Bottom row is fully visible
        directions.remove('down') if 'down' in directions else None
    if not np.any(np.isnan(grid[:, 0])):  # Leftmost column is fully visible
        directions.remove('left') if 'left' in directions else None
    if not np.any(np.isnan(grid[:, cols-1])):  # Rightmost column is fully visible
        directions.remove('right') if 'right' in directions else None
    
    # NEW: Remove directions that would go off the grid edge based on window position
    if window_coords:
        (min_row, min_col), (max_row, max_col) = window_coords
        
        # Remove 'up' if already at top edge
        if min_row == 0 and 'up' in directions:
            directions.remove('up')
            
        # Remove 'down' if already at bottom edge
        if max_row >= rows - 1 and 'down' in directions:
            directions.remove('down')
            
        # Remove 'left' if already at left edge
        if min_col == 0 and 'left' in directions:
            directions.remove('left')
            
        # Remove 'right' if already at right edge  
        if max_col >= cols - 1 and 'right' in directions:
            directions.remove('right')
    
    # If all directions are blocked, just choose a random direction
    if not directions:
        directions = ['up', 'right', 'down', 'left']
        if verbose_logging:
            logger.info("All directions blocked, using all directions with random weights")
    
    # Add weighted randomness with bias toward directions with more NaN values
    scores = {dir: np.random.random() * 0.7 for dir in directions}  # 70% randomness
    
    # Add 30% weight based on unexplored cells
    for dir in directions:
        if dir == 'up' and 'up' in scores:
            scores['up'] += 0.3 * np.sum(np.isnan(grid[0, :])) / cols
        elif dir == 'down' and 'down' in scores:
            scores['down'] += 0.3 * np.sum(np.isnan(grid[rows-1, :])) / cols
        elif dir == 'left' and 'left' in scores:
            scores['left'] += 0.3 * np.sum(np.isnan(grid[:, 0])) / rows
        elif dir == 'right' and 'right' in scores:
            scores['right'] += 0.3 * np.sum(np.isnan(grid[:, cols-1])) / rows
    
    # Avoid directions with inf values (found magic squares)
    if 'up' in scores and np.any(np.isinf(grid[0, :])):
        scores['up'] -= 1.0
    if 'down' in scores and np.any(np.isinf(grid[rows-1, :])):
        scores['down'] -= 1.0
    if 'left' in scores and np.any(np.isinf(grid[:, 0])):
        scores['left'] -= 1.0
    if 'right' in scores and np.any(np.isinf(grid[:, cols-1])):
        scores['right'] -= 1.0
    
    if verbose_logging:
        logger.info(f"Random walk scores: {scores}")
        
    # Return direction with highest score
    return max(scores, key=scores.get)


def _spiral_search_strategy(grid, window_coords=None, verbose_logging=False):
    """
    Strategy 3: Spiral search - Tries to explore in a spiral pattern 
    to systematically cover the board
    
    Args:
        grid: 2D numpy array representing the current visible grid
        window_coords: Tuple of ((min_row, min_col), (max_row, max_col)) representing the current view window
        verbose_logging: Boolean flag to enable detailed logging
        
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    rows, cols = grid.shape
    
    # First check if we can determine our position in the grid
    # by counting NaN values in each edge
    top_none = np.sum(np.isnan(grid[0, :]))
    bottom_none = np.sum(np.isnan(grid[rows-1, :]))
    left_none = np.sum(np.isnan(grid[:, 0]))
    right_none = np.sum(np.isnan(grid[:, cols-1]))
    
    # Create a simple spiral pattern based on the current state
    # The pattern is: right → down → left → up → right → ...
    
    # Get the window position to avoid moving off the grid
    at_top_edge = False
    at_bottom_edge = False
    at_left_edge = False
    at_right_edge = False
    
    if window_coords:
        (min_row, min_col), (max_row, max_col) = window_coords
        at_top_edge = (min_row == 0)
        at_bottom_edge = (max_row >= rows - 1)
        at_left_edge = (min_col == 0)
        at_right_edge = (max_col >= cols - 1)
    
    # No NaNs on the right edge means we should go down next
    if right_none == 0 and bottom_none > 0 and not at_bottom_edge:
        # Check for inf values (found magic squares) in bottom direction
        if not np.any(np.isinf(grid[rows-1, :])):
            return 'down'
    # No NaNs on the bottom edge means we should go left next
    elif bottom_none == 0 and left_none > 0 and not at_left_edge:
        # Check for inf values (found magic squares) in left direction
        if not np.any(np.isinf(grid[:, 0])):
            return 'left'
    # No NaNs on the left edge means we should go up next
    elif left_none == 0 and top_none > 0 and not at_top_edge:
        # Check for inf values (found magic squares) in up direction
        if not np.any(np.isinf(grid[0, :])):
            return 'up'
    # No NaNs on the top edge means we should go right next
    elif top_none == 0 and right_none > 0 and not at_right_edge:
        # Check for inf values (found magic squares) in right direction
        if not np.any(np.isinf(grid[:, cols-1])):
            return 'right'
    
    # If the spiral pattern isn't clear, fall back to the exploration strategy
    # with a bias toward completing the current spiral movement
    
    # Count NaN values on each edge to determine where to go
    directions = ['up', 'right', 'down', 'left']
    scores = {
        'up': top_none * 2,
        'right': right_none * 2,
        'down': bottom_none * 2,
        'left': left_none * 2
    }
    
    # Penalize directions with inf values (found magic squares)
    if np.any(np.isinf(grid[0, :])):
        scores['up'] -= 10
    if np.any(np.isinf(grid[rows-1, :])):
        scores['down'] -= 10
    if np.any(np.isinf(grid[:, 0])):
        scores['left'] -= 10
    if np.any(np.isinf(grid[:, cols-1])):
        scores['right'] -= 10
    
    # Penalize directions that would go off the grid
    if at_top_edge:
        scores['up'] -= 50
    if at_bottom_edge:
        scores['down'] -= 50
    if at_left_edge:
        scores['left'] -= 50
    if at_right_edge:
        scores['right'] -= 50
    
    # Add small randomness to avoid getting stuck
    for dir in directions:
        scores[dir] += np.random.randint(0, 2)
    
    if verbose_logging:
        logger.info(f"Spiral search scores: {scores}")
        
    return max(scores, key=scores.get)


def _pattern_detection_strategy(grid, window_coords=None, verbose_logging=False):
    """
    Strategy: Pattern detection - Prioritizes unexplored areas but also looks for
    patterns in row and column sums that might indicate magic squares.
    
    Specifically checks:
    - The two rightmost columns for matching sums
    - The two top rows for matching sums
    - The two bottom rows for matching sums
    - If window coordinates are provided, checks for potential magic squares within the window
    
    Args:
        grid: 2D numpy array representing the current visible grid
        window_coords: Tuple of ((min_row, min_col), (max_row, max_col)) representing the current view window
        verbose_logging: Boolean flag to enable detailed logging
        
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    rows, cols = grid.shape
    directions = ['up', 'right', 'down', 'left']
    scores = {dir: 0 for dir in directions}
    
    # First, prioritize unexplored areas (base exploration strategy)
    # Get edges based on current position
    top_edge = grid[0, :]
    bottom_edge = grid[rows-1, :]
    left_edge = grid[:, 0]
    right_edge = grid[:, cols-1]
    
    # Count NaN values in each direction
    none_count_up = np.sum(np.isnan(top_edge))
    none_count_down = np.sum(np.isnan(bottom_edge))
    none_count_left = np.sum(np.isnan(left_edge))
    none_count_right = np.sum(np.isnan(right_edge))
    
    # Assign base scores based on unexplored cells
    scores['up'] = none_count_up * 2
    scores['down'] = none_count_down * 2
    scores['left'] = none_count_left * 2
    scores['right'] = none_count_right * 2
    
    # Penalize directions with no unexplored cells
    if none_count_up == 0:
        scores['up'] -= 5
    if none_count_down == 0:
        scores['down'] -= 5
    if none_count_left == 0:
        scores['left'] -= 5
    if none_count_right == 0:
        scores['right'] -= 5
    
    # Avoid directions with inf values (found magic squares)
    if np.any(np.isinf(top_edge)):
        scores['up'] -= 15
    if np.any(np.isinf(bottom_edge)):
        scores['down'] -= 15
    if np.any(np.isinf(left_edge)):
        scores['left'] -= 15
    if np.any(np.isinf(right_edge)):
        scores['right'] -= 15
    
    # Avoid going off the grid edge based on window position
    if window_coords:
        (min_row, min_col), (max_row, max_col) = window_coords
        
        # Penalize 'up' if already at top edge
        if min_row == 0:
            scores['up'] -= 50
            
        # Penalize 'down' if already at bottom edge
        if max_row >= rows - 1:
            scores['down'] -= 50
            
        # Penalize 'left' if already at left edge
        if min_col == 0:
            scores['left'] -= 50
            
        # Penalize 'right' if already at right edge  
        if max_col >= cols - 1:
            scores['right'] -= 50
    
    # Now add the pattern detection logic
    
    # Check for column pattern: compare the two rightmost visible columns
    if cols >= 2 and none_count_right == 0:
        # Only check if both columns are fully visible (no NaN values)
        if not np.any(np.isnan(grid[:, -1])) and not np.any(np.isnan(grid[:, -2])):
            rightmost_sum = np.nansum(grid[:, -1])
            second_rightmost_sum = np.nansum(grid[:, -2])
            # If sums are close (within 10% of each other), favor going right
            if abs(rightmost_sum - second_rightmost_sum) < 0.1 * rightmost_sum:
                scores['right'] += 8
    
    # Check for top row pattern: compare the two topmost visible rows
    if rows >= 2 and none_count_up == 0:
        # Only check if both rows are fully visible (no NaN values)
        if not np.any(np.isnan(grid[0, :])) and not np.any(np.isnan(grid[1, :])):
            top_row_sum = np.nansum(grid[0, :])
            second_row_sum = np.nansum(grid[1, :])
            # If sums are close (within 10% of each other), favor going up
            if abs(top_row_sum - second_row_sum) < 0.1 * top_row_sum:
                scores['up'] += 8
    
    # Check for bottom row pattern: compare the two bottommost visible rows
    if rows >= 2 and none_count_down == 0:
        # Only check if both rows are fully visible (no NaN values)
        if not np.any(np.isnan(grid[-1, :])) and not np.any(np.isnan(grid[-2, :])):
            bottom_row_sum = np.nansum(grid[-1, :])
            second_bottom_row_sum = np.nansum(grid[-2, :])
            # If sums are close (within 10% of each other), favor going down
            if abs(bottom_row_sum - second_bottom_row_sum) < 0.1 * bottom_row_sum:
                scores['down'] += 8
    
    # Similarly, check left columns for patterns
    if cols >= 2 and none_count_left == 0:
        if not np.any(np.isnan(grid[:, 0])) and not np.any(np.isnan(grid[:, 1])):
            leftmost_sum = np.nansum(grid[:, 0])
            second_leftmost_sum = np.nansum(grid[:, 1])
            if abs(leftmost_sum - second_leftmost_sum) < 0.1 * leftmost_sum:
                scores['left'] += 8
    
    # If window coordinates are provided, use them for advanced pattern detection
    if window_coords:
        (min_row, min_col), (max_row, max_col) = window_coords
        window_height = max_row - min_row + 1
        window_width = max_col - min_col + 1
        
        # Check for potential 3x3 magic squares within the visible window
        if window_height >= 3 and window_width >= 3:
            # Get visible grid section
            visible_grid = grid[min_row:max_row+1, min_col:max_col+1]
            
            # Check if we have enough valid (non-NaN) values to analyze
            valid_cells = np.sum(~np.isnan(visible_grid))
            if valid_cells >= 6:  # Need at least 6 values to detect patterns
                
                # Check row sums
                row_sums = np.nansum(visible_grid, axis=1)
                valid_rows = np.sum(~np.isnan(row_sums))
                if valid_rows >= 2:
                    # Compare adjacent row sums
                    for i in range(valid_rows - 1):
                        if (not np.isnan(row_sums[i]) and not np.isnan(row_sums[i+1]) and 
                            abs(row_sums[i] - row_sums[i+1]) < 0.1 * row_sums[i]):
                            # Detect if pattern suggests moving up or down
                            if i == 0 and none_count_up > 0:
                                scores['up'] += 10
                            elif i == valid_rows - 2 and none_count_down > 0:
                                scores['down'] += 10
                
                # Check column sums
                col_sums = np.nansum(visible_grid, axis=0)
                valid_cols = np.sum(~np.isnan(col_sums))
                if valid_cols >= 2:
                    # Compare adjacent column sums
                    for i in range(valid_cols - 1):
                        if (not np.isnan(col_sums[i]) and not np.isnan(col_sums[i+1]) and 
                            abs(col_sums[i] - col_sums[i+1]) < 0.1 * col_sums[i]):
                            # Detect if pattern suggests moving left or right
                            if i == 0 and none_count_left > 0:
                                scores['left'] += 10
                            elif i == valid_cols - 2 and none_count_right > 0:
                                scores['right'] += 10
    
    # Add small randomness to avoid getting stuck
    for dir in directions:
        scores[dir] += np.random.randint(0, 3)
    
    if verbose_logging:
        logger.info(f"Pattern detection scores: {scores}")
        
    # Return direction with highest score
    return max(scores, key=scores.get) 