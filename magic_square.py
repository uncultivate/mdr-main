import numpy as np
import logging
import importlib.util
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_next_direction(grid, strategy='random_walk', found_squares=None, window_coords=None, verbose_logging=False):
    """
    Function to determine next direction based on the chosen strategy.
    Does NOT perform magic square detection.
    
    Args:
        grid: 2D numpy array representing the current visible grid (with NaN for unexplored areas)
        strategy: String indicating which search strategy to use
        found_squares: List of previously found magic square positions to avoid re-detection
        window_coords: Tuple of ((min_row, min_col), (max_row, max_col)) representing the current view window
        verbose_logging: Boolean flag to enable detailed logging
        
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    try:
        if found_squares is None:
            found_squares = []
            
        # Validate grid is a proper 2D numpy array
        if not isinstance(grid, np.ndarray) or len(grid.shape) != 2:
            logger.error(f"Invalid grid format in calculate_next_direction: shape={getattr(grid, 'shape', 'None')}")
            return "right"
            
        # Convert None values to NaN if present
        grid = np.array(grid, dtype=float)
        grid = np.where(grid == None, np.nan, grid)
        
        logger.info(f"Calculating next direction for grid shape: {grid.shape}")
        
        # Validate window_coords before using
        valid_window_coords = False
        if window_coords:
            try:
                if (isinstance(window_coords, (list, tuple)) and len(window_coords) == 2 and
                    isinstance(window_coords[0], (list, tuple)) and len(window_coords[0]) == 2 and
                    isinstance(window_coords[1], (list, tuple)) and len(window_coords[1]) == 2):
                    valid_window_coords = True
                else:
                    logger.error(f"Invalid window_coords format: {window_coords}")
            except Exception as e:
                logger.error(f"Error validating window_coords: {str(e)}")
                
        if valid_window_coords:
            (min_row, min_col), (max_row, max_col) = window_coords
            window_height = max_row - min_row + 1
            window_width = max_col - min_col + 1
            window_center_row = min_row + window_height // 2
            window_center_col = min_col + window_width // 2
            logger.info(f"WINDOW POSITION: Center=({window_center_row},{window_center_col}), Bounds=({min_row},{min_col}) to ({max_row},{max_col})")
        
        if verbose_logging:
            logger.info(f"Grid exploration status: {100 * (1 - np.sum(np.isnan(grid)) / grid.size):.1f}% explored")
            if valid_window_coords:
                 logger.info(f"Search window: {window_height}x{window_width} at ({min_row},{min_col}) to ({max_row},{max_col})")
    
        # Mark already found magic squares with inf to help search strategies avoid them
        marked_grid = grid.copy()
        if found_squares:
            for idx, square in enumerate(found_squares):
                square_type = square.get('type')
                position = square.get('position')
                if square_type and position:
                    try:
                        row_start, col_start = position
                        size = 3 if square_type == '3x3' else 5
                        if (row_start >= 0 and row_start + size <= marked_grid.shape[0] and 
                            col_start >= 0 and col_start + size <= marked_grid.shape[1]):
                            marked_grid[row_start:row_start+size, col_start:col_start+size] = np.inf
                            if verbose_logging:
                                logger.info(f"Marked revealed square {idx+1}: type={square_type} at {position}")
                    except Exception as e:
                        logger.error(f"Error marking revealed square {square}: {str(e)}")
        
        # Strategy selection based on the strategy name
        if strategy == 'random_walk':
            next_direction = _random_walk_strategy(marked_grid, window_coords if valid_window_coords else None, verbose_logging)
        elif strategy == 'directed_exploration':
            next_direction = _directed_exploration_strategy(marked_grid, window_coords if valid_window_coords else None, verbose_logging)
        elif strategy == 'adam_l':
            next_direction = _adam_l_strategy(marked_grid, window_coords if valid_window_coords else None, verbose_logging)
        elif strategy == 'meredith_n':
            next_direction = _meredith_n_zigzag(marked_grid, window_coords if valid_window_coords else None, verbose_logging)
        elif strategy == 'james_c':
            next_direction = _james_c_proximity(marked_grid, window_coords if valid_window_coords else None, verbose_logging)
        elif strategy == 'jake_p':
            next_direction = _jake_p_snake(marked_grid, window_coords if valid_window_coords else None, verbose_logging)
        elif strategy == 'jono_s':
            next_direction = _jono_s_vacuum(marked_grid, window_coords if valid_window_coords else None, verbose_logging)
        elif strategy == 'zigzag_search':
            next_direction = _zigzag_search(marked_grid, window_coords if valid_window_coords else None, verbose_logging)
        else:
            # Check if it's a custom strategy ID
            if strategy.startswith('custom_'):
                logger.info(f"Attempting to use custom strategy: {strategy}")
                next_direction = _execute_custom_strategy(strategy, marked_grid, window_coords if valid_window_coords else None, verbose_logging)
            else:
                # Fallback to random walk if strategy is unknown and not custom
                logger.warning(f"Unknown strategy: {strategy}, falling back to random walk")
                next_direction = _random_walk_strategy(marked_grid, window_coords if valid_window_coords else None, verbose_logging)
        
        if verbose_logging:
            logger.info(f"Using strategy '{strategy}', chose direction: {next_direction}")
            
        return next_direction

    except Exception as e:
        logger.error(f"Error in calculate_next_direction: {str(e)}")
        return "right" # Fallback direction


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
        return _random_walk_strategy(grid, window_coords, verbose_logging)  # Fallback
    
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
            return _random_walk_strategy(grid, window_coords, verbose_logging)  # Fallback
            
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
            return _random_walk_strategy(grid, window_coords, verbose_logging)  # Fallback
        
        if verbose_logging:
            logger.info(f"Custom strategy {function_name} decided to go: {direction}")
            
        return direction
        
    except Exception as e:
        logger.error(f"Error executing custom strategy: {str(e)}")
        return _random_walk_strategy(grid, window_coords, verbose_logging)  # Fallback


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


def _adam_l_strategy(grid, window_coords=None, verbose_logging=False):
    """
    Custom search strategy by Adam L (Blackjack inspired).
    Args:
        grid: 2D numpy array with NaN for unexplored cells and inf for known magic squares.
        window_coords: Tuple containing window coordinates ((min_row, min_col), (max_row, max_col)). Required.
        verbose_logging: Boolean flag (currently unused in this specific function).
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    import numpy as np

    if window_coords is None:
        logger.warning("Adam L strategy requires window_coords, falling back to random walk.")
        return _random_walk_strategy(grid, None, verbose_logging)

    # Extract window coordinates
    (min_row, min_col), (max_row, max_col) = window_coords

    # Get current window dimensions and center
    window_height = max_row - min_row + 1
    window_width = max_col - min_col + 1
    center_row = min_row + window_height // 2
    center_col = min_col + window_width // 2

    # Get number of rows and columns in the grid
    rows, cols = grid.shape

    # --- Safely get neighbor values --- 
    options = {}
    # Up
    if center_row > 0:
        options['up'] = grid[center_row-1, center_col]
    # Down
    if center_row < rows - 1:
       options['down'] = grid[center_row+1, center_col]
    # Left
    if center_col > 0:
       options['left'] = grid[center_row, center_col-1]
    # Right
    if center_col < cols - 1:
       options['right'] = grid[center_row, center_col+1]
       
    # Remove directions blocked by grid edges (redundant with checks above, but safe)
    if min_row <= 0:
        if 'up' in options: del(options['up'])
    if max_row >= rows-1:
        if 'down' in options: del(options['down'])
    if min_col <= 0:
        if 'left' in options: del(options['left'])
    if max_col >= cols-1:
        if 'right' in options: del(options['right'])

    # Filter out NaN/Inf values and create valid options
    valid_options = {d: v for d, v in options.items() if v is not None and not np.isnan(v) and not np.isinf(v)}

    if not valid_options:
        # Fallback if no valid moves (e.g., surrounded by NaNs/Infs or edges)
        logger.info("Adam L: No valid moves found, choosing random fallback.")
        all_directions = [d for d in ['up', 'down', 'left', 'right'] 
                          if (d != 'up' or min_row > 0) and 
                             (d != 'down' or max_row < rows-1) and 
                             (d != 'left' or min_col > 0) and 
                             (d != 'right' or max_col < cols-1)]
        return np.random.choice(all_directions) if all_directions else 'right'

    # --- Blackjack Logic --- 
    hit_me = np.random.randint(low=1, high=12)
    print(f'Adam L drew a {hit_me}!') # Use print for user feedback in this strategy

    go = np.random.choice(list(valid_options.keys())) # Default random choice from valid
    best = 99999
    for direction in valid_options:
        number = valid_options[direction]
        try:
            numeric_number = float(number)
            diff = 21 - numeric_number - hit_me
            print(f'Adam L: {direction} ({numeric_number}) gets you {diff} from 21')
            if diff < 0:
                print('Adam L: Bust!')
                continue
            elif diff < best:
                best = diff
                go = direction
        except (ValueError, TypeError):
            print(f"Adam L: Skipping direction {direction} due to invalid number: {number}")
            continue

    print(f'Adam L: Going {go}')
    return go


def _meredith_n_zigzag(grid, window_coords=None, verbose_logging=False):
    """
    Custom search strategy by Meredith N (Inner Zigzag).
    Attempts a zigzag pattern while trying to avoid grid borders.
    Args:
        grid: 2D numpy array with NaN/inf for unexplored/revealed squares.
        window_coords: Tuple ((min_row, min_col), (max_row, max_col)). Required.
        verbose_logging: Boolean flag.
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    import numpy as np

    if window_coords is None:
        logger.warning("Meredith N strategy requires window_coords, falling back to random walk.")
        return _random_walk_strategy(grid, None, verbose_logging)

    # Extract window coordinates
    (min_row, min_col), (max_row, max_col) = window_coords

    # Get current window dimensions and center
    window_height = max_row - min_row + 1
    window_width = max_col - min_col + 1
    center_row = min_row + window_height // 2
    center_col = min_col + window_width // 2

    # Get number of rows and columns in the grid
    rows, cols = grid.shape

    # Define inner boundaries (avoiding row/col 0 and rows-1/cols-1)
    inner_min_row = 1
    inner_max_row = rows - 2
    inner_min_col = 1
    inner_max_col = cols - 2
    
    # Check if we are within the defined inner boundaries - if not, try to move towards them
    if center_row < inner_min_row:
        return 'down'
    if center_row > inner_max_row:
        return 'up'
    if center_col < inner_min_col:
        return 'right'
    if center_col > inner_max_col:
        return 'left'

    # Inner Zigzag Logic
    if center_row % 2 == 0: # Even rows (relative to inner grid? Let's assume absolute row index)
        if center_col < inner_max_col:
            # Check if right is blocked by revealed square
            if center_col + 1 < cols and not np.isinf(grid[center_row, center_col + 1]):
                return 'right'
            elif center_row < inner_max_row and not np.isinf(grid[center_row + 1, center_col]): # If right blocked, try down
                 return 'down'
            else: # If both blocked, fallback needed
                 logger.info("Meredith N: Right/Down blocked on even row, fallback.")
                 # Simple fallback: try left if possible, else up
                 if center_col > inner_min_col and not np.isinf(grid[center_row, center_col - 1]): return 'left'
                 if center_row > inner_min_row and not np.isinf(grid[center_row - 1, center_col]): return 'up'
                 return 'down' # Last resort
        elif center_row < inner_max_row: # At right inner edge, move down if possible
             if not np.isinf(grid[center_row + 1, center_col]):
                 return 'down'
             else: # If down blocked, fallback
                 logger.info("Meredith N: Down blocked at right edge, fallback.")
                 if center_col > inner_min_col and not np.isinf(grid[center_row, center_col - 1]): return 'left'
                 if center_row > inner_min_row and not np.isinf(grid[center_row - 1, center_col]): return 'up'
                 return 'left' # Last resort
        else: # At bottom-right inner corner
             logger.info("Meredith N: At bottom-right inner corner on even row, fallback.")
             if center_col > inner_min_col and not np.isinf(grid[center_row, center_col - 1]): return 'left'
             if center_row > inner_min_row and not np.isinf(grid[center_row - 1, center_col]): return 'up'
             return 'up' # Last resort
    else: # Odd rows
        if center_col > inner_min_col:
            # Check if left is blocked
            if center_col - 1 >= 0 and not np.isinf(grid[center_row, center_col - 1]):
                return 'left'
            elif center_row < inner_max_row and not np.isinf(grid[center_row + 1, center_col]): # If left blocked, try down
                return 'down'
            else: # If both blocked, fallback
                 logger.info("Meredith N: Left/Down blocked on odd row, fallback.")
                 if center_col < inner_max_col and not np.isinf(grid[center_row, center_col + 1]): return 'right'
                 if center_row > inner_min_row and not np.isinf(grid[center_row - 1, center_col]): return 'up'
                 return 'down' # Last resort
        elif center_row < inner_max_row: # At left inner edge, move down if possible
            if not np.isinf(grid[center_row + 1, center_col]):
                return 'down'
            else: # If down blocked, fallback
                 logger.info("Meredith N: Down blocked at left edge, fallback.")
                 if center_col < inner_max_col and not np.isinf(grid[center_row, center_col + 1]): return 'right'
                 if center_row > inner_min_row and not np.isinf(grid[center_row - 1, center_col]): return 'up'
                 return 'right' # Last resort
        else: # At bottom-left inner corner
             logger.info("Meredith N: At bottom-left inner corner on odd row, fallback.")
             if center_col < inner_max_col and not np.isinf(grid[center_row, center_col + 1]): return 'right'
             if center_row > inner_min_row and not np.isinf(grid[center_row - 1, center_col]): return 'up'
             return 'up' # Last resort
             
    # Default fallback if logic above fails (shouldn't happen with inner boundary checks)
    logger.warning("Meredith N: Reached end of logic, using default fallback.")
    return _random_walk_strategy(grid, window_coords, verbose_logging)


def _james_c_proximity(grid, window_coords=None, verbose_logging=False):
    """
    Custom search strategy by James C (Proximity Explorer).
    Prioritizes moving towards the highest concentration of nearby unexplored cells (NaNs),
    weighted by inverse square distance from the window center.
    Args:
        grid: 2D numpy array with NaN/inf.
        window_coords: Tuple ((min_row, min_col), (max_row, max_col)). Required.
        verbose_logging: Boolean flag.
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    import numpy as np

    if window_coords is None:
        logger.warning("James C strategy requires window_coords, falling back to random walk.")
        return _random_walk_strategy(grid, None, verbose_logging)

    rows, cols = grid.shape
    (min_row, min_col), (max_row, max_col) = window_coords

    # Calculate midpoint of window
    # Use float division for potentially more accurate center, then floor
    window_mid_row = (min_row + max_row) / 2.0
    window_mid_col = (min_col + max_col) / 2.0

    # Grid of NaNs (True where NaN, False otherwise)
    # Important: Also consider inf as explored/not a target
    nan_mask = np.isnan(grid) & ~np.isinf(grid)

    # Create distance squared array, handle potential division by zero at center
    # Use np.meshgrid for clarity and efficiency
    jj, ii = np.meshgrid(np.arange(cols), np.arange(rows))
    # Calculate squared distance, add epsilon to avoid division by zero
    dist_sq = (ii - window_mid_row)**2 + (jj - window_mid_col)**2 + 1e-9 
    # Inverse distance squared score (higher score for closer NaNs)
    inv_dist_sq_score = 1.0 / dist_sq

    # Apply NaN mask to get scores only for unexplored cells
    score_array = inv_dist_sq_score * nan_mask

    # Calculate summed scores for each direction relative to the window bounds
    # Ensure bounds are valid before summing
    up_weighting = score_array[0:min_row, :].sum() if min_row > 0 else 0
    down_weighting = score_array[max_row+1:, :].sum() if max_row < rows - 1 else 0
    left_weighting = score_array[:, 0:min_col].sum() if min_col > 0 else 0
    right_weighting = score_array[:, max_col+1:].sum() if max_col < cols - 1 else 0

    weighting_dict = {
        'up': up_weighting,
        'down': down_weighting,
        'left': left_weighting,
        'right': right_weighting
    }
    
    # Filter out directions that are impossible (at edge)
    possible_directions = {}
    if min_row > 0: possible_directions['up'] = weighting_dict['up']
    if max_row < rows - 1: possible_directions['down'] = weighting_dict['down']
    if min_col > 0: possible_directions['left'] = weighting_dict['left']
    if max_col < cols - 1: possible_directions['right'] = weighting_dict['right']

    if not possible_directions:
        logger.warning("James C: No possible directions with unexplored cells. Fallback.")
        # Fallback: Use random walk logic without the NaN bias
        scores = {dir: np.random.random() for dir in ['up', 'down', 'left', 'right']}
        if min_row <= 0: scores.pop('up', None)
        if max_row >= rows - 1: scores.pop('down', None)
        if min_col <= 0: scores.pop('left', None)
        if max_col >= cols - 1: scores.pop('right', None)
        return max(scores, key=scores.get) if scores else 'right' # Absolute fallback

    # Find the direction with the highest score among possible directions
    direction = max(possible_directions, key=possible_directions.get)

    if verbose_logging:
        logger.info(f"James C Proximity Scores: Up={up_weighting:.2f}, Down={down_weighting:.2f}, Left={left_weighting:.2f}, Right={right_weighting:.2f}")
        logger.info(f"James C chose direction: {direction}")

    return direction 


def _jake_p_snake(grid, window_coords=None, verbose_logging=False):
    """
    Custom search strategy by Jake P (The Snake).
    Follows a strict zigzag pattern, moving right on even rows, left on odd rows.
    Includes boundary checks and fallback logic.
    Args:
        grid: 2D numpy array (used for shape only in this version).
        window_coords: Tuple ((min_row, min_col), (max_row, max_col)). Required.
        verbose_logging: Boolean flag.
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    import numpy as np

    if window_coords is None:
        logger.warning("Jake P strategy requires window_coords, falling back to random walk.")
        return _random_walk_strategy(grid, None, verbose_logging)

    # Check Boundaries Helper
    def valid_coord(row_index, column_index, total_rows, total_columns):
        is_row_valid = 0 <= row_index < total_rows
        is_col_valid = 0 <= column_index < total_columns
        return is_row_valid and is_col_valid

    # Current State
    (window_min_row, window_min_col), (window_max_row, window_max_col) = window_coords
    window_height = window_max_row - window_min_row + 1
    window_width = window_max_col - window_min_col + 1
    window_centre_row = window_min_row + window_height // 2
    window_centre_column = window_min_col + window_width // 2
    grid_total_rows, grid_total_columns = grid.shape

    # Determine Move Dir based on Snake Pattern
    edge_margin = 1 # How close to the edge before turning
    is_at_right_boundary = window_centre_column >= grid_total_columns - 1 - edge_margin
    is_at_left_boundary = window_centre_column <= edge_margin
    is_at_bottom_boundary = window_centre_row >= grid_total_rows - 1 - edge_margin # Check if at the bottom edge

    next_direction = 'down' # Default if at an edge and can't move sideways
    is_centre_row_even = window_centre_row % 2 == 0

    if is_centre_row_even: # Even rows -> Move Right
        if not is_at_right_boundary:
            next_direction = 'right'
        elif not is_at_bottom_boundary: # Hit right edge, move down if not at bottom
            next_direction = 'down'
        else: # Hit bottom-right corner, must move left to continue pattern
            next_direction = 'left'
    else: # Odd rows -> Move Left
        if not is_at_left_boundary:
            next_direction = 'left'
        elif not is_at_bottom_boundary: # Hit left edge, move down if not at bottom
            next_direction = 'down'
        else: # Hit bottom-left corner, must move right to continue pattern
            next_direction = 'right'

    # --- Validate the chosen direction --- 
    potential_next_row, potential_next_col = window_centre_row, window_centre_column
    if next_direction == 'up': potential_next_row -= 1
    elif next_direction == 'down': potential_next_row += 1
    elif next_direction == 'left': potential_next_col -= 1
    elif next_direction == 'right': potential_next_col += 1

    # If the initially chosen direction is invalid (off-grid)
    if not valid_coord(potential_next_row, potential_next_col, grid_total_rows, grid_total_columns):
        if verbose_logging:
             logger.warning(f"Jake P: Initial direction {next_direction} is invalid. Finding alternative.")
        
        possible_moves = ['right', 'down', 'left', 'up'] # Order preference for fallback
        found_valid = False
        for alternative_direction in possible_moves:
            if alternative_direction == next_direction: continue # Skip the one we know is bad
            
            alt_row, alt_col = window_centre_row, window_centre_column
            if alternative_direction == 'up': alt_row -= 1
            elif alternative_direction == 'down': alt_row += 1
            elif alternative_direction == 'left': alt_col -= 1
            elif alternative_direction == 'right': alt_col += 1

            if valid_coord(alt_row, alt_col, grid_total_rows, grid_total_columns):
                next_direction = alternative_direction
                found_valid = True
                if verbose_logging:
                    logger.info(f"Jake P: Found valid fallback direction: {next_direction}")
                break # Use first valid alternative found
        
        if not found_valid:
             logger.error("Jake P: Could not find any valid move! Defaulting to 'down'.")
             next_direction = 'down' # Should not happen if grid > 1x1

    if verbose_logging:
        logger.info(f"Jake P chose direction: {next_direction}")

    return next_direction 


def _jono_s_vacuum(grid, window_coords=None, verbose_logging=False):
    """
    Custom search strategy by Jono S (Robot Vacuum) - NaN Concentration.
    Prioritizes moving towards the global direction (up/down/left/right of window) 
    with the highest concentration of unexplored (NaN) cells.
    Avoids obstacles (inf) and grid edges.
    Args:
        grid: 2D numpy array with NaN/inf.
        window_coords: Tuple ((min_row, min_col), (max_row, max_col)). Required.
        verbose_logging: Boolean flag.
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    import numpy as np

    if window_coords is None:
        logger.warning("Jono S strategy requires window_coords, falling back to random walk.")
        return _random_walk_strategy(grid, None, verbose_logging)

    rows, cols = grid.shape
    (min_row, min_col), (max_row, max_col) = window_coords

    # Get current window center (needed for checking immediate obstacles)
    window_height = max_row - min_row + 1
    window_width = max_col - min_col + 1
    center_row = min_row + window_height // 2
    center_col = min_col + window_width // 2

    # --- 1. Calculate NaN Concentration in Global Regions --- 
    region_scores = {
        'up': 0.0,
        'down': 0.0,
        'left': 0.0,
        'right': 0.0
    }

    # Function to calculate concentration safely
    def calculate_concentration(sub_grid):
        if sub_grid.size == 0:
            return 0.0
        num_nans = np.sum(np.isnan(sub_grid))
        # Ignore inf cells in concentration calculation (treat as explored)
        num_valid_cells = np.sum(~np.isinf(sub_grid))
        if num_valid_cells == 0:
             return 0.0 # Avoid division by zero if region only contains inf
        # Concentration is NaNs / (Total cells - Inf cells)
        return num_nans / float(num_valid_cells) 

    # Calculate concentration for each region relative to window *bounds*
    if min_row > 0:
        region_up = grid[0:min_row, :]
        region_scores['up'] = calculate_concentration(region_up)
        
    if max_row < rows - 1:
        region_down = grid[max_row + 1:, :]
        region_scores['down'] = calculate_concentration(region_down)
        
    if min_col > 0:
        region_left = grid[:, 0:min_col]
        region_scores['left'] = calculate_concentration(region_left)
        
    if max_col < cols - 1:
        region_right = grid[:, max_col + 1:]
        region_scores['right'] = calculate_concentration(region_right)

    # --- 2. Check Validity of Immediate Moves --- 
    valid_moves = {}
    # Up
    if center_row > 0 and not np.isinf(grid[center_row - 1, center_col]):
        valid_moves['up'] = region_scores['up']
    # Down
    if center_row < rows - 1 and not np.isinf(grid[center_row + 1, center_col]):
        valid_moves['down'] = region_scores['down']
    # Left
    if center_col > 0 and not np.isinf(grid[center_row, center_col - 1]):
        valid_moves['left'] = region_scores['left']
    # Right
    if center_col < cols - 1 and not np.isinf(grid[center_row, center_col + 1]):
        valid_moves['right'] = region_scores['right']

    if not valid_moves:
        logger.warning("Jono S: No valid moves available (blocked by obstacles/edges)! Defaulting right.")
        return 'right'
        
    # --- 3. Choose Direction with Highest Concentration Score --- 
    # Find the maximum score among valid moves
    max_score = -1.0
    if valid_moves:
         # Ensure we handle the case where all scores might be 0
         scores_list = list(valid_moves.values())
         if scores_list: # Check if the list is not empty
             max_score = max(scores_list)

    # Define a preferred order for tie-breaking
    preference_order = ['up', 'left', 'down', 'right']
         
    # If max score is positive, choose the best direction based on preference order
    if max_score > 0:
        best_directions = [direction for direction, score in valid_moves.items() if score == max_score]
        
        # Apply preference order for tie-breaking
        chosen_direction = best_directions[0] # Default to the first one found
        for pref_dir in preference_order:
            if pref_dir in best_directions:
                chosen_direction = pref_dir
                break # Found the most preferred direction

        if verbose_logging:
            scores_str = ", ".join([f"{d}={s:.2f}" for d, s in region_scores.items()])
            valid_scores_str = ", ".join([f"{d}={s:.2f}" for d, s in valid_moves.items()])
            logger.info(f"Jono S Concentrations: [{scores_str}] | Valid Moves: [{valid_scores_str}] | Best (Score={max_score:.2f}): {best_directions} | Preferred & Chosen: {chosen_direction}")
        return chosen_direction
    else:
        # --- 4. Fallback: No NaN concentration found, move randomly to any valid neighbor --- 
        fallback_directions = list(valid_moves.keys())
        # Apply preference order to fallback as well for consistency
        chosen_direction = fallback_directions[0] if fallback_directions else 'right' # Default fallback
        for pref_dir in preference_order:
             if pref_dir in fallback_directions:
                  chosen_direction = pref_dir
                  break
        # chosen_direction = np.random.choice(fallback_directions) # Old random fallback
        if verbose_logging:
             logger.info(f"Jono S: No NaN concentration detected. Fallback moving {chosen_direction} from {fallback_directions}.")
        return chosen_direction 


def _zigzag_search(grid, window_coords=None, verbose_logging=False):
    """
    Strategy by Tony R (Zigzag Search).
    Explore systematically from left to right, top to bottom, 
    reversing direction on alternate rows.
    Args:
        grid: 2D numpy array (used for shape only).
        window_coords: Tuple ((min_row, min_col), (max_row, max_col)). Required.
        verbose_logging: Boolean flag.
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    import numpy as np

    if window_coords is None:
        logger.warning("Zigzag Search strategy requires window_coords, falling back to random walk.")
        return _random_walk_strategy(grid, None, verbose_logging)

    rows, cols = grid.shape
    (min_row, min_col), (max_row, max_col) = window_coords

    # Get current window center
    window_height = max_row - min_row + 1
    window_width = max_col - min_col + 1
    center_row = min_row + window_height // 2
    center_col = min_col + window_width // 2

    # --- Zigzag Logic --- 
    # Define boundary checks - Note: uses 1 and cols-2 as turning points
    is_at_right_turn_point = (center_col >= cols - 2)
    is_at_left_turn_point = (center_col <= 1)
    is_at_bottom = (center_row >= rows - 1) # Check if at the very bottom

    if is_at_bottom:
        # If at the bottom, behavior might need adjustment based on desired end state
        # For now, let's try to move horizontally if possible, otherwise default
        if center_row % 2 == 0: # Even row, last move was likely right
             if not is_at_left_turn_point: return 'left'
        else: # Odd row, last move was likely left
             if not is_at_right_turn_point: return 'right'
        # If stuck horizontally at bottom, maybe go up?
        if center_row > 0: return 'up'
        return 'right' # Absolute fallback

    # Normal zigzag pattern
    if center_row % 2 == 0: # Even row -> move right until the turning point
      if not is_at_right_turn_point:
        if verbose_logging: logger.info("Zigzag: Even row, moving right.")
        return 'right'
      else: # At turning point, move down
        if verbose_logging: logger.info("Zigzag: Even row, turning down.")
        return 'down'
    else: # Odd row -> move left until the turning point
      if not is_at_left_turn_point:
        if verbose_logging: logger.info("Zigzag: Odd row, moving left.")
        return 'left'
      else: # At turning point, move down
        if verbose_logging: logger.info("Zigzag: Odd row, turning down.")
        return 'down'

    return chosen_direction

