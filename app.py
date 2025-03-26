from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from magic_square import detect_magic_square_and_navigate
import logging
import importlib.util
import sys
import uuid
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS to allow requests from both localhost and Vercel
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://mdr-main.vercel.app",  # Your main Vercel domain
    "https://*.vercel.app"  # For preview deployments
]

# Configure CORS with all necessary headers
CORS(app, 
     resources={
         r"/*": {
             "origins": allowed_origins,
             "methods": ["GET", "POST", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization"],
             "expose_headers": ["Content-Type"],
             "supports_credentials": True,
             "max_age": 3600
         }
     })

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    # Check if the origin matches any of our allowed patterns
    allowed = False
    for allowed_origin in allowed_origins:
        if allowed_origin == origin or (allowed_origin.startswith('https://') and allowed_origin.endswith('.vercel.app')):
            allowed = True
            break
    
    if allowed:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Keep track of game state
game_state = {
    'steps': 0,
    'max_steps': 100,
    'magic_squares_found': 0,
    'total_magic_squares': 0,
    'game_over': False,
    'search_strategy': 'exploration',  # Default search strategy
    'found_squares': [],  # List to store positions of found magic squares
    'verbose_logging': True  # Default verbose logging to True for diagnostics
}

# Storage for custom strategies
custom_strategies = {}

@app.route('/api/toggle_verbose_logging', methods=['POST'])
def toggle_verbose_logging():
    """Toggle verbose logging on/off"""
    data = request.json
    verbose = data.get('verbose', True)  # Default to True if not specified
    
    game_state['verbose_logging'] = verbose
    logger.info(f"Verbose logging set to: {verbose}")
    
    return jsonify({
        'status': 'success',
        'verbose_logging': game_state['verbose_logging']
    })

@app.route('/api/detect_magic_square', methods=['POST'])
def detect_magic_square():
    data = request.json
    
    # Convert grid to float array to handle NaN values properly
    grid = np.array(data['grid'], dtype=float)
    strategy = data.get('strategy', game_state['search_strategy'])
    
    # Extract window coordinates if provided
    window_coords = data.get('window_coords')
    
    # Log grid information
    logger.info(f"Grid shape before conversion: {grid.shape}")
    logger.info(f"Grid type: {type(grid)}, contains NaN: {np.any(np.isnan(grid))}")
    if window_coords:
        (min_row, min_col), (max_row, max_col) = window_coords
        window_height = max_row - min_row + 1
        window_width = max_col - min_col + 1
        window_center_row = min_row + window_height // 2
        window_center_col = min_col + window_width // 2
        logger.info(f"WINDOW POSITION IN API: Center=({window_center_row},{window_center_col}), " +
                   f"Bounds=({min_row},{min_col}) to ({max_row},{max_col}), " +
                   f"Size={window_height}x{window_width}")
    
    # Add verbose logging of the actual grid content if enabled
    if game_state['verbose_logging']:
        # Log information about found squares
        if game_state['found_squares']:
            logger.info(f"Found squares list contains {len(game_state['found_squares'])} entries:")
            for i, square in enumerate(game_state['found_squares']):
                logger.info(f"  Square {i+1}: {square}")
        else:
            logger.info("No magic squares have been found yet")
        
        # Format the grid for logging - limit to first few rows/columns if large
        if grid.shape[0] > 10 or grid.shape[1] > 10:
            visible_grid = grid[:min(10, grid.shape[0]), :min(10, grid.shape[1])]
            logger.info(f"Grid preview (first 10x10 or smaller):\n{np.array2string(visible_grid, precision=1, separator=', ', threshold=100, edgeitems=3)}")
        else:
            logger.info(f"Grid content:\n{np.array2string(grid, precision=1, separator=', ', threshold=100)}")
        
        # Log detailed window coordinates
        if window_coords:
            (min_row, min_col), (max_row, max_col) = window_coords
            window_height = max_row - min_row + 1
            window_width = max_col - min_col + 1
            logger.info(f"Window dimensions: {window_height}x{window_width}")
            logger.info(f"Window region: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
    
    # Validate the grid shape
    rows, cols = grid.shape
    if rows != cols:
        logger.warning(f"Grid is not square: {grid.shape}")
    
    # Store the strategy in game state
    game_state['search_strategy'] = strategy
    
    # Use combined function to check for magic square and get next direction
    is_magic, next_direction, square_info = detect_magic_square_and_navigate(
        grid, 
        strategy, 
        game_state['found_squares'],
        window_coords,
        verbose_logging=game_state['verbose_logging']  # Pass verbose flag to magic_square module
    )
    
    # Update game state
    game_state['steps'] += 1
    
    if is_magic:
        # Only count the magic square if it hasn't been found before
        if square_info and square_info not in game_state['found_squares']:
            game_state['magic_squares_found'] += 1
            # Add the found square to our tracking list
            game_state['found_squares'].append(square_info)
            logger.info(f"New magic square detected at {square_info}! Total found: {game_state['magic_squares_found']}")
            
            # Verify the square_info format is correct
            if 'type' not in square_info or 'position' not in square_info:
                logger.error(f"Invalid square_info format! Expected 'type' and 'position' keys but got: {square_info}")
            elif not isinstance(square_info['position'], tuple) or len(square_info['position']) != 2:
                logger.error(f"Invalid position format in square_info! Expected tuple of length 2 but got: {square_info['position']}")
            else:
                logger.info(f"Square info format is valid: {square_info}")
        elif square_info:
            logger.info(f"Magic square already found before: {square_info}")
        else:
            logger.warning("Magic square detected but no square_info provided")
    
    # Check if game should end
    if game_state['steps'] >= game_state['max_steps'] or game_state['magic_squares_found'] >= game_state['total_magic_squares']:
        game_state['game_over'] = True
        logger.info(f"Game over! Steps: {game_state['steps']}, Magic squares found: {game_state['magic_squares_found']}")
    
    response_data = {
        'is_magic_square': is_magic,
        'next_direction': next_direction,
        'steps': game_state['steps'],
        'magic_squares_found': game_state['magic_squares_found'],
        'game_over': game_state['game_over'],
        'strategy': strategy,
        'square_info': square_info if is_magic else None
    }
    
    logger.info(f"Response: is_magic_square={is_magic}, next_direction={next_direction}")
    
    return jsonify(response_data)

@app.route('/api/reset_game', methods=['POST'])
def reset_game():
    data = request.json
    total_magic_squares = data.get('total_magic_squares', 3)
    strategy = data.get('strategy', 'exploration')
    
    logger.info(f"Resetting game with {total_magic_squares} magic squares and '{strategy}' strategy")
    
    # Save current verbose_logging setting
    verbose_setting = game_state['verbose_logging']
    
    game_state['steps'] = 0
    game_state['max_steps'] = 100
    game_state['magic_squares_found'] = 0
    game_state['total_magic_squares'] = total_magic_squares
    game_state['game_over'] = False
    game_state['search_strategy'] = strategy
    game_state['found_squares'] = []  # Reset the list of found squares
    
    # Restore verbose_logging setting
    game_state['verbose_logging'] = verbose_setting
    
    return jsonify({
        'status': 'game_reset',
        'total_magic_squares': total_magic_squares,
        'strategy': strategy
    })

@app.route('/api/add_custom_strategy', methods=['POST'])
def add_custom_strategy():
    data = request.json
    name = data.get('name', '').strip()
    author = data.get('author', '').strip()
    code = data.get('code', '').strip()
    
    if not name:
        return jsonify({'error': 'Strategy name is required'}), 400
        
    if not author:
        return jsonify({'error': 'Author name is required'}), 400
        
    if not code:
        return jsonify({'error': 'Function code is required'}), 400
    
    # Generate a unique ID for this strategy
    strategy_id = f"custom_{uuid.uuid4().hex[:8]}"
    
    # Basic validation checks
    if not code.startswith('def '):
        return jsonify({'error': 'The code must define a function using the "def" keyword'}), 400
    
    # Extract function name for use in validation
    match = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\(', code)
    if not match:
        return jsonify({'error': 'Could not find function definition in code'}), 400
    
    function_name = match.group(1)
    
    # Check for grid parameter
    if 'grid' not in code.split('(')[1].split(')')[0]:
        return jsonify({'error': 'The function must accept a "grid" parameter'}), 400
    
    # Check for return statement
    if 'return' not in code:
        return jsonify({'error': 'The function must include a return statement'}), 400
    
    # Create a temporary file for the function
    temp_filename = f"temp_strategy_{strategy_id}.py"
    
    try:
        with open(temp_filename, 'w') as f:
            f.write(f"""import numpy as np

{code}

def validate_{function_name}():
    # Create a test grid
    test_grid = np.ones((5, 5))
    # Mark some cells as unexplored
    test_grid[0, 0] = np.nan
    
    # Create test window coordinates
    test_window_coords = ((0, 0), (4, 4))
    
    try:
        # Call the function with proper exception handling
        function_result = {function_name}(test_grid, test_window_coords)
        
        # Check if the result is None
        if function_result is None:
            return False, "Function returned None. It must return one of 'up', 'down', 'left', 'right'"
        
        # Check the return value
        if not isinstance(function_result, str):
            return False, f"Function must return a string, got {{type(function_result).__name__}}"
            
        # Check that the result is a valid direction
        if function_result not in ['up', 'down', 'left', 'right']:
            return False, f"Function must return one of 'up', 'down', 'left', 'right', got '{{function_result}}'"
            
        return True, "Function validated successfully"
    except Exception as e:
        return False, f"Error during function execution: {{str(e)}}"
""")
        
        # Dynamically import the temporary module
        spec = importlib.util.spec_from_file_location(f"temp_strategy_{strategy_id}", temp_filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"temp_strategy_{strategy_id}"] = module
        spec.loader.exec_module(module)
        
        # Validate the function
        validation_func = getattr(module, f"validate_{function_name}")
        is_valid, validation_message = validation_func()
        
        if not is_valid:
            os.remove(temp_filename)
            return jsonify({'error': validation_message}), 400
        
        # Store the strategy
        custom_strategies[strategy_id] = {
            'name': name,
            'author': author,
            'code': code,
            'function_name': function_name,
            'module_path': temp_filename
        }
        
        # Update available strategies
        logger.info(f"New custom strategy added: {name} by {author} (ID: {strategy_id})")
        
        return jsonify({
            'id': strategy_id,
            'name': name,
            'author': author,
            'description': f"Custom strategy by {author}"
        })
        
    except Exception as e:
        logger.error(f"Error adding custom strategy: {str(e)}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return jsonify({'error': f"Server error while processing your function: {str(e)}"}), 500

@app.route('/api/available_strategies', methods=['GET'])
def get_strategies():
    """Return the list of available search strategies"""
    logger.info("Returning available search strategies")
    
    # Combine built-in and custom strategies
    strategies = [
        {
            'id': 'exploration',
            'name': 'Exploration Priority',
            'description': 'Prioritizes unexplored areas of the grid'
        },
        {
            'id': 'random_walk',
            'name': 'Random Walk',
            'description': 'Uses randomness with a bias toward unexplored areas'
        },
        {
            'id': 'spiral',
            'name': 'Spiral Search',
            'description': 'Explores in a spiral pattern to systematically cover the board'
        },
        {
            'id': 'pattern_detection',
            'name': 'Pattern Detection',
            'description': 'Searches for matching row and column sums that might indicate magic squares'
        }
    ]
    
    # Add custom strategies
    for strategy_id, strategy_info in custom_strategies.items():
        strategies.append({
            'id': strategy_id,
            'name': f"{strategy_info['name']} (by {strategy_info['author']})",
            'description': f"Custom strategy by {strategy_info['author']}"
        })
    
    return jsonify({
        'strategies': strategies
    })

if __name__ == '__main__':
    import os
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    # In production, bind to 0.0.0.0 to accept connections from any IP
    host = '0.0.0.0' if os.environ.get('PRODUCTION', False) else '127.0.0.1'
    app.run(host=host, port=port, debug=os.environ.get('DEBUG', False)) 