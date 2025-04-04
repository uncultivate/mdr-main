from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from magic_square import calculate_next_direction
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
    'search_strategy': 'random_walk',  # Default search strategy
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

@app.route('/api/get_next_move', methods=['POST'])
def get_next_move():
    data = request.json
    
    # Convert grid to float array
    try:
        grid_data = data['grid']
        if not isinstance(grid_data, list) or len(grid_data) == 0 or not isinstance(grid_data[0], list):
            raise ValueError("Invalid grid format")
        grid = np.array(grid_data, dtype=float)
        if len(grid.shape) != 2:
            raise ValueError(f"Invalid grid dimensions: {grid.shape}")
    except (KeyError, ValueError) as e:
        logger.error(f"Error processing grid data: {str(e)}")
        return jsonify({'error': str(e), 'next_direction': 'right', 'steps': game_state['steps']}), 400

    strategy = data.get('strategy', game_state['search_strategy'])
    window_coords = data.get('window_coords')
    found_squares = data.get('found_squares', []) # Get revealed squares from frontend

    # Validate window_coords format if provided
    if window_coords:
        try:
            if (not isinstance(window_coords, list) or len(window_coords) != 2 or
                not isinstance(window_coords[0], list) or not isinstance(window_coords[1], list) or
                len(window_coords[0]) != 2 or len(window_coords[1]) != 2):
                logger.warning(f"Invalid window_coords format: {window_coords}, ignoring.")
                window_coords = None
        except Exception as e:
            logger.error(f"Error validating window_coords: {str(e)}, ignoring.")
            window_coords = None

    # Store the strategy in game state
    game_state['search_strategy'] = strategy
    
    # Update steps count - ONLY update steps here
    game_state['steps'] += 1
    
    # Check game over condition based on steps ONLY
    if game_state['steps'] >= game_state['max_steps']:
        game_state['game_over'] = True
        logger.info(f"Game over! Steps limit reached: {game_state['steps']}")

    # Calculate the next direction using the strategy
    
    next_direction = calculate_next_direction(
        grid,
        strategy,
        found_squares, # Pass revealed squares to strategy
        window_coords,
        game_state['verbose_logging']
    )

    response_data = {
        'next_direction': next_direction,
        'steps': game_state['steps'],
        'game_over': game_state['game_over'] # Let frontend know if step limit reached
    }

    logger.info(f"Response: next_direction={next_direction}, steps={game_state['steps']}, game_over={game_state['game_over']}")

    return jsonify(response_data)

@app.route('/api/reset_game', methods=['POST'])
def reset_game():
    data = request.json
    total_magic_squares = data.get('total_magic_squares', 3)
    strategy = data.get('strategy', 'random_walk')
    
    logger.info(f"Resetting game with {total_magic_squares} magic squares and '{strategy}' strategy")
    
    # Save current verbose_logging setting
    verbose_setting = game_state['verbose_logging']
    
    game_state.clear() # Clear the whole state
    game_state.update({
        'steps': 0,
        'max_steps': 100,
        'magic_squares_found': 0, # Frontend now tracks this
        'total_magic_squares': total_magic_squares, # Still needed for frontend win condition
        'game_over': False,
        'search_strategy': strategy,
        'found_squares': [],  # This might not be needed anymore
        'verbose_logging': verbose_setting
    })
    
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
    
    # Check for window_coords parameter
    if 'window_coords' not in code.split('(')[1].split(')')[0]:
        return jsonify({'error': 'The function must accept a "window_coords" parameter'}), 400
    
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
    except TypeError as e:
        if "missing 1 required positional argument" in str(e):
            return False, f"Function is missing required parameters. Make sure it accepts both 'grid' and 'window_coords'"
        return False, f"Error during function execution: {{str(e)}}"
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
            'id': 'random_walk',
            'name': 'Randy S (Random Walk)',
            'description': 'Uses randomness with a bias toward unexplored areas'
        },
        {
            'id': 'zigzag_search',
            'name': 'Tony R (Zigzag Search)',
            'description': 'Searches in a zigzag pattern to systematically cover the board'
        },

        {
            'id': 'adam_l',
            'name': 'Adam L (Blackjack)',
            'description': 'Blackjack-inspired strategy by Adam L.'
        },
        {
            'id': 'meredith_n',
            'name': 'Meredith N (Inner Zigzag)',
            'description': 'Zigzag pattern avoiding grid borders.'
        },
        {
            'id': 'james_c',
            'name': 'James C (Proximity Explorer)',
            'description': 'Moves towards closest unexplored cells.'
        },
        {
            'id': 'jake_p',
            'name': 'Jake P (The Snake)',
            'description': 'Zigzag pattern with boundary checks.'
        },
        {
            'id': 'jono_s',
            'name': 'Jono S (Robot Vacuum)',
            'description': 'Prioritizes unexplored cells (NaN), avoids obstacles (inf).'
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