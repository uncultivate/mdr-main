# Macrodata Refinement Challenge

A tribute to the popular TV Show - Severance, this challenge focuses on finding magic squares in a partially revealed grid using Python and NumPy.

## Challenge Overview

As a Data Engineer in the Macrodata Refinement section, your task is to develop search strategies to navigate through a partially revealed grid and find hidden magic squares. The challenge tests your ability to work with arrays/matrices in NumPy and design efficient search algorithms.

### Key Features

- Grid sizes: 10x10, 20x20, and 30x30
- Multiple magic squares (3x3 and 5x5) to find
- Step limit for search completion
- Points awarded for:
  - Finding magic squares
  - Efficiency (fewer moves)
  - Complete detection of all magic squares

## Function Requirements

Your search strategy must be implemented as a Python function with the following signature:

```python
def your_strategy_name(grid, window_coords):
    """
    Custom search strategy to find magic squares in a partially revealed grid
    
    Args:
        grid: 2D numpy array with NaN for unexplored cells and inf for known magic squares
        window_coords: Tuple containing window coordinates ((min_row, min_col), (max_row, max_col))
        
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    # Your code here
    return 'down'  # Return a direction
```

### Parameter Examples

#### Grid Parameter
The `grid` parameter is a 2D NumPy array where:
- Unexplored cells are represented by `NaN`
- Known magic squares are represented by `inf`
- Explored cells contain numbers

Example grid:
```python
import numpy as np

grid = np.array([
    [1, 2, 3, np.nan, np.nan],
    [4, 5, 6, np.nan, np.nan],
    [7, 8, 9, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan, np.nan]
])
```

#### Window Coordinates Parameter
The `window_coords` parameter is a tuple containing the current view window coordinates:
```python
window_coords = ((0, 0), (2, 2))  # Example: 3x3 window starting at (0,0)
```

In this example:
- `min_row = 0`: Top row of the window
- `min_col = 0`: Leftmost column of the window
- `max_row = 2`: Bottom row of the window
- `max_col = 2`: Rightmost column of the window

### Magic Square Detection

A magic square is a grid (3x3 or 5x5) where:
- The sum of numbers in each row is equal
- The sum of numbers in each column is equal
- The sum of numbers in each diagonal is equal

Example 3x3 magic square:
```python
magic_square = np.array([
    [8, 1, 6],
    [3, 5, 7],
    [4, 9, 2]
])
# All rows, columns, and diagonals sum to 15
```

## Development Tips

1. Use NumPy's `isnan()` and `isinf()` functions to check for unexplored cells and known magic squares
2. Enable logging in the developer console for detailed information
3. Consider the window coordinates when planning your search strategy
4. Test your strategy with different grid sizes and magic square quantities

## Example Strategy

Here's a basic example of a search strategy:

```python
def example_strategy(grid, window_coords):
    """
    Example search strategy that prioritizes unexplored areas
    
    Args:
        grid: 2D numpy array with NaN for unexplored cells
        window_coords: Tuple containing window coordinates ((min_row, min_col), (max_row, max_col))
        
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    import numpy as np
    
    # Extract window coordinates
    (min_row, min_col), (max_row, max_col) = window_coords
    
    # Get current window dimensions and center
    window_height = max_row - min_row + 1
    window_width = max_col - min_col + 1
    center_row = min_row + window_height // 2
    center_col = min_col + window_width // 2
    
    # Get number of rows and columns in the grid
    rows, cols = grid.shape
    
    # Count unexplored cells in each direction
    up_nans = np.sum(np.isnan(grid[0, :]))
    down_nans = np.sum(np.isnan(grid[-1, :]))
    left_nans = np.sum(np.isnan(grid[:, 0]))
    right_nans = np.sum(np.isnan(grid[:, -1]))
    
    # Choose direction with most unexplored cells
    directions = {
        'up': up_nans,
        'down': down_nans,
        'left': left_nans,
        'right': right_nans
    }
    
    return max(directions.items(), key=lambda x: x[1])[0]
```

## Getting Started

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install numpy
   ```
3. Implement your search strategy
4. Test your strategy with different grid configurations
5. Submit your solution

Good luck with the challenge! 