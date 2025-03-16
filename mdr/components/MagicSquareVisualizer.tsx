'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { RotateCcw, Wand2, Play, Pause, Code, X } from 'lucide-react';

const MAX_NUM = 26;
const MAGIC_CONSTANT5 = 65;
const MAGIC_CONSTANT3 = 27;

// We assume each cell is 48px (h-12 / w-12 corresponds to 48px at a default base)
const CELL_SIZE = 48;

// API base URL - configurable for production
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000').replace(/\/+$/, '');

// Log the API URL when the component mounts (development only)
if (process.env.NODE_ENV === 'development') {
  console.log('API Base URL:', API_BASE_URL);
}

type Coordinates = [number, number];
type Grid = number[][];

interface MagicGrid {
  row: number;
  col: number;
  size: number;
  revealed: boolean;
}

interface Strategy {
  id: string;
  name: string;
  description: string;
}

const generateMagicSumGrid = (dim: number, magicConstant?: number): Grid => {
  if (dim % 2 === 0) {
    throw new Error("Dimension must be odd to generate a magic square.");
  }

  const totalNumbers = dim * dim;
  let start: number;
  
  if (magicConstant === undefined) {
    // Default behavior: start at 1 (i.e. a classical magic square)
    start = 1;
  } else {
    // For a magic square with numbers starting at 'start', the magic constant is:
    // magicConstant = dim * (2 * start + (dim * dim - 1)) / 2
    // To have only positive integers, start must be at least 1.
    const minValidMagicConstant = (dim * (dim * dim + 1)) / 2;
    if (magicConstant < minValidMagicConstant) {
      throw new Error(
        `Provided magic constant must be at least ${minValidMagicConstant} for a ${dim}x${dim} magic square with positive integers.`
      );
    }
    // Solve for start: start = (2*magicConstant/dim - (dim*dim - 1)) / 2
    const calculatedStart = (2 * magicConstant / dim - (totalNumbers - 1)) / 2;
    if (calculatedStart < 1 || calculatedStart % 1 !== 0) {
      throw new Error("Invalid magic constant: the computed starting number is not a positive integer.");
    }
    start = calculatedStart;
  }

  const square: number[][] = new Array(dim);
  for (let i = 0; i < dim; i++) {
    square[i] = new Array(dim);
  }

  // Start from the middle of the top row.
  let pointRow = 0;
  let pointCol = Math.floor(dim / 2);

  for (let num = start; num < start + totalNumbers; num++) {
    square[pointRow][pointCol] = num;

    // Move diagonally up and to the right.
    let newPointRow = pointRow - 1;
    let newPointCol = pointCol + 1;

    // Wrap around if necessary.
    if (newPointRow < 0) newPointRow = dim - 1;
    if (newPointCol >= dim) newPointCol = 0;

    // If the chosen cell is already occupied, move directly down instead.
    if (square[newPointRow][newPointCol] !== undefined) {
      newPointRow = pointRow + 1;
      newPointCol = pointCol;
      if (newPointRow >= dim) newPointRow = 0;
    }

    pointRow = newPointRow;
    pointCol = newPointCol;
  }

  return square;
};

// Helper function to calculate window coordinates based on center
const calculateWindowCoords = (center: Coordinates, boardSize: number): [Coordinates, Coordinates] => {
  // Use 5 by default; if the center touches an edge, use 3.
  const windowRows = (center[0] === 0 || center[0] === boardSize - 1) ? 3 : 5;
  const windowCols = (center[1] === 0 || center[1] === boardSize - 1) ? 3 : 5;

  let minRow = center[0] - Math.floor(windowRows / 2);
  let minCol = center[1] - Math.floor(windowCols / 2);
  let maxRow = minRow + windowRows - 1;
  let maxCol = minCol + windowCols - 1;

  if (center[0] === 0) {
    minRow = 0;
    maxRow = windowRows - 1;
  } else if (center[0] === boardSize - 1) {
    maxRow = boardSize - 1;
    minRow = boardSize - windowRows;
  } else {
    if (minRow < 0) {
      minRow = 0;
      maxRow = windowRows - 1;
    }
    if (maxRow >= boardSize) {
      maxRow = boardSize - 1;
      minRow = boardSize - windowRows;
    }
  }

  if (center[1] === 0) {
    minCol = 0;
    maxCol = windowCols - 1;
  } else if (center[1] === boardSize - 1) {
    maxCol = boardSize - 1;
    minCol = boardSize - windowCols;
  } else {
    if (minCol < 0) {
      minCol = 0;
      maxCol = windowCols - 1;
    }
    if (maxCol >= boardSize) {
      maxCol = boardSize - 1;
      minCol = boardSize - windowCols;
    }
  }
  return [[minRow, minCol], [maxRow, maxCol]];
};

interface MagicSquareVisualizerProps {
  onError?: (message: string) => void;
}

const MagicSquareVisualizer: React.FC<MagicSquareVisualizerProps> = ({ onError }) => {
  // Backend state
  const [aiMode, setAiMode] = useState<boolean>(false);
  const [gameStats, setGameStats] = useState({
    steps: 0,
    magicSquaresFound: 0,
    gameOver: false
  });

  // UI state from ArrayVisualizer
  const [boardSize, setBoardSize] = useState<number>(10);
  const [magic5Qty, setMagic5Qty] = useState<number>(1);
  const [magic3Qty, setMagic3Qty] = useState<number>(1);
  const [toastMessage, setToastMessage] = useState<string>("");
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard'>('easy');
  const [highlightCells, setHighlightCells] = useState<Coordinates[]>([]);
  const [forceScale, setForceScale] = useState(true);
  const [verboseLogging, setVerboseLogging] = useState<boolean>(false);
  
  // Search strategy state
  const [searchStrategy, setSearchStrategy] = useState<string>('exploration');
  const [availableStrategies, setAvailableStrategies] = useState<Strategy[]>([
    { id: 'exploration', name: 'Exploration Priority', description: 'Prioritizes unexplored areas of the grid' },
    { id: 'random_walk', name: 'Random Walk', description: 'Uses randomness with a bias toward unexplored areas' },
    { id: 'spiral', name: 'Spiral Search', description: 'Explores in a spiral pattern to systematically cover the board' }
  ]);

  // Grid state
  const [fullArray, setFullArray] = useState<Grid>([]);
  const [currentArray, setCurrentArray] = useState<(number | null)[][]>([]);
  const [coords, setCoords] = useState<[Coordinates, Coordinates]>([[0, 0], [4, 4]]);
  const [magicGrids, setMagicGrids] = useState<MagicGrid[]>([]);
  const [windowCenter, setWindowCenter] = useState<Coordinates>([2, 2]);
  
  // Create a ref to the grid container
  const gridRef = useRef<HTMLDivElement>(null);
  
  // Custom function state
  const [showCustomFunctionForm, setShowCustomFunctionForm] = useState(false);
  const [customFunctionName, setCustomFunctionName] = useState('');
  const [customFunctionCode, setCustomFunctionCode] = useState('');
  const [customFunctionAuthor, setCustomFunctionAuthor] = useState('');
  const [customFunctionError, setCustomFunctionError] = useState<string | null>(null);
  
  // Fetch available strategies from the backend
  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/available_strategies`, {
          credentials: 'include'
        });
        if (response.ok) {
          const data = await response.json();
          setAvailableStrategies(data.strategies);
        }
      } catch (error) {
        console.error('Error fetching strategies:', error);
      }
    };
    
    fetchStrategies();
  }, []);

  // This handler updates every cell's transform based on mouse distance.
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!gridRef.current) return;
    const cells = gridRef.current.children;
    for (let i = 0; i < cells.length; i++) {
      const cell = cells[i] as HTMLElement;
      const rect = cell.getBoundingClientRect();
      const cellCenterX = rect.left + rect.width / 2;
      const cellCenterY = rect.top + rect.height / 2;
      const dx = e.pageX - cellCenterX;
      const dy = e.pageY - cellCenterY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      // Calculate mouse scaling factor
      const mouseScale = Math.max(1, Math.min(1 + ((70 - distance) / 175), 2));
      // Retrieve the base scale stored on each cell (set during rendering)
      const baseScale = parseFloat(cell.getAttribute('data-base-scale') || "1");
      const combinedScale = baseScale * mouseScale;
      cell.style.transform = `scale(${combinedScale})`;
    }
  };

  useEffect(() => {
    if (difficulty === 'medium') {
      const newHighlights: Coordinates[] = magicGrids.map(grid => [
        grid.row + Math.floor(Math.random() * grid.size),
        grid.col + Math.floor(Math.random() * grid.size)
      ]);
      setHighlightCells(newHighlights);
    } else {
      setHighlightCells([]);
    }
  }, [magicGrids, difficulty]);

  const unmask = (direction: 'up' | 'down' | 'left' | 'right') => {
    let newCenter: Coordinates = [...windowCenter] as Coordinates;

    switch (direction) {
      case 'up':
        if (newCenter[0] > 0) newCenter[0] = newCenter[0] - 1;
        break;
      case 'down':
        if (newCenter[0] < boardSize - 1) newCenter[0] = newCenter[0] + 1;
        break;
      case 'left':
        if (newCenter[1] > 0) newCenter[1] = newCenter[1] - 1;
        break;
      case 'right':
        if (newCenter[1] < boardSize - 1) newCenter[1] = newCenter[1] + 1;
        break;
      default:
        break;
    }

    const newCoords = calculateWindowCoords(newCenter, boardSize);
    setWindowCenter(newCenter);
    setCoords(newCoords);

    const newArray = currentArray.map(row => [...row]);
    const [[newMinRow, newMinCol], [newMaxRow, newMaxCol]] = newCoords;
    for (let i = newMinRow; i <= newMaxRow; i++) {
      for (let j = newMinCol; j <= newMaxCol; j++) {
        newArray[i][j] = fullArray[i][j];
      }
    }
    setCurrentArray(newArray);
    
    // Call the backend after unmasking
    if (!aiMode) detectMagicSquare();
  };

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Only process keyboard events if not in a text input or textarea
      const target = event.target as HTMLElement;
      const isInputActive = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA';
      
      // If we're in a form input, don't process arrow keys or Enter
      if (isInputActive) return;
      
      switch (event.key) {
        case 'ArrowUp':
          unmask('up');
          break;
        case 'ArrowDown':
          unmask('down');
          break;
        case 'ArrowLeft':
          unmask('left');
          break;
        case 'ArrowRight':
          unmask('right');
          break;
        case 'Enter': { // Change from 'Spacebar' to 'Enter'
          event.preventDefault();
          const [[minRow, minCol], [maxRow, maxCol]] = coords;
          const visibleMagicGrids = magicGrids.filter(grid => {
            const gridCenterRow = grid.row + Math.floor(grid.size / 2);
            const gridCenterCol = grid.col + Math.floor(grid.size / 2);
            return gridCenterRow >= minRow && gridCenterRow <= maxRow &&
                   gridCenterCol >= minCol && gridCenterCol <= maxCol &&
                   !grid.revealed;
          });
          if (visibleMagicGrids.length === 0) break;
          const gridToReveal = visibleMagicGrids.reduce((closest, current) => {
            const currentCenterRow = current.row + Math.floor(current.size / 2);
            const currentCenterCol = current.col + Math.floor(current.size / 2);
            const closestCenterRow = closest.row + Math.floor(closest.size / 2);
            const closestCenterCol = closest.col + Math.floor(closest.size / 2);
            const currentDistance = Math.abs(currentCenterRow - windowCenter[0]) + Math.abs(currentCenterCol - windowCenter[1]);
            const closestDistance = Math.abs(closestCenterRow - windowCenter[0]) + Math.abs(closestCenterCol - windowCenter[1]);
            return currentDistance < closestDistance ? current : closest;
          }, visibleMagicGrids[0]);

          setMagicGrids(prev =>
            prev.map(grid => {
              const gridCenterRow = grid.row + Math.floor(grid.size / 2);
              const gridCenterCol = grid.col + Math.floor(grid.size / 2);
              if (gridCenterRow === gridToReveal.row + Math.floor(gridToReveal.size / 2) &&
                  gridCenterCol === gridToReveal.col + Math.floor(gridToReveal.size / 2)) {
                return { ...grid, revealed: true };
              }
              return grid;
            })
          );
          break;
        }
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [coords, magicGrids, currentArray, fullArray, windowCenter, boardSize]);

  const generateArray = () => {
    const arr: Grid = Array(boardSize).fill(0).map(() => 
      Array(boardSize).fill(0).map(() => Math.floor(Math.random() * MAX_NUM))
    );
    setFullArray(arr);
    
    const initialCenter: Coordinates = [2, 2];
    const initialCoords = calculateWindowCoords(initialCenter, boardSize);
    const [[initialMinRow, initialMinCol], [initialMaxRow, initialMaxCol]] = initialCoords;
    
    const maskedArr: (number | null)[][] = arr.map((row, i) => 
      row.map((val, j) => 
        (i >= initialMinRow && i <= initialMaxRow && j >= initialMinCol && j <= initialMaxCol)
          ? val
          : null
      )
    );
    setCurrentArray(maskedArr);
    
    setCoords(initialCoords);
    setWindowCenter(initialCenter);
    setMagicGrids([]);
    setForceScale(true);
    setToastMessage("");
  };

  const resetGame = async () => {
    // Don't regenerate magic squares, only reset backend state
    // Remove this line: generateBoardWithMagicGrids();
    
    // Reset frontend state without changing magic squares
    const initialCenter: Coordinates = [2, 2];
    const initialCoords = calculateWindowCoords(initialCenter, boardSize);
    const [[initialMinRow, initialMinCol], [initialMaxRow, initialMaxCol]] = initialCoords;
    
    // Reset the current array to only show the initial window view
    const maskedArr: (number | null)[][] = fullArray.map((row, i) => 
      row.map((val, j) => 
        (i >= initialMinRow && i <= initialMaxRow && j >= initialMinCol && j <= initialMaxCol)
          ? val
          : null
      )
    );
    setCurrentArray(maskedArr);
    setCoords(initialCoords);
    setWindowCenter(initialCenter);
    setForceScale(true);
    
    // Also reset the revealed state of magic grids
    setMagicGrids(prev => 
      prev.map(grid => ({ ...grid, revealed: false }))
    );
    
    // Reset backend state
    try {
      const response = await fetch(`${API_BASE_URL}/api/reset_game`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ 
          total_magic_squares: magic3Qty + magic5Qty,
          strategy: searchStrategy
        }), 
      });
      
      const data = await response.json();
      setGameStats({
        steps: 0,
        magicSquaresFound: 0,
        gameOver: false
      });
      setAiMode(false);
    } catch (error) {
      console.error('Error resetting game:', error);
      setToastMessage("Error connecting to backend server");
      setTimeout(() => setToastMessage(""), 4000);
    }
  };

  const generateBoardWithMagicGrids = () => {
    const arr: Grid = Array(boardSize)
      .fill(0)
      .map(() =>
        Array(boardSize)
          .fill(0)
          .map(() => Math.floor(Math.random() * MAX_NUM))
      );

    const initialCenter: Coordinates = [2, 2];
    const initialCoords = calculateWindowCoords(initialCenter, boardSize);
    const [[initialMinRow, initialMinCol], [initialMaxRow, initialMaxCol]] = initialCoords;
    const maskedArr: (number | null)[][] = arr.map((row, i) =>
      row.map((val, j) =>
        (i >= initialMinRow && i <= initialMaxRow && j >= initialMinCol && j <= initialMaxCol)
          ? val
          : null
      )
    );

    // Prepare to add magic grids.
    const small_size = 3;
    const large_size = 5;
    const magicGridLarge = generateMagicSumGrid(large_size, MAGIC_CONSTANT5);
    const magicGridSmall = generateMagicSumGrid(small_size, MAGIC_CONSTANT3);
    const newMagicGrids: MagicGrid[] = [];
    const newFullArray = arr.map(row => [...row]);

    const gridsToPlace = [
      { name: 'large', grid: magicGridLarge, qty: magic5Qty, size: large_size },
      { name: 'small', grid: magicGridSmall, qty: magic3Qty, size: small_size },
    ];

    let placedLarge = 0;
    let placedSmall = 0;

    for (const { name, grid, qty, size } of gridsToPlace) {
      for (let count = 0; count < qty; count++) {
        let placed = false;
        let tries = 0;
        while (!placed && tries < 20) {
          const startRow = Math.floor(Math.random() * (boardSize - size));
          const startCol = Math.floor(Math.random() * (boardSize - size));

          const overlap = newMagicGrids.some(existing => {
            return !(
              startRow + size <= existing.row ||
              existing.row + existing.size <= startRow ||
              startCol + size <= existing.col ||
              existing.col + existing.size <= startCol
            );
          });

          if (!overlap) {
            for (let i = 0; i < size; i++) {
              for (let j = 0; j < size; j++) {
                newFullArray[startRow + i][startCol + j] = grid[i][j];
              }
            }
            
            newMagicGrids.push({
              row: startRow,
              col: startCol,
              size: size,
              revealed: false,
            });
            placed = true;
            if (name === 'large') placedLarge++;
            else if (name === 'small') placedSmall++;
          }
          tries++;
        }
      }
    }
    
    const newCurrentArray = newFullArray.map((row, i) =>
      row.map((val, j) =>
        (i >= initialMinRow && i <= initialMaxRow && j >= initialMinCol && j <= initialMaxCol)
          ? val
          : null
      )
    );

    setFullArray(newFullArray);
    setCurrentArray(newCurrentArray);
    setCoords(initialCoords);
    setWindowCenter(initialCenter);
    setMagicGrids(newMagicGrids);
    setForceScale(true);

    const totalRequestedLarge = magic5Qty;
    const totalRequestedSmall = magic3Qty;
    const message = `Successfully placed ${placedLarge} of ${totalRequestedLarge} large magic square(s), and ${placedSmall} of ${totalRequestedSmall} small magic square(s).`;
    setToastMessage(message);
    setTimeout(() => setToastMessage(""), 4000);
  };

  const detectMagicSquare = async () => {
    // Use NaN instead of null for unexplored cells
    const gridToSend = currentArray.map(row => 
      row.map(val => val === null ? Number.NaN : val)
    );
    
    try {
      console.log('Attempting to connect to backend at:', `${API_BASE_URL}/api/detect_magic_square`);
      const response = await fetch(`${API_BASE_URL}/api/detect_magic_square`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          grid: gridToSend,
          strategy: searchStrategy,
          window_coords: coords
        }),
        credentials: 'include'
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      setGameStats({
        steps: data.steps,
        magicSquaresFound: data.magic_squares_found,
        gameOver: data.game_over
      });
      
      if (data.is_magic_square && data.square_info) {
        // Highlight the magic square found
        setToastMessage("Magic square found!");
        setTimeout(() => setToastMessage(""), 2000);
        
        // Find all potential magic squares in the current view window
        const [[minRow, minCol], [maxRow, maxCol]] = coords;
        
        // Log the found square information
        console.log(`Backend found a magic square: ${JSON.stringify(data.square_info)}`);
        
        // Extract the position and size from the square_info
        const { type, position } = data.square_info;
        const [squareRow, squareCol] = position;
        const squareSize = type === '3x3' ? 3 : 5;
        
        // Calculate the center of the found magic square
        const centerRow = squareRow + Math.floor(squareSize / 2);
        const centerCol = squareCol + Math.floor(squareSize / 2);
        
        // Mark this square in our current grid view if it's visible
        if (centerRow >= minRow && centerRow <= maxRow && 
            centerCol >= minCol && centerCol <= maxCol) {
          
          // Create a new copy of the current array
          const newArray = currentArray.map(row => [...row]);
          
          // Mark all cells in the found magic square with Infinity, not just the center
          for (let r = squareRow; r < squareRow + squareSize; r++) {
            for (let c = squareCol; c < squareCol + squareSize; c++) {
              // Check if this cell is within the visible window and has a valid value
              if (r >= minRow && r <= maxRow && c >= minCol && c <= maxCol && 
                  newArray[r] && newArray[r][c] !== null) {
                console.log(`Marking found magic square cell at [${r}, ${c}] with Infinity`);
                newArray[r][c] = Infinity;
              }
            }
          }
          
          setCurrentArray(newArray);
        }
        
        // Check if the center of the found square is in the visible window
        const isCenterVisible = centerRow >= minRow && centerRow <= maxRow && 
                               centerCol >= minCol && centerCol <= maxCol;
        
        if (isCenterVisible) {
          // Get all unrevealed magic grids in the current window whose center matches
          // the detected magic square's center
          const visibleMagicGrids = magicGrids.filter(grid => {
            const gridCenterRow = grid.row + Math.floor(grid.size / 2);
            const gridCenterCol = grid.col + Math.floor(grid.size / 2);
            
            // Check if this grid matches the detected magic square's position
            const isMatch = Math.abs(gridCenterRow - centerRow) <= 1 && 
                           Math.abs(gridCenterCol - centerCol) <= 1 &&
                           grid.size === squareSize &&
                           !grid.revealed;
            
            return isMatch;
          });
          
          if (visibleMagicGrids.length > 0) {
            // Choose the closest magic grid to the found square
            const gridToReveal = visibleMagicGrids[0];
            
            // Reveal the chosen magic grid
            setMagicGrids(prev =>
              prev.map(grid => {
                if (grid.row === gridToReveal.row && grid.col === gridToReveal.col) {
                  return { ...grid, revealed: true };
                }
                return grid;
              })
            );
            
            console.log(`Magic square found and revealed at row: ${gridToReveal.row}, col: ${gridToReveal.col}`);
          } else {
            console.log("Magic square detected but couldn't match it to a known magic grid in the current view");
          }
        } else {
          console.log("Magic square found but it's not in the current visible window");
        }
      }
      
      if (aiMode && !data.game_over) {
        // Move in the suggested direction
        unmask(data.next_direction as 'up' | 'down' | 'left' | 'right');
      }
      
      return data;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      console.error('Error connecting to backend:', error);
      setToastMessage(`Error connecting to backend: ${errorMessage}`);
      setTimeout(() => setToastMessage(""), 4000);
      if (onError) onError(errorMessage);
      return null;
    }
  };
  
  const toggleAIMode = async () => {
    const newAiMode = !aiMode;
    setAiMode(newAiMode);
    
    if (newAiMode) {
      // Start AI playing
      const data = await detectMagicSquare();
      if (data && !data.game_over) {
        unmask(data.next_direction as 'up' | 'down' | 'left' | 'right');
      }
    }
  };
  
  // Effect for AI mode moves
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (aiMode && !gameStats.gameOver) {
      timer = setTimeout(async () => {
        const data = await detectMagicSquare();
        if (data && !data.game_over) {
          unmask(data.next_direction as 'up' | 'down' | 'left' | 'right');
        }
      }, 200); // Move every 200ms (5 times per second) instead of 1000ms
    }
    
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [aiMode, coords, gameStats.gameOver]);
  
  // Initialize on component mount
  useEffect(() => {
    resetGame();
  }, []);

  const [minCoords, maxCoords] = coords;

  const validateAndAddCustomFunction = () => {
    if (!customFunctionName.trim()) {
      setCustomFunctionError("Function name cannot be empty");
      return;
    }
    
    if (!customFunctionAuthor.trim()) {
      setCustomFunctionError("Please enter your name as the author");
      return;
    }
    
    if (!customFunctionCode.trim()) {
      setCustomFunctionError("Function code cannot be empty");
      return;
    }
    
    // Basic validation that it looks like a function
    if (!customFunctionCode.includes('def') || !customFunctionCode.includes('grid') || !customFunctionCode.includes('return')) {
      setCustomFunctionError("Your function should have a 'def' statement, accept a 'grid' parameter, and include a 'return' statement");
      return;
    }
    
    // Submit to backend for validation and registration
    fetch(`${API_BASE_URL}/api/add_custom_strategy`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify({
        name: customFunctionName,
        author: customFunctionAuthor,
        code: customFunctionCode
      }),
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        setCustomFunctionError(data.error);
      } else {
        // Success - add to available strategies
        const newStrategy = {
          id: data.id,
          name: `${customFunctionName} (by ${customFunctionAuthor})`,
          description: data.description || 'Custom search strategy'
        };
        
        setAvailableStrategies(prev => [...prev, newStrategy]);
        setSearchStrategy(data.id);
        setCustomFunctionError(null);
        setShowCustomFunctionForm(false);
        setToastMessage(`Custom strategy "${customFunctionName}" added successfully!`);
        setTimeout(() => setToastMessage(""), 4000);
        
        // Reset form
        setCustomFunctionName('');
        setCustomFunctionCode('');
        setCustomFunctionAuthor('');
      }
    })
    .catch(error => {
      console.error('Error adding custom function:', error);
      setCustomFunctionError("Error submitting function to server. Please check your connection.");
    });
  };

  const toggleVerboseLogging = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/toggle_verbose_logging`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ 
          verbose: !verboseLogging 
        }),
      });
      
      const data = await response.json();
      setVerboseLogging(data.verbose_logging);
      setToastMessage(`Verbose logging ${data.verbose_logging ? 'enabled' : 'disabled'}`);
      setTimeout(() => setToastMessage(""), 3000);
    } catch (error) {
      console.error('Error toggling verbose logging:', error);
      setToastMessage("Error connecting to backend server");
      setTimeout(() => setToastMessage(""), 4000);
    }
  };

  return (
    <Card className="p-6 Card" style={{ background: "#00203a", color: "#24e4f2" }}>
      <CardContent>
        <div className="flex justify-between mb-4">
          <div className="text-white">
            Steps: {gameStats.steps}/100 | Magic Squares: {gameStats.magicSquaresFound}
          </div>
          <Button
            variant="outline"
            onClick={toggleAIMode}
            className="bg-[#00203a] hover:bg-[#001f2f] text-[#24e4f2] hover:text-[#24e4f2]"
            disabled={gameStats.gameOver}
          >
            {aiMode ? <Pause className="h-4 w-4 mr-2" /> : <Play className="h-4 w-4 mr-2" />}
            {aiMode ? 'Stop AI' : 'Start AI'}
          </Button>
        </div>
          
        {gameStats.gameOver && (
          <div className="bg-purple-900 p-3 rounded mb-4 text-white text-center">
            Game Over! Final Score: {gameStats.magicSquaresFound} magic squares found in {gameStats.steps} steps.  <Button
              variant="outline"
              onClick={resetGame}
              className="mt-2 bg-purple-800 hover:bg-purple-700 text-white hover:text-white"
            >
              New Game
            </Button>
          </div>
        )}
          
        <div className="relative">
          <div 
            ref={gridRef}
            onMouseMove={handleMouseMove}
            className={`grid gap-1 array-grid ${forceScale ? 'force-scale' : ''}`}
            style={{ gridTemplateColumns: `repeat(${boardSize}, minmax(0, 1fr))` }}
            onMouseEnter={() => setForceScale(false)}
          >
            {currentArray.map((row, i) =>
              row.map((val, j) => {
                const isWithinWindow = i >= minCoords[0] && i <= maxCoords[0] && j >= minCoords[1] && j <= maxCoords[1];
                const isCenterCell = (i === windowCenter[0] && j === windowCenter[1]);

                const inMagicGrid = magicGrids.some(grid =>
                  i >= grid.row && i < grid.row + grid.size &&
                  j >= grid.col && j < grid.col + grid.size
                );

                let magicTextClass = "";
                if (val !== null) {
                  if (val === Infinity) {
                    // Special styling for found magic square centers
                    magicTextClass = "text-purple-600 font-bold";
                  } else if (difficulty === 'easy' && inMagicGrid) {
                    magicTextClass = "text-pink-600";
                  } else if (difficulty === 'medium' && highlightCells.some(([r, c]) => r === i && c === j)) {
                    magicTextClass = "text-pink-600";
                  }
                }

                if (magicGrids.some(grid =>
                    i >= grid.row && i < grid.row + grid.size &&
                    j >= grid.col && j < grid.col + grid.size && grid.revealed
                )) {
                  magicTextClass = "text-green-600";
                }

                const magicCenterGrid = magicGrids.find(grid =>
                  i === grid.row + Math.floor(grid.size / 2) &&
                  j === grid.col + Math.floor(grid.size / 2)
                );

                // Compute the "base scale" based on grid–position relative to the current sliding window.
                let baseScale = 1;
                if (isWithinWindow) {
                  const windowHeight = maxCoords[0] - minCoords[0] + 1;
                  const windowWidth = maxCoords[1] - minCoords[1] + 1;
                  const centerRowIndex = windowCenter[0] - minCoords[0];
                  const centerColIndex = windowCenter[1] - minCoords[1];
                  const relativeRow = i - minCoords[0];
                  const relativeCol = j - minCoords[1];
                  const dist = Math.max(
                    Math.abs(relativeRow - centerRowIndex),
                    Math.abs(relativeCol - centerColIndex)
                  );
                  if (dist === 0) {
                    baseScale = 3.0;
                  } else if (dist === 1) {
                    baseScale = 1.8;
                  } else {
                    baseScale = 1.2;
                  }
                }

                const cellBg = val === Infinity ? 'bg-purple-900' : 'bg-transparent';

                return (
                  <div
                    key={`${i}-${j}`}
                    onMouseEnter={
                      isWithinWindow
                        ? () => {
                            /* Remove these lines that change position on mouse hover */
                            /* if (i === minCoords[0]) unmask('up');
                            if (i === maxCoords[0]) unmask('down');
                            if (j === minCoords[1]) unmask('left');
                            if (j === maxCoords[1]) unmask('right'); */
                          }
                        : undefined
                    }
                    onClick={
                      magicCenterGrid && !magicCenterGrid.revealed
                        ? () => {
                            setMagicGrids(prev =>
                              prev.map(grid => {
                                const cRow = grid.row + Math.floor(grid.size / 2);
                                const cCol = grid.col + Math.floor(grid.size / 2);
                                if (i === cRow && j === cCol) {
                                  return { ...grid, revealed: true };
                                }
                                return grid;
                              })
                            );
                          }
                        : undefined
                    }
                    className={`
                      h-12 w-12 flex items-center justify-center
                      bg-transparent
                      ${isWithinWindow ? 'ring-[#24e4f2]' : ''}
                      ${val === Infinity ? 'text-green-600' : magicTextClass ? magicTextClass : "text-[#24e4f2]"}
                      ${isCenterCell ? 'center-window' : ''}
                    `}
                    data-base-scale={baseScale}
                    style={{ transform: `scale(${baseScale})` }}
                  >
                    {val !== null ? (val === Infinity ? fullArray[i][j] : val) : ''}
                  </div>
                );
              })
            )}
          </div>
          
          <div className="flex gap-4 justify-center mt-4">
            <Button
              variant="outline"
              onClick={resetGame}
              className="bg-[#00203a] hover:bg-[#001f2f] text-[#24e4f2] hover:text-[#24e4f2]"
            >
              <RotateCcw className="h-4 w-4 mr-2" /> New Game
            </Button>
            
            <Button
              variant="outline"
              onClick={generateBoardWithMagicGrids}
              className="bg-[#00203a] hover:bg-[#001f2f] text-[#24e4f2] hover:text-[#24e4f2]"
            >
              <Wand2 className="h-4 w-4 mr-2" /> Generate Magic Squares
            </Button>
            
            <Button
              variant="outline"
              onClick={() => setShowCustomFunctionForm(true)}
              className="bg-[#00203a] hover:bg-[#001f2f] text-[#24e4f2] hover:text-[#24e4f2]"
            >
              <Code className="h-4 w-4 mr-2" /> Add Custom Strategy
            </Button>

            <Button
              variant="outline"
              onClick={toggleVerboseLogging}
              className={`${verboseLogging ? 'bg-[#004060]' : 'bg-[#00203a]'} hover:bg-[#001f2f] text-[#24e4f2] hover:text-[#24e4f2]`}
            >
              {verboseLogging ? 'Disable Logging' : 'Enable Logging'}
            </Button>
          </div>
        </div>

        {/* Divider */}
        <div className="mt-6 mb-4 border-t border-gray-500"></div>

        {/* Controls moved to the bottom, arranged in a column */}
        <div className="flex flex-col gap-4">
          <div>
            <label className="mr-2">Board Size:</label>
            <input
              type="number"
              value={boardSize}
              min="5"
              max="50"
              onChange={(e) => setBoardSize(parseInt(e.target.value) || 5)}
              className="p-1 rounded border border-gray-300 text-black"
            />
          </div>
          <div>
            <label className="mr-2">Large Magic Squares Qty:</label>
            <input
              type="number"
              value={magic5Qty}
              min="0"
              onChange={(e) => setMagic5Qty(parseInt(e.target.value) || 0)}
              className="p-1 rounded border border-gray-300 text-black"
            />
          </div>
          <div>
            <label className="mr-2">Small Magic Squares Qty:</label>
            <input
              type="number"
              value={magic3Qty}
              min="0"
              onChange={(e) => setMagic3Qty(parseInt(e.target.value) || 0)}
              className="p-1 rounded border border-gray-300 text-black"
            />
          </div>
          <div>
            <label className="mr-2">Hints:</label>
            <select
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value as 'easy' | 'medium' | 'hard')}
              className="p-1 rounded border border-gray-300 text-black"
            >
              <option value="easy">Reveal All</option>
              <option value="medium">Reveal One</option>
              <option value="hard">Hidden</option>
            </select>
          </div>
          <div>
            <label className="mr-2">Search Strategy:</label>
            <select
              value={searchStrategy}
              onChange={(e) => setSearchStrategy(e.target.value)}
              className="p-1 rounded border border-gray-300 text-black"
            >
              {availableStrategies.map(strategy => (
                <option key={strategy.id} value={strategy.id} title={strategy.description}>
                  {strategy.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Toast message */}
        {toastMessage && (
          <div className="fixed bottom-4 right-4 bg-yellow-300 text-black px-4 py-2 rounded shadow-lg">
            {toastMessage}
          </div>
        )}

        {/* Custom function form */}
        {showCustomFunctionForm && (
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div className="bg-[#001525] p-6 rounded-lg shadow-lg w-2/3 max-w-3xl">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-[#24e4f2]">Add Custom Search Strategy</h2>
                <Button 
                  variant="ghost" 
                  onClick={() => setShowCustomFunctionForm(false)}
                  className="text-[#24e4f2] hover:bg-[#00203a]"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              
              {customFunctionError && (
                <div className="bg-red-900 text-white p-3 rounded mb-4">
                  {customFunctionError}
                </div>
              )}
              
              <div className="mb-4">
                <label className="block mb-2">Your Name:</label>
                <input
                  type="text"
                  value={customFunctionAuthor}
                  onChange={(e) => setCustomFunctionAuthor(e.target.value)}
                  placeholder="Enter your name"
                  className="w-full p-2 bg-[#002040] text-white border border-gray-600 rounded"
                />
              </div>
              
              <div className="mb-4">
                <label className="block mb-2">Strategy Name:</label>
                <input
                  type="text"
                  value={customFunctionName}
                  onChange={(e) => setCustomFunctionName(e.target.value)}
                  placeholder="Name your custom strategy"
                  className="w-full p-2 bg-[#002040] text-white border border-gray-600 rounded"
                />
              </div>
              
              <div className="mb-4">
                <label className="block mb-2">Python Function Code:</label>
                <textarea
                  value={customFunctionCode}
                  onChange={(e) => setCustomFunctionCode(e.target.value)}
                  placeholder={`def custom_search_strategy(grid, window_coords):
    """
    Your custom search strategy
    
    Args:
        grid: 2D numpy array with NaN for unexplored cells
        window_coords: Tuple containing window coordinates ((min_row, min_col), (max_row, max_col))
        
    Returns:
        str: Direction to move next ('up', 'down', 'left', 'right')
    """
    rows, cols = grid.shape
    
    # Extract window coordinates
    (min_row, min_col), (max_row, max_col) = window_coords
    
    # Your code here
    # Use window coordinates to make smarter movement decisions
    
    return 'up'  # Return a direction`}
                  className="w-full h-64 p-2 font-mono bg-[#002040] text-white border border-gray-600 rounded"
                />
              </div>
              
              <div className="flex justify-end">
                <Button
                  variant="outline"
                  onClick={() => setShowCustomFunctionForm(false)}
                  className="mr-2 bg-[#00203a] hover:bg-[#001f2f] text-[#24e4f2] hover:text-[#24e4f2]"
                >
                  Cancel
                </Button>
                <Button
                  variant="outline"
                  onClick={validateAndAddCustomFunction}
                  className="bg-[#004060] hover:bg-[#005070] text-[#24e4f2] hover:text-[#24e4f2]"
                >
                  Add Strategy
                </Button>
              </div>
            </div>
          </div>
        )}

        <style jsx global>{`
          @import url('https://unpkg.com/open-props/easings.min.css');

          .Card {
              cursor: url('data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%2248%22%20height%3D%2248%22%20viewBox%3D%22-12%20-12%2048%2048%22%3E%3Cpath%20fill%3D%22%2300203a%22%20stroke%3D%22%2324e4f2%22%20stroke-width%3D%222%22%20d%3D%22M5.5%203.21V20.8c0%20.45.54.67.85.35l4.86-4.86a.5.5%200%200%201%20.35-.15h6.87a.5.5%200%200%200%20.35-.85L6.35%202.85a.5.5%200%200%200-.85.35Z%22/%3E%3C/svg%3E') 16 16, auto;
          }
          .Card:active {
              cursor: url('data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%2248%22%20height%3D%2248%22%20viewBox%3D%22-12%20-12%2048%2048%22%3E%3Cpath%20fill%3D%22%2300203a%22%20stroke%3D%22%2324e4f2%22%20stroke-width%3D%222%22%20d%3D%22M5.5%203.21V20.8c0%20.45.54.67.85.35l4.86-4.86a.5.5%200%200%201%20.35-.15h6.87a.5.5%200%200%200%20.35-.85L6.35%202.85a.5.5%200%200%200-.85.35Z%22/%3E%3C/svg%3E') 16 16, auto;
          }
          .relative {}

          .relative:active {}

          /* The original CSS scaling rules are preserved here but will be overridden by our inline transforms. */
          @media (hover: hover) and (prefers-reduced-motion: no-preference) {
            .array-grid > div {
              transform-origin: center;
              transition: transform 0.3s var(--ease-3);
            }
          }
        `}</style>
      </CardContent>
    </Card>
  );
};

export default MagicSquareVisualizer; 