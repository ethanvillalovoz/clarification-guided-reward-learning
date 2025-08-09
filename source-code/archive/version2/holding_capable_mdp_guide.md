# Multi-Object Preference Learning in Gridworld

## Object Holding Capability

Version 2 introduces the ability to handle multiple objects and visually indicate when the robot is holding a specific object. This feature adds significant complexity to the state space but enables more realistic interaction scenarios.

## Adding New Objects

To add a new object type to the simulation, follow these steps:

### 1. Prepare Images

Each object requires four distinct images:
- Regular object photo 
- Object rotated 180 degrees
- Regular photo with red borders (indicating holding state)
- Photo rotated 180 degrees with red borders (indicating holding state)

All images should be placed in the `data/images/` directory.

### 2. Update Code Configuration

Modify the following sections in the code:

#### Object Constants
```python
# Add new color/type constants as needed
RED = 1
YELLOW = 2
BLUE = 3  # New color example

CUP = 1
BOWL = 2  # New object type example

# Add to the color dictionary
COLOR_DICTIONARY = {1: 'red', 2: 'yellow', 3: 'blue'}
```

#### Update Reward Functions
Add sublist entries to these parameters:
```python
reward_weights = [
    [-10, -2, -2, -2],  # Red cup reward weights
    [-2, -2, 3, 4],     # Yellow cup reward weights
    [5, -3, 1, -2]      # New object reward weights
]

true_f_idx = [
    [1, 1, 1, 1],  # All features matter for red cup
    [1, 1, 1, 1],  # All features matter for yellow cup
    [1, 1, 1, 1]   # Feature importance for new object
]
```

#### Update Object List
```python
object_type_tuple = [RED_CUP, YELLOW_CUP, BLUE_BOWL]  # Add new object
```

### 3. Modify Rendering Function

Update the `render()` method to include image paths and rendering logic for the new object:

```python
# Add new image paths
path_blue = 'data/images/bluebowl.jpeg'
path180_blue = 'data/images/bluebowl_180.jpeg'
path_blue_holding = 'data/images/bluebowl_holding.jpeg'
path180_blue_holding = 'data/images/bluebowl_180_holding.jpeg'

# Add rendering condition for new object
elif type_o == (3, 2):  # Blue bowl
    if orientation == 0:
        if is_holding:
            ab = AnnotationBbox(getImage(path_blue_holding), (loc[0], loc[1]), frameon=False)
        else:
            ab = AnnotationBbox(getImage(path_blue), (loc[0], loc[1]), frameon=False)
    else:
        if is_holding:
            ab = AnnotationBbox(getImage(path180_blue_holding), (loc[0], loc[1]), frameon=False)
        else:
            ab = AnnotationBbox(getImage(path180_blue), (loc[0], loc[1]), frameon=False)
```

## Computational Performance

As you increase the environment grid size, computation time increases exponentially due to the state space growth.

### Hardware Reference

Testing was performed on a laptop with the following specifications:
- 2.6 GHz 6-Core Intel Core i7
- AMD Radeon Pro 5500M 8 GB, Intel UHD Graphics 630 1536 MB
- 32 GB 2667 MHz DDR4

### Performance Results

The largest environment configuration that completed in reasonable time had these settings:

```python
self.x_min = -2
self.x_max = 3
self.y_min = -2
self.y_max = 3
```

Performance metrics:
- Total states enumerated: 14,400
- Value iteration iterations: 10
- Total computation time: ~7 minutes

### State Space Growth

| Grid Size | Objects | Actions | Approx. States | Est. Time |
|-----------|---------|---------|----------------|-----------|
| 5×5       | 1       | 7       | 500            | <1 min    |
| 5×5       | 2       | 14      | 14,400         | 7 min     |
| 7×7       | 2       | 14      | 38,416         | 25+ min   |

## Future Improvements

1. **Automated Image Handling**: Create a more robust system for adding images for new objects
   - Consider implementing an object class with image loading methods
   - Use configuration files to specify image paths rather than hardcoding

2. **State Space Optimization**:
   - Implement hierarchical planning to reduce computation time
   - Use approximate value iteration for larger environments
   - Consider prioritized sweeping for faster convergence

3. **Parallel Processing**:
   - Parallelize state enumeration and value iteration
   - Consider GPU acceleration for matrix operations

4. **Dynamic Feature Extraction**:
   - Create a modular feature extraction system rather than hardcoded features
   - Allow for runtime addition of new features