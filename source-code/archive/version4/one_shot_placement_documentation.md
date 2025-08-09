# One-Shot Placement MDP - Version 4 Notes

## Key Changes from Version 3

- **One-Shot Object Placement**: Once an object is placed down, it cannot be picked up again
- **Enhanced Exit Handling**: Modified to handle more diverse human preference structures
- **Comprehensive Unit Tests**: Added test cases with multiple object combinations and preference trees

## Implementation Analysis

### Object State Representation

Each object in the environment now maintains four state attributes:
```python
state = {
    'pos': (x, y),         # Position coordinates in grid
    'orientation': np.pi,  # 0 or π (0° or 180°)
    'holding': False,      # Whether agent is holding object
    'done': False          # Whether object has been placed permanently
}
```

### State Space Analysis

Adding the one-shot placement constraint affects state space size:

- **Without constraint**: 296 states
- **With constraint**: 198 states

For reference, in previous versions with the same environment bounds:
```python
self.x_min = -3
self.x_max = 4
self.y_min = -3
self.y_max = 4
```
Single object environments generated 196 states.

The one-shot constraint reduces exploration paths but maintains a similar state count because it adds a new binary attribute (`done`) to track placement status.

## Unit Test Results

Tests conducted using multiple preference structures and object configurations in a 5×5 grid:
```python
self.x_min = -2
self.x_max = 2
self.y_min = -2
self.y_max = 2
```

### Test Configurations

| Configuration | Objects | State Count | Status |
|---------------|---------|------------|--------|
| Single Object | 1 | 66 | ✓ |
| Two Objects | 2 | 1,500 | ✓ |
| Two Identical Objects | 2 | 1,500 | ✓ |
| Three Objects | 3 | 28,700 | ✓ |

### Preference Structures

Tests used five different preference tree structures with varying complexity:

#### 1. Ethan's Preferences
```python
f_Ethan = {
    'pref_values': {
        "cup": {
            'red': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 41},
            'yellow': {'Q1': 45, 'Q2': -1, 'Q3': 100, 'Q4': -1}
        },
        'bowl': {
            'glass': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 10},
            'plastic': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': -1},
            'china': {'Q1': -1, 'Q2': 5, 'Q3': 90, 'Q4': 5}
        }
    }
}
```

#### 2. Michelle's Preferences
```python
f_Michelle = {
    'pref_values': {
        'glass': {
            'red': {'Q1': 80, 'Q2': 1100, 'Q3': -1, 'Q4': 41},
            'yellow': {'Q1': 80, 'Q2': 100, 'Q3': 1111, 'Q4': 41}
        },
        'china': {
            'cup': {'Q1': 80, 'Q2': 100, 'Q3': -1, 'Q4': 41},
            'bowl': {
                'red': {'Q1': 80, 'Q2': 100, 'Q3': -1, 'Q4': 410},
                'yellow': {'Q1': 80, 'Q2': 100, 'Q3': 221, 'Q4': 41},
                'purple': {'Q1': 80, 'Q2': 100, 'Q3': -1, 'Q4': 410}
            }
        },
        'plastic': {'Q1': 80, 'Q2': -1, 'Q3': -1, 'Q4': 41}
    }
}
```

#### 3. Annika's Preferences
```python
f_Annika = {
    'pref_values': {
        'cup': {
            'glass': {
                'red': {'Q1': 10, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                'yellow': {'Q1': 20, 'Q2': -1, 'Q3': 15, 'Q4': 0}
            },
            'plastic': {
                'red': {'Q1': 5, 'Q2': -1, 'Q3': -1, 'Q4': 25}
            }
        },
        'bowl': {
            'purple': {
                'glass': {'Q1': 30, 'Q2': -1, 'Q3': 20, 'Q4': 10},
                'plastic': {'Q1': 5, 'Q2': -1, 'Q3': 10, 'Q4': 15}
            }
        }
    }
}
```

#### 4. Admoni's Preferences
```python
f_Admoni = {
    'pref_values': {
        'red': {
            'cup': {
                'glass': {'Q1': 10, 'Q2': -1, 'Q3': 5, 'Q4': 20},
                'china': {'Q1': -1, 'Q2': 30, 'Q3': -1, 'Q4': 15}
            },
            'bowl': {
                'plastic': {'Q1': 10, 'Q2': 5, 'Q3': 15, 'Q4': -1}
            }
        },
        'yellow': {
            'bowl': {'glass': {'Q1': 25, 'Q2': -1, 'Q3': 10, 'Q4': 20}}
        }
    }
}
```

#### 5. Simmons's Preferences
```python
f_Simmons = {
    'pref_values': {
        'plastic': {
            'cup': {
                'red': {'Q1': 10, 'Q2': 20, 'Q3': 15, 'Q4': 25},
                'yellow': {'Q1': 5, 'Q2': 10, 'Q3': 0, 'Q4': 5}
            },
            'bowl': {'Q1': 5, 'Q2': 0, 'Q3': -1, 'Q4': 20}
        },
        'glass': {
            'bowl': {
                'red': {'Q1': 30, 'Q2': -1, 'Q3': 5, 'Q4': 10}
            }
        }
    }
}
```

## Object Properties

For testing, we used a variety of objects with different properties:

### Object Types
- cup
- bowl

### Materials
- glass
- china
- plastic

### Colors
- yellow
- red
- purple

## Test Execution Details

### Ethan's Preference Tests

**Environment Configuration:**
```python
self.x_min = -2
self.x_max = 2
self.y_min = -2
self.y_max = 2
```

1. One object
- obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
- Found states:  66
- Completed: YES

2. Two objects
- obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
- obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass', 'object_label': 2}
- Found states:  1500
- Completed: YES

3. Two of the same object
- obj_3 = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 3}
- obj_3_dup = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 4}
- Found states:  1500
- Completed: YES

4. Three objects
- obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
- obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass', 'object_label': 2}
- obj_3 = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 3}
- Found states:  28700
- Completed: YES


### Michelle's Preference Tests

**Environment Configuration:**
```python
self.x_min = -2
self.x_max = 2
self.y_min = -2
self.y_max = 2
```

1. One object
- obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
- Found states:  66
- Completed: YES

2. Two objects
- obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
- obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass', 'object_label': 2}
- Found states:  1500
- Completed: YES

3. Two of the same object
- obj_3 = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 3}
- obj_3_dup = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 4}
- Found states:  1500
- Completed: YES

4. Three objects
- obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
- obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass', 'object_label': 2}
- obj_3 = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 3}
- Found states:  28700
- Completed: YES


### Annika's Preference Tests

**Environment Configuration:**
```python
self.x_min = -2
self.x_max = 2
self.y_min = -2
self.y_max = 2
```

1. One object
- obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
- Found states:  66
- Completed: YES

2. Two objects
- obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
- obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass', 'object_label': 2}
- Found states:  1500
- Completed: YES

3. Two of the same object
- obj_3 = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 3}
- obj_3_dup = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 4}
- Found states:  1500
- Completed: YES

4. Three objects
- obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
- obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass', 'object_label': 2}
- obj_3 = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 3}
- Found states:  28700
- Completed: YES

## Performance Analysis

### State Growth Analysis

As the number of objects increases, the state space grows exponentially:

| Objects | State Count | Growth Factor |
|---------|------------|---------------|
| 1       | 66         | -             |
| 2       | 1,500      | 22.7x         |
| 3       | 28,700     | 19.1x         |

This exponential growth aligns with theoretical expectations for multi-object MDPs.

### Placement Strategy Impact

The one-shot placement constraint significantly reduces the exploration paths in the state space. Once an object is placed, those states are effectively removed from further exploration, resulting in more direct paths to goal states.

### Runtime Performance

All test configurations completed successfully with reasonable runtime performance, even with the 3-object configuration that generated nearly 30,000 states.

## Preference Tree Observations

1. **Depth vs. Width**: Preference trees with greater depth (like Annika's) lead to more complex decision-making processes but don't necessarily increase state space size.

2. **Value Distribution**: Trees with highly variable values across quadrants (like Michelle's preferences with values >1000) result in more distinct optimal policies.

3. **Material-First vs. Color-First Trees**: The ordering of attributes in the preference tree (whether color or material appears higher in the hierarchy) affects which clarification questions would be most valuable.

## Future Work

1. **Clarification-Based Guidance**: Implement interactive clarification questioning based on tree structure to guide object placement.

2. **Optimal Policy Extraction**: Develop methods to extract human-interpretable rules from learned policies.

3. **Multi-Agent Extensions**: Extend the model to handle conflicting preferences from multiple agents.

4. **Dynamic Environment Support**: Support for environments where preferences may change over time.