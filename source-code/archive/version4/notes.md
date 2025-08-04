Here is notes that I am doing for version 4 from version 3:

Version 4 changes:
---
- Adding condition that once an object is placed down it cannot be picked up again
- Creating exit condition in step given state to handle more diverse human preferences
- Adding unit tests (multiple object combinations + different preference trees)

**Key Observation**
By adding the condition of once an object has been placed down it cannot be picked up again, it is only adding a couple states instead of cutting down on states as we expect. It does do some sort of restriction as if we do add another parameter per object it doesn't generate the full number of states.

Here is the attributes each objects as in version 4:

```
            state['pos'] = copy.deepcopy(self.initial_object_locs[obj])  # type tuple (color, type) to location
            state['orientation'] = np.pi  # The orientation options are 0 or pi, which correspond to 0 or
            # 180 degrees for the cup
            state['holding'] = False
            state['done'] = False
```

Number of states w/o constraint: `Found states:  296`
Number of states w/ constraints: `Found states:  198`

So in previous versions, the number of states with 1 object with the same enviroment of:
    self.x_min = -3
    self.x_max = 4
    self.y_min = -3
    self.y_max = 4

Will generate 196 states. So as we can see with the constraint it doesn't trim down the base line number of states.

Unittest Cases
----
In Version 4 I created unittests. This has multiple abstract preferenes to test our program to the fullest with one, two, and three objects.

Here are the human preferences we are using:
```python
f_Ethan = {'pref_values': {"cup": {'red': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 41},
                                   'yellow': {'Q1': 45, 'Q2': -1, 'Q3': 100, 'Q4': -1}},
                           'bowl': {'glass': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                    'plastic': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': -1},
                                    'china': {'Q1': -1, 'Q2': 5, 'Q3': 90, 'Q4': 5}}}}


f_Michelle = {'pref_values': {'glass': {'red': {'Q1': 80, 'Q2': 1100, 'Q3': -1, 'Q4': 41},
                                        'yellow': {'Q1': 80, 'Q2': 100, 'Q3': 1111, 'Q4': 41}},
                              'china': {'cup': {'Q1': 80, 'Q2': 100, 'Q3': -1, 'Q4': 41},
                                        'bowl': {'red': {'Q1': 80, 'Q2': 100, 'Q3': -1, 'Q4': 410},
                                                 'yellow': {'Q1': 80, 'Q2': 100, 'Q3': 221, 'Q4': 41}}},
                              'plastic': {'Q1': 80, 'Q2': -1, 'Q3': -1, 'Q4': 41}}}


f_Annika = {'pref_values': {'cup': {'glass': {'red': {'Q1': 10, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                               'yellow': {'Q1': 20, 'Q2': -1, 'Q3': 15, 'Q4': 0}},
                                    'plastic': {'red': {'Q1': 5, 'Q2': -1, 'Q3': -1, 'Q4': 25}}},
                         'bowl': {'red': {'glass': {'Q1': 30, 'Q2': -1, 'Q3': 20, 'Q4': 10},
                                          'plastic': {'Q1': 5, 'Q2': -1, 'Q3': 10, 'Q4': 15}}}}}


f_Admoni = {'pref_values': {'red': {'cup': {'glass': {'Q1': 10, 'Q2': -1, 'Q3': 5, 'Q4': 20},
                                             'china': {'Q1': -1, 'Q2': 30, 'Q3': -1, 'Q4': 15}},
                                  'bowl': {'plastic': {'Q1': 10, 'Q2': 5, 'Q3': 15, 'Q4': -1}}},
                          'yellow': {'bowl': {'glass': {'Q1': 25, 'Q2': -1, 'Q3': 10, 'Q4': 20}}}}}


f_Simmons = {'pref_values': {'plastic': {'cup': {'red': {'Q1': 10, 'Q2': 20, 'Q3': 15, 'Q4': 25},
                                                 'yellow': {'Q1': 5, 'Q2': 10, 'Q3': 0, 'Q4': 5}},
                                        'bowl': {'Q1': 5, 'Q2': 0, 'Q3': -1, 'Q4': 20}},
                              'glass': {'bowl': {'red': {'Q1': 30, 'Q2': -1, 'Q3': 5, 'Q4': 10}}}}}
```

The objects we are going to do for these tests are:
- One object
- Two objects
- Two of the same object
- Three objects

In each of the test there will be a blend of objects, materials, and colors. The possible options for each are the following:

objects:
- cup
- bowl

materials:
- glass
- china
- plastic

color:
- yellow
- red

test_f_Ethan
-----
- set environment limits
    self.x_min = -2
    self.x_max = 2
    self.y_min = -2
    self.y_max = 2
- f_Ethan = {'pref_values': {"cup": {'red': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 41},
                                       'yellow': {'Q1': 45, 'Q2': -1, 'Q3': 100, 'Q4': -1}},
                               'bowl': {'glass': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                        'plastic': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': -1},
                                        'china': {'Q1': -1, 'Q2': 5, 'Q3': 90, 'Q4': 5}}}}

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


test_f_Michelle
-----
- set environment limits
    self.x_min = -2
    self.x_max = 2
    self.y_min = -2
    self.y_max = 2
- f_Michelle = {'pref_values': {'glass': {'red': {'Q1': 80, 'Q2': 1100, 'Q3': -1, 'Q4': 41},
                                            'yellow': {'Q1': 80, 'Q2': 100, 'Q3': 1111, 'Q4': 41}},
                                  'china': {'cup': {'Q1': 80, 'Q2': 100, 'Q3': -1, 'Q4': 41},
                                            'bowl': {'red': {'Q1': 80, 'Q2': 100, 'Q3': -1, 'Q4': 410},
                                                     'yellow': {'Q1': 80, 'Q2': 100, 'Q3': 221, 'Q4': 41},
                                                     'purple': {'Q1': 80, 'Q2': 100, 'Q3': -1, 'Q4': 410}}},
                                  'plastic': {'Q1': 80, 'Q2': -1, 'Q3': -1, 'Q4': 41}}}

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


test_f_Annika
-----
- set environment limits
    self.x_min = -2
    self.x_max = 2
    self.y_min = -2
    self.y_max = 2
- f_Annika = {'pref_values': {'cup': {'glass': {'red': {'Q1': 10, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                                  'yellow': {'Q1': 20, 'Q2': -1, 'Q3': 15, 'Q4': 0}},
                                        'plastic': {'red': {'Q1': 5, 'Q2': -1, 'Q3': -1, 'Q4': 25}}},
                                'bowl': {'purple': {'glass': {'Q1': 30, 'Q2': -1, 'Q3': 20, 'Q4': 10},
                                                 'plastic': {'Q1': 5, 'Q2': -1, 'Q3': 10, 'Q4': 15}}}}}

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

<!-- test_f_Admoni
-----
- set environment limits
    self.x_min = -2
    self.x_max = 2
    self.y_min = -2
    self.y_max = 2
- f_Admoni = {'pref_values': {'red': {'cup': {'glass': {'Q1': 10, 'Q2': -1, 'Q3': 5, 'Q4': 20},
                                                'china': {'Q1': -1, 'Q2': 30, 'Q3': -1, 'Q4': 15}},
                                        'bowl': {'plastic': {'Q1': 10, 'Q2': 5, 'Q3': 15, 'Q4': -1}}},
                                'yellow': {'bowl': {'glass': {'Q1': 25, 'Q2': -1, 'Q3': 10, 'Q4': 20}},
                                           'cup': {'glass': {'Q1': 10, 'Q2': -1, 'Q3': 5, 'Q4': 20}}}}}

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
- Completed: YES -->