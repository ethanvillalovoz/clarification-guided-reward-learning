Here is notes that I am doing for version 3 from version 2:

- Previously our features from version 2 were [orientation, red prox, blue prox, pos y]
- Now our features are [Q1: num, Q2: num, Q3: num, Q4: num]
- This is extracted from a new idea of having a user preference tree as show in the image here: ![pref_values_example](/ethan_RISS_work/version3/pictures_for_notes_md/pref_values_example.png)
- Deleting `featurized_states()`
    - We are no longer using distance of the red or blue centroid to be used as features of the reward function
    - Instead we are just using a new function called `lookup_quadrant_reward()` which will look up a user's preference values tree to extract the reward value for where the object is for that given state
- The axis lines on the grid of the simulation is offset by 0.5 in the X direction and -0.5 in the y direction to distinguish where the objects are on in which quadrant when on the traiditonal axis lines: ![new_grid_format](/ethan_RISS_work/version3/pictures_for_notes_md/Code%20scratch%20book-6.jpg)

Stress Testing
=======

1. Test 1
-----
- Two objects:
    - obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass'}
    - obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass'}
- Enviroment Size:
    self.x_min = -2
    self.x_max = 3
    self.y_min = -2
    self.y_max = 3
- f_Ethan = {'pref_values': {"cup": {'red': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 41},
                                       'yellow': {'Q1': 45, 'Q2': -1, 'Q3': 100, 'Q4': -1}},
                               'bowl': {'glass': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                        'plastic': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': -1},
                                        'china': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 5}}}}
- Results:
    - Found states:  3600
    - Was able to be completed

2. Test 2
-----
- Two objects:
    - obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
    - obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass', 'object_label': 2}
- Enviroment Size:
    self.x_min = -3
    self.x_max = 4
    self.y_min = -3
    self.y_max = 4
- f_Ethan = {'pref_values': {"cup": {'red': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 41},
                                       'yellow': {'Q1': 45, 'Q2': -1, 'Q3': 100, 'Q4': -1}},
                               'bowl': {'glass': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                        'plastic': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': -1},
                                        'china': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 5}}}}
- Results:
    - Found states:  14112
    - Was able to be completed

**This test also works for if the two objects are the same**

3. Test 3
-----
- Two objects:
    - obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
    - obj_3 = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 3}
- Enviroment Size:
    self.x_min = -3
    self.x_max = 4
    self.y_min = -3
    self.y_max = 4
- f_Ethan = {'pref_values': {"cup": {'red': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 41},
                                       'yellow': {'Q1': 45, 'Q2': -1, 'Q3': 100, 'Q4': -1}},
                               'bowl': {'glass': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                        'plastic': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': -1},
                                        'china': {'Q1': -1, 'Q2': 5, 'Q3': 90, 'Q4': 5}}}}
- Results:
    - Found states:  14112
    - Was able to be completed

4. Test 4
-----
- One object:
    - obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
- Enviroment Size:
    self.x_min = -3
    self.x_max = 4
    self.y_min = -3
    self.y_max = 4
- f_Ethan = {'pref_values': {"cup": {'red': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 41},
                                       'yellow': {'Q1': 45, 'Q2': -1, 'Q3': 100, 'Q4': -1}},
                               'bowl': {'glass': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                        'plastic': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': -1},
                                        'china': {'Q1': -1, 'Q2': 5, 'Q3': 90, 'Q4': 5}}}}
- Results:
    - Found states:  196
    - Was able to be completed

**The number of states found here does match with the number of states visited generated from `reduced_custom_mdp`!**

5. Test 5
-----
- One object:
    - obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
    - obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass', 'object_label': 2}
    - obj_3 = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 3}
- Enviroment Size:
    self.x_min = -2
    self.x_max = 3
    self.y_min = -2
    self.y_max = 3
- f_Ethan = {'pref_values': {"cup": {'red': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 41},
                                       'yellow': {'Q1': 45, 'Q2': -1, 'Q3': 100, 'Q4': -1}},
                               'bowl': {'glass': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                        'plastic': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': -1},
                                        'china': {'Q1': -1, 'Q2': 5, 'Q3': 90, 'Q4': 5}}}}
- Results:
    - Found states:  193200
    - Was NOT able to be completed. Computer crashed when doing vectorized calculations. Even when making the grid smaller.

6. Test 6
-----
- One object:
    - obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
- Enviroment Size:
    self.x_min = -2
    self.x_max = 3
    self.y_min = -2
    self.y_max = 3
- f_Ethan = {'pref_values': {"cup": {'red': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 41},
                                       'yellow': {'Q1': 45, 'Q2': -1, 'Q3': 100, 'Q4': -1}},
                               'bowl': {'glass': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                        'plastic': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': -1},
                                        'china': {'Q1': -1, 'Q2': 5, 'Q3': 90, 'Q4': 5}}}}
- Results:
    - Found states:  100
    - Was able to be completed
