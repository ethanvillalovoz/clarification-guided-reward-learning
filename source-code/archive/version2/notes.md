Version 2 has the ability to handle multiple objects and show when the robot is holding specific object. To add an additional object you must find a photo and create 4 independent changes:

- regular photo
- photo rotated 180 degrees 
- regular photo with red boarders
- photo rotated 180 degrees with red boarders

Additional changes you must make to the code to add a new object to create addition sub-lists for `reward_weights` and `true_f_idx` with the respective preferences you would want.

Finally, you would need to hard code the new images to be able to be rendered in `def render(self, current_state, timestep):`. By following the current format this will work.

Computation time:
---------
As you increase the enviroment limits (grid boundaries), computation time increases very fast. For my laptop, that has the following specs:

- 2.6 GHz 6-Core Intel Core i7
- AMD Radeon Pro 5500M 8 GB, Intel UHD Graphics 630 1536 MB
- 32 GB 2667 MHz DDR4

The largest enviroment settings I was able to achieve was:

```
self.x_min = -2
self.x_max = 3
self.y_min = -2
self.y_max = 3
```

With this, it took a computation time of almost 7 minutes with the following attributes:

- Found states:  14400
- vi iterations: 10


For future updates:
--------

- Create more robust way to adding images for new objects