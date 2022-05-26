# Robotic Autonomy - MAE 6260

Jeronimo Cox - wrr9nb Tyler Haden - tjh8ap May 10, 2022

## Tracking

Autonomous robots are only as useful as their ability to interact with and navigate within the environment around it. However, correct actions can only be performed when the environment is accurately understood by the robot. Due to the dynamic nature of interacting environments, and noisy or error prone sensor data collection, techniques must be developed to ensure accurate belief. [1]
One example of a kind of belief necessary for robotic operation is the tracking of a target object, as in the generalized tracking problem. Objects come in many forms leading to diverse types of motion and ways for the robot to perceive them. A common yet robust method for tracking the location of an object is Recursive Bayesian Estimation (RBE).
We propose using the Extended Kalman Filter as the method for estimation. It can model the location and orientation of a non-linear moving object as a Gaussian probability distribution, and is able to account for the uncertainty of detection from noisy or incomplete sensors.

### Problem Statement

Our project requires the accurate tracking of a 15 cm diameter ball via a depth camera. The ball will be carried by a person while walking around, and may be tossed or rolled around the room. Figures 1 and 2 demonstrate what this might look like. The person won’t be attempting to evade the ball’s detection and we can assume that they know the ball is being detected.
There are several issues that present themselves in this situation. Sometimes the ball is partly or fully occluded, even when it is within frame. This would happen if the person is not directly holding the ball towards the camera, or walks behind an obstacle. The ball’s motion may
take on different modes depending on if it stationary, carried, or tossed.



## Motion Model

Tracking of the ball requires the utilization of a dynamic model to account for the different possibilities of motion of the ball. Two of the most basic motion models similar to a linear random walk but with directional changes are the Ackerman steering model and the differential steering model.
Of the potential motion models that can be used to represent the motion, the Ackerman motion model was selected as it is reflective of a more consistent linear forward motion with gradual change in heading, rather than differential steering which models a more dynamic heading change.

The state for the Ackerman motion model is mapped to the horizontal plane of the 3 dimensional space that the ball might move through. As the state changes at the rate at which the robot/detector is working, equation 1 is used to model the state-to-state change. As the positional rates of change are not reflected in the state (just position), information on the rates of positional change that the detected object is attainable by comparing the k-1 step to step k being currently conducted.

The BU term of the motion model equation 1 is representative of control inputs (equation 2) to the tracked object, represented as U, and how the inputs relate/affect the motion with the B matrix. As the control inputs to the ball and how Tomo or Sid might change the trajectory are unknown, the control inputs are simply neglected (made equal to zero to not produce motion model with control inputs), leaving the AXk−1 matrix and w as motion disturbance, making the motion model Gaussian. The A matrix serves at the State Transition matrix, representative of the inertia and instantaneous motion of the detected object, derived with the Jacobian of the state. The reasoning of A including more factors than an identity matrix is that using an identity matrix as A serves as a way of saying that step to step, there is no expected positional changes, and that the state doesn’t change, which is untrue.
The Jacobian is calculated with the last state recorded, and the predicted state, giving us a 3x3 matrix (for each of the motion model states mapped onto global frame) relating each degree of freedom to the others utilizing the Ackerman steering model.
The Transition matrix A within the motion model equation (equation 1) is calculated as

A=I−J

which includes the identity matrix as the Jacobian is effectively a measure of rate of change, but the transition includes the fact that its the rate of change, from the state of the last time step.

## Sensor Model

Our understanding of the world is limited to the resolution of detail that we can observe. Understanding the limitations of the d435i camera is crucial in identifying how accurate our estimation of the state of tracked ball. The depth camera is able to tell us how far away estimated points on the ball are away from the camera. Along with the positioning of the object within the collected image frame, the depth and the estimated positioning in reference to the camera can be found.

### Camera

The Intel® RealSenseTM Depth Camera D435i uses stereo vision with an infrared projector for depth perception. It additionally has a 2MP resolution RGB camera [2]. The makers of the camera provide ROS wrappers that allow quick integration. It gives the option for the depth and RGB frames to be aligned when published, which allows us to directly correlate per pixel [3]. The default published frame resolution is 640x480, which is precise enough for our use case and is small enough to allow for fast processing.
In order to determine horizontal and vertical positions of the pixels in the frame, we found the angle of spread of the image. We used a protractor centered at the location of the camera to measure angles.
This figure shows our findings. We took measurements on each side and averaged them to get 50° x 43° total. Note that this does not match the technical specs (69° × 42°) which can be attributed to alignment distortion and measurement error.

For a given pixel, we now have its depth D along with apparent yaw θy and pitch θz angle. This lets us calculate the pixel’s position in space relative to the camera. The coordinate axes are oriented such that x faces directly away from the camera and z points up.


### Color Filtering

Due to the ball’s bright color, we could quickly locate the ball within the RGB frame. First, thresholding pixels based on their HSV values. The Hue Saturation Value color space transforms the RGB color space to align relevant color metrics along the axes. It has been shown to provide more information for content-based image retrieval applications [4].
The target balls came in five bright colors: purple, blue, green, orange, and yellow. To most effectively determine the proper HSV thresholds, we conducted a survey of the colors detected on the ball and in the background.
Figure 5: Measured hues of balls vs background.
Figure 5 shows the background hues at the bottom. Above are the different hues for each ball color. It’s clear that some of the balls overlap in hues with the background. This would make it difficult to use this thresholding method. We

found that purple was the least overlapping color, although we selected thresholds for all colors to allow inter-operability. For full thresholds see the color appendix.
Hue Sat. Value Lower 113 35 40
Upper 145 160 240

Once we filtered the pixels based on HSV, we used OpenCV’s findContours() function to find best fit outlines around the ball. This returns the
center coordinates and apparent ball radius.

### Radius Regression

Another form of filtering we use to improve the robustness of detection is fitting the apparent radius and expected radius of the ball in frame. This method allows us to remove many false negatives when the size of the contour is very different to what we should expect at that distance.
We expect the size of the apparent radius R based on distance D to follow the following equation.

The values of a and b can be calculated by regression on data collected with no false positives. We use these values to filter out false positives where the difference is no larger than 15 pixels.


### Probabilistic Sensor

Due to persistent noise associated with pixel and depth accuracy, measured location of the ball is not perfect. The Kalman Filter allows us to represent this uncertainty as a Gaussian centered at the ball. Variances represent the spread of possible actual locations compared to observed location. Co-variances represent how these uncertainties are affected by other dimensions, however these would likely be very small and we only calculate variances.
Distance is measured by the stereoscopic camera. These cameras have inherent error that increases as the distance increases. This error rate increases exponentially [5]. Additionally, the manufacturers list the error as 2% [2]. With these assumptions, we can determine the variance in depth.

At 1 meter, the variance in depth is 2 cm. This matches the 2% error that is listed on the spec sheet. Although it is unclear what they mean. At 5 meters, the variance grows to 34 cm which is over twice the diameter of the ball.
The other two measurements we are making is vertical and horizontal alignment. Pixel distortion and partial occlusion contribute to inaccuracy. For example if a hand is covering part of the ball, the contour might capture only part of the pixels. Additionally, in extreme lighting conditions, the ball’s position might be off.

This would lead to a rather uniform variance along these axis. However there is additional positional error when the ball is close to the edge of the frame. We introduce a polynomial function that acts as a kind of gradual step function from baseline to slightly increased when the ball is detected at the edges.

Where x is a variable that ranges from -1 to 1 representing the yaw or pitch percentage in that direction.


## Bayesian Estimation

We propose the use of an Extended Kalman Filter. While the represented states are Gaussian given the uncertainties including the motion disturbance and limitations of our observation based on resolutions of sensors, the dynamics of our system is non linear. As our model is representative of a non-linear system, the Extended Kalman Filter is selected.

Associated with Kalman Filtering is two tasks that occur each time step, a prediction of how the state of the tracked object will change based on it’s previous motion, and a correction of the predicted state utilizing the observation of the tracked object using whatever selected sensor suite is observed within the sensor model. The prediction step consists of providing a predicted state given the previous state, utilizing the form seen in equation 5, along with a production of a new covariance. The derivation of the next time’s state is produced as a function similar to that of the motion model. Covariance of the predicted step increases from the covariance of the last step, as equation 6 is reflective that uncertainty of the whereabouts of the tracked object increases. The estimated next state given a current state will always be more uncertain than the belief of the last step, as without making an observation to verify the prediction, the prediction belief is only a hypothetical belief with no verification that that belief is what will occur.

For supplying an adjustment to the belief of the state of a tracked object, a correction is made utilizing the most recent observation of the state
of the tracked object. Within the correction step,
the state of the tracked object is corrected from
our prior-made predicted state using the Kalman
gain calculated with equation 7 to reflect the weight-the case that the ball or object of interest being of the correction based on our observation’s potential inaccuracy. With the matrix of Kalman Gains for each of the degrees of freedom of the system, The correction to the belief produced by the prediction can be applied utilizing the following equations to make the correction step.

Using equation 8, the belief of the predicted state is corrected utilizing the observed measurement, in our case the data perceived with the camera. The h function in equation 8 is representative of the transformation between the sensor position and the position of the tracked object. As the location of our sensor is constant, the C matrix is an identity matrix signifying that the state of the camera doesn’t change. With correction of the sate and covariance, the Extended Kalman Filter is ready for state estimation.

## Control Algorithm

The method of control involved leveraging ROS’s open source navigational stack for providing a path planning pipeline between our goal pose provider and our chassis’s drive controller. The Bayesian estimation used to estimate the pose of the ball along with localization provided with rtabmap produced end goal positions for the navigational stack within the globalized or map frame of space. As the goal to be achieved with the robot is to chase the ball

For each step that a prediction is made based on the prior belief of state, an n-step look ahead
is saved. The n-step look ahead is n state predictions ahead of the current time step of what might the object of interest might do given its current state. The n-step look ahead is used for comes out of view. Whether the object leaves from the field of view or the view of the object becomes obstructed by obstacles, an estimated motion given a belief produced by the objects last observed actions gives an estimation to the objects unobserved whereabouts. For however long ahead the actions of the object are predicted, at least some motion may be predicted.

The method used was to input last known positions of the object until the object is unseen, and then to feed the positions of the predicted locations for each time step that the object is unseen, using the last predicted position after using exhausting predicted steps. As the object was assumed to be most likely observable from its last known positions, our implemented chase and search given navigation stack’s spinning feature once goal positions are met makes it possible to look for the ball when the ball isn’t observed. Expecting that the chase and leader robots move at roughly the same speed, no correction for stabilized chassis acceleration is needed other than tuning the max linear and angular acceleration and velocities within a configuration file.

### Minimizing Uncertainty

We already have a notion for uncertainty in ball belief. We have used the magnitude of the covariance matrix representing the Gaussian random variable of the ball location. However this isonlyindirectlyusedtoinferbeliefuncertainty. The true certainty or uncertainty is defined as the entropy associated with the probability density function. In cases where the belief is non Gaussian, covariance can no longer be used. The differential entropy of the density function of all possible states is defined as follows.

However there is a closed form solution for when this probability density function is Gaussian.

The aim of the control algorithm is to minimize this entropy. We propose the strategy of defining a preferred position and orientation relative to the estimated ball belief, that minimizes this entropy. Figure 10 demonstrates how the ball is most likely detected right in front of the robot, where the camera is facing it. This simplifies our control problem down to approaching the ball but stopping short to leave the ball in view.
