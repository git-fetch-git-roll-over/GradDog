Further Extensions
===========================================


Improved performance
-------------------------
We reduced the number of repeated calculations by storing a function's local partial derivatives in a pandas dataframe. However, this approach slowed down our forward mode implementation. So we will focus on an improved data storage and processing method that increases speed.

Higher-order derivatives
-------------------------
The Hessian enables us to calculate second-order derivatives for functions with vector inputs and scalar outputs. However, this project could be extended to include derivatives beyond the second-order, as well as second-order derivatives for functions with vectors outputs as well as scalar outputs.

Increased user flexibility
-------------------------
Currently our package allows users to calculate derivatives without having to specify what type of inputs the function takes or what dimension the funcion is - graddog infers this information from the user-fed function and seed implicitly. To continue this design choice, we could extend our project by broadening the types of inputs our code can can implicitly from the user beyond scalars and iterables.

More user-facing tools
-------------------------
For this iteration of GradDog, we developed tools for the user to explore relevant features of scalar functions such as tangent lines and intervals within which extreme values lie. We intend to develop more functions for the user to visualize other relevant aspects of functions and their derivatives, particularly for functions with 2 or 3 inputs.


