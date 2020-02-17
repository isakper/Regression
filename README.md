* Lydelsen för projektet [finner ni här](https://github.com/yourbasic/grudat18/blob/master/individuellt-projekt.md).
* Denna README skall ersättas med en schysst beskrivning av ditt bibliotek.
* Ta hjälp av 
  [Markdown Cheat Sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
  för formatering av dokumentet.


## Package Reg

This package provides a method for creating a predictive model using linear or logistic regression.  

The package include two classes; one for linear regression and one for logistic regression, both of which uses a typical gradient descent method. If your prediction model should predict something continuous (for example pricing of property), the linear regression class is suggested. Logistic regression class is suggested for binary outcomes, such as when trying to classify images.

Package needs numpy

## Class lin_reg_layer:

### __init__(self, num_in, num_out, bias_node = True)
Creates a linear regression layer. Calls method "new_coef" to create matrix of random values aka the coefficients.

        
### new_coef
Creates a matrix with random float values between 0 and 1.

### _add_bias
Adds a column of ones to matrix

### feed_for
Does the feed forward process. Input some data and this function returns predicted value(s) using the coefficients.

### grad_descent
Does a gradient descent step, changing the coefficients.

### coef_ret
returns the coefficients

### cost
Calculates and returns the cost

## Class log_reg_layer:

### __init__
Creates a logistic regression layer. Calls method "new_coef" to create matrix of random values aka the coefficients.

        
### new_coef
Creates a matrix with random float values between 0 and 1.

### _add_bias
Adds a column of ones to matrix

### feed_for
Does the feed forward process. Input some data and this function returns predicted value(s) using the coefficients.

### grad_descent
Does a gradient descent step, changing the coefficients.

### coef_ret
returns the coefficients

### cost
Calculates and returns the cost





        


