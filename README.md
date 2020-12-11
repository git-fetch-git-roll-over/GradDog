# GradDog Package
[GradDog Documentation](https://graddog.readthedocs.io/en/latest/)  
The GradDog package does automatic differentiation for humans.  
  
[![codecov](https://codecov.io/gh/git-fetch-git-roll-over/GradDog/branch/master/graph/badge.svg)](https://codecov.io/gh/git-fetch-git-roll-over/GradDog)
[![](https://github.com/git-fetch-git-roll-over/GradDog/workflows/GradDog%20Automatic%20Differentiator/badge.svg)](https://github.com/git-fetch-git-roll-over/GradDog/actions)

## CS107: Systems Development for Computational Sciences
Project members: Ivan Shu, Max Cembalest, Seeam Noor, and Peyton Benac  
Harvard University Fall 2020

## Broader Impact and Inclusivity Statement
The ``GradDog`` package is able to calculate both derivateives through automatic differentiation in both `forward mode` and `reverse mode`. It calculates to machine precision and saves a great amount of computational costs compared to both conventional finite differences and symbolic derivatives methods. However, one downside to note is that ``GradDog`` does not keep track of the mathmatical formula that composes the derivative matrix. If the user were a student, who were trying to use this package for education purpose to understand the process of automatic differentiation, this package might mitigate the overall learning experience. ``GradDog`` is simply designed and developed to provide a convenient avenue to calculate derivatives given any numerical functions. It is meant to act as a small tool to help to solve users' questions.  In writing our documentation and designing our package, we have attempted to reduce the number of assumptions we are making about a user's background.  We do not believe that this package has risks of any major negative impacts, as it does not, for example, replace any existing jobs or access sensitive user information.

The ``GradDog`` package is an open source project and welcomes any contributors from all over the world with different background. The four major developers of ``GradDog`` are either undergraduate or graduate students at Harvard University, an environment that promotes diversity. We will treat every pull request equally, with exactly the same review and approval process. Each time, when a pull request is created by an outside contributor, all the main developers will schedule a time to review it together. We will be making every effort to make sure we are only examing the code based on its idea rather than who initiated the request. If there are any ambiguities or issues about the code, we will reach out to the contributors and make sure to address the misunderstandings or any questions they have. This serves our larger goal of contributing to the movement to make open source code development more inclusive.
