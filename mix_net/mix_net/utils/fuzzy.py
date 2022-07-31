"""This script implements some (base) classes for fuzzy inference.
The hierarhy from bottom up is as follows:
    - FuzzyMembershipFunction
    - FuzzyImplication
    - FuzzyInference

A fuzzy membership function takes in some values and produces a
membership value for a given class.
    Example: Is x a big value? A fuzzy membership function is defined
        That answers this question with a value between 0 and 1 for
        any given x.

A fuzzy implication contains several membership functions, which
are the so called evidences for an implication.
    Example: The color is red, the price is high, is it a Ferrary?
        The implication takes in all the data and evaluates each
        membership function (Is the color red? Is the price high?)
        Based on the resulting values and a combination rule, it
        draws the implication, what is the chance, that the car was
        a Ferrary.

A fuzzy inference takes several fuzzy implications and draws the overall
conclusion. Each implication has its own confidence, whether the car
was a Ferrary. The inference takes these values and determines the overall
conclusion with the help of a defined operation. (e.g. max operation)
"""


class FuzzyMembershipFunction:
    """Base class for implementing a fuzzy membership function
    class for a specific task.

    A Fuzzy Memmbership function has got a form:
        X: A(x) --> [0, 1]
    That is, it is a function that maps the input variable from
    the domain X to the [0, 1] interval, hence saying, up to which
    degree does a point x belong to a class A.

    Hence, what has to be implemented in a child class is a function
    that takes in a value (or tuple of values) x and produces a
    membership value as an output.
    """

    def __call__(self, x):
        """This is the function that has to be implemented.
        It gives back the membership value corresponding to the
        value x.

        args:
            x: dict, that contains the values, based on which
                the membership value is determined.

        returns:
            membership value.
        """
        raise NotImplementedError


class FuzzyImplication:
    """Base class for implementing a fuzzy implication class
    for a specific task.

    A Fuzzy implication has got the form:
        IF U1 is A1 and U2 is A2 ... and Un is An THEN V is B

    therefore, a FuzzyImplication class has got some MembershipFunctions,
    which represent the Ui is Ai evidences and a relation operator, which
    determines the membership value of V in B. (up to which extent does V
    belong to the class B.)

    Hence, the function, that has to be implemented, is the one that takes
    the variable x, which is given to A1 ... An for evaluation and then based
    on the membership functions determines the value of B.
    In the easyest default version, this done by taking the minimum among the
    evidences, that is B = min{A1 is U1, ... , An is Un}.
    """

    def __init__(self, evidences):
        """Initializes a FuzzyImplication object.

        args:
            evidences: list of FuzzyMembershipFunction objects
        """

        self._evidences = evidences

    def __call__(self, x):
        """This is the function that can be overriden on demand.
        It generates the implication based on the observation x.
        In this default version the x observation is given to each
        evidence membership function and then the implication is
        determined by minimum operation.

        args:
            x: dict, that contains all the necessary values for each
                of the membership functions. Every membership function
                gets this x dictionary.
        """

        membership_values = [
            membership_function(x) for membership_function in self._evidences
        ]

        return min(membership_values)


class FuzzyInference:
    """Base class for implementing a Fuzzy Inference class
    for a specific task.

    It takes several Fuzzy Implications and determines the
    overall conclusion. Formally, given:
        IF U11 is A11 and ... and U1n is A1n THEN V is B1
        IF U21 is A21 and ... and U2k is A2k THEN V is B2
        ...
        IF UM1 is AM1 and ... and UMp is AMp THEN V is BM
    and we know the results of all these implications, that is
    we know the Bi values, what is the overall conclusion that
    can be derived from these:
        F(B1, B2, ..., BM) = B

    In the default implementation, the inference is carried out
    by the max operator, that is:
        B = max{B1, B2, ..., BM}
    This can however be overriden in child classes.
    """

    def __init__(self, implications):
        """Initializes a FuzzyInference object.

        args:
            implications: The list of FuzzyImplication objects.
        """

        self._implications = implications

    def __call__(self, x):
        """Carries ot the inferenc based on the observation x.

        In the default implementation it amounts to getting the
        implications from all of the FuzzyImplication objects
        and carrying out a max operation on the results.

        args:
            x: dict, that contains all the necessary values for each
                of the membership functions. Every implication and hence
                every membership function gets this x dictionary.
        """

        implications_results = [implication(x) for implication in self._implications]

        return max(implications_results)
