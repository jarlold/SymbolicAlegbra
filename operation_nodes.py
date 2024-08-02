def is_float(a):
    try:
        float(a)
        return True
    except:
        return False

class SyntaxNode:
    def __init__(self, arg1, arg2):
        self.arg1, self.arg2 = arg1, arg2
        self.symbol = None
        self.communative = False

    def __str__(self):
       return "( {} {} {} )".format(self.symbol, str(self.arg1), str(self.arg2)) 

    def equal_to(self, t2):
        if issubclass(self.arg1.__class__, SyntaxNode):


class Addition(SyntaxNode):
    def __init__(self, arg1, arg2):
        super().__init__(arg1, arg2)
        self.symbol = "+"
        self.communative = True

    def perform(self):
        # If they're operations, perform them
        arg1 = self.arg1
        if issubclass(self.arg1.__class__, SyntaxNode):
            arg1 = self.arg1.perform()

        arg2 = self.arg2
        if issubclass(self.arg2.__class__, SyntaxNode):
            arg2 = self.arg2.perform()

        # If both results are numeric, we can actually add them!
        if is_float(str(arg1)) and is_float(str(arg2)):
            return float(arg1) + float(arg2)

        # Otherwise just leave it as symbols
        return Addition(arg1, arg2)

class Subtraction(SyntaxNode):
    def __init__(self, arg1, arg2):
        super().__init__(arg1, arg2)
        self.symbol = "-"

    def perform(self):
        # If they're operations, perform them
        arg1 = self.arg1
        if issubclass(self.arg1.__class__, SyntaxNode):
            arg1 = self.arg1.perform()

        arg2 = self.arg2
        if issubclass(self.arg2.__class__, SyntaxNode):
            arg2 = self.arg2.perform()

        # If both results are numeric, we can actually add them!
        if is_float(str(arg1)) and is_float(str(arg2)):
            return float(arg1) - float(arg2)

        if arg1 == arg2:
            return 0

        # Otherwise just leave it as symbols
        return Subtraction(arg1, arg2)


class Multiplication(SyntaxNode):
    def __init__(self, arg1, arg2):
        super().__init__(arg1, arg2)
        self.symbol = "*"
        self.communative = True

    def perform(self):
        # If they're operations, perform them
        arg1 = self.arg1
        if issubclass(self.arg1.__class__, SyntaxNode):
            arg1 = self.arg1.perform()

        arg2 = self.arg2
        if issubclass(self.arg2.__class__, SyntaxNode):
            arg2 = self.arg2.perform()

        # If both results are numeric, we can actually multiply them!
        if is_float(str(arg1)) and is_float(str(arg2)):
            return float(arg1) * float(arg2)

        if arg1 == 0 or arg2 == 0:
            return 0

        # Otherwise just leave it as symbols
        return Multiplication(arg1, arg2)


class Power(SyntaxNode):
    def __init__(self, arg1, arg2):
        super().__init__(arg1, arg2)
        self.symbol = "^"

    def perform(self):
        # If they're operations, perform them
        arg1 = self.arg1
        if issubclass(self.arg1.__class__, SyntaxNode):
            arg1 = self.arg1.perform()

        arg2 = self.arg2
        if issubclass(self.arg2.__class__, SyntaxNode):
            arg2 = self.arg2.perform()

        # If both results are numeric, we can actually multiply them!
        if is_float(str(arg1)) and is_float(str(arg2)):
            return float(arg1) ** float(arg2)

        # Zero to the power of anything is 0
        if arg1 == 0:
            return 0

        # 1 to the power of anything is 1
        if arg1 == 1:
            return 1

        # Anything to the power of 0 is 1
        if arg2 == 0:
            return 1

        # Anything to the power of 1 is itself
        if arg2 == 1:
            return arg1

        # Otherwise just leave it as symbols
        return Power(arg1, arg2)


