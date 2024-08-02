from operation_nodes import *


t1 = Subtraction("C", Multiplication("A", "B"))
t2 = Subtraction("C", Multiplication("B", "A"))

print(trees_are_equal(t1, t2))


if __name__ == "__main__":
    i = input("> ")
    i = i.strip().replace("{", "(").replace("[", "(").replace("}", ")").replace("]", "]")
    term = eval(i)
    print(term)
    print(term.perform())

