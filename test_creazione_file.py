
class SampleClassCreation:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello, {self.name}!" + " " + self.is_adult()

    def is_adult(self):
        if self.age is None:
            raise ValueError("Age is not set")
        elif self.age >= 18:
            return "You are an adult."
        elif self.age < 0:
            raise ValueError("Age cannot be negative")
        else:
            return "You are not an adult."

if __name__ == "__main__":
    saluto = SampleClassCreation("Marco", 25)
    print(saluto.greet())