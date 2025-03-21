from state.State import UserState

class Graph:
    def __init__(self):
        self.state = UserState()

    def transition(self, user_input):
        if self.state.user_type is None:
            # Identify user type (Buyer or Supplier)
            if "negotiate" in user_input.lower():
                self.state.set_user_type("Supplier")
            else:
                self.state.set_user_type("Buyer")

        # Further transitions based on user input can be added here
        return self.state
